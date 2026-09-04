use candle_core::{Device, Result, Tensor};
use std::cmp::Ordering;
use std::time::{Duration, Instant};

const DEFAULT_WARMUP: usize = 20;
const DEFAULT_ITERS: usize = 100;

#[derive(Clone, Copy)]
enum CaseKind {
    ConvTranspose1D {
        batch: usize,
        c_in: usize,
        c_out: usize,
        len: usize,
        kernel: usize,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    },
    ConvTranspose2D {
        batch: usize,
        c_in: usize,
        c_out: usize,
        h: usize,
        w: usize,
        kernel: usize,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    },
}

#[derive(Clone, Copy)]
struct BenchCase {
    name: &'static str,
    kind: CaseKind,
}

#[derive(Clone, Copy)]
enum Path {
    Legacy,
    RawCuda,
    Cudnn,
}

impl Path {
    fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::RawCuda => "raw_cuda",
            Self::Cudnn => "cudnn",
        }
    }
}

struct EnvGuard {
    saved: Vec<(&'static str, Option<String>)>,
}

impl EnvGuard {
    fn for_path(path: Path) -> Self {
        const KEYS: [&str; 3] = [
            "CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL",
            "CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT",
            "CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT",
        ];
        let saved = KEYS
            .iter()
            .map(|&key| (key, std::env::var(key).ok()))
            .collect::<Vec<_>>();
        for key in KEYS {
            std::env::remove_var(key);
        }
        match path {
            Path::Legacy => {}
            Path::RawCuda => {
                std::env::set_var("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL", "1");
                std::env::set_var("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT", "1");
            }
            Path::Cudnn => {
                std::env::set_var("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT", "1");
            }
        }
        Self { saved }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in self.saved.drain(..) {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

fn deterministic(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * mul) % 101) as isize + bias) as f32 / 64.0)
        .collect()
}

fn tensors(case: BenchCase, device: &Device) -> Result<(Tensor, Tensor)> {
    match case.kind {
        CaseKind::ConvTranspose1D {
            batch,
            c_in,
            c_out,
            len,
            kernel,
            groups,
            ..
        } => {
            let x = Tensor::from_vec(
                deterministic(batch * c_in * len, 37, -50),
                (batch, c_in, len),
                device,
            )?;
            let c_out_group = c_out / groups;
            let k = Tensor::from_vec(
                deterministic(c_in * c_out_group * kernel, 53, -50),
                (c_in, c_out_group, kernel),
                device,
            )?;
            Ok((x, k))
        }
        CaseKind::ConvTranspose2D {
            batch,
            c_in,
            c_out,
            h,
            w,
            kernel,
            groups,
            ..
        } => {
            let x = Tensor::from_vec(
                deterministic(batch * c_in * h * w, 37, -50),
                (batch, c_in, h, w),
                device,
            )?;
            let c_out_group = c_out / groups;
            let k = Tensor::from_vec(
                deterministic(c_in * c_out_group * kernel * kernel, 53, -50),
                (c_in, c_out_group, kernel, kernel),
                device,
            )?;
            Ok((x, k))
        }
    }
}

fn legacy(case: BenchCase, x: &Tensor, kernel: &Tensor) -> Result<Tensor> {
    match case.kind {
        CaseKind::ConvTranspose1D {
            padding,
            output_padding,
            stride,
            dilation,
            groups,
            ..
        } => {
            let xs = x.chunk(groups, 1)?;
            let ks = kernel.chunk(groups, 0)?;
            let ys = xs
                .iter()
                .zip(&ks)
                .map(|(xg, kg)| {
                    xg.conv_transpose1d(kg, padding, output_padding, stride, dilation, 1)
                })
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&ys, 1)
        }
        CaseKind::ConvTranspose2D {
            padding,
            output_padding,
            stride,
            dilation,
            groups,
            ..
        } => {
            let xs = x.chunk(groups, 1)?;
            let ks = kernel.chunk(groups, 0)?;
            let ys = xs
                .iter()
                .zip(&ks)
                .map(|(xg, kg)| {
                    xg.conv_transpose2d(kg, padding, output_padding, stride, dilation)
                })
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&ys, 1)
        }
    }
}

fn native(case: BenchCase, x: &Tensor, kernel: &Tensor) -> Result<Tensor> {
    match case.kind {
        CaseKind::ConvTranspose1D {
            padding,
            output_padding,
            stride,
            dilation,
            groups,
            ..
        } => x.conv_transpose1d(kernel, padding, output_padding, stride, dilation, groups),
        CaseKind::ConvTranspose2D {
            padding,
            output_padding,
            stride,
            dilation,
            groups,
            ..
        } => x.conv_transpose2d_with_groups(
            kernel,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        ),
    }
}

fn run_path(path: Path, case: BenchCase, x: &Tensor, kernel: &Tensor) -> Result<Tensor> {
    let _guard = EnvGuard::for_path(path);
    match path {
        Path::Legacy => legacy(case, x, kernel),
        Path::RawCuda | Path::Cudnn => native(case, x, kernel),
    }
}

fn max_abs_rel(lhs: &Tensor, rhs: &Tensor) -> Result<(f32, f32)> {
    if lhs.dims() != rhs.dims() {
        candle_core::bail!(
            "shape mismatch in benchmark parity: {:?} vs {:?}",
            lhs.dims(),
            rhs.dims()
        )
    }
    let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
    let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (&a, &b) in lhs.iter().zip(&rhs) {
        let abs = (a - b).abs();
        let rel = abs / b.abs().max(1e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    Ok((max_abs, max_rel))
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn measure(
    path: Path,
    case: BenchCase,
    x: &Tensor,
    kernel: &Tensor,
    device: &Device,
    warmup: usize,
    iters: usize,
) -> Result<(Duration, Duration, Duration)> {
    let _guard = EnvGuard::for_path(path);
    for _ in 0..warmup {
        let y = match path {
            Path::Legacy => legacy(case, x, kernel)?,
            Path::RawCuda | Path::Cudnn => native(case, x, kernel)?,
        };
        device.synchronize()?;
        std::hint::black_box(y);
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        device.synchronize()?;
        let start = Instant::now();
        let y = match path {
            Path::Legacy => legacy(case, x, kernel)?,
            Path::RawCuda | Path::Cudnn => native(case, x, kernel)?,
        };
        device.synchronize()?;
        let elapsed = start.elapsed();
        std::hint::black_box(y);
        samples.push(elapsed);
    }
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Ok((
        percentile(&samples, 0.50),
        percentile(&samples, 0.10),
        percentile(&samples, 0.90),
    ))
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn parse_count(flag: &str, default: usize) -> usize {
    let args = std::env::args().collect::<Vec<_>>();
    args.windows(2)
        .find_map(|pair| (pair[0] == flag).then(|| pair[1].parse::<usize>().ok()).flatten())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    let warmup = parse_count("--warmup", DEFAULT_WARMUP);
    let iters = parse_count("--iters", DEFAULT_ITERS);
    if warmup == 0 || iters == 0 {
        candle_core::bail!("--warmup and --iters must both be greater than zero")
    }

    let device = Device::new_cuda(0)?;
    let cases = [
        BenchCase {
            name: "convt1d-g2",
            kind: CaseKind::ConvTranspose1D {
                batch: 1,
                c_in: 64,
                c_out: 96,
                len: 128,
                kernel: 3,
                padding: 1,
                output_padding: 1,
                stride: 2,
                dilation: 1,
                groups: 2,
            },
        },
        BenchCase {
            name: "convt1d-g8",
            kind: CaseKind::ConvTranspose1D {
                batch: 1,
                c_in: 64,
                c_out: 64,
                len: 128,
                kernel: 3,
                padding: 1,
                output_padding: 1,
                stride: 2,
                dilation: 1,
                groups: 8,
            },
        },
        BenchCase {
            name: "convt1d-depthwise-g64",
            kind: CaseKind::ConvTranspose1D {
                batch: 1,
                c_in: 64,
                c_out: 64,
                len: 128,
                kernel: 3,
                padding: 1,
                output_padding: 1,
                stride: 2,
                dilation: 1,
                groups: 64,
            },
        },
        BenchCase {
            name: "convt2d-g2",
            kind: CaseKind::ConvTranspose2D {
                batch: 1,
                c_in: 64,
                c_out: 96,
                h: 32,
                w: 32,
                kernel: 3,
                padding: 1,
                output_padding: 1,
                stride: 2,
                dilation: 1,
                groups: 2,
            },
        },
        BenchCase {
            name: "convt2d-g8",
            kind: CaseKind::ConvTranspose2D {
                batch: 1,
                c_in: 64,
                c_out: 64,
                h: 32,
                w: 32,
                kernel: 3,
                padding: 1,
                output_padding: 1,
                stride: 2,
                dilation: 1,
                groups: 8,
            },
        },
        BenchCase {
            name: "convt2d-depthwise-g64",
            kind: CaseKind::ConvTranspose2D {
                batch: 1,
                c_in: 64,
                c_out: 64,
                h: 32,
                w: 32,
                kernel: 3,
                padding: 1,
                output_padding: 1,
                stride: 2,
                dilation: 1,
                groups: 64,
            },
        },
    ];

    println!("=== GROUPED TRANSPOSE BENCH V4-B ===");
    println!("device={:?}", device.location());
    println!("dtype=f32");
    println!("warmup={warmup}");
    println!("iters={iters}");
    println!(
        "cuda_compute_cap={}",
        std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "auto".into())
    );
    println!("measurement=api_wall_time_with_device_sync");

    let mut all_pass = true;
    for case in cases {
        let (x, kernel) = tensors(case, &device)?;
        let reference = run_path(Path::Legacy, case, &x, &kernel)?;
        device.synchronize()?;
        let raw = run_path(Path::RawCuda, case, &x, &kernel)?;
        device.synchronize()?;
        let cudnn = run_path(Path::Cudnn, case, &x, &kernel)?;
        device.synchronize()?;

        let (raw_abs, raw_rel) = max_abs_rel(&raw, &reference)?;
        let (cudnn_abs, cudnn_rel) = max_abs_rel(&cudnn, &reference)?;
        let raw_parity = raw_abs <= 1e-4 || raw_rel <= 1e-4;
        let cudnn_parity = cudnn_abs <= 1e-4 || cudnn_rel <= 1e-4;
        all_pass &= raw_parity && cudnn_parity;

        let (legacy_med, legacy_p10, legacy_p90) = measure(
            Path::Legacy,
            case,
            &x,
            &kernel,
            &device,
            warmup,
            iters,
        )?;
        let (raw_med, raw_p10, raw_p90) = measure(
            Path::RawCuda,
            case,
            &x,
            &kernel,
            &device,
            warmup,
            iters,
        )?;
        let (cudnn_med, cudnn_p10, cudnn_p90) = measure(
            Path::Cudnn,
            case,
            &x,
            &kernel,
            &device,
            warmup,
            iters,
        )?;

        println!();
        println!("CASE {}", case.name);
        println!(
            "{} median_ms={:.6} p10_ms={:.6} p90_ms={:.6}",
            Path::Legacy.name(),
            ms(legacy_med),
            ms(legacy_p10),
            ms(legacy_p90)
        );
        println!(
            "{} median_ms={:.6} p10_ms={:.6} p90_ms={:.6} speedup={:.3}x max_abs={:.8} max_rel={:.8} parity={}",
            Path::RawCuda.name(),
            ms(raw_med),
            ms(raw_p10),
            ms(raw_p90),
            ms(legacy_med) / ms(raw_med),
            raw_abs,
            raw_rel,
            raw_parity
        );
        println!(
            "{} median_ms={:.6} p10_ms={:.6} p90_ms={:.6} speedup={:.3}x max_abs={:.8} max_rel={:.8} parity={}",
            Path::Cudnn.name(),
            ms(cudnn_med),
            ms(cudnn_p10),
            ms(cudnn_p90),
            ms(legacy_med) / ms(cudnn_med),
            cudnn_abs,
            cudnn_rel,
            cudnn_parity
        );
    }

    println!();
    println!("STATUS={}", if all_pass { "PASS" } else { "FAIL" });
    if !all_pass {
        candle_core::bail!("grouped transpose benchmark numerical parity failed")
    }
    Ok(())
}
