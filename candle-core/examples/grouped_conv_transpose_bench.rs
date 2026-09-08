use candle_core::{Device, Result, Tensor};
use std::cmp::Ordering;
use std::time::Instant;

const DEFAULT_WARMUP: usize = 8;
const DEFAULT_ITERS: usize = 40;
const DEFAULT_INNER: usize = 32;
const DEFAULT_MAX_DRIFT_PCT: f64 = 5.0;
const DEFAULT_PROMOTION_MARGIN_PCT: f64 = 10.0;
const GROUPS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

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

impl CaseKind {
    fn dim(self) -> &'static str {
        match self {
            Self::ConvTranspose1D { .. } => "1d",
            Self::ConvTranspose2D { .. } => "2d",
        }
    }

    fn groups(self) -> usize {
        match self {
            Self::ConvTranspose1D { groups, .. } | Self::ConvTranspose2D { groups, .. } => groups,
        }
    }
}

#[derive(Clone, Copy)]
struct BenchCase {
    name: &'static str,
    kind: CaseKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    Legacy,
    RawCuda,
    Cudnn,
}

impl Path {
    fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::RawCuda => "raw",
            Self::Cudnn => "cudnn",
        }
    }
}

struct EnvGuard {
    saved: Vec<(&'static str, Option<String>)>,
}

impl EnvGuard {
    fn for_path(path: Path) -> Self {
        const KEYS: [&str; 4] = [
            "CANDLE_GROUPED_TRANSPOSE_DISPATCH",
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
                std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "raw");
                std::env::set_var("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT", "1");
            }
            Path::Cudnn => {
                std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "cudnn");
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

#[derive(Clone, Copy)]
struct TimingStats {
    median_us: f64,
    p10_us: f64,
    p90_us: f64,
}

impl TimingStats {
    fn print(self, phase: &str, path: Path) {
        println!(
            "PHASE phase={} path={} median_us={:.6} p10_us={:.6} p90_us={:.6}",
            phase,
            path.name(),
            self.median_us,
            self.p10_us,
            self.p90_us
        );
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
                .map(|(xg, kg)| xg.conv_transpose2d(kg, padding, output_padding, stride, dilation))
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

fn execute(path: Path, case: BenchCase, x: &Tensor, kernel: &Tensor) -> Result<Tensor> {
    match path {
        Path::Legacy => legacy(case, x, kernel),
        Path::RawCuda | Path::Cudnn => native(case, x, kernel),
    }
}

fn run_path(path: Path, case: BenchCase, x: &Tensor, kernel: &Tensor) -> Result<Tensor> {
    let _guard = EnvGuard::for_path(path);
    execute(path, case, x, kernel)
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

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn batched_launches(
    path: Path,
    case: BenchCase,
    x: &Tensor,
    kernel: &Tensor,
    inner: usize,
) -> Result<Vec<Tensor>> {
    let mut outputs = Vec::with_capacity(inner);
    for _ in 0..inner {
        outputs.push(execute(path, case, x, kernel)?);
    }
    Ok(outputs)
}

fn measure_batched(
    path: Path,
    case: BenchCase,
    x: &Tensor,
    kernel: &Tensor,
    device: &Device,
    warmup: usize,
    iters: usize,
    inner: usize,
) -> Result<TimingStats> {
    let _guard = EnvGuard::for_path(path);

    for _ in 0..warmup {
        let outputs = batched_launches(path, case, x, kernel, inner)?;
        device.synchronize()?;
        std::hint::black_box(outputs);
    }

    let mut samples_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        device.synchronize()?;
        let start = Instant::now();
        let outputs = batched_launches(path, case, x, kernel, inner)?;
        device.synchronize()?;
        let per_launch_us = start.elapsed().as_secs_f64() * 1_000_000.0 / inner as f64;
        std::hint::black_box(outputs);
        samples_us.push(per_launch_us);
    }

    samples_us.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Ok(TimingStats {
        median_us: percentile(&samples_us, 0.50),
        p10_us: percentile(&samples_us, 0.10),
        p90_us: percentile(&samples_us, 0.90),
    })
}

fn parse_count(flag: &str, default: usize) -> usize {
    let args = std::env::args().collect::<Vec<_>>();
    args.windows(2)
        .find_map(|pair| {
            (pair[0] == flag)
                .then(|| pair[1].parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(default)
}

fn parse_f64(flag: &str, default: f64) -> f64 {
    let args = std::env::args().collect::<Vec<_>>();
    args.windows(2)
        .find_map(|pair| {
            (pair[0] == flag)
                .then(|| pair[1].parse::<f64>().ok())
                .flatten()
        })
        .unwrap_or(default)
}

fn has_flag(flag: &str) -> bool {
    std::env::args().any(|arg| arg == flag)
}

fn cases() -> Vec<BenchCase> {
    let mut cases = Vec::with_capacity(GROUPS.len() * 2);
    for &groups in &GROUPS {
        let name = match groups {
            1 => "convt1d-g1",
            2 => "convt1d-g2",
            4 => "convt1d-g4",
            8 => "convt1d-g8",
            16 => "convt1d-g16",
            32 => "convt1d-g32",
            64 => "convt1d-g64-depthwise",
            _ => unreachable!(),
        };
        cases.push(BenchCase {
            name,
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
                groups,
            },
        });
    }
    for &groups in &GROUPS {
        let name = match groups {
            1 => "convt2d-g1",
            2 => "convt2d-g2",
            4 => "convt2d-g4",
            8 => "convt2d-g8",
            16 => "convt2d-g16",
            32 => "convt2d-g32",
            64 => "convt2d-g64-depthwise",
            _ => unreachable!(),
        };
        cases.push(BenchCase {
            name,
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
                groups,
            },
        });
    }
    cases
}

fn stabilized_target(case: BenchCase) -> bool {
    let groups = case.kind.groups();
    match case.kind.dim() {
        "1d" => groups == 1 || groups == 2,
        "2d" => groups == 1 || groups == 16 || groups == 32 || groups == 64,
        _ => false,
    }
}

fn median3(a: f64, b: f64, c: f64) -> f64 {
    let mut values = [a, b, c];
    values.sort_unstable_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));
    values[1]
}

fn relative_drift_pct(pre: f64, post: f64) -> f64 {
    let center = (pre + post) * 0.5;
    if center == 0.0 {
        0.0
    } else {
        (post - pre).abs() / center * 100.0
    }
}

fn winner(raw_us: f64, cudnn_us: f64) -> Path {
    if raw_us < cudnn_us {
        Path::RawCuda
    } else {
        Path::Cudnn
    }
}

fn main() -> Result<()> {
    let warmup = parse_count("--warmup", DEFAULT_WARMUP);
    let iters = parse_count("--iters", DEFAULT_ITERS);
    let inner = parse_count("--inner", DEFAULT_INNER);
    let max_drift_pct = parse_f64("--max-drift-pct", DEFAULT_MAX_DRIFT_PCT);
    let promotion_margin_pct = parse_f64("--promotion-margin-pct", DEFAULT_PROMOTION_MARGIN_PCT);
    let full_frontier = has_flag("--full-frontier");

    if warmup == 0 || iters == 0 || inner == 0 {
        candle_core::bail!("--warmup, --iters and --inner must all be greater than zero")
    }
    if max_drift_pct < 0.0 || promotion_margin_pct < 0.0 {
        candle_core::bail!("drift and promotion margin percentages must be non-negative")
    }

    let device = Device::new_cuda(0)?;
    println!("=== GROUPED TRANSPOSE BENCH V4-B2.1 STABILIZED A/B/A FRONTIER ===");
    println!("device={:?}", device.location());
    println!("dtype=f32");
    println!("warmup_samples={warmup}");
    println!("timed_samples={iters}");
    println!("launches_per_sample={inner}");
    println!("max_drift_pct={max_drift_pct:.3}");
    println!("promotion_margin_pct={promotion_margin_pct:.3}");
    println!("full_frontier={full_frontier}");
    println!(
        "cuda_compute_cap={}",
        std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "auto".into())
    );
    println!("measurement=batched_api_wall_time_one_sync_per_sample");
    println!("sequence=raw_a,cudnn_b,raw_a2,cudnn_a,raw_b,cudnn_a2");
    println!("auto_dispatch_rule_modified=false");

    let selected_cases = cases()
        .into_iter()
        .filter(|case| full_frontier || stabilized_target(*case))
        .collect::<Vec<_>>();

    let mut numerical_pass = true;
    for case in selected_cases {
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
        let parity = raw_parity && cudnn_parity;
        numerical_pass &= parity;

        println!();
        println!("CASE {}", case.name);
        println!(
            "PARITY raw_abs={:.8} raw_rel={:.8} cudnn_abs={:.8} cudnn_rel={:.8} pass={}",
            raw_abs, raw_rel, cudnn_abs, cudnn_rel, parity
        );

        let raw_a = measure_batched(
            Path::RawCuda,
            case,
            &x,
            &kernel,
            &device,
            warmup,
            iters,
            inner,
        )?;
        let cudnn_b = measure_batched(
            Path::Cudnn,
            case,
            &x,
            &kernel,
            &device,
            warmup,
            iters,
            inner,
        )?;
        let raw_a2 = measure_batched(
            Path::RawCuda,
            case,
            &x,
            &kernel,
            &device,
            warmup,
            iters,
            inner,
        )?;

        let cudnn_a = measure_batched(
            Path::Cudnn,
            case,
            &x,
            &kernel,
            &device,
            warmup,
            iters,
            inner,
        )?;
        let raw_b = measure_batched(
            Path::RawCuda,
            case,
            &x,
            &kernel,
            &device,
            warmup,
            iters,
            inner,
        )?;
        let cudnn_a2 = measure_batched(
            Path::Cudnn,
            case,
            &x,
            &kernel,
            &device,
            warmup,
            iters,
            inner,
        )?;

        raw_a.print("raw_a", Path::RawCuda);
        cudnn_b.print("cudnn_b", Path::Cudnn);
        raw_a2.print("raw_a2", Path::RawCuda);
        cudnn_a.print("cudnn_a", Path::Cudnn);
        raw_b.print("raw_b", Path::RawCuda);
        cudnn_a2.print("cudnn_a2", Path::Cudnn);

        let raw_consensus_us = median3(raw_a.median_us, raw_b.median_us, raw_a2.median_us);
        let cudnn_consensus_us =
            median3(cudnn_a.median_us, cudnn_b.median_us, cudnn_a2.median_us);
        let raw_p90_us = median3(raw_a.p90_us, raw_b.p90_us, raw_a2.p90_us);
        let cudnn_p90_us = median3(cudnn_a.p90_us, cudnn_b.p90_us, cudnn_a2.p90_us);

        let raw_drift_pct = relative_drift_pct(raw_a.median_us, raw_a2.median_us);
        let cudnn_drift_pct = relative_drift_pct(cudnn_a.median_us, cudnn_a2.median_us);
        let drift_pass =
            raw_drift_pct <= max_drift_pct && cudnn_drift_pct <= max_drift_pct;

        let consensus_winner = winner(raw_consensus_us, cudnn_consensus_us);
        let round1_winner = winner(raw_a.median_us, cudnn_a.median_us);
        let round2_winner = winner(raw_b.median_us, cudnn_b.median_us);
        let round3_winner = winner(raw_a2.median_us, cudnn_a2.median_us);
        let direction_consistent = round1_winner == consensus_winner
            && round2_winner == consensus_winner
            && round3_winner == consensus_winner;

        let winner_speedup = if consensus_winner == Path::RawCuda {
            cudnn_consensus_us / raw_consensus_us
        } else {
            raw_consensus_us / cudnn_consensus_us
        };
        let gain_pct = (winner_speedup - 1.0) * 100.0;
        let margin_pass = gain_pct >= promotion_margin_pct;

        let p90_speedup = if consensus_winner == Path::RawCuda {
            cudnn_p90_us / raw_p90_us
        } else {
            raw_p90_us / cudnn_p90_us
        };
        let p90_non_regression = p90_speedup >= 1.0;

        let promotion_gate =
            parity && drift_pass && direction_consistent && margin_pass && p90_non_regression;

        println!(
            "STABILIZED_FRONTIER dim={} groups={} winner={} raw_us={:.6} cudnn_us={:.6} winner_speedup={:.4}x gain_pct={:.3} raw_drift_pct={:.3} cudnn_drift_pct={:.3} direction_consistent={} p90_speedup={:.4}x parity={} promotion_gate={}",
            case.kind.dim(),
            case.kind.groups(),
            consensus_winner.name(),
            raw_consensus_us,
            cudnn_consensus_us,
            winner_speedup,
            gain_pct,
            raw_drift_pct,
            cudnn_drift_pct,
            direction_consistent,
            p90_speedup,
            parity,
            if promotion_gate { "PASS" } else { "HOLD" }
        );
        println!(
            "GATE dim={} groups={} drift_pass={} margin_pass={} p90_non_regression={} max_drift_pct={:.3} required_gain_pct={:.3}",
            case.kind.dim(),
            case.kind.groups(),
            drift_pass,
            margin_pass,
            p90_non_regression,
            max_drift_pct,
            promotion_margin_pct
        );
    }

    println!();
    println!("STATUS={}", if numerical_pass { "PASS" } else { "FAIL" });
    if !numerical_pass {
        candle_core::bail!("grouped transpose stabilized frontier numerical parity failed")
    }
    Ok(())
}
