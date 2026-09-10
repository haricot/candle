use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::{CudaStorage, WrapErr};
use candle_core::{CpuStorage, CustomOp2, DType, Device, Layout, Result, Shape, Tensor};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use std::cmp::Ordering;
use std::time::{Duration, Instant};

const DEFAULT_WARMUP_MS: f64 = 500.0;
const DEFAULT_ITERS: usize = 40;
const DEFAULT_INNER: usize = 32;
const DEFAULT_MAX_DRIFT_PCT: f64 = 5.0;
const DEFAULT_MIN_INTEGRATED_SPEEDUP_X: f64 = 1.10;

#[derive(Clone, Copy, Debug)]
struct Case {
    c: usize,
    h: usize,
    w: usize,
}

const CASES: [Case; 4] = [
    Case {
        c: 48,
        h: 64,
        w: 48,
    },
    Case {
        c: 96,
        h: 32,
        w: 24,
    },
    Case {
        c: 192,
        h: 16,
        w: 12,
    },
    Case { c: 384, h: 8, w: 6 },
];

#[derive(Clone, Copy, Debug)]
enum Mode {
    Current,
    AsdCandidate,
}

impl Mode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::AsdCandidate => "asd_candidate",
        }
    }
}

#[derive(Clone, Copy)]
struct TimingStats {
    median_us: f64,
    p10_us: f64,
    p90_us: f64,
}

struct RawExactDw5x5;

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

fn deterministic(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * mul) % 127) as isize + bias) as f32 / 128.0)
        .collect()
}

fn tensors(case: Case, device: &Device) -> Result<(Tensor, Tensor)> {
    let x = Tensor::from_vec(
        deterministic(case.c * case.h * case.w, 37, -63),
        (1, case.c, case.h, case.w),
        device,
    )?;
    let kernel = Tensor::from_vec(
        deterministic(case.c * 25, 53, -63),
        (case.c, 1, 5, 5),
        device,
    )?;
    Ok((x, kernel))
}

fn exact_call(
    case: Case,
    input_contiguous: bool,
    input_start_offset: usize,
    weight_contiguous: bool,
    weight_start_offset: usize,
) -> candle_kernels::asd_exact_conv2d::ExactConv2dCall {
    candle_kernels::asd_exact_conv2d::ExactConv2dCall {
        batch: 1,
        c_in: case.c,
        c_out: case.c,
        spatial0: case.h,
        spatial1: case.w,
        weight0: case.c,
        weight1: 1,
        weight2: 5,
        weight3: 5,
        groups: case.c,
        kernel: 5,
        stride: 1,
        padding: 2,
        dilation: 1,
        dtype: "f32",
        input_contiguous,
        input_start_offset,
        weight_contiguous,
        weight_start_offset,
    }
}

fn lookup(case: Case) -> Option<candle_kernels::asd_exact_conv2d::ExactAsdMatch> {
    candle_kernels::asd_exact_conv2d::lookup(exact_call(case, true, 0, true, 0))
}

impl CustomOp2 for RawExactDw5x5 {
    fn name(&self) -> &'static str {
        "asd-v2-exact-dw5x5-raw-timed"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_l: &Layout,
        _kernel: &CpuStorage,
        _kernel_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("exact DW5x5 ASD timed validation is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_l: &Layout,
        kernel: &CudaStorage,
        kernel_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        if input.dtype() != DType::F32 || kernel.dtype() != DType::F32 {
            candle_core::bail!("exact DW5x5 ASD raw candidate requires f32")
        }
        if !input_l.is_contiguous()
            || input_l.start_offset() != 0
            || !kernel_l.is_contiguous()
            || kernel_l.start_offset() != 0
        {
            candle_core::bail!("exact DW5x5 ASD raw candidate requires contiguous_zero_offset")
        }
        let dims = input_l.dims();
        let wdims = kernel_l.dims();
        if dims.len() != 4 || wdims.len() != 4 {
            candle_core::bail!("exact DW5x5 ASD raw candidate rank mismatch")
        }
        let case = Case {
            c: dims[1],
            h: dims[2],
            w: dims[3],
        };
        let Some(matched) = lookup(case) else {
            candle_core::bail!("raw exact custom op called for an ASD V2 domain miss")
        };
        if !candle_kernels::asd_exact_conv2d::VALIDATION_BUILD {
            candle_core::bail!("ASD V2 candidate requires validation build")
        }
        if matched.selected_backend != "raw_cuda" || dims[0] != 1 || wdims != [case.c, 1, 5, 5] {
            candle_core::bail!("exact DW5x5 ASD raw candidate contract mismatch")
        }

        let dev = input.device().clone();
        let src = input.as_cuda_slice::<f32>()?;
        let weight = kernel.as_cuda_slice::<f32>()?;
        let out = unsafe { dev.alloc::<f32>(case.c * case.h * case.w)? };
        let func = dev.get_or_load_func("asd_dw5x5_conv2d_f32", &candle_kernels::ASD_DW5X5)?;
        let cfg = LaunchConfig {
            grid_dim: (
                case.w.div_ceil(16) as u32,
                case.h.div_ceil(8) as u32,
                case.c as u32,
            ),
            block_dim: (16, 8, 1),
            shared_mem_bytes: 0,
        };
        let channels = case.c as u64;
        let height = case.h as u64;
        let width = case.w as u64;
        let mut builder = func.builder();
        builder.arg(&channels);
        builder.arg(&height);
        builder.arg(&width);
        builder.arg(src);
        builder.arg(weight);
        builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(out, dev),
            Shape::from((1, case.c, case.h, case.w)),
        ))
    }
}

fn current(x: &Tensor, kernel: &Tensor, groups: usize) -> Result<Tensor> {
    x.conv2d(kernel, 2, 1, 1, groups)
}

fn candidate(case: Case, x: &Tensor, kernel: &Tensor, trace: bool) -> Result<Tensor> {
    if let Some(matched) = lookup(case) {
        if trace {
            println!(
                "SELECT mode=asd_candidate selected=raw reason=exact_asd policy_id={} decision_id={} state={}",
                matched.policy_id, matched.decision_id, matched.state
            );
        }
        let out = x.apply_op2_no_bwd(kernel, &RawExactDw5x5)?;
        if trace {
            println!(
                "SUBMIT mode=asd_candidate submitted_backend=raw launch_submission=success policy_id={} decision_id={}",
                matched.policy_id, matched.decision_id
            );
        }
        Ok(out)
    } else {
        if trace {
            println!("SELECT mode=asd_candidate selected=current reason=exact_miss");
        }
        let out = current(x, kernel, case.c)?;
        if trace {
            println!(
                "SUBMIT mode=asd_candidate submitted_backend=current launch_submission=success"
            );
        }
        Ok(out)
    }
}

fn run(mode: Mode, case: Case, x: &Tensor, kernel: &Tensor, trace: bool) -> Result<Tensor> {
    match mode {
        Mode::Current => {
            if trace {
                println!("SELECT mode=current selected=current reason=reference");
            }
            let out = current(x, kernel, case.c)?;
            if trace {
                println!("SUBMIT mode=current submitted_backend=current launch_submission=success");
            }
            Ok(out)
        }
        Mode::AsdCandidate => candidate(case, x, kernel, trace),
    }
}

fn max_abs_rel(lhs: &Tensor, rhs: &Tensor) -> Result<(f32, f32)> {
    if lhs.dims() != rhs.dims() {
        candle_core::bail!("shape mismatch: {:?} vs {:?}", lhs.dims(), rhs.dims())
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
    mode: Mode,
    case: Case,
    x: &Tensor,
    kernel: &Tensor,
    inner: usize,
) -> Result<Vec<Tensor>> {
    let mut outputs = Vec::with_capacity(inner);
    for _ in 0..inner {
        outputs.push(run(mode, case, x, kernel, false)?);
    }
    Ok(outputs)
}

fn timed_warmup(
    phase: &str,
    mode: Mode,
    case: Case,
    x: &Tensor,
    kernel: &Tensor,
    device: &Device,
    warmup_ms: f64,
    inner: usize,
) -> Result<()> {
    let requested = Duration::from_secs_f64(warmup_ms / 1000.0);
    let started = Instant::now();
    let mut batches = 0usize;
    let mut launches = 0usize;
    loop {
        let outputs = batched_launches(mode, case, x, kernel, inner)?;
        device.synchronize()?;
        std::hint::black_box(outputs);
        batches += 1;
        launches += inner;
        if started.elapsed() >= requested {
            break;
        }
    }
    println!(
        "WARMUP phase={} mode={} requested_ms={:.3} actual_ms={:.3} batches={} launches={}",
        phase,
        mode.as_str(),
        warmup_ms,
        started.elapsed().as_secs_f64() * 1000.0,
        batches,
        launches
    );
    Ok(())
}

fn measure(
    phase: &str,
    mode: Mode,
    case: Case,
    x: &Tensor,
    kernel: &Tensor,
    device: &Device,
    warmup_ms: f64,
    iters: usize,
    inner: usize,
) -> Result<TimingStats> {
    timed_warmup(phase, mode, case, x, kernel, device, warmup_ms, inner)?;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        device.synchronize()?;
        let start = Instant::now();
        let outputs = batched_launches(mode, case, x, kernel, inner)?;
        device.synchronize()?;
        samples.push(start.elapsed().as_secs_f64() * 1_000_000.0 / inner as f64);
        std::hint::black_box(outputs);
    }
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Ok(TimingStats {
        median_us: percentile(&samples, 0.50),
        p10_us: percentile(&samples, 0.10),
        p90_us: percentile(&samples, 0.90),
    })
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

fn print_phase(name: &str, mode: Mode, stats: TimingStats) {
    println!(
        "PHASE phase={} mode={} median_us={:.6} p10_us={:.6} p90_us={:.6}",
        name,
        mode.as_str(),
        stats.median_us,
        stats.p10_us,
        stats.p90_us
    );
}

fn main() -> Result<()> {
    let warmup_ms = parse_f64("--warmup-ms", DEFAULT_WARMUP_MS);
    let iters = parse_count("--iters", DEFAULT_ITERS);
    let inner = parse_count("--inner", DEFAULT_INNER);
    let max_drift_pct = parse_f64("--max-drift-pct", DEFAULT_MAX_DRIFT_PCT);
    let min_speedup_x = parse_f64(
        "--min-integrated-speedup-x",
        DEFAULT_MIN_INTEGRATED_SPEEDUP_X,
    );
    if !warmup_ms.is_finite() || warmup_ms <= 0.0 || iters == 0 || inner == 0 {
        candle_core::bail!("--warmup-ms, --iters and --inner must be greater than zero")
    }
    if max_drift_pct < 0.0 || !min_speedup_x.is_finite() || min_speedup_x <= 1.0 {
        candle_core::bail!("invalid validation thresholds")
    }
    if !candle_kernels::asd_exact_conv2d::VALIDATION_BUILD {
        candle_core::bail!(
            "grouped_conv2d_asd_validate_timed requires an ASD V2 candidate build with CANDLE_ASD_VALIDATION=1"
        )
    }

    let device = Device::new_cuda(0)?;
    println!("=== ASD-V2 DW5X5 TIME-EQUIVALENT INTEGRATED CANDIDATE VALIDATION ===");
    println!("scope=validation_only");
    println!("production_dispatch_modified=false");
    println!("device={:?}", device.location());
    println!(
        "cuda_build_compute_cap={}",
        candle_kernels::CUDA_BUILD_COMPUTE_CAP
    );
    println!("cases={}", CASES.len());
    println!("warmup_policy=time_equivalent_per_backend");
    println!("warmup_ms={warmup_ms:.3}");
    println!("timed_samples={iters}");
    println!("launches_per_sample={inner}");
    println!("max_drift_pct={max_drift_pct:.3}");
    println!("minimum_integrated_speedup_x={min_speedup_x:.4}");
    println!("sequence=current_a,asd_b,current_a2,asd_a,current_b,asd_a2");

    let mut all_pass = true;
    for case in CASES {
        let (x, kernel) = tensors(case, &device)?;
        let matched = lookup(case).ok_or_else(|| {
            candle_core::Error::Msg(
                format!(
                    "missing exact ASD V2 decision for c={} h={} w={}",
                    case.c, case.h, case.w
                )
                .into(),
            )
        })?;
        println!(
            "\nCASE c={} h={} w={} policy_id={} decision_id={} state={} required_speedup_x={:.4}",
            case.c,
            case.h,
            case.w,
            matched.policy_id,
            matched.decision_id,
            matched.state,
            matched.min_integrated_speedup_x,
        );

        println!("=== BACKEND IDENTITY PROBE ===");
        let current_out = run(Mode::Current, case, &x, &kernel, true)?;
        device.synchronize()?;
        println!("COMPLETION mode=current synchronized=true");
        let candidate_out = run(Mode::AsdCandidate, case, &x, &kernel, true)?;
        device.synchronize()?;
        println!("COMPLETION mode=asd_candidate synchronized=true");
        let (max_abs, max_rel) = max_abs_rel(&candidate_out, &current_out)?;
        let parity = max_abs <= 1e-4 || max_rel <= 1e-4;
        println!(
            "PARITY current_vs_asd max_abs={:.8} max_rel={:.8} pass={}",
            max_abs, max_rel, parity
        );

        let current_a = measure(
            "current_a",
            Mode::Current,
            case,
            &x,
            &kernel,
            &device,
            warmup_ms,
            iters,
            inner,
        )?;
        let asd_b = measure(
            "asd_b",
            Mode::AsdCandidate,
            case,
            &x,
            &kernel,
            &device,
            warmup_ms,
            iters,
            inner,
        )?;
        let current_a2 = measure(
            "current_a2",
            Mode::Current,
            case,
            &x,
            &kernel,
            &device,
            warmup_ms,
            iters,
            inner,
        )?;
        let asd_a = measure(
            "asd_a",
            Mode::AsdCandidate,
            case,
            &x,
            &kernel,
            &device,
            warmup_ms,
            iters,
            inner,
        )?;
        let current_b = measure(
            "current_b",
            Mode::Current,
            case,
            &x,
            &kernel,
            &device,
            warmup_ms,
            iters,
            inner,
        )?;
        let asd_a2 = measure(
            "asd_a2",
            Mode::AsdCandidate,
            case,
            &x,
            &kernel,
            &device,
            warmup_ms,
            iters,
            inner,
        )?;

        print_phase("current_a", Mode::Current, current_a);
        print_phase("asd_b", Mode::AsdCandidate, asd_b);
        print_phase("current_a2", Mode::Current, current_a2);
        print_phase("asd_a", Mode::AsdCandidate, asd_a);
        print_phase("current_b", Mode::Current, current_b);
        print_phase("asd_a2", Mode::AsdCandidate, asd_a2);

        let current_us = median3(
            current_a.median_us,
            current_a2.median_us,
            current_b.median_us,
        );
        let asd_us = median3(asd_b.median_us, asd_a.median_us, asd_a2.median_us);
        let current_p90 = median3(current_a.p90_us, current_a2.p90_us, current_b.p90_us);
        let asd_p90 = median3(asd_b.p90_us, asd_a.p90_us, asd_a2.p90_us);
        let current_drift_pct = relative_drift_pct(current_a.median_us, current_b.median_us);
        let asd_drift_pct = relative_drift_pct(asd_b.median_us, asd_a2.median_us);
        let speedup_x = current_us / asd_us;
        let latency_reduction_pct = (1.0 - asd_us / current_us) * 100.0;
        let drift_pass = current_drift_pct <= max_drift_pct && asd_drift_pct <= max_drift_pct;
        let performance_pass = speedup_x >= min_speedup_x;
        let p90_non_regression = asd_p90 <= current_p90;
        let pass = parity && drift_pass && performance_pass && p90_non_regression;
        all_pass &= pass;

        println!(
            "INTEGRATED_RESULT c={} h={} w={} current_us={:.6} asd_us={:.6} speedup_x={:.6} latency_reduction_pct={:.3} current_drift_pct={:.3} asd_drift_pct={:.3} current_p90_us={:.6} asd_p90_us={:.6}",
            case.c,
            case.h,
            case.w,
            current_us,
            asd_us,
            speedup_x,
            latency_reduction_pct,
            current_drift_pct,
            asd_drift_pct,
            current_p90,
            asd_p90
        );
        println!(
            "GATE c={} parity={} drift={} integrated_speedup={} p90_non_regression={} required_speedup_x={:.4} pass={}",
            case.c,
            parity,
            drift_pass,
            performance_pass,
            p90_non_regression,
            min_speedup_x,
            pass
        );
    }

    println!("\n=== NEGATIVE DOMAIN PROBE ===");
    let miss = Case {
        c: 64,
        h: 32,
        w: 24,
    };
    let (x_miss, k_miss) = tensors(miss, &device)?;
    let lookup_miss = lookup(miss).is_none();
    println!("DOMAIN_LOOKUP c=64 h=32 w=24 miss={lookup_miss}");
    let current_miss = run(Mode::Current, miss, &x_miss, &k_miss, false)?;
    let candidate_miss = run(Mode::AsdCandidate, miss, &x_miss, &k_miss, true)?;
    device.synchronize()?;
    let (miss_abs, miss_rel) = max_abs_rel(&candidate_miss, &current_miss)?;
    let domain_miss_parity = miss_abs <= 1e-4 || miss_rel <= 1e-4;
    let domain_pass = lookup_miss && domain_miss_parity;
    println!(
        "DOMAIN_MISS current_vs_candidate max_abs={:.8} max_rel={:.8} parity={} expected_backend=current pass={}",
        miss_abs, miss_rel, domain_miss_parity, domain_pass
    );
    all_pass &= domain_pass;

    println!(
        "\nMULTI_POINT_GATE cases={} domain_miss={} production_activation=false warmup_policy=time_equivalent_per_backend pass={}",
        CASES.len(), domain_pass, all_pass
    );
    println!("STATUS={}", if all_pass { "PASS" } else { "HOLD" });
    if !all_pass {
        candle_core::bail!("ASD V2 DW5x5 time-equivalent integrated candidate validation failed")
    }
    Ok(())
}
