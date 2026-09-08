use candle_core::{Device, Result, Tensor};
use std::cmp::Ordering;
use std::time::Instant;

const DEFAULT_WARMUP: usize = 8;
const DEFAULT_ITERS: usize = 40;
const DEFAULT_INNER: usize = 32;
const DEFAULT_MAX_DRIFT_PCT: f64 = 5.0;
const DEFAULT_MIN_INTEGRATED_SPEEDUP_X: f64 = 1.10;

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

struct EnvGuard {
    saved: Vec<(&'static str, Option<String>)>,
}

impl EnvGuard {
    fn new(mode: Mode, trace: bool) -> Self {
        const KEYS: [&str; 4] = [
            "CANDLE_ASD_EXACT_DISABLE",
            "CANDLE_ASD_EXACT_TRACE",
            "CANDLE_GROUPED_TRANSPOSE_DISPATCH",
            "CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL",
        ];
        let saved = KEYS
            .iter()
            .map(|&key| (key, std::env::var(key).ok()))
            .collect::<Vec<_>>();
        for key in KEYS {
            std::env::remove_var(key);
        }
        if matches!(mode, Mode::Current) {
            std::env::set_var("CANDLE_ASD_EXACT_DISABLE", "1");
        }
        if trace {
            std::env::set_var("CANDLE_ASD_EXACT_TRACE", "1");
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
        .map(|i| (((i * mul) % 101) as isize + bias) as f32 / 64.0)
        .collect()
}

fn tensors(groups: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let batch = 1usize;
    let c_in = 64usize;
    let c_out = 64usize;
    let len = 128usize;
    let kernel = 3usize;
    let x = Tensor::from_vec(
        deterministic(batch * c_in * len, 37, -50),
        (batch, c_in, len),
        device,
    )?;
    let weight = Tensor::from_vec(
        deterministic(c_in * (c_out / groups) * kernel, 53, -50),
        (c_in, c_out / groups, kernel),
        device,
    )?;
    Ok((x, weight))
}

fn call(x: &Tensor, kernel: &Tensor, groups: usize) -> Result<Tensor> {
    x.conv_transpose1d(kernel, 1, 1, 2, 1, groups)
}

fn run(mode: Mode, trace: bool, x: &Tensor, kernel: &Tensor, groups: usize) -> Result<Tensor> {
    let _guard = EnvGuard::new(mode, trace);
    call(x, kernel, groups)
}

fn max_abs_rel(lhs: &Tensor, rhs: &Tensor) -> Result<(f32, f32)> {
    if lhs.dims() != rhs.dims() {
        candle_core::bail!(
            "shape mismatch in ASD validation: {:?} vs {:?}",
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
    x: &Tensor,
    kernel: &Tensor,
    groups: usize,
    inner: usize,
) -> Result<Vec<Tensor>> {
    let mut outputs = Vec::with_capacity(inner);
    for _ in 0..inner {
        outputs.push(call(x, kernel, groups)?);
    }
    Ok(outputs)
}

fn measure(
    mode: Mode,
    x: &Tensor,
    kernel: &Tensor,
    groups: usize,
    device: &Device,
    warmup: usize,
    iters: usize,
    inner: usize,
) -> Result<TimingStats> {
    let _guard = EnvGuard::new(mode, false);
    for _ in 0..warmup {
        let outputs = batched_launches(x, kernel, groups, inner)?;
        device.synchronize()?;
        std::hint::black_box(outputs);
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        device.synchronize()?;
        let start = Instant::now();
        let outputs = batched_launches(x, kernel, groups, inner)?;
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
    let warmup = parse_count("--warmup", DEFAULT_WARMUP);
    let iters = parse_count("--iters", DEFAULT_ITERS);
    let inner = parse_count("--inner", DEFAULT_INNER);
    let max_drift_pct = parse_f64("--max-drift-pct", DEFAULT_MAX_DRIFT_PCT);
    let min_speedup_x = parse_f64(
        "--min-integrated-speedup-x",
        DEFAULT_MIN_INTEGRATED_SPEEDUP_X,
    );
    if warmup == 0 || iters == 0 || inner == 0 {
        candle_core::bail!("--warmup, --iters and --inner must be greater than zero")
    }
    if max_drift_pct < 0.0 || !min_speedup_x.is_finite() || min_speedup_x <= 1.0 {
        candle_core::bail!("invalid validation thresholds")
    }

    let device = Device::new_cuda(0)?;
    let (x, kernel) = tensors(2, &device)?;

    println!("=== ASD-V1-R3 INTEGRATED EXACT CANDIDATE VALIDATION ===");
    println!("device={:?}", device.location());
    println!("signature=ct1d:f32:b1:c64:l128:w64x32x3:g2:k3:s2:p1:op1:d1:contiguous_zero_offset");
    println!("reference_backend=current");
    println!("candidate_backend=raw_cuda");
    println!("warmup_samples={warmup}");
    println!("timed_samples={iters}");
    println!("launches_per_sample={inner}");
    println!("max_drift_pct={max_drift_pct:.3}");
    println!("minimum_integrated_speedup_x={min_speedup_x:.4}");
    println!("sequence=current_a,asd_b,current_a2,asd_a,current_b,asd_a2");

    println!("\n=== BACKEND IDENTITY PROBE ===");
    println!("PROBE mode=current expected_selected=cudnn expected_submitted=cudnn");
    let current = run(Mode::Current, true, &x, &kernel, 2)?;
    device.synchronize()?;
    println!("COMPLETION mode=current synchronized=true");

    println!("PROBE mode=asd_candidate expected_selected=raw expected_submitted=raw");
    let candidate = run(Mode::AsdCandidate, true, &x, &kernel, 2)?;
    device.synchronize()?;
    println!("COMPLETION mode=asd_candidate synchronized=true");

    let (max_abs, max_rel) = max_abs_rel(&candidate, &current)?;
    let parity = max_abs <= 1e-4 || max_rel <= 1e-4;
    println!(
        "PARITY current_vs_asd max_abs={:.8} max_rel={:.8} pass={}",
        max_abs, max_rel, parity
    );

    println!("\n=== NEGATIVE DOMAIN PROBE ===");
    let (x_g4, kernel_g4) = tensors(4, &device)?;
    println!("PROBE groups=4 expected_reason=auto_rule expected_submitted=cudnn");
    let _domain_miss = run(Mode::AsdCandidate, true, &x_g4, &kernel_g4, 4)?;
    device.synchronize()?;
    println!("DOMAIN_MISS groups=4 synchronized=true expected_backend=current");

    println!("\n=== INTEGRATED A/B/A ===");
    let current_a = measure(Mode::Current, &x, &kernel, 2, &device, warmup, iters, inner)?;
    let asd_b = measure(
        Mode::AsdCandidate,
        &x,
        &kernel,
        2,
        &device,
        warmup,
        iters,
        inner,
    )?;
    let current_a2 = measure(Mode::Current, &x, &kernel, 2, &device, warmup, iters, inner)?;
    let asd_a = measure(
        Mode::AsdCandidate,
        &x,
        &kernel,
        2,
        &device,
        warmup,
        iters,
        inner,
    )?;
    let current_b = measure(Mode::Current, &x, &kernel, 2, &device, warmup, iters, inner)?;
    let asd_a2 = measure(
        Mode::AsdCandidate,
        &x,
        &kernel,
        2,
        &device,
        warmup,
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
    let throughput_increase_pct = (speedup_x - 1.0) * 100.0;

    let drift_pass = current_drift_pct <= max_drift_pct && asd_drift_pct <= max_drift_pct;
    let performance_pass = speedup_x >= min_speedup_x;
    let p90_non_regression = asd_p90 <= current_p90;
    let pass = parity && drift_pass && performance_pass && p90_non_regression;

    println!(
        "INTEGRATED_RESULT current_us={:.6} asd_us={:.6} speedup_x={:.6} latency_reduction_pct={:.3} throughput_increase_pct={:.3} current_drift_pct={:.3} asd_drift_pct={:.3} current_p90_us={:.6} asd_p90_us={:.6}",
        current_us,
        asd_us,
        speedup_x,
        latency_reduction_pct,
        throughput_increase_pct,
        current_drift_pct,
        asd_drift_pct,
        current_p90,
        asd_p90
    );
    println!(
        "GATE parity={} drift={} integrated_speedup={} p90_non_regression={} required_speedup_x={:.4}",
        parity, drift_pass, performance_pass, p90_non_regression, min_speedup_x
    );
    println!("STATUS={}", if pass { "PASS" } else { "HOLD" });

    Ok(())
}
