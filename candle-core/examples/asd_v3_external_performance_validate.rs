use candle_core::{Device, Result, Tensor};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const IMPLEMENTATION_ID: &str =
    "candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256";
const DEFAULT_WARMUP_MS: f64 = 500.0;
const DEFAULT_ITERS: usize = 40;
const DEFAULT_INNER: usize = 32;
const DEFAULT_MAX_DRIFT_PCT: f64 = 5.0;
const DEFAULT_MIN_SPEEDUP_X: f64 = 1.01;

#[derive(Clone, Copy, Debug)]
struct TimingStats {
    median_us: f64,
    p10_us: f64,
    p90_us: f64,
}

#[derive(Clone, Copy)]
struct MeasureConfig {
    warmup_ms: f64,
    iters: usize,
    inner: usize,
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

fn tensors(device: &Device) -> Result<(Tensor, Tensor)> {
    let x = Tensor::from_vec(
        deterministic(1 * 128 * 32, 37, -50),
        (1, 128, 32),
        device,
    )?;
    let w = Tensor::from_vec(
        deterministic(128 * 64 * 3, 53, -50),
        (128, 64, 3),
        device,
    )?;
    Ok((x, w))
}

fn call(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.conv_transpose1d(w, 1, 1, 2, 1, 2)
}

fn module_path(dir: &Path) -> PathBuf {
    dir.join(format!("{IMPLEMENTATION_ID}.cubin"))
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).map_err(|err| {
        candle_core::Error::Msg(format!("failed to read {}: {err}", path.display()))
    })?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn gpu_uuid(device: &Device) -> Result<String> {
    let uuid = device
        .as_cuda_device()?
        .cuda_stream()
        .context()
        .uuid()
        .map_err(|err| candle_core::Error::Msg(format!("failed to read CUDA UUID: {err}")))?;
    let hex = uuid
        .bytes
        .iter()
        .map(|byte| format!("{:02x}", *byte as u8))
        .collect::<String>();
    if hex.len() != 32 {
        candle_core::bail!("unexpected CUDA UUID length {}", hex.len())
    }
    Ok(format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    ))
}

fn max_abs_rel(lhs: &Tensor, rhs: &Tensor) -> Result<(f32, f32)> {
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

fn median3(a: f64, b: f64, c: f64) -> f64 {
    let mut values = [a, b, c];
    values.sort_unstable_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));
    values[1]
}

fn relative_drift_pct(first: f64, last: f64) -> f64 {
    let center = (first + last) * 0.5;
    if center == 0.0 {
        0.0
    } else {
        (last - first).abs() / center * 100.0
    }
}

fn timed_warmup(
    phase: &str,
    dir: &Path,
    x: &Tensor,
    w: &Tensor,
    device: &Device,
    warmup_ms: f64,
    inner: usize,
) -> Result<()> {
    std::env::set_var("CANDLE_ASD_MODULE_DIR", dir);
    let requested = Duration::from_secs_f64(warmup_ms / 1000.0);
    let started = Instant::now();
    let mut batches = 0usize;
    let mut launches = 0usize;
    loop {
        let mut outputs = Vec::with_capacity(inner);
        for _ in 0..inner {
            outputs.push(call(x, w)?);
        }
        device.synchronize()?;
        std::hint::black_box(outputs);
        batches += 1;
        launches += inner;
        if started.elapsed() >= requested {
            break;
        }
    }
    println!(
        "WARMUP phase={} requested_ms={:.3} actual_ms={:.3} batches={} launches={}",
        phase,
        warmup_ms,
        started.elapsed().as_secs_f64() * 1000.0,
        batches,
        launches
    );
    Ok(())
}

fn settle_pair(
    dir_a: &Path,
    dir_b: &Path,
    x: &Tensor,
    w: &Tensor,
    device: &Device,
    warmup_ms: f64,
    inner: usize,
) -> Result<()> {
    println!("=== NON-AUTHORITATIVE SETTLING ===");
    println!("settling_policy=time_equivalent_pair");
    timed_warmup("settle_a", dir_a, x, w, device, warmup_ms, inner)?;
    timed_warmup("settle_b", dir_b, x, w, device, warmup_ms, inner)?;
    device.synchronize()?;
    println!("settling_complete=true");
    Ok(())
}

fn measure(
    phase: &str,
    dir: &Path,
    x: &Tensor,
    w: &Tensor,
    device: &Device,
    cfg: MeasureConfig,
) -> Result<TimingStats> {
    timed_warmup(phase, dir, x, w, device, cfg.warmup_ms, cfg.inner)?;

    let mut samples = Vec::with_capacity(cfg.iters);
    for _ in 0..cfg.iters {
        device.synchronize()?;
        let start = Instant::now();
        let mut outputs = Vec::with_capacity(cfg.inner);
        for _ in 0..cfg.inner {
            outputs.push(call(x, w)?);
        }
        device.synchronize()?;
        samples.push(start.elapsed().as_secs_f64() * 1_000_000.0 / cfg.inner as f64);
        std::hint::black_box(outputs);
    }

    samples.sort_unstable_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));
    Ok(TimingStats {
        median_us: percentile(&samples, 0.50),
        p10_us: percentile(&samples, 0.10),
        p90_us: percentile(&samples, 0.90),
    })
}

fn print_phase(name: &str, artifact: &str, stats: TimingStats) {
    println!(
        "PHASE phase={name} artifact={artifact} median_us={:.6} p10_us={:.6} p90_us={:.6}",
        stats.median_us, stats.p10_us, stats.p90_us
    );
}

fn main() -> Result<()> {
    let dir_a = std::env::var_os("CANDLE_ASD_MODULE_DIR_A")
        .map(PathBuf::from)
        .ok_or_else(|| candle_core::Error::Msg("missing CANDLE_ASD_MODULE_DIR_A".into()))?;
    let dir_b = std::env::var_os("CANDLE_ASD_MODULE_DIR_B")
        .map(PathBuf::from)
        .ok_or_else(|| candle_core::Error::Msg("missing CANDLE_ASD_MODULE_DIR_B".into()))?;

    let warmup_ms = parse_f64("--warmup-ms", DEFAULT_WARMUP_MS);
    let iters = parse_count("--iters", DEFAULT_ITERS);
    let inner = parse_count("--inner", DEFAULT_INNER);
    let max_drift_pct = parse_f64("--max-drift-pct", DEFAULT_MAX_DRIFT_PCT);
    let min_speedup_x = parse_f64("--min-speedup-x", DEFAULT_MIN_SPEEDUP_X);
    if !warmup_ms.is_finite() || warmup_ms <= 0.0 || iters == 0 || inner == 0 {
        candle_core::bail!("--warmup-ms, --iters and --inner must be greater than zero")
    }
    if max_drift_pct < 0.0 || !min_speedup_x.is_finite() || min_speedup_x <= 0.0 {
        candle_core::bail!("invalid benchmark thresholds")
    }

    let path_a = module_path(&dir_a);
    let path_b = module_path(&dir_b);
    if !path_a.is_file() || !path_b.is_file() {
        candle_core::bail!("both A and B CUBIN files must exist")
    }

    let artifact_a_sha256 = sha256_file(&path_a)?;
    let artifact_b_sha256 = sha256_file(&path_b)?;
    if artifact_a_sha256 == artifact_b_sha256 {
        candle_core::bail!("A and B must be distinct CUDA artifacts")
    }

    std::env::set_var("CANDLE_ASD_EXACT_TRACE", "0");
    std::env::set_var("CANDLE_SM61_EXACT_GROUPED_TRACE", "0");
    std::env::remove_var("CANDLE_SM61_EXACT_GROUPED_DISABLE");

    let device = Device::new_cuda(0)?;
    let gpu_uuid = gpu_uuid(&device)?;
    let (x, w) = tensors(&device)?;

    // One synchronized parity probe before timing.
    std::env::set_var("CANDLE_ASD_MODULE_DIR", &dir_a);
    let out_a = call(&x, &w)?;
    device.synchronize()?;
    std::env::set_var("CANDLE_ASD_MODULE_DIR", &dir_b);
    let out_b = call(&x, &w)?;
    device.synchronize()?;
    let (max_abs, max_rel) = max_abs_rel(&out_a, &out_b)?;
    let parity_pass = max_abs <= 1e-5 || max_rel <= 1e-5;

    let cfg = MeasureConfig {
        warmup_ms,
        iters,
        inner,
    };

    println!("=== ASD V3 EXTERNAL PERFORMANCE VALIDATION ===");
    println!("implementation_id={IMPLEMENTATION_ID}");
    println!("gpu_uuid={gpu_uuid}");
    println!("artifact_a_sha256={artifact_a_sha256}");
    println!("artifact_b_sha256={artifact_b_sha256}");
    println!("warmup_policy=time_equivalent_per_backend");
    println!("warmup_ms={warmup_ms:.3}");
    println!("settling_policy=time_equivalent_pair_non_authoritative");
    println!("timed_samples={iters}");
    println!("launches_per_sample={inner}");
    println!("max_drift_pct={max_drift_pct:.3}");
    println!("min_speedup_x={min_speedup_x:.6}");
    println!("sequence=a1,b1,a2,b2,a3,b3");
    println!("PARITY max_abs={max_abs:.8} max_rel={max_rel:.8} pass={parity_pass}");

    settle_pair(&dir_a, &dir_b, &x, &w, &device, warmup_ms, inner)?;

    println!("=== AUTHORITATIVE SEQUENCE ===");
    let a1 = measure("a1", &dir_a, &x, &w, &device, cfg)?;
    let b1 = measure("b1", &dir_b, &x, &w, &device, cfg)?;
    let a2 = measure("a2", &dir_a, &x, &w, &device, cfg)?;
    let b2 = measure("b2", &dir_b, &x, &w, &device, cfg)?;
    let a3 = measure("a3", &dir_a, &x, &w, &device, cfg)?;
    let b3 = measure("b3", &dir_b, &x, &w, &device, cfg)?;

    print_phase("a1", &artifact_a_sha256, a1);
    print_phase("b1", &artifact_b_sha256, b1);
    print_phase("a2", &artifact_a_sha256, a2);
    print_phase("b2", &artifact_b_sha256, b2);
    print_phase("a3", &artifact_a_sha256, a3);
    print_phase("b3", &artifact_b_sha256, b3);

    let a_us = median3(a1.median_us, a2.median_us, a3.median_us);
    let b_us = median3(b1.median_us, b2.median_us, b3.median_us);
    let a_p90 = median3(a1.p90_us, a2.p90_us, a3.p90_us);
    let b_p90 = median3(b1.p90_us, b2.p90_us, b3.p90_us);
    let a_drift_pct = relative_drift_pct(a1.median_us, a3.median_us);
    let b_drift_pct = relative_drift_pct(b1.median_us, b3.median_us);
    let speedup_x = a_us / b_us;
    let latency_change_pct = (b_us / a_us - 1.0) * 100.0;

    let drift_pass = a_drift_pct <= max_drift_pct && b_drift_pct <= max_drift_pct;
    let p90_pass = b_p90 <= a_p90;
    let speedup_pass = speedup_x >= min_speedup_x;
    let harness_pass = parity_pass && drift_pass;
    let candidate_pass = harness_pass && p90_pass && speedup_pass;
    let decision = if candidate_pass {
        "PROMOTE_CANDIDATE"
    } else {
        "REJECT_CANDIDATE"
    };
    let status = if harness_pass { "PASS" } else { "HOLD" };

    println!(
        "PERFORMANCE_RESULT a_us={a_us:.6} b_us={b_us:.6} speedup_x={speedup_x:.6} latency_change_pct={latency_change_pct:.3} a_drift_pct={a_drift_pct:.3} b_drift_pct={b_drift_pct:.3} a_p90_us={a_p90:.6} b_p90_us={b_p90:.6}"
    );
    println!(
        "GATE parity={parity_pass} drift={drift_pass} speedup={speedup_pass} p90_non_regression={p90_pass} required_speedup_x={min_speedup_x:.6}"
    );
    println!("STATUS={status}");
    println!("DECISION={decision}");

    let parity_evidence_sha256 = match std::env::var_os("CANDLE_ASD_PARITY_EVIDENCE") {
        Some(path) => sha256_file(Path::new(&path))?,
        None => "none".to_owned(),
    };

    let evidence = format!(
        "ASD-CUDA-PERFORMANCE-EVIDENCE-V1\nimplementation_id={IMPLEMENTATION_ID}\narchitecture=sm61\ngpu_uuid={gpu_uuid}\nartifact_a_sha256={artifact_a_sha256}\nartifact_b_sha256={artifact_b_sha256}\nparity_evidence_sha256={parity_evidence_sha256}\nwarmup_policy=time_equivalent_per_backend\nwarmup_ms={warmup_ms:.3}\nsettling_policy=time_equivalent_pair_non_authoritative\nsettling_ms_per_artifact={warmup_ms:.3}\ntimed_samples={iters}\nlaunches_per_sample={inner}\nsequence=a1,b1,a2,b2,a3,b3\nmax_abs={max_abs:.8}\nmax_rel={max_rel:.8}\nparity_pass={parity_pass}\na1_median_us={:.6}\nb1_median_us={:.6}\na2_median_us={:.6}\nb2_median_us={:.6}\na3_median_us={:.6}\nb3_median_us={:.6}\na_consensus_us={a_us:.6}\nb_consensus_us={b_us:.6}\na_p90_us={a_p90:.6}\nb_p90_us={b_p90:.6}\na_drift_pct={a_drift_pct:.3}\nb_drift_pct={b_drift_pct:.3}\nspeedup_x={speedup_x:.6}\nmin_speedup_x={min_speedup_x:.6}\ndrift_pass={drift_pass}\np90_non_regression={p90_pass}\nspeedup_pass={speedup_pass}\nstatus={status}\ndecision={decision}\n",
        a1.median_us,
        b1.median_us,
        a2.median_us,
        b2.median_us,
        a3.median_us,
        b3.median_us,
    );
    let evidence_sha256 = Sha256::digest(evidence.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    println!("performance_evidence_sha256={evidence_sha256}");

    if let Some(path) = std::env::var_os("CANDLE_ASD_PERF_EVIDENCE_OUT") {
        let path = PathBuf::from(path);
        use std::io::Write as _;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|err| {
                candle_core::Error::Msg(format!(
                    "refusing to overwrite performance evidence {}: {err}",
                    path.display()
                ))
            })?;
        file.write_all(evidence.as_bytes()).map_err(|err| {
            candle_core::Error::Msg(format!(
                "failed to write performance evidence {}: {err}",
                path.display()
            ))
        })?;
        println!("performance_evidence_file={}", path.display());
    }

    if !harness_pass {
        candle_core::bail!("ASD V3 performance harness is unstable; do not use this evidence")
    }

    Ok(())
}
