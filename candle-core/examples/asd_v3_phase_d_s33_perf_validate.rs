use candle_core::{Device, Result, Tensor};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const IMPLEMENTATION_ID: &str =
    "candle.sm61-exact-grouped.dynamic.ct1d-s33-g2-u1-b256";

const DEFAULT_WARMUP_MS: f64 = 750.0;
const DEFAULT_ITERS: usize = 80;
const DEFAULT_INNER: usize = 64;

#[derive(Clone, Copy, Debug)]
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

fn percentile(samples: &[f64], q: f64) -> f64 {
    let mut values = samples.to_vec();
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * q).round() as usize;
    values[index]
}

fn deterministic(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * mul) % 101) as isize + bias) as f32 / 64.0)
        .collect()
}

fn required_env_path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| candle_core::Error::Msg(format!("missing required environment variable {name}")))
}

fn call(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.conv_transpose1d(w, 1, 1, 2, 1, 2)
}

fn warmup(
    x: &Tensor,
    w: &Tensor,
    device: &Device,
    requested_ms: f64,
    inner: usize,
) -> Result<()> {
    let requested = Duration::from_secs_f64(requested_ms / 1000.0);
    let start = Instant::now();
    let mut launches = 0usize;
    while start.elapsed() < requested {
        let mut outputs = Vec::with_capacity(inner);
        for _ in 0..inner {
            outputs.push(call(x, w)?);
        }
        device.synchronize()?;
        launches += inner;
    }
    println!(
        "WARMUP requested_ms={requested_ms:.3} actual_ms={:.3} launches={launches}",
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

fn measure(
    x: &Tensor,
    w: &Tensor,
    device: &Device,
    iters: usize,
    inner: usize,
) -> Result<TimingStats> {
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let mut outputs = Vec::with_capacity(inner);
        for _ in 0..inner {
            outputs.push(call(x, w)?);
        }
        device.synchronize()?;
        samples.push(start.elapsed().as_secs_f64() * 1_000_000.0 / inner as f64);
    }
    Ok(TimingStats {
        median_us: percentile(&samples, 0.50),
        p10_us: percentile(&samples, 0.10),
        p90_us: percentile(&samples, 0.90),
    })
}

fn main() -> Result<()> {
    let runtime_profile = required_env_path("CANDLE_ASD_RUNTIME_PROFILE")?;
    let module_dir = required_env_path("CANDLE_ASD_MODULE_DIR")?;
    let manifest = module_dir.join(format!("{IMPLEMENTATION_ID}.manifest"));
    let cubin = module_dir.join(format!("{IMPLEMENTATION_ID}.cubin"));

    if !runtime_profile.is_file() || !manifest.is_file() || !cubin.is_file() {
        candle_core::bail!(
            "missing S33 runtime control-plane profile={} manifest={} cubin={}",
            runtime_profile.display(),
            manifest.display(),
            cubin.display()
        )
    }

    let warmup_ms = parse_f64("--warmup-ms", DEFAULT_WARMUP_MS);
    let iters = parse_count("--iters", DEFAULT_ITERS);
    let inner = parse_count("--inner", DEFAULT_INNER);
    if !warmup_ms.is_finite() || warmup_ms <= 0.0 || iters == 0 || inner == 0 {
        candle_core::bail!("invalid benchmark configuration")
    }

    std::env::remove_var("CANDLE_ASD_EXACT_DISABLE");
    std::env::remove_var("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL");
    std::env::remove_var("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "auto");

    let device = Device::new_cuda(0)?;
    let x = Tensor::from_vec(
        deterministic(128 * 33, 37, -50),
        (1, 128, 33),
        &device,
    )?;
    let w = Tensor::from_vec(
        deterministic(128 * 64 * 3, 53, -50),
        (128, 64, 3),
        &device,
    )?;

    device.as_cuda_device()?.refresh_asd();

    let cold = call(&x, &w)?;
    device.synchronize()?;
    let dims = cold.dims3()?;
    if dims != (1, 128, 66) {
        candle_core::bail!("unexpected S33 output shape {dims:?}")
    }

    let source = device
        .as_cuda_device()?
        .asd_resolved_module_source(IMPLEMENTATION_ID)?
        .unwrap_or("none");
    if source != "external_cubin" {
        candle_core::bail!("S33 source is {source}, expected external_cubin")
    }

    warmup(&x, &w, &device, warmup_ms, inner)?;
    let stats = measure(&x, &w, &device, iters, inner)?;

    let source_after = device
        .as_cuda_device()?
        .asd_resolved_module_source(IMPLEMENTATION_ID)?
        .unwrap_or("none");
    if source_after != "external_cubin" {
        candle_core::bail!(
            "S33 source changed after timing: {source_after}, expected external_cubin"
        )
    }

    println!("=== ASD V3 PHASE D S33 WARM-PATH PERFORMANCE ===");
    println!("runtime_profile={}", runtime_profile.display());
    println!("module_dir={}", module_dir.display());
    println!("implementation_id={IMPLEMENTATION_ID}");
    println!("source={source_after}");
    println!("warmup_ms={warmup_ms:.3}");
    println!("timed_samples={iters}");
    println!("launches_per_sample={inner}");
    println!("median_us={:.6}", stats.median_us);
    println!("p10_us={:.6}", stats.p10_us);
    println!("p90_us={:.6}", stats.p90_us);
    println!("STATUS=PASS");
    Ok(())
}
