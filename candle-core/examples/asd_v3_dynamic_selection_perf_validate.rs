use candle_core::{Device, Result, Tensor};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const EMBEDDED_DECISION_ID: &str = "ct1d-sm61-s32-g8-raw-exact";
const EMBEDDED_IMPLEMENTATION_ID: &str =
    "candle.sm61-exact-grouped.ct1d-s32-g8-u1-b256";
const DYNAMIC_DECISION_ID: &str = "ct1d-sm61-s32-g8-runtime-additive-perf";
const DYNAMIC_IMPLEMENTATION_ID: &str =
    "candle.sm61-exact-grouped.runtime-perf.ct1d-s32-g8-u1-b256";
const DYNAMIC_CANDIDATE_ID: &str = "runtime-perf-ct1d-s32-g8-u1-b256";
const ENTRY: &str = "flow_v0322_ct1d_s32_g8_u1_b256";
const SIGNATURE: &str = "op=conv_transpose1d,dim=1,batch=1,c_in=128,c_out=128,spatial=32,weight_shape=128x16x3,groups=8,kernel=3,stride=2,padding=1,output_padding=1,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset";

const DEFAULT_WARMUP_MS: f64 = 500.0;
const DEFAULT_ITERS: usize = 60;
const DEFAULT_INNER: usize = 64;
const DEFAULT_MAX_OVERHEAD_PCT: f64 = 1.0;
const DEFAULT_MAX_DRIFT_PCT: f64 = 5.0;

#[derive(Clone, Copy, Debug)]
enum PathKind {
    Embedded,
    Dynamic,
}

impl PathKind {
    fn name(self) -> &'static str {
        match self {
            Self::Embedded => "embedded",
            Self::Dynamic => "dynamic",
        }
    }

    fn configure(self, device: &Device) -> Result<()> {
        match self {
            Self::Embedded => std::env::remove_var("CANDLE_SM61_EXACT_GROUPED_DISABLE"),
            Self::Dynamic => std::env::set_var("CANDLE_SM61_EXACT_GROUPED_DISABLE", "1"),
        }
        device.as_cuda_device()?.refresh_asd_runtime_policy();
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct Stats {
    median_us: f64,
    p10_us: f64,
    p90_us: f64,
}

struct TempTree(PathBuf);

impl Drop for TempTree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
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

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn deterministic(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * mul) % 101) as isize + bias) as f32 / 64.0)
        .collect()
}

fn tensors(device: &Device) -> Result<(Tensor, Tensor)> {
    let x = Tensor::from_vec(
        deterministic(128 * 32, 37, -50),
        (1, 128, 32),
        device,
    )?;
    let w = Tensor::from_vec(
        deterministic(128 * 16 * 3, 53, -50),
        (128, 16, 3),
        device,
    )?;
    Ok((x, w))
}

fn call(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.conv_transpose1d(w, 1, 1, 2, 1, 8)
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

fn exact_call() -> candle_kernels::asd_exact::ExactOperationCall {
    candle_kernels::asd_exact::ExactOperationCall {
        op: candle_kernels::asd_exact::ExactOperation::ConvTranspose1d,
        dim: 1,
        batch: 1,
        c_in: 128,
        c_out: 128,
        spatial0: 32,
        spatial1: 0,
        weight_rank: 3,
        weight0: 128,
        weight1: 16,
        weight2: 3,
        weight3: 0,
        groups: 8,
        kernel: 3,
        stride: 2,
        padding: 1,
        output_padding: 1,
        dilation: 1,
        dtype: "f32",
        input_contiguous: true,
        input_start_offset: 0,
        weight_contiguous: true,
        weight_start_offset: 0,
    }
}

fn dynamic_identity(evidence_sha256: &str) -> String {
    let canonical = format!(
        "ASD-DECISION-V1\n\
id={DYNAMIC_DECISION_ID}\n\
state=promoted\n\
provider=raw_cuda\n\
op=conv_transpose1d\n\
dim=1\n\
batch=1\n\
c_in=128\n\
c_out=128\n\
spatial=32\n\
weight_shape=128x16x3\n\
groups=8\n\
kernel=3\n\
stride=2\n\
padding=1\n\
output_padding=1\n\
dilation=1\n\
dtype=f32\n\
input_layout=contiguous_zero_offset\n\
weight_layout=contiguous_zero_offset\n\
implementation_id={DYNAMIC_IMPLEMENTATION_ID}\n\
evidence_sha256={evidence_sha256}\n\
min_integrated_speedup_x=none\n"
    );
    sha256_hex(canonical.as_bytes())
}

fn source_cubin() -> Result<PathBuf> {
    let root = candle_kernels::asd_paths::artifacts_dir("sm61").ok_or_else(|| {
        candle_core::Error::Msg("unable to resolve ASD artifacts directory".into())
    })?;
    let path = root.join(format!("{EMBEDDED_IMPLEMENTATION_ID}.cubin"));
    if !path.is_file() {
        candle_core::bail!("missing canonical G8 CUBIN {}", path.display())
    }
    Ok(path)
}

fn setup_dynamic(
    root: &Path,
    profile_id: &str,
    gpu_uuid: &str,
    evidence_sha256: &str,
    cubin_bytes: &[u8],
) -> Result<(PathBuf, PathBuf, String)> {
    let module_dir = root.join("modules");
    std::fs::create_dir_all(&module_dir)?;
    let extension_path = root.join("extensions.asd");
    let dynamic_cubin = module_dir.join(format!("{DYNAMIC_IMPLEMENTATION_ID}.cubin"));
    let dynamic_manifest = module_dir.join(format!("{DYNAMIC_IMPLEMENTATION_ID}.manifest"));
    let embedded_cubin = module_dir.join(format!("{EMBEDDED_IMPLEMENTATION_ID}.cubin"));
    let embedded_manifest_source = candle_kernels::asd_paths::artifacts_dir("sm61")
        .map(|dir| dir.join(format!("{EMBEDDED_IMPLEMENTATION_ID}.manifest")))
        .ok_or_else(|| candle_core::Error::Msg("unable to resolve canonical manifest".into()))?;
    let embedded_manifest = module_dir.join(format!("{EMBEDDED_IMPLEMENTATION_ID}.manifest"));

    let cubin_sha256 = sha256_hex(cubin_bytes);
    let decision_identity = dynamic_identity(evidence_sha256);

    std::fs::write(&dynamic_cubin, cubin_bytes)?;
    std::fs::write(&embedded_cubin, cubin_bytes)?;
    std::fs::copy(&embedded_manifest_source, &embedded_manifest)?;

    std::fs::write(
        &extension_path,
        format!(
            "ASD-EXACT-EXTENSIONS-V1\n\
base_profile_id={profile_id}\n\
target.architecture=sm61\n\
target.gpu_uuid={gpu_uuid}\n\
decision|{DYNAMIC_DECISION_ID}|promoted|raw_cuda|{SIGNATURE}|impl={DYNAMIC_IMPLEMENTATION_ID}|evidence={evidence_sha256}|min_integrated_speedup_x=none\n"
        ),
    )?;
    std::fs::write(
        &dynamic_manifest,
        format!(
            "ASD-CUDA-MODULE-V1\n\
abi_version=1\n\
implementation_id={DYNAMIC_IMPLEMENTATION_ID}\n\
architecture=sm61\n\
artifact_kind=cubin\n\
entry={ENTRY}\n\
artifact_sha256={cubin_sha256}\n\
candidate_id={DYNAMIC_CANDIDATE_ID}\n\
kernel_abi=asd.xwo.f32.v1\n\
output_count=8192\n\
grid_x=32\n\
block_x=256\n\
shared_mem_bytes=0\n\
decision_identity_sha256={decision_identity}\n"
        ),
    )?;

    Ok((extension_path, module_dir, decision_identity))
}

fn warmup(
    kind: PathKind,
    x: &Tensor,
    w: &Tensor,
    device: &Device,
    warmup_ms: f64,
    inner: usize,
) -> Result<()> {
    kind.configure(device)?;
    let requested = Duration::from_secs_f64(warmup_ms / 1000.0);
    let started = Instant::now();
    let mut launches = 0usize;
    loop {
        let mut outputs = Vec::with_capacity(inner);
        for _ in 0..inner {
            outputs.push(call(x, w)?);
        }
        device.synchronize()?;
        std::hint::black_box(outputs);
        launches += inner;
        if started.elapsed() >= requested {
            break;
        }
    }
    println!(
        "WARMUP path={} requested_ms={:.3} actual_ms={:.3} launches={}",
        kind.name(),
        warmup_ms,
        started.elapsed().as_secs_f64() * 1000.0,
        launches
    );
    Ok(())
}

fn measure(
    phase: &str,
    kind: PathKind,
    x: &Tensor,
    w: &Tensor,
    device: &Device,
    iters: usize,
    inner: usize,
) -> Result<Stats> {
    kind.configure(device)?;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        device.synchronize()?;
        let start = Instant::now();
        let mut outputs = Vec::with_capacity(inner);
        for _ in 0..inner {
            outputs.push(call(x, w)?);
        }
        device.synchronize()?;
        samples.push(start.elapsed().as_secs_f64() * 1_000_000.0 / inner as f64);
        std::hint::black_box(outputs);
    }
    samples.sort_unstable_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));
    let stats = Stats {
        median_us: percentile(&samples, 0.50),
        p10_us: percentile(&samples, 0.10),
        p90_us: percentile(&samples, 0.90),
    };
    println!(
        "PHASE phase={phase} path={} median_us={:.6} p10_us={:.6} p90_us={:.6}",
        kind.name(),
        stats.median_us,
        stats.p10_us,
        stats.p90_us
    );
    Ok(stats)
}

fn main() -> Result<()> {
    let warmup_ms = parse_f64("--warmup-ms", DEFAULT_WARMUP_MS);
    let iters = parse_count("--iters", DEFAULT_ITERS);
    let inner = parse_count("--inner", DEFAULT_INNER);
    let max_overhead_pct = parse_f64("--max-overhead-pct", DEFAULT_MAX_OVERHEAD_PCT);
    let max_drift_pct = parse_f64("--max-drift-pct", DEFAULT_MAX_DRIFT_PCT);
    let reverse_phase_order = has_flag("--reverse-phase-order");
    if !warmup_ms.is_finite()
        || warmup_ms <= 0.0
        || iters == 0
        || inner == 0
        || !max_overhead_pct.is_finite()
        || max_overhead_pct < 0.0
        || !max_drift_pct.is_finite()
        || max_drift_pct < 0.0
    {
        candle_core::bail!("invalid benchmark configuration")
    }

    let profile_id = candle_kernels::asd_exact::PROFILE_ID.ok_or_else(|| {
        candle_core::Error::Msg("benchmark requires embedded Exact Profile".into())
    })?;
    let embedded_uuid = candle_kernels::asd_exact::TARGET_GPU_UUID.ok_or_else(|| {
        candle_core::Error::Msg("benchmark requires device-scoped Exact Profile".into())
    })?;

    std::env::remove_var("CANDLE_SM61_EXACT_GROUPED_DISABLE");
    std::env::set_var("CANDLE_ASD_EXACT_TRACE", "0");
    std::env::set_var("CANDLE_SM61_EXACT_GROUPED_TRACE", "0");
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_TRACE", "0");
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "auto");
    std::env::remove_var("CANDLE_ASD_EXACT_DISABLE");
    std::env::remove_var("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL");
    std::env::remove_var("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT");

    let device = Device::new_cuda(0)?;
    let gpu_uuid = gpu_uuid(&device)?;
    if gpu_uuid != embedded_uuid {
        candle_core::bail!("GPU UUID mismatch actual={} embedded={}", gpu_uuid, embedded_uuid)
    }
    let embedded = match candle_kernels::asd_exact::lookup(exact_call(), Some(&gpu_uuid)) {
        Some(candle_kernels::asd_exact::ExactMatch::Proven(matched)) => matched,
        _ => candle_core::bail!("missing embedded G8 decision"),
    };
    if embedded.decision_id != EMBEDDED_DECISION_ID
        || embedded.implementation_id != EMBEDDED_IMPLEMENTATION_ID
    {
        candle_core::bail!("unexpected embedded G8 decision")
    }

    let source_cubin = source_cubin()?;
    let cubin_bytes = std::fs::read(&source_cubin)?;
    let root = std::env::temp_dir().join(format!(
        "candle-asd-dynamic-selection-perf-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let _cleanup = TempTree(root.clone());
    let (extension_path, module_dir, dynamic_identity) = setup_dynamic(
        &root,
        profile_id,
        &gpu_uuid,
        embedded.evidence_sha256,
        &cubin_bytes,
    )?;

    std::env::set_var("CANDLE_ASD_RUNTIME_PROFILE", &extension_path);
    std::env::set_var("CANDLE_ASD_MODULE_DIR", &module_dir);
    device.as_cuda_device()?.refresh_asd();

    let (x, w) = tensors(&device)?;

    // Prime both paths and resolve both CUDA functions before measurement.
    PathKind::Embedded.configure(&device)?;
    let embedded_out = call(&x, &w)?;
    device.synchronize()?;
    PathKind::Dynamic.configure(&device)?;
    let dynamic_out = call(&x, &w)?;
    device.synchronize()?;

    let embedded_flat = embedded_out.flatten_all()?.to_vec1::<f32>()?;
    let dynamic_flat = dynamic_out.flatten_all()?.to_vec1::<f32>()?;
    let max_abs = embedded_flat
        .iter()
        .zip(&dynamic_flat)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    if max_abs != 0.0 {
        candle_core::bail!("embedded/dynamic parity failure max_abs={max_abs}")
    }

    let embedded_source = device
        .as_cuda_device()?
        .asd_resolved_module_source(EMBEDDED_IMPLEMENTATION_ID)?
        .unwrap_or("none");
    let dynamic_source = device
        .as_cuda_device()?
        .asd_resolved_module_source(DYNAMIC_IMPLEMENTATION_ID)?
        .unwrap_or("none");
    if embedded_source != "external_cubin" || dynamic_source != "external_cubin" {
        candle_core::bail!(
            "unexpected sources embedded={} dynamic={}",
            embedded_source,
            dynamic_source
        )
    }

    // Equal settling before authoritative alternating phases.
    warmup(PathKind::Embedded, &x, &w, &device, warmup_ms, inner)?;
    warmup(PathKind::Dynamic, &x, &w, &device, warmup_ms, inner)?;

    println!("=== ASD V3 DYNAMIC SELECTION PERFORMANCE VALIDATION ===");
    println!("gpu_uuid={gpu_uuid}");
    println!("base_profile_id={profile_id}");
    println!("embedded_decision_id={EMBEDDED_DECISION_ID}");
    println!("dynamic_decision_id={DYNAMIC_DECISION_ID}");
    println!("embedded_implementation_id={EMBEDDED_IMPLEMENTATION_ID}");
    println!("dynamic_implementation_id={DYNAMIC_IMPLEMENTATION_ID}");
    println!("dynamic_decision_identity_sha256={dynamic_identity}");
    println!("artifact_sha256={}", sha256_hex(&cubin_bytes));
    println!("embedded_source={embedded_source}");
    println!("dynamic_source={dynamic_source}");
    println!("same_cuda_artifact=true");
    println!("PARITY max_abs={max_abs:.8} pass=true");
    println!("warmup_ms={warmup_ms:.3}");
    println!("timed_samples={iters}");
    println!("launches_per_sample={inner}");
    println!("max_overhead_pct={max_overhead_pct:.3}");
    println!("max_drift_pct={max_drift_pct:.3}");
    let (e1, d1, e2, d2, e3, d3) = if reverse_phase_order {
        println!("sequence=d1,e1,d2,e2,d3,e3");
        let d1 = measure("d1", PathKind::Dynamic, &x, &w, &device, iters, inner)?;
        let e1 = measure("e1", PathKind::Embedded, &x, &w, &device, iters, inner)?;
        let d2 = measure("d2", PathKind::Dynamic, &x, &w, &device, iters, inner)?;
        let e2 = measure("e2", PathKind::Embedded, &x, &w, &device, iters, inner)?;
        let d3 = measure("d3", PathKind::Dynamic, &x, &w, &device, iters, inner)?;
        let e3 = measure("e3", PathKind::Embedded, &x, &w, &device, iters, inner)?;
        (e1, d1, e2, d2, e3, d3)
    } else {
        println!("sequence=e1,d1,e2,d2,e3,d3");
        let e1 = measure("e1", PathKind::Embedded, &x, &w, &device, iters, inner)?;
        let d1 = measure("d1", PathKind::Dynamic, &x, &w, &device, iters, inner)?;
        let e2 = measure("e2", PathKind::Embedded, &x, &w, &device, iters, inner)?;
        let d2 = measure("d2", PathKind::Dynamic, &x, &w, &device, iters, inner)?;
        let e3 = measure("e3", PathKind::Embedded, &x, &w, &device, iters, inner)?;
        let d3 = measure("d3", PathKind::Dynamic, &x, &w, &device, iters, inner)?;
        (e1, d1, e2, d2, e3, d3)
    };

    let embedded_us = median3(e1.median_us, e2.median_us, e3.median_us);
    let dynamic_us = median3(d1.median_us, d2.median_us, d3.median_us);
    let overhead_pct = (dynamic_us / embedded_us - 1.0) * 100.0;
    let embedded_drift_pct = relative_drift_pct(e1.median_us, e3.median_us);
    let dynamic_drift_pct = relative_drift_pct(d1.median_us, d3.median_us);
    let drift_pass =
        embedded_drift_pct <= max_drift_pct && dynamic_drift_pct <= max_drift_pct;
    let overhead_pass = overhead_pct <= max_overhead_pct;
    let pass = drift_pass && overhead_pass;

    println!("embedded_consensus_median_us={embedded_us:.6}");
    println!("dynamic_consensus_median_us={dynamic_us:.6}");
    println!("dynamic_overhead_pct={overhead_pct:.6}");
    println!("embedded_drift_pct={embedded_drift_pct:.6}");
    println!("dynamic_drift_pct={dynamic_drift_pct:.6}");
    println!("drift_pass={drift_pass}");
    println!("overhead_pass={overhead_pass}");
    println!("steady_state_filesystem_io=false");
    println!("steady_state_manifest_parse=false");
    println!("steady_state_artifact_sha=false");
    println!(
        "RESULT={}",
        if pass {
            "NO_MEASURABLE_SELECTION_REGRESSION"
        } else {
            "SELECTION_REGRESSION_OR_UNSTABLE"
        }
    );
    println!("STATUS={}", if pass { "PASS" } else { "FAIL" });

    if !pass {
        candle_core::bail!(
            "dynamic selection performance gate failed overhead_pct={:.6} embedded_drift_pct={:.6} dynamic_drift_pct={:.6}",
            overhead_pct,
            embedded_drift_pct,
            dynamic_drift_pct
        )
    }

    Ok(())
}
