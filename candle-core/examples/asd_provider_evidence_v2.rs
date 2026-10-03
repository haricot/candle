use candle_core::{Device, Result, Tensor};
use cudarc::cudnn::safe::{ConvBackwardData, Cudnn};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

const RAW_MODULE_ABI_VERSION: u32 = 1;
const CHALLENGER_IMPLEMENTATION_ID: &str = "candle.cudnn.grouped-transpose.v1";
const PROTOCOL_ID: &str = "provider-evidence-v2";
const HARNESS_REVISION: &str = "provider-evidence-v2-ct1d-matrix-r3";

#[derive(Clone, Copy, Debug)]
struct Ct1dCase {
    decision_id: &'static str,
    implementation_id: &'static str,
    candidate_id: &'static str,
    entry: &'static str,
    groups: usize,
}

impl Ct1dCase {
    const fn weight_c_out_per_group(self) -> usize {
        128 / self.groups
    }

    fn exact_signature(self) -> String {
        format!(
            "op=conv_transpose1d,dim=1,batch=1,c_in=128,c_out=128,spatial=32,weight_shape=128x{}x3,groups={},kernel=3,stride=2,padding=1,output_padding=1,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset",
            self.weight_c_out_per_group(),
            self.groups
        )
    }
}

const CT1D_CASES: &[Ct1dCase] = &[
    Ct1dCase {
        decision_id: "ct1d-sm61-s32-g2-raw-exact",
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256",
        candidate_id: "ct1d-s32-g2-u1-b256",
        entry: "flow_v0322_ct1d_s32_g2_u1_b256",
        groups: 2,
    },
    Ct1dCase {
        decision_id: "ct1d-sm61-s32-g4-raw-exact",
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g4-u1-b256",
        candidate_id: "ct1d-s32-g4-u1-b256",
        entry: "flow_v0322_ct1d_s32_g4_u1_b256",
        groups: 4,
    },
    Ct1dCase {
        decision_id: "ct1d-sm61-s32-g8-raw-exact",
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g8-u1-b256",
        candidate_id: "ct1d-s32-g8-u1-b256",
        entry: "flow_v0322_ct1d_s32_g8_u1_b256",
        groups: 8,
    },
    Ct1dCase {
        decision_id: "ct1d-sm61-s32-g16-raw-exact",
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g16-u1-b256",
        candidate_id: "ct1d-s32-g16-u1-b256",
        entry: "flow_v0322_ct1d_s32_g16_u1_b256",
        groups: 16,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Purpose {
    PerformanceChallenge,
    FallbackQualification,
}

impl Purpose {
    const fn as_str(self) -> &'static str {
        match self {
            Self::PerformanceChallenge => "performance-challenge",
            Self::FallbackQualification => "fallback-qualification",
        }
    }
}

const DEFAULT_WARMUP_MS: f64 = 500.0;
const DEFAULT_ITERS: usize = 40;
const DEFAULT_INNER: usize = 32;
const DEFAULT_MAX_DRIFT_PCT: f64 = 5.0;
const DEFAULT_MIN_SPEEDUP_X: f64 = 1.01;

#[derive(Clone, Copy, Debug)]
enum Provider {
    IncumbentRaw,
    ChallengerCudnn,
}

impl Provider {
    const fn execution_provider(self) -> &'static str {
        match self {
            Self::IncumbentRaw => "raw_cuda",
            Self::ChallengerCudnn => "cudnn",
        }
    }

    const fn implementation_id(self, case: Ct1dCase) -> &'static str {
        match self {
            Self::IncumbentRaw => case.implementation_id,
            Self::ChallengerCudnn => CHALLENGER_IMPLEMENTATION_ID,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Replication {
    R1,
    R2,
}

impl Replication {
    const fn as_str(self) -> &'static str {
        match self {
            Self::R1 => "r1",
            Self::R2 => "r2",
        }
    }

    const fn phase_sequence(self) -> &'static str {
        match self {
            Self::R1 => "a1,b1,b2,a2,a3,b3",
            Self::R2 => "b1,a1,a2,b2,b3,a3",
        }
    }

    const fn settling_sequence(self) -> &'static str {
        match self {
            Self::R1 => "a,b",
            Self::R2 => "b,a",
        }
    }
}

#[derive(Clone, Debug)]
struct GitSnapshot {
    commit: String,
    clean: bool,
    status: &'static str,
}

#[derive(Clone, Debug)]
struct GpuWorkloadSnapshot {
    status: &'static str,
    active_compute_processes: usize,
    utilization_pct: Option<u64>,
    clean: bool,
}

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

#[derive(Clone, Debug)]
struct TelemetrySnapshot {
    status: &'static str,
    gpu_temp_c: Option<f64>,
    sm_clock_mhz: Option<u64>,
    memory_clock_mhz: Option<u64>,
    power_w: Option<f64>,
    pstate: Option<String>,
    throttle_mask: Option<String>,
    thermal_state: &'static str,
    throttle_state: &'static str,
}

impl TelemetrySnapshot {
    fn unavailable() -> Self {
        Self {
            status: "unavailable",
            gpu_temp_c: None,
            sm_clock_mhz: None,
            memory_clock_mhz: None,
            power_w: None,
            pstate: None,
            throttle_mask: None,
            thermal_state: "UNKNOWN",
            throttle_state: "UNKNOWN",
        }
    }
}

#[derive(Clone, Debug)]
struct CudnnIdentity {
    version_raw: usize,
    algorithm_id: i32,
    algorithm_debug: String,
    workspace_bytes: usize,
    identity: String,
}

#[derive(Clone, Debug)]
struct RawIdentity {
    source: &'static str,
    identity: String,
    artifact_sha256: String,
    proof_status: &'static str,
    verified: bool,
}

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
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

fn parse_path(flag: &str) -> Option<PathBuf> {
    let args = std::env::args().collect::<Vec<_>>();
    args.windows(2)
        .find_map(|pair| (pair[0] == flag).then(|| PathBuf::from(&pair[1])))
}

fn parse_value(flag: &str) -> Option<String> {
    let args = std::env::args().collect::<Vec<_>>();
    args.windows(2)
        .find_map(|pair| (pair[0] == flag).then(|| pair[1].clone()))
}

fn parse_case() -> Result<Ct1dCase> {
    let requested = parse_value("--decision")
        .unwrap_or_else(|| CT1D_CASES[0].decision_id.to_owned());
    CT1D_CASES
        .iter()
        .copied()
        .find(|case| case.decision_id == requested)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "invalid --decision {requested:?}; expected one of {}",
                CT1D_CASES
                    .iter()
                    .map(|case| case.decision_id)
                    .collect::<Vec<_>>()
                    .join(",")
            ))
        })
}

fn parse_purpose() -> Result<Purpose> {
    match parse_value("--purpose").as_deref() {
        None | Some("performance-challenge") => Ok(Purpose::PerformanceChallenge),
        Some("fallback-qualification") => Ok(Purpose::FallbackQualification),
        Some(other) => candle_core::bail!(
            "invalid --purpose {other:?}; expected performance-challenge or fallback-qualification"
        ),
    }
}

fn parse_replication() -> Result<Option<Replication>> {
    let args = std::env::args().collect::<Vec<_>>();
    let value = args
        .windows(2)
        .find_map(|pair| (pair[0] == "--replication").then(|| pair[1].as_str()));
    match value {
        None => Ok(None),
        Some("r1") => Ok(Some(Replication::R1)),
        Some("r2") => Ok(Some(Replication::R2)),
        Some(other) => candle_core::bail!("invalid --replication {other:?}; expected r1 or r2"),
    }
}

fn git_snapshot() -> GitSnapshot {
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());

    let status = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok());

    match (commit, status) {
        (Some(commit), Some(status)) => GitSnapshot {
            commit,
            clean: status.trim().is_empty(),
            status: "ok",
        },
        (commit, _) => GitSnapshot {
            commit: commit.unwrap_or_else(|| "unavailable".to_owned()),
            clean: false,
            status: "unavailable",
        },
    }
}

fn query_gpu_workload(gpu_uuid: &str) -> GpuWorkloadSnapshot {
    let apps = match Command::new("nvidia-smi")
        .arg("-i")
        .arg(gpu_uuid)
        .arg("--query-compute-apps=pid,process_name")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => {
            return GpuWorkloadSnapshot {
                status: "unavailable",
                active_compute_processes: 0,
                utilization_pct: None,
                clean: false,
            };
        }
    };
    let apps_stdout = match String::from_utf8(apps.stdout) {
        Ok(stdout) => stdout,
        Err(_) => {
            return GpuWorkloadSnapshot {
                status: "unavailable",
                active_compute_processes: 0,
                utilization_pct: None,
                clean: false,
            };
        }
    };
    let active_compute_processes = apps_stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !line.to_ascii_lowercase().contains("no running"))
        .count();

    let utilization_pct = Command::new("nvidia-smi")
        .arg("-i")
        .arg(gpu_uuid)
        .arg("--query-gpu=utilization.gpu")
        .arg("--format=csv,noheader,nounits")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|stdout| stdout.lines().find_map(|line| line.trim().parse::<u64>().ok()));

    let status = if utilization_pct.is_some() {
        "ok"
    } else {
        "partial"
    };
    let clean = active_compute_processes == 0
        && utilization_pct.is_some_and(|utilization| utilization <= 1);

    GpuWorkloadSnapshot {
        status,
        active_compute_processes,
        utilization_pct,
        clean,
    }
}

fn sha256_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn input_identity(case: Ct1dCase) -> (String, String) {
    let x = deterministic(128 * 32, 37, -50);
    let w = deterministic(128 * case.weight_c_out_per_group() * 3, 53, -50);
    (sha256_f32(&x), sha256_f32(&w))
}

fn deterministic(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * mul) % 101) as isize + bias) as f32 / 64.0)
        .collect()
}

fn tensors(device: &Device, case: Ct1dCase) -> Result<(Tensor, Tensor)> {
    let x = Tensor::from_vec(deterministic(128 * 32, 37, -50), (1, 128, 32), device)?;
    let w = Tensor::from_vec(
        deterministic(128 * case.weight_c_out_per_group() * 3, 53, -50),
        (128, case.weight_c_out_per_group(), 3),
        device,
    )?;
    Ok((x, w))
}

fn call(x: &Tensor, w: &Tensor, case: Ct1dCase) -> Result<Tensor> {
    x.conv_transpose1d(w, 1, 1, 2, 1, case.groups)
}

fn configure_provider(provider: Provider) {
    // Provider measurements must never inherit tracing that writes from the
    // hot execution path.
    std::env::remove_var("CANDLE_ASD_EXACT_TRACE");
    std::env::remove_var("CANDLE_SM61_EXACT_GROUPED_TRACE");
    std::env::remove_var("CANDLE_GROUPED_TRANSPOSE_TRACE");
    std::env::remove_var("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL");
    std::env::remove_var("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_ASD_EXACT_DISABLE");

    match provider {
        Provider::IncumbentRaw => std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "auto"),
        Provider::ChallengerCudnn => {
            std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "cudnn")
        }
    }
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

fn exact_call(case: Ct1dCase) -> candle_kernels::asd_exact::ExactOperationCall {
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
        weight1: case.weight_c_out_per_group(),
        weight2: 3,
        weight3: 0,
        groups: case.groups,
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

fn incumbent_profile_match(
    gpu_uuid: &str,
    case: Ct1dCase,
) -> Result<candle_kernels::asd_exact::ExactAsdMatch> {
    if candle_kernels::asd_exact::PROFILE_ID.is_none() {
        candle_core::bail!(
            "Provider Evidence V2 requires an embedded ASD Exact Profile; rebuild with CANDLE_ASD_EXACT_POLICY pointing to the Stage2F production profile"
        )
    }
    if candle_kernels::asd_exact::TARGET_GPU_UUID.is_none() {
        candle_core::bail!(
            "Provider Evidence V2 requires a device-scoped ASD Exact Profile with target GPU UUID"
        )
    }

    let matched = match candle_kernels::asd_exact::lookup(exact_call(case), Some(gpu_uuid)) {
        Some(candle_kernels::asd_exact::ExactMatch::Proven(matched)) => matched,
        Some(candle_kernels::asd_exact::ExactMatch::Unproven(_)) => {
            candle_core::bail!("{} Exact Profile match is not proven", case.decision_id)
        }
        None => candle_core::bail!("missing Exact Profile decision {}", case.decision_id),
    };
    if matched.decision_id != case.decision_id
        || matched.state != "promoted"
        || matched.execution_provider != candle_kernels::asd_exact::ExactExecutionProvider::RawCuda
        || matched.implementation_id != case.implementation_id
    {
        candle_core::bail!(
            "unexpected CT1D incumbent profile decision expected={} id={} state={} provider={} impl={}",
            case.decision_id,
            matched.decision_id,
            matched.state,
            matched.execution_provider.as_str(),
            matched.implementation_id
        )
    }
    Ok(matched)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn raw_artifact_root() -> Option<PathBuf> {
    std::env::var_os("CANDLE_ASD_MODULE_DIR")
        .filter(|root| !root.is_empty())
        .map(PathBuf::from)
        .or_else(|| candle_kernels::asd_paths::artifacts_dir("sm61"))
}

fn external_manifest_verified(
    artifact_path: &Path,
    artifact_kind: &str,
    artifact_sha256: &str,
    case: Ct1dCase,
) -> bool {
    let manifest_path = artifact_path.with_extension("manifest");
    let Ok(source) = std::fs::read_to_string(manifest_path) else {
        return false;
    };
    let mut lines = source.lines();
    if lines.next() != Some("ASD-CUDA-MODULE-V1") {
        return false;
    }
    let mut fields = std::collections::BTreeMap::<&str, &str>::new();
    for raw in lines {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return false;
        };
        fields.insert(key.trim(), value.trim());
    }
    fields.get("abi_version").copied() == Some("1")
        && fields.get("implementation_id").copied() == Some(case.implementation_id)
        && fields.get("architecture").copied() == Some("sm61")
        && fields.get("artifact_kind").copied() == Some(artifact_kind)
        && fields.get("entry").copied() == Some(case.entry)
        && fields.get("artifact_sha256").copied() == Some(artifact_sha256)
}

fn raw_identity(case: Ct1dCase) -> Result<RawIdentity> {
    if let Some(root) = raw_artifact_root() {
        for (extension, source_name) in [("cubin", "external_cubin"), ("ptx", "external_ptx")] {
            let path = root.join(format!("{}.{extension}", case.implementation_id));
            if !path.is_file() {
                continue;
            }
            let bytes = std::fs::read(&path).map_err(|err| {
                candle_core::Error::Msg(format!(
                    "failed to read incumbent ASD artifact {}: {err}",
                    path.display()
                ))
            })?;
            let artifact_sha256 = sha256_bytes(&bytes);
            let verified = external_manifest_verified(&path, extension, &artifact_sha256, case);
            return Ok(RawIdentity {
                source: source_name,
                identity: format!(
                    "raw_cuda:{source_name}:abi={RAW_MODULE_ABI_VERSION}:implementation={}:entry={}:artifact_sha256={artifact_sha256}",
                    case.implementation_id,
                    case.entry
                ),
                artifact_sha256,
                proof_status: if verified {
                    "external_artifact_verified"
                } else {
                    "external_artifact_unverified"
                },
                verified,
            });
        }
    }

    if env_truthy("CANDLE_ASD_BUILTIN_RAW_DISABLE") {
        candle_core::bail!(
            "Provider Evidence V2 incumbent raw implementation is unavailable: no external artifact and builtin raw is disabled"
        )
    }

    let ptx = candle_kernels::sm61_exact_grouped_ptx(case.candidate_id)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "missing builtin PTX for {}",
                case.implementation_id
            ))
        })?;
    let artifact_sha256 = sha256_bytes(ptx.as_bytes());
    Ok(RawIdentity {
        source: "builtin_ptx",
        identity: format!(
            "raw_cuda:builtin_ptx:abi={RAW_MODULE_ABI_VERSION}:implementation={}:entry={}:artifact_sha256={artifact_sha256}",
            case.implementation_id,
            case.entry
        ),
        artifact_sha256,
        proof_status: "historical_evidence_bound",
        verified: true,
    })
}

fn cudnn_identity(device: &Device, case: Ct1dCase) -> Result<CudnnIdentity> {
    let dev = device.as_cuda_device()?;
    let cudnn = Cudnn::new(dev.cuda_stream())?;
    let mut conv = cudnn.create_conv2d::<f32>(
        [1, 0],
        [2, 1],
        [1, 1],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    conv.set_group_count(case.groups as i32)?;

    let dx = cudnn.create_4d_tensor::<f32>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [1, 128, 64, 1],
    )?;
    let w = cudnn.create_4d_filter::<f32>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [128, case.weight_c_out_per_group() as i32, 3, 1],
    )?;
    let dy = cudnn.create_4d_tensor::<f32>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [1, 128, 32, 1],
    )?;

    let op = ConvBackwardData {
        conv: &conv,
        dx: &dx,
        w: &w,
        dy: &dy,
    };
    let algo = op.pick_algorithm()?;
    let workspace_bytes = op.get_workspace_size(algo)?;
    let algorithm_id = algo as i32;
    let algorithm_debug = format!("{algo:?}");
    let version_raw = unsafe { cudarc::cudnn::sys::cudnnGetVersion() };

    let identity = format!(
        "cudnn:runtime_version={version_raw}:operation=conv_backward_data:algorithm_id={algorithm_id}:workspace_bytes={workspace_bytes}:dtype=f32:n=1:ci=128:co=128:l=32:groups={}:k=3:s=2:p=1:op=1:d=1",
        case.groups
    );

    Ok(CudnnIdentity {
        version_raw,
        algorithm_id,
        algorithm_debug,
        workspace_bytes,
        identity,
    })
}

fn max_abs_rel(lhs: &Tensor, rhs: &Tensor) -> Result<(f32, f32)> {
    if lhs.dims() != rhs.dims() {
        candle_core::bail!(
            "provider parity shape mismatch {:?} vs {:?}",
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
    provider: Provider,
    x: &Tensor,
    w: &Tensor,
    device: &Device,
    case: Ct1dCase,
    warmup_ms: f64,
    inner: usize,
) -> Result<()> {
    configure_provider(provider);
    let requested = Duration::from_secs_f64(warmup_ms / 1000.0);
    let started = Instant::now();
    let mut batches = 0usize;
    let mut launches = 0usize;
    loop {
        let mut outputs = Vec::with_capacity(inner);
        for _ in 0..inner {
            outputs.push(call(x, w, case)?);
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
        "WARMUP phase={phase} provider={} requested_ms={warmup_ms:.3} actual_ms={:.3} batches={batches} launches={launches}",
        provider.execution_provider(),
        started.elapsed().as_secs_f64() * 1000.0,
    );
    Ok(())
}

fn settle_pair(
    replication: Replication,
    x: &Tensor,
    w: &Tensor,
    device: &Device,
    case: Ct1dCase,
    warmup_ms: f64,
    inner: usize,
) -> Result<()> {
    println!("=== NON-AUTHORITATIVE SETTLING ===");
    println!("settling_policy=time_equivalent_pair_balanced_by_replication");
    println!("settling_sequence={}", replication.settling_sequence());
    let settle = |phase: &str, provider: Provider| {
        timed_warmup(phase, provider, x, w, device, case, warmup_ms, inner)
    };
    match replication {
        Replication::R1 => {
            settle("settle_a", Provider::IncumbentRaw)?;
            settle("settle_b", Provider::ChallengerCudnn)?;
        }
        Replication::R2 => {
            settle("settle_b", Provider::ChallengerCudnn)?;
            settle("settle_a", Provider::IncumbentRaw)?;
        }
    }
    device.synchronize()?;
    println!("settling_complete=true");
    Ok(())
}

fn measure(
    phase: &str,
    provider: Provider,
    x: &Tensor,
    w: &Tensor,
    device: &Device,
    case: Ct1dCase,
    cfg: MeasureConfig,
) -> Result<TimingStats> {
    timed_warmup(
        phase,
        provider,
        x,
        w,
        device,
        case,
        cfg.warmup_ms,
        cfg.inner,
    )?;

    let mut samples = Vec::with_capacity(cfg.iters);
    for _ in 0..cfg.iters {
        device.synchronize()?;
        let start = Instant::now();
        let mut outputs = Vec::with_capacity(cfg.inner);
        for _ in 0..cfg.inner {
            outputs.push(call(x, w, case)?);
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

fn print_phase(
    name: &str,
    provider: Provider,
    case: Ct1dCase,
    identity: &str,
    stats: TimingStats,
) {
    println!(
        "PHASE phase={name} provider={} implementation={} identity={} median_us={:.6} p10_us={:.6} p90_us={:.6}",
        provider.execution_provider(),
        provider.implementation_id(case),
        identity,
        stats.median_us,
        stats.p10_us,
        stats.p90_us
    );
}

fn parse_optional_f64(value: &str) -> Option<f64> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("N/A") {
        None
    } else {
        value.parse().ok()
    }
}

fn parse_optional_u64(value: &str) -> Option<u64> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("N/A") {
        None
    } else {
        value.parse().ok()
    }
}

fn parse_throttle_mask(value: &str) -> Option<u64> {
    let value = value.trim();
    let hex = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))?;
    u64::from_str_radix(hex, 16).ok()
}

fn classify_throttle(mask: Option<u64>) -> (&'static str, &'static str) {
    let Some(mask) = mask else {
        return ("UNKNOWN", "UNKNOWN");
    };
    let thermal = mask & (0x20 | 0x40) != 0;
    let power = mask & (0x04 | 0x80) != 0;
    let thermal_state = if thermal {
        "THERMAL_THROTTLE_ACTIVE"
    } else {
        "NO_THERMAL_THROTTLE_OBSERVED"
    };
    let throttle_state = match (thermal, power, mask) {
        (true, true, _) => "THERMAL_AND_POWER",
        (true, false, _) => "THERMAL",
        (false, true, _) => "POWER",
        (false, false, 0) => "NONE",
        (false, false, 1) => "IDLE",
        _ => "OTHER_ACTIVE",
    };
    (thermal_state, throttle_state)
}

fn query_telemetry(gpu_uuid: &str) -> TelemetrySnapshot {
    const QUERY: &str =
        "temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,pstate,clocks_throttle_reasons.active";
    let output = match Command::new("nvidia-smi")
        .arg("-i")
        .arg(gpu_uuid)
        .arg(format!("--query-gpu={QUERY}"))
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => return TelemetrySnapshot::unavailable(),
    };
    let stdout = match String::from_utf8(output.stdout) {
        Ok(stdout) => stdout,
        Err(_) => return TelemetrySnapshot::unavailable(),
    };
    let Some(line) = stdout.lines().find(|line| !line.trim().is_empty()) else {
        return TelemetrySnapshot::unavailable();
    };
    let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
    if fields.len() != 6 {
        return TelemetrySnapshot::unavailable();
    }
    let throttle_mask = (!fields[5].is_empty() && !fields[5].eq_ignore_ascii_case("N/A"))
        .then(|| fields[5].to_owned());
    let (thermal_state, throttle_state) =
        classify_throttle(throttle_mask.as_deref().and_then(parse_throttle_mask));
    TelemetrySnapshot {
        status: "ok",
        gpu_temp_c: parse_optional_f64(fields[0]),
        sm_clock_mhz: parse_optional_u64(fields[1]),
        memory_clock_mhz: parse_optional_u64(fields[2]),
        power_w: parse_optional_f64(fields[3]),
        pstate: (!fields[4].is_empty() && !fields[4].eq_ignore_ascii_case("N/A"))
            .then(|| fields[4].to_owned()),
        throttle_mask,
        thermal_state,
        throttle_state,
    }
}

fn fmt_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "na".to_owned())
}

fn fmt_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "na".to_owned())
}

fn fmt_string(value: Option<&str>) -> &str {
    value.unwrap_or("na")
}

fn print_telemetry(phase: &str, snapshot: &TelemetrySnapshot) {
    println!(
        "TELEMETRY phase={phase} observe_only=true status={} gpu_temp_c={} sm_clock_mhz={} memory_clock_mhz={} power_w={} pstate={} throttle_mask={} thermal_state={} throttle_state={}",
        snapshot.status,
        fmt_f64(snapshot.gpu_temp_c),
        fmt_u64(snapshot.sm_clock_mhz),
        fmt_u64(snapshot.memory_clock_mhz),
        fmt_f64(snapshot.power_w),
        fmt_string(snapshot.pstate.as_deref()),
        fmt_string(snapshot.throttle_mask.as_deref()),
        snapshot.thermal_state,
        snapshot.throttle_state,
    );
}

fn max_f64(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn min_u64(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn max_u64(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn telemetry_status(start: &TelemetrySnapshot, end: &TelemetrySnapshot) -> &'static str {
    match (start.status, end.status) {
        ("ok", "ok") => "complete",
        ("unavailable", "unavailable") => "unavailable",
        _ => "partial",
    }
}

fn main() -> Result<()> {
    let case = parse_case()?;
    let purpose = parse_purpose()?;
    let warmup_ms = parse_f64("--warmup-ms", DEFAULT_WARMUP_MS);
    let iters = parse_count("--iters", DEFAULT_ITERS);
    let inner = parse_count("--inner", DEFAULT_INNER);
    let max_drift_pct = parse_f64("--max-drift-pct", DEFAULT_MAX_DRIFT_PCT);
    let min_speedup_x = parse_f64("--min-speedup-x", DEFAULT_MIN_SPEEDUP_X);
    let evidence_out = parse_path("--evidence-out");
    let replication = parse_replication()?;
    let effective_replication = replication.unwrap_or(Replication::R1);
    let git = git_snapshot();
    let headless_display_env =
        std::env::var_os("DISPLAY").is_none() && std::env::var_os("WAYLAND_DISPLAY").is_none();
    let target_gpu_uuid = candle_kernels::asd_exact::TARGET_GPU_UUID.unwrap_or("unavailable");
    let gpu_workload = query_gpu_workload(target_gpu_uuid);
    let raw = raw_identity(case)?;
    let protocol_defaults = warmup_ms == DEFAULT_WARMUP_MS
        && iters == DEFAULT_ITERS
        && inner == DEFAULT_INNER
        && max_drift_pct == DEFAULT_MAX_DRIFT_PCT
        && min_speedup_x == DEFAULT_MIN_SPEEDUP_X;
    let authoritative_protocol = protocol_defaults
        && replication.is_some()
        && git.status == "ok"
        && git.clean
        && headless_display_env
        && gpu_workload.status == "ok"
        && gpu_workload.clean
        && raw.verified;

    if !warmup_ms.is_finite() || warmup_ms <= 0.0 || iters == 0 || inner == 0 {
        candle_core::bail!("--warmup-ms, --iters and --inner must be greater than zero")
    }
    if max_drift_pct < 0.0 || !min_speedup_x.is_finite() || min_speedup_x <= 0.0 {
        candle_core::bail!("invalid Provider Evidence thresholds")
    }

    std::env::remove_var("CANDLE_ASD_EXACT_TRACE");
    std::env::remove_var("CANDLE_SM61_EXACT_GROUPED_TRACE");
    std::env::remove_var("CANDLE_GROUPED_TRANSPOSE_TRACE");

    let device = Device::new_cuda(0)?;
    let gpu_uuid = gpu_uuid(&device)?;
    let profile_match = incumbent_profile_match(&gpu_uuid, case)?;
    let raw_identity = raw.identity.as_str();
    let raw_artifact_sha256 = raw.artifact_sha256.as_str();
    let raw_source = raw.source;
    let cudnn = cudnn_identity(&device, case)?;
    let (input_sha256, weight_sha256) = input_identity(case);
    let (x, w) = tensors(&device, case)?;

    configure_provider(Provider::IncumbentRaw);
    let out_a = call(&x, &w, case)?;
    device.synchronize()?;
    configure_provider(Provider::ChallengerCudnn);
    let out_b = call(&x, &w, case)?;
    device.synchronize()?;
    let (max_abs, max_rel) = max_abs_rel(&out_a, &out_b)?;
    let parity_pass = max_abs <= 1e-5 || max_rel <= 1e-5;

    let cfg = MeasureConfig {
        warmup_ms,
        iters,
        inner,
    };

    println!("=== ASD PROVIDER EVIDENCE V2 ===");
    println!("protocol={PROTOCOL_ID}");
    println!("harness_revision={HARNESS_REVISION}");
    println!("purpose={}", purpose.as_str());
    println!("measurement_plane=production_path");
    println!("provider_hot_execution_measured=false");
    println!("hot_path_trace_expected=false");
    println!(
        "replication={}",
        replication.map(Replication::as_str).unwrap_or("unspecified")
    );
    println!("source_commit={}", git.commit);
    println!("source_tree_status={}", git.status);
    println!("source_tree_clean={}", git.clean);
    println!("headless_display_env={headless_display_env}");
    println!("gpu_workload_preflight_status={}", gpu_workload.status);
    println!(
        "gpu_workload_preflight_active_compute_processes={}",
        gpu_workload.active_compute_processes
    );
    println!(
        "gpu_workload_preflight_utilization_pct={}",
        fmt_u64(gpu_workload.utilization_pct)
    );
    println!("gpu_workload_preflight_clean={}", gpu_workload.clean);
    println!("workload_class=exact_operator_microbenchmark");
    println!("model_phase_semantics=not_applicable");
    println!("input_sha256={input_sha256}");
    println!("weight_sha256={weight_sha256}");
    println!("profile_id={}", profile_match.profile_id);
    println!("decision_id={}", profile_match.decision_id);
    println!("ct1d_groups={}", case.groups);
    println!("exact_signature={}", case.exact_signature());
    println!("architecture=sm61");
    println!("gpu_uuid={gpu_uuid}");
    println!("incumbent_execution_provider=raw_cuda");
    println!("incumbent_implementation_id={}", case.implementation_id);
    println!("incumbent_identity={raw_identity}");
    println!("incumbent_raw_source={}", raw.source);
    println!("incumbent_raw_proof_status={}", raw.proof_status);
    println!("incumbent_raw_identity_verified={}", raw.verified);
    println!("incumbent_artifact_sha256={raw_artifact_sha256}");
    println!(
        "incumbent_profile_evidence_sha256={}",
        profile_match.evidence_sha256
    );
    println!("challenger_execution_provider=cudnn");
    println!("challenger_implementation_id={CHALLENGER_IMPLEMENTATION_ID}");
    println!("challenger_identity={}", cudnn.identity);
    println!("cudnn_version_raw={}", cudnn.version_raw);
    println!("cudnn_algorithm_id={}", cudnn.algorithm_id);
    println!("cudnn_algorithm_debug={}", cudnn.algorithm_debug);
    println!("cudnn_workspace_bytes={}", cudnn.workspace_bytes);
    println!("cache_state=warm_after_settling");
    println!("cold_cache_measured=false");
    println!("residency_cuda_context=warm_after_settling");
    println!("residency_raw_module=warm_after_settling");
    println!("residency_raw_module_source={raw_source}");
    println!("residency_cudnn_handle=thread_local_cached");
    println!("residency_cudnn_descriptors=recreated_per_call");
    println!("residency_cudnn_algorithm=repicked_per_call");
    println!("residency_cudnn_workspace=allocated_per_call");
    println!("latency_kind=per_launch_from_batched_wall_clock");
    println!("throughput_kind=serial_repeated_exact_op");
    println!("aggregation=median_of_three_phase_medians");
    println!("best_run_selection=forbidden");
    println!("warmup_policy=time_equivalent_per_provider");
    println!("warmup_ms={warmup_ms:.3}");
    println!("settling_policy=time_equivalent_pair_balanced_by_replication_non_authoritative");
    println!("timed_samples={iters}");
    println!("launches_per_sample={inner}");
    println!("max_drift_pct={max_drift_pct:.3}");
    println!("min_speedup_x={min_speedup_x:.6}");
    println!("sequence={}", effective_replication.phase_sequence());
    println!("replication_pairing=deterministic_balanced");
    println!(
        "PROTOCOL_GATE defaults={} replication={} source_clean={} headless={} gpu_idle={} raw_identity_verified={}",
        protocol_defaults,
        replication.is_some(),
        git.clean,
        headless_display_env,
        gpu_workload.clean,
        raw.verified
    );
    println!("authoritative_protocol={authoritative_protocol}");
    println!("PARITY max_abs={max_abs:.8} max_rel={max_rel:.8} pass={parity_pass}");
    println!("telemetry_policy=boundary_snapshots_non_authoritative");
    println!("telemetry_observe_only=true");

    let telemetry_start = query_telemetry(&gpu_uuid);
    print_telemetry("start", &telemetry_start);

    settle_pair(
        effective_replication,
        &x,
        &w,
        &device,
        case,
        warmup_ms,
        inner,
    )?;

    println!("=== AUTHORITATIVE SEQUENCE ===");
    let (a1, b1, a2, b2, a3, b3) = match effective_replication {
        Replication::R1 => {
            let a1 = measure("a1", Provider::IncumbentRaw, &x, &w, &device, case, cfg)?;
            let b1 = measure("b1", Provider::ChallengerCudnn, &x, &w, &device, case, cfg)?;
            let b2 = measure("b2", Provider::ChallengerCudnn, &x, &w, &device, case, cfg)?;
            let a2 = measure("a2", Provider::IncumbentRaw, &x, &w, &device, case, cfg)?;
            let a3 = measure("a3", Provider::IncumbentRaw, &x, &w, &device, case, cfg)?;
            let b3 = measure("b3", Provider::ChallengerCudnn, &x, &w, &device, case, cfg)?;
            (a1, b1, a2, b2, a3, b3)
        }
        Replication::R2 => {
            let b1 = measure("b1", Provider::ChallengerCudnn, &x, &w, &device, case, cfg)?;
            let a1 = measure("a1", Provider::IncumbentRaw, &x, &w, &device, case, cfg)?;
            let a2 = measure("a2", Provider::IncumbentRaw, &x, &w, &device, case, cfg)?;
            let b2 = measure("b2", Provider::ChallengerCudnn, &x, &w, &device, case, cfg)?;
            let b3 = measure("b3", Provider::ChallengerCudnn, &x, &w, &device, case, cfg)?;
            let a3 = measure("a3", Provider::IncumbentRaw, &x, &w, &device, case, cfg)?;
            (a1, b1, a2, b2, a3, b3)
        }
    };

    let telemetry_end = query_telemetry(&gpu_uuid);
    print_telemetry("end", &telemetry_end);

    print_phase("a1", Provider::IncumbentRaw, case, &raw_identity, a1);
    print_phase("b1", Provider::ChallengerCudnn, case, &cudnn.identity, b1);
    print_phase("a2", Provider::IncumbentRaw, case, &raw_identity, a2);
    print_phase("b2", Provider::ChallengerCudnn, case, &cudnn.identity, b2);
    print_phase("a3", Provider::IncumbentRaw, case, &raw_identity, a3);
    print_phase("b3", Provider::ChallengerCudnn, case, &cudnn.identity, b3);

    let a_us = median3(a1.median_us, a2.median_us, a3.median_us);
    let b_us = median3(b1.median_us, b2.median_us, b3.median_us);
    let a_p90 = median3(a1.p90_us, a2.p90_us, a3.p90_us);
    let b_p90 = median3(b1.p90_us, b2.p90_us, b3.p90_us);
    let a_drift_pct = relative_drift_pct(a1.median_us, a3.median_us);
    let b_drift_pct = relative_drift_pct(b1.median_us, b3.median_us);
    let speedup_x = a_us / b_us;
    let latency_change_pct = (b_us / a_us - 1.0) * 100.0;
    let incumbent_ops_per_s = 1_000_000.0 / a_us;
    let challenger_ops_per_s = 1_000_000.0 / b_us;

    let drift_pass = a_drift_pct <= max_drift_pct && b_drift_pct <= max_drift_pct;
    let p90_pass = b_p90 <= a_p90;
    let speedup_pass = speedup_x >= min_speedup_x;
    let harness_pass = parity_pass && drift_pass;
    let candidate_pass = harness_pass && p90_pass && speedup_pass;
    let promotion_result = if !authoritative_protocol {
        "MEASUREMENT_ONLY"
    } else if candidate_pass {
        "PROMOTE_CHALLENGER_SIGNAL"
    } else if harness_pass {
        "REJECT_CHALLENGER"
    } else {
        "HOLD"
    };
    let fallback_qualification = if purpose != Purpose::FallbackQualification {
        "NOT_REQUESTED"
    } else if !authoritative_protocol {
        "MEASUREMENT_ONLY"
    } else if harness_pass {
        "QUALIFIED_REPLICATION"
    } else {
        "NOT_QUALIFIED_REPLICATION"
    };
    let decision = match purpose {
        Purpose::PerformanceChallenge => promotion_result,
        Purpose::FallbackQualification => fallback_qualification,
    };
    let status = if harness_pass { "PASS" } else { "HOLD" };

    println!(
        "PERFORMANCE_RESULT measurement_plane=production_path incumbent_us={a_us:.6} challenger_us={b_us:.6} incumbent_ops_per_s={incumbent_ops_per_s:.3} challenger_ops_per_s={challenger_ops_per_s:.3} speedup_x={speedup_x:.6} challenger_latency_change_pct={latency_change_pct:.3} incumbent_drift_pct={a_drift_pct:.3} challenger_drift_pct={b_drift_pct:.3} incumbent_p90_us={a_p90:.6} challenger_p90_us={b_p90:.6}"
    );
    println!(
        "GATE parity={parity_pass} drift={drift_pass} speedup={speedup_pass} p90_non_regression={p90_pass} required_speedup_x={min_speedup_x:.6}"
    );
    println!("STATUS={status}");
    println!("PROMOTION_RESULT={promotion_result}");
    println!("FALLBACK_QUALIFICATION={fallback_qualification}");
    println!("DECISION={decision}");

    let telemetry_status = telemetry_status(&telemetry_start, &telemetry_end);
    let gpu_temp_c_max_observed = max_f64(telemetry_start.gpu_temp_c, telemetry_end.gpu_temp_c);
    let sm_clock_mhz_min_observed =
        min_u64(telemetry_start.sm_clock_mhz, telemetry_end.sm_clock_mhz);
    let sm_clock_mhz_max_observed =
        max_u64(telemetry_start.sm_clock_mhz, telemetry_end.sm_clock_mhz);
    let memory_clock_mhz_min_observed = min_u64(
        telemetry_start.memory_clock_mhz,
        telemetry_end.memory_clock_mhz,
    );
    let memory_clock_mhz_max_observed = max_u64(
        telemetry_start.memory_clock_mhz,
        telemetry_end.memory_clock_mhz,
    );
    let power_w_max_observed = max_f64(telemetry_start.power_w, telemetry_end.power_w);
    let purpose_name = purpose.as_str();
    let exact_signature = case.exact_signature();
    let ct1d_groups = case.groups;
    let incumbent_implementation_id = case.implementation_id;
    let incumbent_entry = case.entry;

    let evidence = format!(
        "ASD-PROVIDER-PERFORMANCE-EVIDENCE-V2\n\
protocol={PROTOCOL_ID}\n\
harness_revision={HARNESS_REVISION}\n\
purpose={purpose_name}\n\
measurement_plane=production_path\n\
provider_hot_execution_measured=false\n\
hot_path_trace_expected=false\n\
replication={}\n\
source_commit={}\n\
source_tree_status={}\n\
source_tree_clean={}\n\
headless_display_env={}\n\
gpu_workload_preflight_status={}\n\
gpu_workload_preflight_active_compute_processes={}\n\
gpu_workload_preflight_utilization_pct={}\n\
gpu_workload_preflight_clean={}\n\
protocol_defaults_pass={protocol_defaults}\n\
workload_class=exact_operator_microbenchmark\n\
model_phase_semantics=not_applicable\n\
input_sha256={}\n\
weight_sha256={}\n\
profile_id={}\n\
decision_id={}\n\
ct1d_groups={ct1d_groups}\n\
architecture=sm61\n\
gpu_uuid={gpu_uuid}\n\
exact_signature={exact_signature}\n\
incumbent_execution_provider=raw_cuda\n\
incumbent_implementation_id={incumbent_implementation_id}\n\
incumbent_identity={raw_identity}\n\
incumbent_raw_abi_version={RAW_MODULE_ABI_VERSION}\n\
incumbent_raw_entry={incumbent_entry}\n\
incumbent_raw_source={}\n\
incumbent_raw_proof_status={}\n\
incumbent_raw_identity_verified={}\n\
incumbent_artifact_sha256={raw_artifact_sha256}\n\
incumbent_profile_evidence_sha256={}\n\
challenger_execution_provider=cudnn\n\
challenger_implementation_id={CHALLENGER_IMPLEMENTATION_ID}\n\
challenger_identity={}\n\
cudnn_version_raw={}\n\
cudnn_operation=conv_backward_data\n\
cudnn_algorithm_id={}\n\
cudnn_algorithm_debug={}\n\
cudnn_workspace_bytes={}\n\
cache_state=warm_after_settling\n\
cold_cache_measured=false\n\
residency_cuda_context=warm_after_settling\n\
residency_raw_module=warm_after_settling\n\
residency_raw_module_source={raw_source}\n\
residency_cudnn_handle=thread_local_cached\n\
residency_cudnn_descriptors=recreated_per_call\n\
residency_cudnn_algorithm=repicked_per_call\n\
residency_cudnn_workspace=allocated_per_call\n\
latency_kind=per_launch_from_batched_wall_clock\n\
throughput_kind=serial_repeated_exact_op\n\
aggregation=median_of_three_phase_medians\n\
best_run_selection=forbidden\n\
warmup_policy=time_equivalent_per_provider\n\
warmup_ms={warmup_ms:.3}\n\
settling_policy=time_equivalent_pair_balanced_by_replication_non_authoritative\n\
settling_sequence={}\n\
settling_ms_per_provider={warmup_ms:.3}\n\
timed_samples={iters}\n\
launches_per_sample={inner}\n\
sequence={}\n\
replication_pairing=deterministic_balanced\n\
authoritative_protocol={authoritative_protocol}\n\
telemetry_policy=boundary_snapshots_non_authoritative\n\
telemetry_observe_only=true\n\
telemetry_source=nvidia-smi\n\
telemetry_status={telemetry_status}\n\
telemetry_start_gpu_temp_c={}\n\
telemetry_start_sm_clock_mhz={}\n\
telemetry_start_memory_clock_mhz={}\n\
telemetry_start_power_w={}\n\
telemetry_start_pstate={}\n\
telemetry_start_throttle_mask={}\n\
telemetry_start_thermal_state={}\n\
telemetry_start_throttle_state={}\n\
telemetry_end_gpu_temp_c={}\n\
telemetry_end_sm_clock_mhz={}\n\
telemetry_end_memory_clock_mhz={}\n\
telemetry_end_power_w={}\n\
telemetry_end_pstate={}\n\
telemetry_end_throttle_mask={}\n\
telemetry_end_thermal_state={}\n\
telemetry_end_throttle_state={}\n\
gpu_temp_c_max_observed={}\n\
sm_clock_mhz_min_observed={}\n\
sm_clock_mhz_max_observed={}\n\
memory_clock_mhz_min_observed={}\n\
memory_clock_mhz_max_observed={}\n\
power_w_max_observed={}\n\
max_abs={max_abs:.8}\n\
max_rel={max_rel:.8}\n\
parity_pass={parity_pass}\n\
a1_median_us={:.6}\n\
b1_median_us={:.6}\n\
a2_median_us={:.6}\n\
b2_median_us={:.6}\n\
a3_median_us={:.6}\n\
b3_median_us={:.6}\n\
incumbent_median_us={a_us:.6}\n\
challenger_median_us={b_us:.6}\n\
incumbent_ops_per_s={incumbent_ops_per_s:.3}\n\
challenger_ops_per_s={challenger_ops_per_s:.3}\n\
speedup_x={speedup_x:.6}\n\
challenger_latency_change_pct={latency_change_pct:.3}\n\
incumbent_drift_pct={a_drift_pct:.3}\n\
challenger_drift_pct={b_drift_pct:.3}\n\
incumbent_p90_us={a_p90:.6}\n\
challenger_p90_us={b_p90:.6}\n\
max_drift_pct={max_drift_pct:.3}\n\
required_speedup_x={min_speedup_x:.6}\n\
drift_pass={drift_pass}\n\
p90_non_regression={p90_pass}\n\
speedup_pass={speedup_pass}\n\
status={status}\n\
promotion_result={promotion_result}\n\
fallback_qualification={fallback_qualification}\n\
fallback_qualification_basis=authoritative_protocol+parity+drift\n\
decision={decision}\n",
        replication.map(Replication::as_str).unwrap_or("unspecified"),
        git.commit,
        git.status,
        git.clean,
        headless_display_env,
        gpu_workload.status,
        gpu_workload.active_compute_processes,
        fmt_u64(gpu_workload.utilization_pct),
        gpu_workload.clean,
        input_sha256,
        weight_sha256,
        profile_match.profile_id,
        profile_match.decision_id,
        raw.source,
        raw.proof_status,
        raw.verified,
        profile_match.evidence_sha256,
        cudnn.identity,
        cudnn.version_raw,
        cudnn.algorithm_id,
        cudnn.algorithm_debug,
        cudnn.workspace_bytes,
        effective_replication.settling_sequence(),
        effective_replication.phase_sequence(),
        fmt_f64(telemetry_start.gpu_temp_c),
        fmt_u64(telemetry_start.sm_clock_mhz),
        fmt_u64(telemetry_start.memory_clock_mhz),
        fmt_f64(telemetry_start.power_w),
        fmt_string(telemetry_start.pstate.as_deref()),
        fmt_string(telemetry_start.throttle_mask.as_deref()),
        telemetry_start.thermal_state,
        telemetry_start.throttle_state,
        fmt_f64(telemetry_end.gpu_temp_c),
        fmt_u64(telemetry_end.sm_clock_mhz),
        fmt_u64(telemetry_end.memory_clock_mhz),
        fmt_f64(telemetry_end.power_w),
        fmt_string(telemetry_end.pstate.as_deref()),
        fmt_string(telemetry_end.throttle_mask.as_deref()),
        telemetry_end.thermal_state,
        telemetry_end.throttle_state,
        fmt_f64(gpu_temp_c_max_observed),
        fmt_u64(sm_clock_mhz_min_observed),
        fmt_u64(sm_clock_mhz_max_observed),
        fmt_u64(memory_clock_mhz_min_observed),
        fmt_u64(memory_clock_mhz_max_observed),
        fmt_f64(power_w_max_observed),
        a1.median_us,
        b1.median_us,
        a2.median_us,
        b2.median_us,
        a3.median_us,
        b3.median_us,
    );

    let evidence_sha256 = sha256_bytes(evidence.as_bytes());
    println!("provider_evidence_schema=ASD-PROVIDER-PERFORMANCE-EVIDENCE-V2");
    println!("provider_evidence_protocol={PROTOCOL_ID}");
    println!(
        "provider_evidence_replication={}",
        replication.map(Replication::as_str).unwrap_or("unspecified")
    );
    println!("provider_evidence_sha256={evidence_sha256}");
    println!("telemetry_observe_only=true");
    println!("telemetry_status={telemetry_status}");
    println!(
        "gpu_temp_c_max_observed={}",
        fmt_f64(gpu_temp_c_max_observed)
    );
    println!(
        "sm_clock_mhz_observed={}..{}",
        fmt_u64(sm_clock_mhz_min_observed),
        fmt_u64(sm_clock_mhz_max_observed)
    );
    println!("thermal_state_end={}", telemetry_end.thermal_state);
    println!("throttle_state_end={}", telemetry_end.throttle_state);

    if let Some(path) = evidence_out {
        std::fs::write(&path, evidence).map_err(|err| {
            candle_core::Error::Msg(format!(
                "failed to write Provider Evidence {}: {err}",
                path.display()
            ))
        })?;
        println!("provider_evidence_file={}", path.display());
    }

    configure_provider(Provider::IncumbentRaw);

    Ok(())
}
