use candle_core::{Device, Result, Tensor};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const EMBEDDED_DECISION_ID: &str = "ct1d-sm61-s32-g8-raw-exact";
const EMBEDDED_IMPLEMENTATION_ID: &str =
    "candle.sm61-exact-grouped.ct1d-s32-g8-u1-b256";
const DYNAMIC_DECISION_ID: &str = "ct1d-sm61-s32-g8-runtime-additive-proof";
const DYNAMIC_IMPLEMENTATION_ID: &str =
    "candle.sm61-exact-grouped.runtime-proof.ct1d-s32-g8-u1-b256";
const DYNAMIC_CANDIDATE_ID: &str = "runtime-proof-ct1d-s32-g8-u1-b256";
const ENTRY: &str = "flow_v0322_ct1d_s32_g8_u1_b256";
const SIGNATURE: &str = "op=conv_transpose1d,dim=1,batch=1,c_in=128,c_out=128,spatial=32,weight_shape=128x16x3,groups=8,kernel=3,stride=2,padding=1,output_padding=1,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset";

struct TempTree(PathBuf);

impl Drop for TempTree {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
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

fn dynamic_decision_identity(evidence_sha256: &str) -> String {
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

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max)
}

fn source_cubin() -> Result<PathBuf> {
    let root = candle_kernels::asd_paths::artifacts_dir("sm61").ok_or_else(|| {
        candle_core::Error::Msg("unable to resolve ASD artifacts directory".into())
    })?;
    let path = root.join(format!("{EMBEDDED_IMPLEMENTATION_ID}.cubin"));
    if !path.is_file() {
        candle_core::bail!(
            "missing canonical G8 external CUBIN {}; install the Phase A sm61 package first",
            path.display()
        )
    }
    Ok(path)
}

fn remove_if_exists(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(candle_core::Error::Msg(format!(
            "failed to remove {}: {err}",
            path.display()
        ))),
    }
}

fn main() -> Result<()> {
    let profile_id = candle_kernels::asd_exact::PROFILE_ID.ok_or_else(|| {
        candle_core::Error::Msg(
            "dynamic no-rebuild validation requires the embedded base Exact Profile".into(),
        )
    })?;
    let embedded_uuid = candle_kernels::asd_exact::TARGET_GPU_UUID.ok_or_else(|| {
        candle_core::Error::Msg(
            "dynamic no-rebuild validation requires a device-scoped base profile".into(),
        )
    })?;

    // Authenticate the existing G8 evidence before intentionally disabling its
    // embedded selector for the additive-runtime proof.
    std::env::remove_var("CANDLE_SM61_EXACT_GROUPED_DISABLE");
    let device = Device::new_cuda(0)?;
    let actual_uuid = gpu_uuid(&device)?;
    if actual_uuid != embedded_uuid {
        candle_core::bail!(
            "GPU UUID mismatch actual={} embedded={}",
            actual_uuid,
            embedded_uuid
        )
    }
    let embedded = match candle_kernels::asd_exact::lookup(exact_call(), Some(&actual_uuid)) {
        Some(candle_kernels::asd_exact::ExactMatch::Proven(matched)) => matched,
        _ => candle_core::bail!("missing embedded G8 exact decision"),
    };
    if embedded.decision_id != EMBEDDED_DECISION_ID
        || embedded.implementation_id != EMBEDDED_IMPLEMENTATION_ID
    {
        candle_core::bail!(
            "unexpected embedded G8 decision id={} implementation={}",
            embedded.decision_id,
            embedded.implementation_id
        )
    }

    let source_cubin = source_cubin()?;
    let cubin_bytes = std::fs::read(&source_cubin).map_err(|err| {
        candle_core::Error::Msg(format!(
            "failed to read source CUBIN {}: {err}",
            source_cubin.display()
        ))
    })?;
    let cubin_sha256 = sha256_hex(&cubin_bytes);
    let decision_identity = dynamic_decision_identity(embedded.evidence_sha256);

    let root = std::env::temp_dir().join(format!(
        "candle-asd-dynamic-no-rebuild-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let _cleanup = TempTree(root.clone());
    let module_dir = root.join("modules");
    std::fs::create_dir_all(&module_dir)?;
    let extension_path = root.join("extensions.asd");
    let dynamic_cubin = module_dir.join(format!("{DYNAMIC_IMPLEMENTATION_ID}.cubin"));
    let dynamic_manifest = module_dir.join(format!("{DYNAMIC_IMPLEMENTATION_ID}.manifest"));

    std::fs::write(
        &extension_path,
        format!(
            "ASD-EXACT-EXTENSIONS-V1\n\
base_profile_id={profile_id}\n\
target.architecture=sm61\n\
target.gpu_uuid={actual_uuid}\n\
decision|{DYNAMIC_DECISION_ID}|promoted|raw_cuda|{SIGNATURE}|impl={DYNAMIC_IMPLEMENTATION_ID}|evidence={}|min_integrated_speedup_x=none\n",
            embedded.evidence_sha256
        ),
    )?;
    std::fs::write(&dynamic_cubin, &cubin_bytes)?;
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

    std::env::set_var("CANDLE_ASD_RUNTIME_PROFILE", &extension_path);
    std::env::set_var("CANDLE_ASD_MODULE_DIR", &module_dir);
    std::env::set_var("CANDLE_SM61_EXACT_GROUPED_DISABLE", "1");
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "auto");
    std::env::remove_var("CANDLE_ASD_EXACT_DISABLE");
    std::env::remove_var("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL");
    std::env::remove_var("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
    std::env::set_var("CANDLE_ASD_EXACT_TRACE", "1");
    std::env::set_var("CANDLE_SM61_EXACT_GROUPED_TRACE", "1");

    let (x, w) = tensors(&device)?;

    // Independent cuDNN reference.
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "cudnn");
    let reference = call(&x, &w)?.to_vec3::<f32>()?;

    // Cold dynamic resolution: profile + manifest + artifact are available.
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "auto");
    let cold = call(&x, &w)?.to_vec3::<f32>()?;
    let cold_source = device
        .as_cuda_device()?
        .asd_resolved_module_source(DYNAMIC_IMPLEMENTATION_ID)?
        .unwrap_or("none");
    if cold_source != "external_cubin" {
        candle_core::bail!("cold dynamic source is {cold_source}, expected external_cubin")
    }

    // Remove every control-plane file used by this dynamic decision. A warm
    // launch can now succeed only from in-memory decision/module caches.
    remove_if_exists(&extension_path)?;
    remove_if_exists(&dynamic_manifest)?;
    remove_if_exists(&dynamic_cubin)?;
    let files_absent = !extension_path.exists()
        && !dynamic_manifest.exists()
        && !dynamic_cubin.exists();
    if !files_absent {
        candle_core::bail!("failed to remove dynamic control-plane files")
    }

    let warm = call(&x, &w)?.to_vec3::<f32>()?;
    let warm_source = device
        .as_cuda_device()?
        .asd_resolved_module_source(DYNAMIC_IMPLEMENTATION_ID)?
        .unwrap_or("none");

    let reference_flat = reference.iter().flatten().flatten().copied().collect::<Vec<_>>();
    let cold_flat = cold.iter().flatten().flatten().copied().collect::<Vec<_>>();
    let warm_flat = warm.iter().flatten().flatten().copied().collect::<Vec<_>>();
    let cold_abs = max_abs(&reference_flat, &cold_flat);
    let warm_abs = max_abs(&reference_flat, &warm_flat);
    let cold_warm_abs = max_abs(&cold_flat, &warm_flat);

    if cold_abs != 0.0 || warm_abs != 0.0 || cold_warm_abs != 0.0 {
        candle_core::bail!(
            "dynamic parity failure reference_vs_cold={} reference_vs_warm={} cold_vs_warm={}",
            cold_abs,
            warm_abs,
            cold_warm_abs
        )
    }
    if warm_source != "external_cubin" {
        candle_core::bail!(
            "warm dynamic source is {warm_source}, expected cached external_cubin"
        )
    }

    // Explicit refresh invalidates both caches. Because the files were deleted,
    // the dynamic implementation must no longer resolve after refresh.
    device.as_cuda_device()?.refresh_asd();
    let after_refresh_source = device
        .as_cuda_device()?
        .asd_resolved_module_source(DYNAMIC_IMPLEMENTATION_ID)?;
    if after_refresh_source.is_some() {
        candle_core::bail!("dynamic module remained resolved after refresh")
    }

    // The operation itself may still succeed through the ordinary generic
    // fallback (for example cuDNN), but it must not resurrect the deleted
    // dynamic implementation.
    let _post_refresh = call(&x, &w)?;
    let resurrected = device
        .as_cuda_device()?
        .asd_resolved_module_source(DYNAMIC_IMPLEMENTATION_ID)?;
    if resurrected.is_some() {
        candle_core::bail!("deleted dynamic implementation was resurrected after refresh")
    }

    println!("=== ASD V3 DYNAMIC NO-REBUILD VALIDATION ===");
    println!("base_profile_id={profile_id}");
    println!("gpu_uuid={actual_uuid}");
    println!("embedded_decision_disabled={EMBEDDED_DECISION_ID}");
    println!("dynamic_decision_id={DYNAMIC_DECISION_ID}");
    println!("dynamic_implementation_id={DYNAMIC_IMPLEMENTATION_ID}");
    println!("implementation_known_to_embedded_catalogue=false");
    println!("decision_identity_sha256={decision_identity}");
    println!("artifact_sha256={cubin_sha256}");
    println!("kernel_abi=asd.xwo.f32.v1");
    println!("cold_source={cold_source}");
    println!("control_plane_files_removed_before_warm=true");
    println!("warm_source={warm_source}");
    println!("warm_launch_after_files_removed=success");
    println!("PARITY reference_vs_cold max_abs={cold_abs:.8} pass=true");
    println!("PARITY reference_vs_warm max_abs={warm_abs:.8} pass=true");
    println!("PARITY cold_vs_warm max_abs={cold_warm_abs:.8} pass=true");
    println!("refresh_asd_invalidated_dynamic_cache=true");
    println!("deleted_dynamic_implementation_resurrected=false");
    println!("hot_path_filesystem_dependency=false");
    println!("candle_rebuild_required_for_new_compatible_kernel=false");
    println!("STATUS=PASS");

    Ok(())
}
