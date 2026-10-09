use candle_core::{Device, Result, Tensor};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const SOURCE_IMPLEMENTATION_ID: &str = "candle.sm61-exact-grouped.dynamic.ct1d-s33-g2-u1-b256";
const GENERIC_IMPLEMENTATION_ID: &str = "thirdparty.generic-exact.ct1d-s33-g2-u1-b256";
const DECISION_ID: &str = "d3-generic-impl-ct1d-s33-g2";
const CANDIDATE_ID: &str = "d3-generic-impl-proof";
const ENTRY: &str = "flow_phase_c_ct1d_s33_g2_u1_b256";
const EVIDENCE_SHA256: &str = "ca9ac9faa8c987f8a4da07ec0834f07e1f2c6d48fb165e4f7e1ca5928103391f";
const SIGNATURE: &str = "op=conv_transpose1d,dim=1,batch=1,c_in=128,c_out=128,spatial=33,weight_shape=128x64x3,groups=2,kernel=3,stride=2,padding=1,output_padding=1,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset";

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

fn actual_uuid(device: &Device) -> Result<String> {
    let uuid = device
        .as_cuda_device()?
        .cuda_stream()
        .context()
        .uuid()
        .map_err(|err| candle_core::Error::Msg(format!("failed to read CUDA UUID: {err:?}")))?;
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

fn runtime_architecture(device: &Device) -> Result<String> {
    let (major, minor) = device
        .as_cuda_device()?
        .cuda_stream()
        .context()
        .compute_capability()
        .map_err(|err| {
            candle_core::Error::Msg(format!("failed to read CUDA compute capability: {err:?}"))
        })?;
    Ok(format!("sm{}", major * 10 + minor))
}

fn decision_identity() -> String {
    let canonical = format!(
        "ASD-DECISION-V1\n\
id={DECISION_ID}\n\
state=promoted\n\
provider=raw_cuda\n\
op=conv_transpose1d\n\
dim=1\n\
batch=1\n\
c_in=128\n\
c_out=128\n\
spatial=33\n\
weight_shape=128x64x3\n\
groups=2\n\
kernel=3\n\
stride=2\n\
padding=1\n\
output_padding=1\n\
dilation=1\n\
dtype=f32\n\
input_layout=contiguous_zero_offset\n\
weight_layout=contiguous_zero_offset\n\
implementation_id={GENERIC_IMPLEMENTATION_ID}\n\
evidence_sha256={EVIDENCE_SHA256}\n\
min_integrated_speedup_x=none\n"
    );
    sha256_hex(canonical.as_bytes())
}

fn call(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.conv_transpose1d(w, 1, 1, 2, 1, 2)
}

fn write_manifest(
    path: &std::path::Path,
    architecture: &str,
    cubin_sha256: &str,
    decision_identity: &str,
) -> Result<()> {
    std::fs::write(
        path,
        format!(
            "ASD-CUDA-MODULE-V1\n\
abi_version=1\n\
implementation_id={GENERIC_IMPLEMENTATION_ID}\n\
architecture={architecture}\n\
artifact_kind=cubin\n\
entry={ENTRY}\n\
artifact_sha256={cubin_sha256}\n\
candidate_id={CANDIDATE_ID}\n\
kernel_abi=asd.xwo.f32.v1\n\
output_count=8448\n\
grid_x=33\n\
block_x=256\n\
shared_mem_bytes=0\n\
decision_identity_sha256={decision_identity}\n"
        ),
    )?;
    Ok(())
}

fn main() -> Result<()> {
    if GENERIC_IMPLEMENTATION_ID.contains("sm61")
        || GENERIC_IMPLEMENTATION_ID.starts_with("candle.sm61-exact-grouped.")
    {
        candle_core::bail!("generic implementation id accidentally encodes sm61")
    }

    let source_dir = std::env::var_os("CANDLE_ASD_MODULE_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            candle_core::Error::Msg(
                "D3 generic CUDA validation requires CANDLE_ASD_MODULE_DIR pointing at the Phase C S33 artifact".into(),
            )
        })?;
    let source_cubin = source_dir.join(format!("{SOURCE_IMPLEMENTATION_ID}.cubin"));
    if !source_cubin.is_file() {
        candle_core::bail!("missing source S33 CUBIN {}", source_cubin.display())
    }
    let cubin_bytes = std::fs::read(&source_cubin)?;
    let cubin_sha256 = sha256_hex(&cubin_bytes);

    let device = Device::new_cuda(0)?;
    let gpu_uuid = actual_uuid(&device)?;
    let architecture = runtime_architecture(&device)?;
    let identity = decision_identity();

    let root =
        std::env::temp_dir().join(format!("candle-asd-d3-generic-cuda-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root)?;
    let _cleanup = TempTree(root.clone());

    let module_dir = root.join("modules");
    std::fs::create_dir_all(&module_dir)?;
    let runtime_profile = root.join("profile.asd");
    let generic_cubin = module_dir.join(format!("{GENERIC_IMPLEMENTATION_ID}.cubin"));
    let generic_manifest = module_dir.join(format!("{GENERIC_IMPLEMENTATION_ID}.manifest"));

    std::fs::write(&generic_cubin, &cubin_bytes)?;
    write_manifest(&generic_manifest, &architecture, &cubin_sha256, &identity)?;
    std::fs::write(
        &runtime_profile,
        format!(
            "ASD-EXACT-EXTENSIONS-V1\n\
base_profile_id=d3-generic-runtime\n\
target.architecture={architecture}\n\
target.gpu_uuid={gpu_uuid}\n\
decision|{DECISION_ID}|promoted|raw_cuda|{SIGNATURE}|impl={GENERIC_IMPLEMENTATION_ID}|evidence={EVIDENCE_SHA256}|min_integrated_speedup_x=none\n"
        ),
    )?;

    std::env::set_var("CANDLE_ASD_RUNTIME_PROFILE", &runtime_profile);
    std::env::set_var("CANDLE_ASD_MODULE_DIR", &module_dir);
    std::env::set_var("CANDLE_ASD_EXACT_TRACE", "1");
    std::env::set_var("CANDLE_ASD_EXACT_CUDA_TRACE", "1");
    std::env::remove_var("CANDLE_ASD_EXACT_DISABLE");
    std::env::remove_var("CANDLE_ASD_EXACT_CUDA_DISABLE");
    std::env::remove_var("CANDLE_SM61_EXACT_GROUPED_DISABLE");
    std::env::remove_var("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL");
    std::env::remove_var("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");

    let x = Tensor::from_vec(deterministic(128 * 33, 37, -50), (1, 128, 33), &device)?;
    let w = Tensor::from_vec(deterministic(128 * 64 * 3, 53, -50), (128, 64, 3), &device)?;

    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "cudnn");
    device.as_cuda_device()?.refresh_asd();
    let reference = call(&x, &w)?;
    device.synchronize()?;

    // A manifest for a different architecture must fail before module launch.
    let wrong_architecture = if architecture == "sm999" {
        "sm998"
    } else {
        "sm999"
    };
    write_manifest(
        &generic_manifest,
        wrong_architecture,
        &cubin_sha256,
        &identity,
    )?;
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "auto");
    device.as_cuda_device()?.refresh_asd();
    let wrong_arch_rejected = match call(&x, &w) {
        Err(_) => true,
        Ok(_) => false,
    };
    if !wrong_arch_rejected {
        candle_core::bail!(
            "manifest architecture {wrong_architecture} was accepted by runtime device {architecture}"
        )
    }

    // Restore the runtime-device architecture and prove that the same generic
    // implementation id now resolves and executes.
    write_manifest(&generic_manifest, &architecture, &cubin_sha256, &identity)?;
    device.as_cuda_device()?.refresh_asd();
    let generic = call(&x, &w)?;
    device.synchronize()?;

    let source = device
        .as_cuda_device()?
        .asd_resolved_module_source(GENERIC_IMPLEMENTATION_ID)?
        .unwrap_or("none");
    let dims = generic.dims3()?;
    let (max_abs, max_rel) = max_abs_rel(&reference, &generic)?;
    let parity = max_abs <= 1e-5 || max_rel <= 1e-5;

    println!("=== ASD V3 PHASE D3 GENERIC CUDA EXECUTOR VALIDATION ===");
    println!("architecture={architecture}");
    println!("gpu_uuid={gpu_uuid}");
    println!("decision_id={DECISION_ID}");
    println!("implementation_id={GENERIC_IMPLEMENTATION_ID}");
    println!("implementation_encodes_sm61=false");
    println!("decision_identity_sha256={identity}");
    println!("artifact_sha256={cubin_sha256}");
    println!("wrong_architecture={wrong_architecture}");
    println!("wrong_architecture_rejected={wrong_arch_rejected}");
    println!("source={source}");
    println!("output_shape={dims:?}");
    println!("max_abs={max_abs:.9}");
    println!("max_rel={max_rel:.9}");
    println!("parity={parity}");

    if source != "external_cubin" {
        candle_core::bail!("generic implementation source is {source}, expected external_cubin")
    }
    if dims != (1, 128, 66) {
        candle_core::bail!("unexpected generic S33 output shape {dims:?}")
    }
    if !parity {
        candle_core::bail!(
            "generic implementation parity failed max_abs={max_abs:.9} max_rel={max_rel:.9}"
        )
    }

    println!("D3_GENERIC_IMPLEMENTATION_ID=PASS");
    println!("D3_RUNTIME_ARCHITECTURE_BINDING=PASS");
    println!("D3_GENERIC_CUDA_EXECUTOR=PASS");
    println!("STATUS=PASS");
    Ok(())
}
