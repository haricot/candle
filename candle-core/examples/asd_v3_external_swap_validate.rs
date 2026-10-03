use candle_core::{Device, Result, Tensor};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const IMPLEMENTATION_ID: &str = "candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256";

fn deterministic(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * mul) % 101) as isize + bias) as f32 / 64.0)
        .collect()
}

fn tensors(device: &Device) -> Result<(Tensor, Tensor)> {
    let x = Tensor::from_vec(deterministic(1 * 128 * 32, 37, -50), (1, 128, 32), device)?;
    let w = Tensor::from_vec(deterministic(128 * 64 * 3, 53, -50), (128, 64, 3), device)?;
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

fn run_external(
    phase: &str,
    dir: &Path,
    x: &Tensor,
    w: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    std::env::set_var("CANDLE_ASD_MODULE_DIR", dir);
    device
        .as_cuda_device()?
        .refresh_asd_module(IMPLEMENTATION_ID)?;
    println!(
        "PHASE={phase} expected_provider=external_cubin module={}",
        module_path(dir).display()
    );
    let out = call(x, w)?;
    device.synchronize()?;
    Ok(out)
}

fn main() -> Result<()> {
    let dir_a = std::env::var_os("CANDLE_ASD_MODULE_DIR_A")
        .map(PathBuf::from)
        .ok_or_else(|| candle_core::Error::Msg("missing CANDLE_ASD_MODULE_DIR_A".into()))?;
    let dir_b = std::env::var_os("CANDLE_ASD_MODULE_DIR_B")
        .map(PathBuf::from)
        .ok_or_else(|| candle_core::Error::Msg("missing CANDLE_ASD_MODULE_DIR_B".into()))?;

    for (name, dir) in [("A", &dir_a), ("B", &dir_b)] {
        let path = module_path(dir);
        if !path.is_file() {
            candle_core::bail!("missing external CUBIN {name}: {}", path.display())
        }
    }

    std::env::set_var("CANDLE_ASD_EXACT_TRACE", "1");
    std::env::set_var("CANDLE_SM61_EXACT_GROUPED_TRACE", "1");
    std::env::remove_var("CANDLE_SM61_EXACT_GROUPED_DISABLE");

    let device = Device::new_cuda(0)?;
    let (x, w) = tensors(&device)?;

    println!("=== ASD V3 EXTERNAL HOT-SWAP VALIDATION ===");
    println!("implementation_id={IMPLEMENTATION_ID}");

    let a1 = run_external("external_a1", &dir_a, &x, &w, &device)?;
    let b = run_external("external_b", &dir_b, &x, &w, &device)?;
    let a2 = run_external("external_a2", &dir_a, &x, &w, &device)?;

    let (ab_abs, ab_rel) = max_abs_rel(&a1, &b)?;
    let (aa_abs, aa_rel) = max_abs_rel(&a1, &a2)?;
    let ab_pass = ab_abs <= 1e-5 || ab_rel <= 1e-5;
    let aa_pass = aa_abs <= 1e-5 || aa_rel <= 1e-5;

    println!("PARITY external_a_vs_b max_abs={ab_abs:.8} max_rel={ab_rel:.8} pass={ab_pass}");
    println!("PARITY external_a1_vs_a2 max_abs={aa_abs:.8} max_rel={aa_rel:.8} pass={aa_pass}");
    let status = if ab_pass && aa_pass { "PASS" } else { "HOLD" };
    println!("STATUS={status}");

    let artifact_a_sha256 = sha256_file(&module_path(&dir_a))?;
    let artifact_b_sha256 = sha256_file(&module_path(&dir_b))?;
    let gpu_uuid = gpu_uuid(&device)?;
    let evidence = format!(
        "ASD-CUDA-VALIDATION-V1\nvalidation=external_hot_swap_parity\nimplementation_id={IMPLEMENTATION_ID}\narchitecture=sm61\ngpu_uuid={gpu_uuid}\nartifact_a_sha256={artifact_a_sha256}\nartifact_b_sha256={artifact_b_sha256}\nmax_abs_a_b={ab_abs:.8}\nmax_rel_a_b={ab_rel:.8}\nmax_abs_a_a2={aa_abs:.8}\nmax_rel_a_a2={aa_rel:.8}\nparity_a_b={ab_pass}\nparity_a_a2={aa_pass}\nstatus={status}\n"
    );
    let evidence_sha256 = Sha256::digest(evidence.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    println!("validation_evidence_sha256={evidence_sha256}");

    if let Some(path) = std::env::var_os("CANDLE_ASD_EVIDENCE_OUT") {
        let path = PathBuf::from(path);
        std::fs::write(&path, &evidence).map_err(|err| {
            candle_core::Error::Msg(format!(
                "failed to write validation evidence {}: {err}",
                path.display()
            ))
        })?;
        println!("validation_evidence_file={}", path.display());
    }

    if !ab_pass || !aa_pass {
        candle_core::bail!("ASD V3 external hot-swap parity failed")
    }
    Ok(())
}
