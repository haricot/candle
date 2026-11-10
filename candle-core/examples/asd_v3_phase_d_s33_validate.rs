use candle_core::{Device, Result, Tensor};
use std::path::PathBuf;

const DECISION_ID: &str = "ct1d-sm61-s33-g2-raw-dynamic";
const IMPLEMENTATION_ID: &str =
    "candle.sm61-exact-grouped.dynamic.ct1d-s33-g2-u1-b256";

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

fn required_env_path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| candle_core::Error::Msg(format!("missing required environment variable {name}")))
}

fn main() -> Result<()> {
    let runtime_profile = required_env_path("CANDLE_ASD_RUNTIME_PROFILE")?;
    let module_dir = required_env_path("CANDLE_ASD_MODULE_DIR")?;
    let cubin = module_dir.join(format!("{IMPLEMENTATION_ID}.cubin"));
    let manifest = module_dir.join(format!("{IMPLEMENTATION_ID}.manifest"));

    if !runtime_profile.is_file() {
        candle_core::bail!(
            "missing Phase D runtime profile {}",
            runtime_profile.display()
        )
    }
    if !cubin.is_file() || !manifest.is_file() {
        candle_core::bail!(
            "missing Phase D S33 artifact or manifest under {}",
            module_dir.display()
        )
    }

    std::env::remove_var("CANDLE_ASD_EXACT_DISABLE");
    std::env::remove_var("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL");
    std::env::remove_var("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT");
    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
    std::env::set_var("CANDLE_ASD_EXACT_TRACE", "1");
    std::env::set_var("CANDLE_SM61_EXACT_GROUPED_TRACE", "1");

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

    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "cudnn");
    device.as_cuda_device()?.refresh_asd_runtime_policy();
    let reference = x.conv_transpose1d(&w, 1, 1, 2, 1, 2)?;
    device.synchronize()?;

    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "auto");
    device.as_cuda_device()?.refresh_asd();
    let dynamic = x.conv_transpose1d(&w, 1, 1, 2, 1, 2)?;
    device.synchronize()?;

    let source = device
        .as_cuda_device()?
        .asd_resolved_module_source(IMPLEMENTATION_ID)?
        .unwrap_or("none");
    let dims = dynamic.dims3()?;
    let (max_abs, max_rel) = max_abs_rel(&reference, &dynamic)?;
    let parity = max_abs <= 1e-5 || max_rel <= 1e-5;

    println!("=== ASD V3 PHASE D S33 MANIFEST-DYNAMIC VALIDATION ===");
    println!("runtime_profile={}", runtime_profile.display());
    println!("module_dir={}", module_dir.display());
    println!("decision_id={DECISION_ID}");
    println!("implementation_id={IMPLEMENTATION_ID}");
    println!("source={source}");
    println!("output_shape={dims:?}");
    println!("max_abs={max_abs:.9}");
    println!("max_rel={max_rel:.9}");
    println!("parity={parity}");

    if source != "external_cubin" {
        candle_core::bail!(
            "S33 dynamic source is {source}, expected external_cubin"
        )
    }
    if dims != (1, 128, 66) {
        candle_core::bail!("unexpected S33 output shape {dims:?}")
    }
    if !parity {
        candle_core::bail!(
            "S33 dynamic parity failed max_abs={max_abs:.9} max_rel={max_rel:.9}"
        )
    }

    println!("D1B_S33_MANIFEST_DYNAMIC=PASS");
    println!("STATUS=PASS");
    Ok(())
}
