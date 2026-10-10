use candle_core::{Device, Result, Tensor};
use std::path::PathBuf;

const IMPLEMENTATION_ID: &str = "candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256";

fn deterministic(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * mul) % 101) as isize + bias) as f32 / 64.0)
        .collect()
}

fn tensors(device: &Device) -> Result<(Tensor, Tensor)> {
    let x = Tensor::from_vec(deterministic(128 * 32, 37, -50), (1, 128, 32), device)?;
    let w = Tensor::from_vec(deterministic(128 * 64 * 3, 53, -50), (128, 64, 3), device)?;
    Ok((x, w))
}

fn call(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.conv_transpose1d(w, 1, 1, 2, 1, 2)
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

fn resolved_source(device: &Device) -> Result<Option<&'static str>> {
    device
        .as_cuda_device()?
        .asd_resolved_module_source(IMPLEMENTATION_ID)
}

fn main() -> Result<()> {
    let module_dir = std::env::var_os("CANDLE_ASD_MODULE_DIR")
        .map(PathBuf::from)
        .or_else(|| candle_kernels::asd_paths::artifacts_dir("sm61"))
        .ok_or_else(|| candle_core::Error::Msg("unable to resolve ASD artifact directory".into()))?;
    let cubin = module_dir.join(format!("{IMPLEMENTATION_ID}.cubin"));
    if !cubin.is_file() {
        candle_core::bail!("missing external CUBIN {}", cubin.display())
    }

    let empty_dir =
        std::env::temp_dir().join(format!("candle-asd-external-missing-{}", std::process::id()));
    if empty_dir.exists() {
        std::fs::remove_dir_all(&empty_dir).map_err(|err| {
            candle_core::Error::Msg(format!("failed to clear {}: {err}", empty_dir.display()))
        })?;
    }
    std::fs::create_dir_all(&empty_dir).map_err(|err| {
        candle_core::Error::Msg(format!("failed to create {}: {err}", empty_dir.display()))
    })?;

    std::env::set_var("CANDLE_ASD_EXACT_TRACE", "1");
    std::env::set_var("CANDLE_SM61_EXACT_GROUPED_TRACE", "1");
    std::env::set_var("CANDLE_GROUPED_TRANSPOSE_DISPATCH", "auto");
    std::env::remove_var("CANDLE_SM61_EXACT_GROUPED_DISABLE");

    let device = Device::new_cuda(0)?;
    let (x, w) = tensors(&device)?;

    println!("=== ASD V3 PHASE B MODULE PROVIDER RELOAD VALIDATION ===");
    println!("implementation_id={IMPLEMENTATION_ID}");
    println!("external_cubin={}", cubin.display());
    println!("builtin_raw_present=false");

    std::env::set_var("CANDLE_ASD_MODULE_DIR", &module_dir);
    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
    device.as_cuda_device()?.refresh_asd();
    let external_a = call(&x, &w)?;
    device.synchronize()?;
    let source_a = resolved_source(&device)?;
    println!("PHASE=external_a source={:?}", source_a);
    if source_a != Some("external_cubin") {
        candle_core::bail!("external_a did not resolve external_cubin")
    }

    std::env::set_var("CANDLE_ASD_MODULE_DIR", &empty_dir);
    std::env::set_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED", "1");
    device.as_cuda_device()?.refresh_asd();
    let fallback_b = call(&x, &w)?;
    device.synchronize()?;
    let source_b = resolved_source(&device)?;
    println!("PHASE=fallback_b raw_source={:?} expected_raw_source=None", source_b);
    if source_b.is_some() {
        candle_core::bail!("fallback_b unexpectedly resolved a raw module")
    }

    let (ab_abs, ab_rel) = max_abs_rel(&external_a, &fallback_b)?;
    let ab_pass = ab_abs <= 1e-5 || ab_rel <= 1e-5;
    println!(
        "PARITY external_a_vs_fallback_b max_abs={ab_abs:.8} max_rel={ab_rel:.8} pass={ab_pass}"
    );

    std::env::set_var("CANDLE_ASD_MODULE_DIR", &module_dir);
    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
    device.as_cuda_device()?.refresh_asd();
    let external_a2 = call(&x, &w)?;
    device.synchronize()?;
    let source_a2 = resolved_source(&device)?;
    println!("PHASE=external_a2 source={:?}", source_a2);
    if source_a2 != Some("external_cubin") {
        candle_core::bail!("external_a2 did not re-resolve external_cubin")
    }

    let (aa_abs, aa_rel) = max_abs_rel(&external_a, &external_a2)?;
    let aa_pass = aa_abs <= 1e-5 || aa_rel <= 1e-5;
    println!(
        "PARITY external_a_vs_a2 max_abs={aa_abs:.8} max_rel={aa_rel:.8} pass={aa_pass}"
    );

    std::env::remove_var("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED");
    std::env::remove_var("CANDLE_ASD_MODULE_DIR");
    device.as_cuda_device()?.refresh_asd();
    let _ = std::fs::remove_dir_all(&empty_dir);

    if !ab_pass || !aa_pass {
        println!("STATUS=HOLD");
        candle_core::bail!("ASD V3 Phase B module-provider reload parity failed")
    }

    println!("STATUS=PASS");
    Ok(())
}
