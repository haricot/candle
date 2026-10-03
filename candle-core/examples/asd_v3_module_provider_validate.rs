use candle_core::{Device, Result, Tensor};
use std::path::PathBuf;

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
    // K00 exact geometry:
    // b1, ci=co=128, l=32, groups=2, k=3, stride=2,
    // padding=1, output_padding=1, dilation=1.
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

fn main() -> Result<()> {
    let module_dir = std::env::var_os("CANDLE_ASD_MODULE_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| {
            candle_core::Error::Msg(
                "CANDLE_ASD_MODULE_DIR must point to the directory containing the external CUBIN"
                    .into(),
            )
        })?;
    let cubin = module_dir.join(format!("{IMPLEMENTATION_ID}.cubin"));
    if !cubin.is_file() {
        candle_core::bail!("missing external CUBIN {}", cubin.display())
    }

    let builtin_only_dir =
        std::env::temp_dir().join(format!("candle-asd-builtin-only-{}", std::process::id()));
    if builtin_only_dir.exists() {
        std::fs::remove_dir_all(&builtin_only_dir).map_err(|err| {
            candle_core::Error::Msg(format!(
                "failed to clear builtin-only ASD directory {}: {err}",
                builtin_only_dir.display()
            ))
        })?;
    }
    std::fs::create_dir_all(&builtin_only_dir).map_err(|err| {
        candle_core::Error::Msg(format!(
            "failed to create builtin-only ASD directory {}: {err}",
            builtin_only_dir.display()
        ))
    })?;

    std::env::set_var("CANDLE_ASD_EXACT_TRACE", "1");
    std::env::set_var("CANDLE_SM61_EXACT_GROUPED_TRACE", "1");
    std::env::remove_var("CANDLE_SM61_EXACT_GROUPED_DISABLE");

    let device = Device::new_cuda(0)?;
    let (x, w) = tensors(&device)?;

    println!("=== ASD V3 MODULE PROVIDER VALIDATION ===");
    println!("implementation_id={IMPLEMENTATION_ID}");
    println!("external_cubin={}", cubin.display());

    // A: force the embedded PTX provider by hiding the external directory.
    std::env::set_var("CANDLE_ASD_MODULE_DIR", &builtin_only_dir);
    device
        .as_cuda_device()?
        .refresh_asd_module(IMPLEMENTATION_ID)?;
    println!(
        "PHASE=builtin_a expected_provider=builtin_ptx external_override={}",
        builtin_only_dir.display()
    );
    let builtin_a = call(&x, &w)?;
    device.synchronize()?;

    // B: restore the external module directory. The same implementation id must
    // resolve to the external CUBIN without rebuilding the Rust crate.
    std::env::set_var("CANDLE_ASD_MODULE_DIR", &module_dir);
    device
        .as_cuda_device()?
        .refresh_asd_module(IMPLEMENTATION_ID)?;
    println!("PHASE=external_b expected_provider=external_cubin");
    let external_b = call(&x, &w)?;
    device.synchronize()?;

    let (ab_abs, ab_rel) = max_abs_rel(&builtin_a, &external_b)?;
    let ab_pass = ab_abs <= 1e-5 || ab_rel <= 1e-5;
    println!("PARITY builtin_vs_external max_abs={ab_abs:.8} max_rel={ab_rel:.8} pass={ab_pass}");

    // A2: return to the builtin provider to prove resolution is not sticky.
    std::env::set_var("CANDLE_ASD_MODULE_DIR", &builtin_only_dir);
    device
        .as_cuda_device()?
        .refresh_asd_module(IMPLEMENTATION_ID)?;
    println!(
        "PHASE=builtin_a2 expected_provider=builtin_ptx external_override={}",
        builtin_only_dir.display()
    );
    let builtin_a2 = call(&x, &w)?;
    device.synchronize()?;

    let (aa_abs, aa_rel) = max_abs_rel(&builtin_a, &builtin_a2)?;
    let aa_pass = aa_abs <= 1e-5 || aa_rel <= 1e-5;
    println!("PARITY builtin_a_vs_a2 max_abs={aa_abs:.8} max_rel={aa_rel:.8} pass={aa_pass}");

    println!(
        "STATUS={}",
        if ab_pass && aa_pass { "PASS" } else { "HOLD" }
    );

    let _ = std::fs::remove_dir_all(&builtin_only_dir);
    if !ab_pass || !aa_pass {
        candle_core::bail!("ASD V3 module-provider parity failed")
    }
    Ok(())
}
