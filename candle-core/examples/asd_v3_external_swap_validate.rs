use candle_core::{Device, Result, Tensor};
use std::path::{Path, PathBuf};

const IMPLEMENTATION_ID: &str =
    "candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256";

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

    println!(
        "PARITY external_a_vs_b max_abs={ab_abs:.8} max_rel={ab_rel:.8} pass={ab_pass}"
    );
    println!(
        "PARITY external_a1_vs_a2 max_abs={aa_abs:.8} max_rel={aa_rel:.8} pass={aa_pass}"
    );
    println!(
        "STATUS={}",
        if ab_pass && aa_pass { "PASS" } else { "HOLD" }
    );

    if !ab_pass || !aa_pass {
        candle_core::bail!("ASD V3 external hot-swap parity failed")
    }
    Ok(())
}
