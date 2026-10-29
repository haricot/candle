use candle_core::{Device, Result, Tensor};

const C: usize = 48;
const H: usize = 64;
const W: usize = 48;

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

fn call(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.conv2d(w, 2, 1, 1, C)
}

fn main() -> Result<()> {
    std::env::set_var("CANDLE_ASD_EXACT_TRACE", "1");
    std::env::set_var("CANDLE_ASD_REAL_DISPATCH_VALIDATION", "1");
    std::env::set_var("CANDLE_ASD_REAL_DISPATCH_REQUIRE_CUDNN", "1");

    let device = Device::new_cuda(0)?;
    let x = Tensor::from_vec(
        deterministic(C * H * W, 37, -50),
        (1, C, H, W),
        &device,
    )?;
    let w = Tensor::from_vec(
        deterministic(C * 5 * 5, 53, -50),
        (C, 1, 5, 5),
        &device,
    )?;

    // Force the ordinary production current path to cuDNN for the reference.
    std::env::set_var("CANDLE_ASD_EXACT_DISABLE", "1");
    device.as_cuda_device()?.refresh_asd();
    let reference = call(&x, &w)?;
    device.synchronize()?;

    // Re-enable runtime Exact Profile authority. If current.asd contains the
    // promoted DW5x5 row, GroupedConv2D::cuda_fwd returns from try_launch_exact
    // before reaching the cuDNN current path.
    std::env::remove_var("CANDLE_ASD_EXACT_DISABLE");
    device.as_cuda_device()?.refresh_asd();
    let exact = call(&x, &w)?;
    device.synchronize()?;

    let (max_abs, max_rel) = max_abs_rel(&reference, &exact)?;
    let parity = max_abs <= 1e-5 || max_rel <= 1e-5;

    println!("=== ASD V3 PHASE D2 DW5X5 RUNTIME PROFILE VALIDATION ===");
    println!("case=c{C}-h{H}-w{W}");
    println!("expected_implementation=candle.depthwise-conv2d-5x5.raw.v1");
    println!("max_abs={max_abs:.8}");
    println!("max_rel={max_rel:.8}");
    println!("parity={parity}");

    if !parity {
        println!("STATUS=HOLD");
        candle_core::bail!(
            "D2 DW5x5 runtime profile parity failed max_abs={max_abs} max_rel={max_rel}"
        )
    }

    println!("D2_DW5X5_RUNTIME_PARITY=PASS");
    println!("STATUS=PASS");
    Ok(())
}
