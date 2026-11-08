use candle::{DType, Device, Result, Tensor};

const SHAPES: &[(usize, usize, usize)] = &[
    (48, 128, 96),
    (24, 128, 96),
    (96, 64, 48),
    (48, 64, 48),
    (192, 32, 24),
    (96, 32, 24),
    (384, 16, 12),
    (192, 16, 12),
    (768, 8, 6),
    (384, 8, 6),
];

fn parity(src: &Tensor, bias: &Tensor, tol: f32) -> Result<(f32, bool)> {
    let (_, c, _, _) = src.dims4()?;
    let reference = src.broadcast_add(&bias.reshape((1, c, 1, 1))?)?.silu()?;
    let production = candle_nn::ops::conv2d_bias_silu(src, bias)?;
    src.device().synchronize()?;
    let diff = (&reference - &production)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    Ok((diff, diff <= tol))
}

fn negative(name: &str, src: &Tensor, bias: &Tensor, tol: f32) -> Result<bool> {
    let (diff, pass) = parity(src, bias, tol)?;
    println!(
        "R4_R1_P1_NEGATIVE_FALLBACK name={name} dtype={:?} shape={:?} contiguous={} start_offset={} bias_contiguous={} bias_start_offset={} max_abs={diff:.8} pass={pass}",
        src.dtype(),
        src.dims(),
        src.is_contiguous(),
        src.layout().start_offset(),
        bias.is_contiguous(),
        bias.layout().start_offset(),
    );
    Ok(pass)
}

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    println!("=== R4-R1-P1 C2 PRODUCTION ACTIVATION VALIDATION ===");
    println!("production_default=true");
    println!("kill_switch=CANDLE_CUDA_BIAS_SILU_DISABLE");
    println!("exact_shapes={}", SHAPES.len());
    println!("negative_dispatch_cases=7");
    let tol = 2.0e-5f32;
    let mut all = true;

    for &(c, h, w) in SHAPES {
        let src = Tensor::randn(0f32, 1f32, (1, c, h, w), &dev)?.to_dtype(DType::F32)?;
        let bias = Tensor::randn(0f32, 0.1f32, c, &dev)?.to_dtype(DType::F32)?;
        let (diff, pass) = parity(&src, &bias, tol)?;
        all &= pass;
        println!(
            "R4_R1_P1_PARITY shape=1x{c}x{h}x{w} elems={} max_abs={diff:.8} tolerance={tol:.8} pass={pass}",
            c * h * w
        );
    }

    let src = Tensor::randn(0f32, 1f32, (1, 32, 16, 16), &dev)?;
    let bias = Tensor::randn(0f32, 0.1f32, 32, &dev)?;
    all &= negative("shape_outside_exact_frontier", &src, &bias, tol)?;

    let src = Tensor::randn(0f32, 1f32, (2, 384, 8, 6), &dev)?;
    let bias = Tensor::randn(0f32, 0.1f32, 384, &dev)?;
    all &= negative("batch_not_one", &src, &bias, tol)?;

    let src = Tensor::randn(0f64, 1f64, (1, 384, 8, 6), &dev)?;
    let bias = Tensor::randn(0f64, 0.1f64, 384, &dev)?;
    let reference = src.broadcast_add(&bias.reshape((1, 384, 1, 1))?)?.silu()?;
    let production = candle_nn::ops::conv2d_bias_silu(&src, &bias)?;
    dev.synchronize()?;
    let diff_f64 = (&reference - &production)?
        .abs()?
        .max_all()?
        .to_scalar::<f64>()?;
    let f64_pass = diff_f64 <= 1.0e-12;
    all &= f64_pass;
    println!(
        "R4_R1_P1_NEGATIVE_FALLBACK name=dtype_not_f32 dtype={:?} shape={:?} max_abs={diff_f64:.16} pass={f64_pass}",
        src.dtype(), src.dims()
    );

    let src = Tensor::randn(0f32, 1f32, (1, 48, 48, 64), &dev)?.transpose(2, 3)?;
    let bias = Tensor::randn(0f32, 0.1f32, 48, &dev)?;
    let contract = !src.is_contiguous();
    let pass = negative("input_noncontiguous", &src, &bias, tol)? && contract;
    all &= pass;
    println!(
        "R4_R1_P1_NEGATIVE_CONTRACT name=input_noncontiguous contract_pass={contract} pass={pass}"
    );

    let src = Tensor::randn(0f32, 1f32, (1, 48, 64, 48), &dev)?;
    let bias_storage = Tensor::randn(0f32, 0.1f32, 49, &dev)?;
    let bias = bias_storage.narrow(0, 1, 48)?;
    let contract = bias.layout().start_offset() != 0;
    let pass = negative("bias_nonzero_offset", &src, &bias, tol)? && contract;
    all &= pass;
    println!(
        "R4_R1_P1_NEGATIVE_CONTRACT name=bias_nonzero_offset contract_pass={contract} pass={pass}"
    );

    let cpu = Device::Cpu;
    let src = Tensor::randn(0f32, 1f32, (1, 384, 8, 6), &cpu)?;
    let bias = Tensor::randn(0f32, 0.1f32, 384, &cpu)?;
    all &= negative("device_not_cuda", &src, &bias, tol)?;

    let tracked_src = Tensor::randn(0f32, 1f32, (1, 48, 64, 48), &dev)?;
    let tracked_bias = Tensor::randn(0f32, 0.1f32, 48, &dev)?;
    let tracked_var = candle::Var::from_tensor(&tracked_src)?;
    let tracked = tracked_var.as_tensor();
    let tracked_ref = tracked
        .broadcast_add(&tracked_bias.reshape((1, 48, 1, 1))?)?
        .silu()?;
    let tracked_out = candle_nn::ops::conv2d_bias_silu(tracked, &tracked_bias)?;
    dev.synchronize()?;
    let tracked_diff = (&tracked_ref - &tracked_out)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    let tracked_pass = tracked.track_op() && tracked_out.track_op() && tracked_diff <= tol;
    all &= tracked_pass;
    println!(
        "R4_R1_P1_NEGATIVE_FALLBACK name=backprop_tracked tracked_input={} tracked_output={} max_abs={tracked_diff:.8} pass={tracked_pass}",
        tracked.track_op(), tracked_out.track_op()
    );
    println!(
        "R4_R1_P1_BACKPROP_FALLBACK tracked_input={} tracked_output={} max_abs={tracked_diff:.8} pass={tracked_pass}",
        tracked.track_op(), tracked_out.track_op()
    );

    println!("R4_R1_P1_PARITY_ALL pass={all}");
    println!("STATUS={}", if all { "PASS" } else { "FAIL" });
    if !all {
        candle::bail!("R4-R1-P1 production activation validation failed")
    }
    Ok(())
}
