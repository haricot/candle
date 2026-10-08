use candle::{DType, Device, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};

const SHAPES: &[(usize, usize, usize)] = &[(48, 64, 48), (96, 32, 24), (192, 16, 12), (384, 8, 6)];

fn deterministic(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * mul) % 127) as isize + bias) as f32 / 128.0)
        .collect()
}

fn make_conv(c: usize, dev: &Device) -> Result<Conv2d> {
    let weight = Tensor::from_vec(deterministic(c * 25, 17, -63), (c, 1, 5, 5), dev)?;
    let bias = Tensor::from_vec(deterministic(c, 29, -31), c, dev)?;
    Ok(Conv2d::new(
        weight,
        Some(bias),
        Conv2dConfig {
            padding: 2,
            stride: 1,
            dilation: 1,
            groups: c,
            cudnn_fwd_algo: None,
        },
    ))
}

fn reference(x: &Tensor, conv: &Conv2d) -> Result<Tensor> {
    let cfg = conv.config();
    let y = x.conv2d_with_algo(
        conv.weight(),
        cfg.padding,
        cfg.stride,
        cfg.dilation,
        cfg.groups,
        cfg.cudnn_fwd_algo,
    )?;
    let Some(bias) = conv.bias() else {
        return y.silu();
    };
    let c = bias.dims1()?;
    y.broadcast_add(&bias.reshape((1, c, 1, 1))?)?.silu()
}

fn check(name: &str, x: &Tensor, conv: &Conv2d, tol: f32) -> Result<bool> {
    let expected = reference(x, conv)?;
    let got = conv.forward_silu(x)?;
    x.device().synchronize()?;
    let diff = (&expected - &got)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let pass = diff <= tol;
    println!(
        "R2C_C3_PARITY name={name} input={:?} weight={:?} max_abs={diff:.8} tolerance={tol:.8} pass={pass}",
        x.dims(),
        conv.weight().dims()
    );
    Ok(pass)
}

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let tol = 2.0e-5f32;
    let mut all = true;
    println!("=== V0.3.19-R2C C3 PRODUCTION VALIDATION ===");
    println!("production_default=true");
    println!("kill_switch=CANDLE_CUDA_DW5X5_BIAS_SILU_DISABLE");
    println!("exact_shapes={}", SHAPES.len());

    for &(c, h, w) in SHAPES {
        let x = Tensor::from_vec(deterministic(c * h * w, 13, -61), (1, c, h, w), &dev)?
            .to_dtype(DType::F32)?;
        let conv = make_conv(c, &dev)?;
        all &= check("exact", &x, &conv, tol)?;
    }

    // Outside exact shape: C3 must fall back.
    let c = 48;
    let x = Tensor::from_vec(deterministic(c * 32 * 24, 11, -59), (1, c, 32, 24), &dev)?;
    all &= check("shape_outside_frontier", &x, &make_conv(c, &dev)?, tol)?;

    // Wrong kernel: never C3.
    let weight = Tensor::from_vec(deterministic(c * 9, 7, -41), (c, 1, 3, 3), &dev)?;
    let bias = Tensor::from_vec(deterministic(c, 19, -23), c, &dev)?;
    let conv = Conv2d::new(
        weight,
        Some(bias),
        Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: c,
            cudnn_fwd_algo: None,
        },
    );
    let x = Tensor::from_vec(deterministic(c * 64 * 48, 5, -53), (1, c, 64, 48), &dev)?;
    all &= check("kernel_not_5x5", &x, &conv, tol)?;

    // Ordinary group=1 convolution: never C3.
    let weight = Tensor::from_vec(deterministic(c * c * 25, 3, -37), (c, c, 5, 5), &dev)?;
    let bias = Tensor::from_vec(deterministic(c, 31, -17), c, &dev)?;
    let conv = Conv2d::new(
        weight,
        Some(bias),
        Conv2dConfig {
            padding: 2,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        },
    );
    all &= check("groups_not_depthwise", &x, &conv, tol)?;

    println!("R2C_C3_PARITY_ALL pass={all}");
    println!("STATUS={}", if all { "PASS" } else { "FAIL" });
    if !all {
        candle::bail!("v0.3.19-r2c C3 production parity failed")
    }
    Ok(())
}
