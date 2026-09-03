use candle_core::{Device, Result, Tensor, Var};

fn deterministic(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * mul) % 101) as isize + bias) as f32 / 64.0)
        .collect()
}

fn assert_close(lhs: &Tensor, rhs: &Tensor, atol: f32, rtol: f32) -> Result<()> {
    assert_eq!(lhs.dims(), rhs.dims());
    let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
    let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
    let mut max_abs = 0f32;
    for (&a, &b) in lhs.iter().zip(&rhs) {
        let abs = (a - b).abs();
        max_abs = max_abs.max(abs);
        assert!(
            abs <= atol + rtol * b.abs(),
            "grouped convolution mismatch: a={a}, b={b}, abs={abs}, max_abs={max_abs}"
        );
    }
    Ok(())
}

fn legacy_conv1d(
    x: &Tensor,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    let xs = x.chunk(groups, 1)?;
    let ks = kernel.chunk(groups, 0)?;
    let ys = xs
        .iter()
        .zip(&ks)
        .map(|(xg, kg)| xg.conv1d(kg, padding, stride, dilation, 1))
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&ys, 1)
}

fn legacy_conv2d(
    x: &Tensor,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    let xs = x.chunk(groups, 1)?;
    let ks = kernel.chunk(groups, 0)?;
    let ys = xs
        .iter()
        .zip(&ks)
        .map(|(xg, kg)| xg.conv2d(kg, padding, stride, dilation, 1))
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&ys, 1)
}

fn check_forward(device: &Device) -> Result<()> {
    for &(batch, c_in, c_out, len, kernel_size, padding, stride, groups) in &[
        (1, 8, 12, 31, 3, 1, 1, 2),
        (3, 16, 24, 29, 5, 2, 2, 4),
        (1, 16, 16, 33, 3, 1, 1, 16),
    ] {
        let x = Tensor::from_vec(
            deterministic(batch * c_in * len, 37, -50),
            (batch, c_in, len),
            device,
        )?;
        let k = Tensor::from_vec(
            deterministic(c_out * (c_in / groups) * kernel_size, 53, -50),
            (c_out, c_in / groups, kernel_size),
            device,
        )?;
        let grouped = x.conv1d(&k, padding, stride, 1, groups)?;
        let legacy = legacy_conv1d(&x, &k, padding, stride, 1, groups)?;
        assert_close(&grouped, &legacy, 1e-4, 1e-4)?;
    }

    for &(batch, c_in, c_out, h, w, ksize, padding, stride, groups) in &[
        (1, 8, 12, 17, 15, 3, 1, 1, 2),
        (1, 16, 24, 18, 16, 3, 1, 2, 4),
        (3, 16, 32, 13, 11, 5, 2, 1, 8),
        (1, 32, 32, 9, 7, 3, 1, 1, 32),
    ] {
        let x = Tensor::from_vec(
            deterministic(batch * c_in * h * w, 37, -50),
            (batch, c_in, h, w),
            device,
        )?;
        let k = Tensor::from_vec(
            deterministic(c_out * (c_in / groups) * ksize * ksize, 53, -50),
            (c_out, c_in / groups, ksize, ksize),
            device,
        )?;
        let grouped = x.conv2d(&k, padding, stride, 1, groups)?;
        let legacy = legacy_conv2d(&x, &k, padding, stride, 1, groups)?;
        assert_close(&grouped, &legacy, 1e-4, 1e-4)?;
    }
    Ok(())
}

fn check_autograd(device: &Device) -> Result<()> {
    let groups = 4;
    let x0 = Tensor::from_vec(
        deterministic(2 * 8 * 9 * 7, 37, -50),
        (2, 8, 9, 7),
        device,
    )?;
    let k0 = Tensor::from_vec(
        deterministic(12 * 2 * 3 * 3, 53, -50),
        (12, 2, 3, 3),
        device,
    )?;

    let x_grouped = Var::from_tensor(&x0)?;
    let k_grouped = Var::from_tensor(&k0)?;
    let y_grouped = x_grouped.conv2d(&k_grouped, 1, 1, 1, groups)?;
    let loss_grouped = y_grouped.sqr()?.sum_all()?;
    let grads_grouped = loss_grouped.backward()?;

    let x_legacy = Var::from_tensor(&x0)?;
    let k_legacy = Var::from_tensor(&k0)?;
    let y_legacy = legacy_conv2d(&x_legacy, &k_legacy, 1, 1, 1, groups)?;
    let loss_legacy = y_legacy.sqr()?.sum_all()?;
    let grads_legacy = loss_legacy.backward()?;

    assert_close(&y_grouped, &y_legacy, 1e-4, 1e-4)?;
    assert_close(
        grads_grouped.get(&x_grouped).expect("grouped input grad"),
        grads_legacy.get(&x_legacy).expect("legacy input grad"),
        1e-4,
        1e-4,
    )?;
    assert_close(
        grads_grouped.get(&k_grouped).expect("grouped kernel grad"),
        grads_legacy.get(&k_legacy).expect("legacy kernel grad"),
        1e-4,
        1e-4,
    )?;
    Ok(())
}

#[test]
fn grouped_conv_cpu_forward_parity() -> Result<()> {
    check_forward(&Device::Cpu)
}

#[test]
fn grouped_conv_cpu_autograd_parity() -> Result<()> {
    check_autograd(&Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn grouped_conv_metal_forward_and_autograd_parity() -> Result<()> {
    let device = Device::new_metal(0)?;
    check_forward(&device)?;
    check_autograd(&device)
}

#[cfg(all(feature = "cuda", feature = "cudnn"))]
#[test]
fn grouped_conv_cuda_cudnn_forward_and_autograd_parity() -> Result<()> {
    let device = Device::new_cuda(0)?;
    check_forward(&device)?;
    check_autograd(&device)
}
