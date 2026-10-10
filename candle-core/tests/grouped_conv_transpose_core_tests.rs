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
            "grouped transpose convolution mismatch: a={a}, b={b}, abs={abs}, max_abs={max_abs}"
        );
    }
    Ok(())
}

fn legacy_conv_transpose1d(
    x: &Tensor,
    kernel: &Tensor,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    let xs = x.chunk(groups, 1)?;
    let ks = kernel.chunk(groups, 0)?;
    let ys = xs
        .iter()
        .zip(&ks)
        .map(|(xg, kg)| xg.conv_transpose1d(kg, padding, output_padding, stride, dilation, 1))
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&ys, 1)
}

fn legacy_conv_transpose2d(
    x: &Tensor,
    kernel: &Tensor,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    let xs = x.chunk(groups, 1)?;
    let ks = kernel.chunk(groups, 0)?;
    let ys = xs
        .iter()
        .zip(&ks)
        .map(|(xg, kg)| xg.conv_transpose2d(kg, padding, output_padding, stride, dilation))
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&ys, 1)
}

fn check_forward(device: &Device) -> Result<()> {
    for &(batch, c_in, c_out, len, ksize, padding, out_pad, stride, dilation, groups) in &[
        (1, 8, 12, 11, 3, 1, 0, 1, 1, 2),
        (3, 8, 16, 7, 3, 1, 1, 2, 1, 4),
        (1, 8, 8, 9, 3, 2, 0, 1, 2, 8),
    ] {
        let x = Tensor::from_vec(
            deterministic(batch * c_in * len, 37, -50),
            (batch, c_in, len),
            device,
        )?;
        let c_out_group = c_out / groups;
        let k = Tensor::from_vec(
            deterministic(c_in * c_out_group * ksize, 53, -50),
            (c_in, c_out_group, ksize),
            device,
        )?;
        let grouped = x.conv_transpose1d(&k, padding, out_pad, stride, dilation, groups)?;
        let legacy = legacy_conv_transpose1d(&x, &k, padding, out_pad, stride, dilation, groups)?;
        assert_close(&grouped, &legacy, 1e-4, 1e-4)?;
    }

    for &(batch, c_in, c_out, h, w, ksize, padding, out_pad, stride, dilation, groups) in &[
        (1, 8, 12, 7, 6, 3, 1, 0, 1, 1, 2),
        (1, 8, 16, 6, 5, 3, 1, 1, 2, 1, 4),
        (3, 8, 8, 5, 4, 3, 2, 0, 1, 2, 8),
    ] {
        let x = Tensor::from_vec(
            deterministic(batch * c_in * h * w, 37, -50),
            (batch, c_in, h, w),
            device,
        )?;
        let c_out_group = c_out / groups;
        let k = Tensor::from_vec(
            deterministic(c_in * c_out_group * ksize * ksize, 53, -50),
            (c_in, c_out_group, ksize, ksize),
            device,
        )?;
        let grouped =
            x.conv_transpose2d_with_groups(&k, padding, out_pad, stride, dilation, groups)?;
        let legacy = legacy_conv_transpose2d(&x, &k, padding, out_pad, stride, dilation, groups)?;
        assert_close(&grouped, &legacy, 1e-4, 1e-4)?;
    }
    Ok(())
}

fn check_autograd_1d(device: &Device) -> Result<()> {
    let groups = 4;
    let x0 = Tensor::from_vec(deterministic(2 * 8 * 7, 37, -50), (2, 8, 7), device)?;
    let k0 = Tensor::from_vec(deterministic(8 * 3 * 3, 53, -50), (8, 3, 3), device)?;

    let x_grouped = Var::from_tensor(&x0)?;
    let k_grouped = Var::from_tensor(&k0)?;
    let y_grouped = x_grouped.conv_transpose1d(&k_grouped, 1, 1, 2, 1, groups)?;
    let loss_grouped = y_grouped.sqr()?.sum_all()?;
    let grads_grouped = loss_grouped.backward()?;

    let x_legacy = Var::from_tensor(&x0)?;
    let k_legacy = Var::from_tensor(&k0)?;
    let y_legacy = legacy_conv_transpose1d(&x_legacy, &k_legacy, 1, 1, 2, 1, groups)?;
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

fn check_autograd_2d(device: &Device) -> Result<()> {
    let groups = 4;
    let x0 = Tensor::from_vec(deterministic(2 * 8 * 5 * 4, 37, -50), (2, 8, 5, 4), device)?;
    let k0 = Tensor::from_vec(deterministic(8 * 3 * 3 * 3, 53, -50), (8, 3, 3, 3), device)?;

    let x_grouped = Var::from_tensor(&x0)?;
    let k_grouped = Var::from_tensor(&k0)?;
    let y_grouped = x_grouped.conv_transpose2d_with_groups(&k_grouped, 1, 1, 2, 1, groups)?;
    let loss_grouped = y_grouped.sqr()?.sum_all()?;
    let grads_grouped = loss_grouped.backward()?;

    let x_legacy = Var::from_tensor(&x0)?;
    let k_legacy = Var::from_tensor(&k0)?;
    let y_legacy = legacy_conv_transpose2d(&x_legacy, &k_legacy, 1, 1, 2, 1, groups)?;
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

fn with_env(vars: &[(&str, &str)], f: impl FnOnce() -> Result<()>) -> Result<()> {
    for (name, value) in vars {
        std::env::set_var(name, value);
    }
    let result = f();
    for (name, _) in vars {
        std::env::remove_var(name);
    }
    result
}

#[test]
fn grouped_conv_transpose_cpu_forward_parity() -> Result<()> {
    with_env(
        &[("CANDLE_CPU_NATIVE_GROUPED_TRANSPOSE_STRICT", "1")],
        || check_forward(&Device::Cpu),
    )
}

#[test]
fn grouped_conv_transpose_cpu_autograd_parity() -> Result<()> {
    with_env(
        &[("CANDLE_CPU_NATIVE_GROUPED_TRANSPOSE_STRICT", "1")],
        || {
            check_autograd_1d(&Device::Cpu)?;
            check_autograd_2d(&Device::Cpu)
        },
    )
}

#[cfg(feature = "metal")]
#[test]
fn grouped_conv_transpose_metal_forward_and_autograd_parity() -> Result<()> {
    with_env(
        &[("CANDLE_METAL_NATIVE_GROUPED_TRANSPOSE_STRICT", "1")],
        || {
            let device = Device::new_metal(0)?;
            check_forward(&device)?;
            check_autograd_1d(&device)?;
            check_autograd_2d(&device)
        },
    )
}

#[cfg(all(feature = "cuda", feature = "cudnn"))]
#[test]
fn grouped_conv_transpose_cuda_forward_and_autograd_parity() -> Result<()> {
    let device = Device::new_cuda(0)?;
    check_forward(&device)?;
    check_autograd_1d(&device)?;
    check_autograd_2d(&device)
}

#[cfg(feature = "cuda")]
#[test]
fn grouped_conv_transpose_cuda_native_kernel_smoke() -> Result<()> {
    with_env(
        &[
            ("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL", "1"),
            ("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT", "1"),
        ],
        || {
            let device = Device::new_cuda(0)?;
            check_forward(&device)?;
            check_autograd_1d(&device)?;
            check_autograd_2d(&device)
        },
    )
}

#[cfg(all(feature = "cuda", feature = "cudnn"))]
#[test]
fn grouped_conv_transpose_cuda_cudnn_native_smoke() -> Result<()> {
    with_env(
        &[("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT", "1")],
        || {
            let device = Device::new_cuda(0)?;

            let x1 = Tensor::from_vec(deterministic(8 * 9, 37, -50), (1, 8, 9), &device)?;
            let k1 = Tensor::from_vec(deterministic(8 * 6 * 3, 53, -50), (8, 6, 3), &device)?;
            let y1 = x1.conv_transpose1d(&k1, 1, 0, 1, 1, 2)?;
            let r1 = legacy_conv_transpose1d(&x1, &k1, 1, 0, 1, 1, 2)?;
            assert_close(&y1, &r1, 1e-4, 1e-4)?;

            let x2 = Tensor::from_vec(deterministic(8 * 7 * 6, 37, -50), (1, 8, 7, 6), &device)?;
            let k2 =
                Tensor::from_vec(deterministic(8 * 6 * 3 * 3, 53, -50), (8, 6, 3, 3), &device)?;
            let y2 = x2.conv_transpose2d_with_groups(&k2, 1, 0, 1, 1, 2)?;
            let r2 = legacy_conv_transpose2d(&x2, &k2, 1, 0, 1, 1, 2)?;
            assert_close(&y2, &r2, 1e-4, 1e-4)
        },
    )
}
