#![cfg(all(feature = "cuda", feature = "cudnn"))]

use anyhow::Result;
use candle_core::{Device, Tensor, Var};

#[derive(Clone, Copy)]
struct Case {
    batch: usize,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
    groups: usize,
}

fn deterministic_values(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let v = ((i * mul) % 101) as isize + bias;
            v as f32 / 64.0
        })
        .collect()
}

fn legacy_grouped_conv2d(
    x: &Tensor,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    groups: usize,
) -> Result<Tensor> {
    let xs = x.chunk(groups, 1)?;
    let ks = kernel.chunk(groups, 0)?;
    let mut ys = Vec::with_capacity(groups);
    for (xg, kg) in xs.iter().zip(&ks) {
        ys.push(xg.conv2d(kg, padding, stride, 1, 1)?);
    }
    Ok(Tensor::cat(&ys, 1)?)
}

fn assert_close(native: &Tensor, legacy: &Tensor) -> Result<()> {
    assert_eq!(native.dims(), legacy.dims());
    let native = native.flatten_all()?.to_vec1::<f32>()?;
    let legacy = legacy.flatten_all()?.to_vec1::<f32>()?;
    for (idx, (&a, &b)) in native.iter().zip(&legacy).enumerate() {
        let abs = (a - b).abs();
        let tol = 1e-4 + 1e-4 * b.abs();
        assert!(
            abs <= tol,
            "grouped cuDNN mismatch at {idx}: native={a}, legacy={b}, abs={abs}, tol={tol}"
        );
    }
    Ok(())
}

// Regression coverage for #3389. PR #3531 addresses the common CUDA depthwise case with a
// differentiable Tensor-level specialization; these cases additionally cover general grouped
// conv2d (c_in/groups > 1) through cuDNN's native group-count support.
#[test]
fn grouped_cudnn_matches_legacy_decomposition() -> Result<()> {
    let device = Device::new_cuda(0)?;
    std::env::set_var("CANDLE_CUDNN_NATIVE_GROUPED_STRICT", "1");

    let cases = [
        Case {
            batch: 1,
            c_in: 16,
            c_out: 24,
            h: 32,
            w: 24,
            kernel: 3,
            padding: 1,
            stride: 1,
            groups: 2,
        },
        Case {
            batch: 1,
            c_in: 32,
            c_out: 48,
            h: 32,
            w: 24,
            kernel: 3,
            padding: 1,
            stride: 2,
            groups: 4,
        },
        Case {
            batch: 3,
            c_in: 32,
            c_out: 64,
            h: 24,
            w: 20,
            kernel: 5,
            padding: 2,
            stride: 1,
            groups: 8,
        },
        Case {
            batch: 1,
            c_in: 48,
            c_out: 48,
            h: 64,
            w: 48,
            kernel: 5,
            padding: 2,
            stride: 1,
            groups: 48,
        },
        Case {
            batch: 3,
            c_in: 96,
            c_out: 96,
            h: 32,
            w: 24,
            kernel: 5,
            padding: 2,
            stride: 1,
            groups: 96,
        },
        Case {
            batch: 1,
            c_in: 384,
            c_out: 384,
            h: 8,
            w: 6,
            kernel: 5,
            padding: 2,
            stride: 1,
            groups: 384,
        },
    ];

    for case in cases {
        let x_len = case.batch * case.c_in * case.h * case.w;
        let k_len = case.c_out * (case.c_in / case.groups) * case.kernel * case.kernel;
        let x = Tensor::from_vec(
            deterministic_values(x_len, 37, -50),
            (case.batch, case.c_in, case.h, case.w),
            &device,
        )?;
        let kernel = Tensor::from_vec(
            deterministic_values(k_len, 53, -50),
            (
                case.c_out,
                case.c_in / case.groups,
                case.kernel,
                case.kernel,
            ),
            &device,
        )?;

        let native = x.conv2d(&kernel, case.padding, case.stride, 1, case.groups)?;
        let legacy = legacy_grouped_conv2d(&x, &kernel, case.padding, case.stride, case.groups)?;
        device.synchronize()?;
        assert_close(&native, &legacy)?;
    }
    Ok(())
}

#[test]
fn grouped_cudnn_preserves_autograd_fallback() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let x = Var::from_vec(
        deterministic_values(4 * 8 * 8, 37, -50),
        (1, 4, 8, 8),
        &device,
    )?;
    let kernel = Var::from_vec(
        deterministic_values(6 * 2 * 3 * 3, 53, -50),
        (6, 2, 3, 3),
        &device,
    )?;

    let loss = x.conv2d(&kernel, 1, 1, 1, 2)?.sqr()?.sum_all()?;
    let grads = loss.backward()?;
    assert!(grads.get(&x).is_some());
    assert!(grads.get(&kernel).is_some());
    Ok(())
}
