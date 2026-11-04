#![cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]

use candle_core::{DType, Device, Result, Tensor};

fn assert_close(got: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        let error = (got - expected).abs();
        assert!(
            got.is_finite() && error <= tolerance,
            "{label}[{index}]: got {got}, expected {expected}, error {error}"
        );
    }
}

#[test]
fn legacy_fp8_cast_roundtrip_cuda() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let input = [-16.0f32, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 16.0];
    let got = Tensor::new(&input, &device)?
        .to_dtype(DType::F8E4M3)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;

    assert_close(&got, &input, 0.0, "f32 -> fp8 -> f32");
    Ok(())
}

#[test]
fn legacy_fp8_unary_ops_cuda() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let input = Tensor::new(&[0.5f32, 1.0, 2.0, 4.0], &device)?.to_dtype(DType::F8E4M3)?;

    let neg = input.neg()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_close(&neg, &[-0.5, -1.0, -2.0, -4.0], 0.0, "neg");

    let recip = input.recip()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_close(&recip, &[2.0, 1.0, 0.5, 0.25], 0.0, "recip");

    let sqrt = input.sqrt()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_close(&sqrt, &[0.70710677, 1.0, 1.4142135, 2.0], 0.13, "sqrt");

    let exp = input.exp()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_close(
        &exp,
        &[1.6487212, 2.7182817, 7.389056, 54.59815],
        2.7,
        "exp",
    );
    Ok(())
}

#[test]
fn legacy_fp8_binary_ops_cuda() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let lhs = Tensor::new(&[-2.0f32, -1.0, 2.0, 4.0], &device)?.to_dtype(DType::F8E4M3)?;
    let rhs = Tensor::new(&[0.5f32, -0.5, 0.5, 2.0], &device)?.to_dtype(DType::F8E4M3)?;

    let add = lhs
        .broadcast_add(&rhs)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    assert_close(&add, &[-1.5, -1.5, 2.5, 6.0], 0.0, "add");

    let sub = lhs
        .broadcast_sub(&rhs)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    assert_close(&sub, &[-2.5, -0.5, 1.5, 2.0], 0.0, "sub");

    let mul = lhs
        .broadcast_mul(&rhs)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    assert_close(&mul, &[-1.0, 0.5, 1.0, 8.0], 0.0, "mul");

    let div = lhs
        .broadcast_div(&rhs)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    assert_close(&div, &[-4.0, 2.0, 4.0, 2.0], 0.0, "div");
    Ok(())
}

#[test]
fn legacy_fp8_reduces_after_widening_cuda() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let input = (0..32)
        .map(|value| value as f32 * 0.125)
        .collect::<Vec<_>>();
    let widened = Tensor::from_vec(input, 32, &device)?
        .to_dtype(DType::F8E4M3)?
        .to_dtype(DType::F32)?;

    let expected = widened.to_vec1::<f32>()?.iter().sum::<f32>();
    let got = widened.sum_all()?.to_scalar::<f32>()?;
    assert_eq!(got, expected);
    Ok(())
}
