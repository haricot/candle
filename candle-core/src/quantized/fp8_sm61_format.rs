//! Host-side FP8 E4M3FN decoding and validation, also available without CUDA.
use crate::Result;
use half::f16;

pub fn e4m3fn_decode(byte: u8) -> f32 {
    let e = ((byte >> 3) & 15) as i32;
    let m = (byte & 7) as i32;
    if e == 15 && m == 7 {
        return f32::NAN;
    }
    let magnitude = if e == 0 {
        (m as f32) * 2f32.powi(-9)
    } else {
        ((8 + m) as f32) * 2f32.powi(e - 10)
    };
    if byte & 128 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

#[cfg(any(test, all(feature = "cuda", feature = "cuda-legacy-fp8")))]
pub(crate) fn checked_f16_weight(value: f32) -> Result<f16> {
    let rounded = f16::from_f32(value);
    if !value.is_finite() || !rounded.is_finite() {
        crate::bail!("FP8 expansion is outside the finite FP16 weight range");
    }
    Ok(rounded)
}

/// Validate and expand an E4M3 inverse-scale view without assuming a model family.
/// Only explicit `weight_scale_inv` conventions are supported, not `weight_scale`.
pub(crate) fn decode_f8e_inverse_scales(
    n: usize,
    k: usize,
    view: &safetensors::tensor::TensorView<'_>,
) -> Result<(Vec<f32>, usize, usize)> {
    if n == 0 || k == 0 {
        crate::bail!("F8E inverse scales require nonempty weight dimensions");
    }
    let scales: Vec<f32> = match format!("{:?}", view.dtype()).as_str() {
        "F32" if view.data().len() % 4 == 0 => view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        "F16" if view.data().len() % 2 == 0 => view
            .data()
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect(),
        "BF16" if view.data().len() % 2 == 0 => view
            .data()
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect(),
        _ => crate::bail!("F8E unsupported inverse-scale dtype or byte count"),
    };
    let shape = view.shape();
    let (bn, bk) = match shape {
        [] | [1] => (n, k),
        [a] if *a == n => (1, k),
        [a, b] if *a == n && *b == 1 => (1, k),
        [a, b] if *a == n.div_ceil(128) && *b == k.div_ceil(128) => (128, 128),
        [a, 1, b, 1] if *a == n.div_ceil(128) && *b == k.div_ceil(128) => (128, 128),
        _ => crate::bail!(
            "F8E unsupported inverse-scale layout {:?} for [{n},{k}]",
            shape
        ),
    };
    if scales.len() != n.div_ceil(bn) * k.div_ceil(bk)
        || scales.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        crate::bail!("F8E scale count, sign, or finiteness failure");
    }
    Ok((scales, bn, bk))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn e4m3fn_all_encodings_match_the_dtype_implementation() {
        for bits in 0..=255u8 {
            let actual = e4m3fn_decode(bits);
            let expected = float8::F8E4M3::from_bits(bits).to_f32();
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits(), "code {bits:#x}");
            }
        }
    }
    #[test]
    fn expanded_weights_reject_finite_values_that_overflow_f16() {
        assert!(checked_f16_weight(448.0 * 200.0).is_err());
        assert!(checked_f16_weight(-448.0 * 200.0).is_err());
        assert!(checked_f16_weight(f32::NAN).is_err());
        assert!(checked_f16_weight(f32::INFINITY).is_err());
        assert_eq!(checked_f16_weight(65504.0).unwrap().to_f32(), 65504.0);
        assert_eq!(checked_f16_weight(-0.0).unwrap().to_bits(), 0x8000);
    }
    #[test]
    fn empty_checkpoint_geometry_is_an_error_not_a_division_by_zero() {
        let bytes = 1f32.to_le_bytes();
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![], &bytes).unwrap();
        assert!(decode_f8e_inverse_scales(0, 128, &view).is_err());
        assert!(decode_f8e_inverse_scales(16, 0, &view).is_err());
    }
}
