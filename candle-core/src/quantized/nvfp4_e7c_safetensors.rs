//! ARCH-E7C: strict ModelOpt NVFP4 safetensors operator reader.
//!
//! Caller supplies the exact shards for `.weight`, `.weight_scale`, and
//! `.weight_scale_2`, optionally resolved through `model.safetensors.index.json`.
//! The memory maps preserve packed bytes and avoid loading entire shards into RAM.
//! Compressed-tensors inverse-scale convention is deliberately NOT guessed.
use crate::Result;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Counts shared by host validation and the kernels' signed 32-bit indexing.
#[cfg_attr(not(feature = "cuda-legacy-fp4"), allow(dead_code))]
pub(crate) struct Nvfp4LaunchShape {
    pub input_elements: usize,
    pub output_elements: usize,
    pub padded_k: usize,
    pub q8_bytes: usize,
}

pub(crate) fn checked_launch_shape(n: usize, k: usize, batch: usize) -> Result<Nvfp4LaunchShape> {
    let limit = i32::MAX as usize;
    if n == 0 || k == 0 || k % 32 != 0 || batch == 0 || batch > 65535 {
        crate::bail!("NVFP4 requires N>0, K divisible by 32, batch 1..65535");
    }
    let count = |a: usize, b: usize| -> Result<usize> {
        match a.checked_mul(b) {
            Some(value) if value <= limit => Ok(value),
            _ => crate::bail!("NVFP4 dimensions exceed CUDA signed 32-bit indexing"),
        }
    };
    count(n, k)?;
    let input_elements = count(batch, k)?;
    let output_elements = count(batch, n)?;
    let padded_k = k
        .checked_add(511)
        .map(|v| v / 512 * 512)
        .filter(|&v| v <= limit)
        .ok_or_else(|| crate::Error::Msg("NVFP4 padded K exceeds CUDA indexing".into()))?;
    let padded_elements = count(batch, padded_k)?;
    let q8_bytes = (padded_elements / 32)
        .checked_mul(36)
        .ok_or_else(|| crate::Error::Msg("NVFP4 Q8 workspace overflow".into()))?;
    Ok(Nvfp4LaunchShape {
        input_elements,
        output_elements,
        padded_k,
        q8_bytes,
    })
}

pub(crate) fn validate_modelopt_payload(
    n: usize,
    k: usize,
    packed: &[u8],
    scales: &[u8],
    global: f32,
) -> Result<()> {
    checked_launch_shape(n, k, 1)?;
    if packed.len() != n * k / 2
        || scales.len() != n * k / 16
        || !global.is_finite()
        || global <= 0.0
    {
        crate::bail!("NVFP4 invalid payload lengths or global multiplier");
    }
    const ABS_DOUBLED: [f32; 8] = [0., 1., 2., 3., 4., 6., 8., 12.];
    for (&scale, block) in scales.iter().zip(packed.chunks_exact(8)) {
        if scale & 0x80 != 0 || scale == 0x7f {
            crate::bail!("NVFP4 invalid signed/NaN E4M3FN scale");
        }
        // Match the kernels' order of operations, including intermediates.
        let base = float8::F8E4M3::from_bits(scale).to_f32() * global * 0.5;
        if !base.is_finite()
            || block.iter().any(|&byte| {
                !(base * ABS_DOUBLED[(byte & 7) as usize]).is_finite()
                    || !(base * ABS_DOUBLED[((byte >> 4) & 7) as usize]).is_finite()
            })
        {
            crate::bail!("NVFP4 dequantized weight exceeds the finite F32 range");
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ModelOptNvfp4Shards {
    pub packed: PathBuf,
    pub scales: PathBuf,
    pub global: PathBuf,
}

impl ModelOptNvfp4Shards {
    pub fn single_file(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        Self {
            packed: path.clone(),
            scales: path.clone(),
            global: path,
        }
    }

    pub fn load(&self, n: usize, k: usize, prefix: &str) -> Result<(Vec<u8>, Vec<u8>, f32)> {
        checked_launch_shape(n, k, 1)?;
        if prefix.is_empty() {
            crate::bail!("E7C invalid ModelOpt operator prefix/dimensions");
        }
        let w = read_tensor(&self.packed, &format!("{prefix}.weight"))?;
        let s = read_tensor(&self.scales, &format!("{prefix}.weight_scale"))?;
        let g = read_tensor(&self.global, &format!("{prefix}.weight_scale_2"))?;
        if (w.dtype != "U8" && w.dtype != "I8")
            || w.shape != [n, k / 2]
            || w.data.len() != n * k / 2
        {
            crate::bail!("E7C ModelOpt packed weight must be U8/I8 [n,k/2], low-even nibble");
        }
        if (s.dtype != "F8_E4M3" && s.dtype != "F8_E4M3FN")
            || s.shape != [n, k / 16]
            || s.data.len() != n * k / 16
        {
            crate::bail!("E7C ModelOpt scale must be F8_E4M3/F8_E4M3FN [n,k/16], row-major");
        }
        if g.dtype != "F32" || (!g.shape.is_empty() && g.shape != [1]) || g.data.len() != 4 {
            crate::bail!("E7C ModelOpt weight_scale_2 must be F32 scalar");
        }
        // Nonnegative finite E4M3 FN codes; reject sign-bit scales and NaNs.
        if s.data.iter().any(|&b| b & 0x80 != 0 || b == 0x7f) {
            crate::bail!("E7C invalid signed/NaN FP8 scale");
        }
        let multiplier = f32::from_le_bytes(g.data.as_slice().try_into().unwrap());
        if !multiplier.is_finite() || multiplier <= 0.0 {
            crate::bail!("E7C invalid global multiplier");
        }
        validate_modelopt_payload(n, k, &w.data, &s.data, multiplier)?;
        Ok((w.data, s.data, multiplier))
    }
}

struct RawTensor {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: String,
}

fn read_tensor(path: &Path, key: &str) -> Result<RawTensor> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(err) => crate::bail!("E7C unable to open {}: {err}", path.display()),
    };
    let mm = match unsafe { memmap2::MmapOptions::new().map(&file) } {
        Ok(mm) => mm,
        Err(err) => crate::bail!("E7C unable to mmap {}: {err}", path.display()),
    };
    let st = match safetensors::SafeTensors::deserialize(&mm) {
        Ok(st) => st,
        Err(err) => crate::bail!("E7C invalid safetensors {}: {err}", path.display()),
    };
    let v = match st.tensor(key) {
        Ok(v) => v,
        Err(err) => crate::bail!("E7C missing tensor {key}: {err}"),
    };
    Ok(RawTensor {
        data: v.data().to_vec(),
        shape: v.shape().to_vec(),
        dtype: format!("{:?}", v.dtype()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_scale_codes_cannot_hide_f32_dequantization_overflow() {
        let packed = [0x77; 16];
        assert!(validate_modelopt_payload(1, 32, &packed, &[0x7e; 2], f32::MAX).is_err());
        assert!(validate_modelopt_payload(1, 32, &packed, &[0x38; 2], 1.0).is_ok());
    }

    #[test]
    fn cuda_index_limits_are_checked_before_allocation() {
        // Each weight matrix fits, but its activation/output launch does not.
        assert!(checked_launch_shape(1, 65536, 32768).is_err());
        assert!(checked_launch_shape(65536, 32, 32768).is_err());
        // Rounding the valid K up to a Q8 workspace would exceed signed i32.
        assert!(checked_launch_shape(1, 2147483616, 1).is_err());
        assert!(checked_launch_shape(1, usize::MAX, 1).is_err());
        assert!(checked_launch_shape(4, 32, 0).is_err());
        assert!(checked_launch_shape(4, 33, 1).is_err());
        let valid = checked_launch_shape(2, 64, 9).unwrap();
        assert_eq!(valid.input_elements, 576);
        assert_eq!(valid.output_elements, 18);
        assert_eq!(valid.padded_k, 512);
        assert_eq!(valid.q8_bytes, 5184);
    }
}
