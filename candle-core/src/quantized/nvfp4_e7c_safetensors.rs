//! ARCH-E7C: strict ModelOpt NVFP4 safetensors operator reader.
//!
//! Caller supplies the exact shards for `.weight`, `.weight_scale`, and
//! `.weight_scale_2`, optionally resolved through `model.safetensors.index.json`.
//! The memory maps preserve packed bytes and avoid loading entire shards into RAM.
//! Compressed-tensors inverse-scale convention is deliberately NOT guessed.
use crate::Result;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ModelOptNvfp4Shards {
    pub packed: PathBuf,
    pub scales: PathBuf,
    pub global: PathBuf,
}

impl ModelOptNvfp4Shards {
    pub fn single_file(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        Self {packed:path.clone(),scales:path.clone(),global:path}
    }

    pub fn load(&self, n: usize, k: usize, prefix: &str) -> Result<(Vec<u8>,Vec<u8>,f32)> {
        if prefix.is_empty() || n == 0 || k < 32 || k % 32 != 0 ||
            n.checked_mul(k).is_none_or(|z| z > i32::MAX as usize) {
            crate::bail!("E7C invalid ModelOpt operator prefix/dimensions");
        }
        let w = read_tensor(&self.packed, &format!("{prefix}.weight"))?;
        let s = read_tensor(&self.scales, &format!("{prefix}.weight_scale"))?;
        let g = read_tensor(&self.global, &format!("{prefix}.weight_scale_2"))?;
        if (w.dtype != "U8" && w.dtype != "I8") || w.shape != [n, k/2] || w.data.len() != n*k/2 {
            crate::bail!("E7C ModelOpt packed weight must be U8/I8 [n,k/2], low-even nibble");
        }
        if (s.dtype != "F8_E4M3" && s.dtype != "F8_E4M3FN") || s.shape != [n,k/16] || s.data.len() != n*k/16 {
            crate::bail!("E7C ModelOpt scale must be F8_E4M3/F8_E4M3FN [n,k/16], row-major");
        }
        if g.dtype != "F32" || (g.shape != [] && g.shape != [1]) || g.data.len() != 4 {
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
        Ok((w.data,s.data,multiplier))
    }
}

struct RawTensor { data: Vec<u8>, shape: Vec<usize>, dtype: String }

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
    Ok(RawTensor {data:v.data().to_vec(),shape:v.shape().to_vec(),dtype:format!("{:?}",v.dtype())})
}
