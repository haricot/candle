//! Reusable CUDA-image transport for *external typed* DKR consumers.
//! Native Candle kernels keep their own sealed catalogue, but use the same
//! device-scoped image policy, CUDA context and admission/fallback rules.
//! Operation manifests, argument ABI and producer attestations belong to callers.

use super::{dkr_image_policy::DkrImageMode, CudaDevice, CudaFunc, DkrLoadedModule, WrapErr};
use crate::Result;
use cudarc::driver::{CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;
use dkr_kernel_registry::{telemetry::{self, Event}, VerifiedArtifact, TYPED_ABI};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, sync::Arc};

const MAX_TYPED_CUBIN_BYTES: usize = 32 * 1024 * 1024;

/// ELF64 little endian; EM_CUDA=190; pre-Blackwell EF_CUDA_SM is e_flags & 0xff.
/// CUDA flag bits 8..15 carry unrelated control flags (LLVM ELF.h).
/// The CUDA driver and function lookup remain the authority for image admission.
fn validate_cubin(image: &[u8], expected_sha: &str, target_cc: u32) -> Result<()> {
    if image.len() < 64 || image.len() > MAX_TYPED_CUBIN_BYTES
        || !image.starts_with(b"\x7fELF") || image[4] != 2 || image[5] != 1
        || u16::from_le_bytes([image[18], image[19]]) != 190
        || (u32::from_le_bytes([image[48], image[49], image[50], image[51]]) & 0xff) != target_cc
    {
        crate::bail!("DKR typed CUBIN ELF architecture/header mismatch");
    }
    let observed = format!("{:x}", Sha256::digest(image));
    if !observed.eq_ignore_ascii_case(expected_sha) {
        crate::bail!("DKR typed CUBIN SHA256 mismatch");
    }
    Ok(())
}

impl CudaDevice {
    /// Load an ABI-verified *typed* PTX/CUBIN pair. The calling adapter verifies
    /// the manifest, typed parameters, and linkage of these two images.
    ///
    /// An exact-target CUBIN is preferred in auto mode; the only silent retry is
    /// verified PTX after CUDA module/function admission fails *before launch*.
    /// The image choice is fixed at device construction. An incomplete CUBIN
    /// manifest or corrupted image is a hard error, never a fallback trigger.
    pub fn get_or_load_dkr_typed_image_func(
        &self,
        artifact: &VerifiedArtifact,
        entry: &str,
        cubin: Option<(&[u8], &str)>,
    ) -> Result<(CudaFunc, bool)> {
        let target = artifact.key().target_cc();
        let sha = artifact.key().sha256();
        let evidence = telemetry::handle(TYPED_ABI, "typed_image", entry, sha, target);
        if !artifact.has_entry(entry) {
            evidence.record(Event::Rejected);
            crate::bail!("DKR typed entry absent: {entry}");
        }
        let (major, minor) = self.context.compute_capability().w()?;
        let device_cc = (major as u32) * 10 + minor as u32;
        if device_cc < target {
            evidence.record(Event::Rejected);
            crate::bail!("DKR typed PTX architecture exceeds CUDA device");
        }
        let cubin_sha = if let Some((image, expected_sha)) = cubin {
            validate_cubin(image, expected_sha, target)?;
            Some(expected_sha.to_ascii_lowercase())
        } else {
            None
        };
        let usable_cubin = if device_cc == target { cubin } else { None };
        if matches!(self.dkr_image_mode, DkrImageMode::Cubin) && usable_cubin.is_none() {
            evidence.record(Event::Rejected);
            crate::bail!("strict DKR typed CUBIN mode requires a verified image for exact GPU SM");
        }
        // A typed image and a native image with the same PTX hash may have
        // different provenance. This namespace deliberately remains separate.
        let key = (artifact.key().clone(), cubin_sha);
        let cache = self.dkr_typed_functions.read().unwrap();
        if let Some(function) = cache.get(&key).and_then(|module| module.functions.get(entry)) {
            evidence.record(Event::FunctionHit);
            return Ok((CudaFunc::new(function.clone(), self.stream.clone()), false));
        }
        drop(cache);
        let mut cache = self.dkr_typed_functions.write().unwrap();
        if let Some(function) = cache.get(&key).and_then(|module| module.functions.get(entry)) {
            evidence.record(Event::FunctionHit);
            return Ok((CudaFunc::new(function.clone(), self.stream.clone()), false));
        }
        evidence.record(Event::FunctionMiss);
        let first = !cache.contains_key(&key);
        let function = if let Some(loaded) = cache.get_mut(&key) {
            let function = loaded.module.load_function(entry).w()?;
            loaded.functions.insert(entry.into(), function.clone());
            function
        } else {
            let load_ptx = || -> Result<(Arc<CudaModule>, CudaFunction)> {
                let module = self.context.load_module(artifact.ptx().into()).w()?;
                let function = module.load_function(entry).w()?;
                if std::env::var_os("CANDLE_DKR_TRACE").is_some() {
                    eprintln!("CANDLE_DKR_TYPED_IMAGE_LOADED target=sm{} format=ptx ptx_sha256={}", target, sha);
                }
                Ok((module, function))
            };
            let load_cubin = |image: &[u8]| -> Result<(Arc<CudaModule>, CudaFunction)> {
                let module = self.context.load_module(Ptx::from_binary(image.to_vec())).w()?;
                let function = module.load_function(entry).w()?;
                if std::env::var_os("CANDLE_DKR_TRACE").is_some() {
                    eprintln!("CANDLE_DKR_TYPED_IMAGE_LOADED target=sm{} format=cubin cubin_sha256={} ptx_sha256={}",
                        target, key.1.as_deref().unwrap_or(""), sha);
                }
                Ok((module, function))
            };
            let (module, function) = match self.dkr_image_mode {
                DkrImageMode::Ptx => load_ptx()?,
                DkrImageMode::Cubin => load_cubin(usable_cubin.expect("strict typed image checked").0)?,
                DkrImageMode::Auto => match usable_cubin {
                    Some((image, _)) => match load_cubin(image) {
                        Ok(value) => value,
                        Err(error) => {
                            // Admission error only, never recovery after execution.
                            eprintln!("CANDLE_DKR_TYPED_IMAGE_FALLBACK target=sm{} reason=module_admission_failure error={}",
                                target, error);
                            load_ptx()?
                        }
                    },
                    None => load_ptx()?,
                },
            };
            cache.insert(key, DkrLoadedModule {
                module,
                functions: HashMap::from([(entry.to_owned(), function.clone())]),
            });
            function
        };
        if first { evidence.record(Event::ModuleLoaded); }
        Ok((CudaFunc::new(function, self.stream.clone()), first))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(cc: u32) -> (Vec<u8>, String) {
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[18..20].copy_from_slice(&190u16.to_le_bytes());
        bytes[48..52].copy_from_slice(&(cc | 0x400).to_le_bytes());
        let sha = format!("{:x}", Sha256::digest(&bytes));
        (bytes, sha)
    }

    // External typed-only deployment intentionally keeps Candle's regular
    // CUDA tensor/FFI producers enabled. Unlike the native sealed catalogue,
    // its feature must not disable standard allocations/conversions/MoE.
    #[cfg(not(feature = "cuda-dkr-catalog"))]
    #[test]
    fn typed_only_preserves_standard_candle_cuda_kernels() {
        assert!(crate::cuda_backend::kernels::BUILTIN_PTX_ENABLED);
        assert!(crate::cuda_backend::kernels::STATIC_FFI_ENABLED);
    }

    #[test]
    fn typed_cubin_requires_exact_physical_target_and_bytes() {
        let (mut binary, hash) = fixture(61);
        assert!(validate_cubin(&binary, &hash, 61).is_ok());
        assert!(validate_cubin(&binary, &hash, 80).is_err());
        // Control flags may change the high byte without changing SM61.
        let (control_flags, control_hash) = fixture(61 | 0x300);
        assert!(validate_cubin(&control_flags, &control_hash, 61).is_ok());
        let (sm75, sm75_hash) = fixture(75);
        assert!(validate_cubin(&sm75, &sm75_hash, 61).is_err());
        binary[60] = 1;
        assert!(validate_cubin(&binary, &hash, 61).is_err());
    }
}
