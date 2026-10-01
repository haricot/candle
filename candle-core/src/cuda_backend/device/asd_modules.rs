//! ASD V3 step 1: resolve promoted implementation ids through CUDA module providers.
//!
//! External modules are intentionally opt-in. Set CANDLE_ASD_MODULE_DIR to a directory
//! containing <implementation_id>.cubin or <implementation_id>.ptx. CUBIN wins when
//! both exist. Missing external artifacts fall back to the embedded SM61 PTX catalogue.

use super::{CudaDevice, CudaFunc};
use crate::cuda_backend::WrapErr;
use crate::{Error, Result};
use cudarc::driver::LaunchConfig;
use cudarc::nvrtc::Ptx;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

#[derive(Clone, Copy, Debug)]
struct AsdKernelSpec {
    implementation_id: &'static str,
    candidate_id: &'static str,
    entry: &'static str,
    output_count: usize,
    grid_x: u32,
    block_x: u32,
}

const IMPLEMENTATIONS: &[AsdKernelSpec] = &[
    AsdKernelSpec {
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256",
        candidate_id: "ct1d-s32-g2-u1-b256",
        entry: "flow_v0322_ct1d_s32_g2_u1_b256",
        output_count: 8192,
        grid_x: 32,
        block_x: 256,
    },
    AsdKernelSpec {
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g4-u1-b256",
        candidate_id: "ct1d-s32-g4-u1-b256",
        entry: "flow_v0322_ct1d_s32_g4_u1_b256",
        output_count: 8192,
        grid_x: 32,
        block_x: 256,
    },
    AsdKernelSpec {
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g8-u1-b256",
        candidate_id: "ct1d-s32-g8-u1-b256",
        entry: "flow_v0322_ct1d_s32_g8_u1_b256",
        output_count: 8192,
        grid_x: 32,
        block_x: 256,
    },
    AsdKernelSpec {
        implementation_id: "candle.sm61-exact-grouped.ct1d-s32-g16-u1-b256",
        candidate_id: "ct1d-s32-g16-u1-b256",
        entry: "flow_v0322_ct1d_s32_g16_u1_b256",
        output_count: 8192,
        grid_x: 32,
        block_x: 256,
    },
    AsdKernelSpec {
        implementation_id: "candle.sm61-exact-grouped.ct2d-s32-g16-u4-b128",
        candidate_id: "ct2d-s32-g16-u4-b128",
        entry: "flow_v0322_ct2d_s32_g16_u4_b128",
        output_count: 524288,
        grid_x: 4096,
        block_x: 128,
    },
    AsdKernelSpec {
        implementation_id: "candle.sm61-exact-grouped.ct2d-s32-g32-u4-b64",
        candidate_id: "ct2d-s32-g32-u4-b64",
        entry: "flow_v0322_ct2d_s32_g32_u4_b64",
        output_count: 524288,
        grid_x: 8192,
        block_x: 64,
    },
    AsdKernelSpec {
        implementation_id: "candle.sm61-exact-grouped.gc1d-l128-g8-u1-b256",
        candidate_id: "gc1d-l128-g8-u1-b256",
        entry: "flow_v0322_gc1d_l128_g8_u1_b256",
        output_count: 8192,
        grid_x: 32,
        block_x: 256,
    },
];

#[derive(Debug)]
enum AsdModuleSource {
    BuiltinPtx(&'static str),
    External {
        path: PathBuf,
        kind: ExternalModuleKind,
    },
}

#[derive(Clone, Copy, Debug)]
enum ExternalModuleKind {
    Cubin,
    Ptx,
}

trait AsdModuleProvider {
    fn resolve(&self, spec: &'static AsdKernelSpec) -> Result<Option<AsdModuleSource>>;
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BuiltinProvider;

impl AsdModuleProvider for BuiltinProvider {
    fn resolve(&self, spec: &'static AsdKernelSpec) -> Result<Option<AsdModuleSource>> {
        Ok(candle_kernels::sm61_exact_grouped_ptx(spec.candidate_id)
            .map(AsdModuleSource::BuiltinPtx))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ExternalCudaModuleProvider {
    root: PathBuf,
}

impl ExternalCudaModuleProvider {
    fn from_env() -> Option<Self> {
        let root = std::env::var_os("CANDLE_ASD_MODULE_DIR")?;
        if root.is_empty() {
            return None;
        }
        Some(Self {
            root: PathBuf::from(root),
        })
    }

    fn candidate_path(&self, implementation_id: &str, extension: &str) -> PathBuf {
        self.root
            .join(format!("{implementation_id}.{extension}"))
    }
}

impl AsdModuleProvider for ExternalCudaModuleProvider {
    fn resolve(&self, spec: &'static AsdKernelSpec) -> Result<Option<AsdModuleSource>> {
        let cubin = self.candidate_path(spec.implementation_id, "cubin");
        if cubin.is_file() {
            return Ok(Some(AsdModuleSource::External {
                path: cubin,
                kind: ExternalModuleKind::Cubin,
            }));
        }

        let ptx = self.candidate_path(spec.implementation_id, "ptx");
        if ptx.is_file() {
            return Ok(Some(AsdModuleSource::External {
                path: ptx,
                kind: ExternalModuleKind::Ptx,
            }));
        }

        Ok(None)
    }
}

pub(crate) struct AsdModuleRegistry<'a> {
    device: &'a CudaDevice,
    builtin: BuiltinProvider,
    external: Option<ExternalCudaModuleProvider>,
}

impl<'a> AsdModuleRegistry<'a> {
    pub(super) fn new(device: &'a CudaDevice) -> Self {
        Self {
            device,
            builtin: BuiltinProvider,
            external: ExternalCudaModuleProvider::from_env(),
        }
    }

    pub(crate) fn resolve(&self, implementation_id: &str) -> Result<AsdCudaImplementation<'a>> {
        let spec = IMPLEMENTATIONS
            .iter()
            .find(|spec| spec.implementation_id == implementation_id)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "unknown Stage2E implementation {implementation_id}"
                ))
            })?;

        if let Some(external) = &self.external {
            if let Some(source) = external.resolve(spec)? {
                return Ok(AsdCudaImplementation {
                    device: self.device,
                    spec,
                    source,
                });
            }
        }

        let source = self.builtin.resolve(spec)?.ok_or_else(|| {
            Error::Msg(format!(
                "missing builtin PTX for ASD implementation {}",
                spec.implementation_id
            ))
        })?;

        Ok(AsdCudaImplementation {
            device: self.device,
            spec,
            source,
        })
    }
}

pub(crate) struct AsdCudaImplementation<'a> {
    device: &'a CudaDevice,
    spec: &'static AsdKernelSpec,
    source: AsdModuleSource,
}

impl AsdCudaImplementation<'_> {
    pub(crate) fn function(&self) -> Result<CudaFunc> {
        match &self.source {
            AsdModuleSource::BuiltinPtx(ptx) => self.device.get_or_load_custom_func(
                self.spec.entry,
                self.spec.candidate_id,
                ptx,
            ),
            AsdModuleSource::External { path, kind } => self.load_external(path, *kind),
        }
    }

    pub(crate) fn output_count(&self) -> usize {
        self.spec.output_count
    }

    pub(crate) fn launch_config(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.spec.grid_x, 1, 1),
            block_dim: (self.spec.block_x, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub(crate) fn candidate_id(&self) -> &'static str {
        self.spec.candidate_id
    }

    pub(crate) fn provider_name(&self) -> &'static str {
        match &self.source {
            AsdModuleSource::BuiltinPtx(_) => "builtin_ptx",
            AsdModuleSource::External {
                kind: ExternalModuleKind::Cubin,
                ..
            } => "external_cubin",
            AsdModuleSource::External {
                kind: ExternalModuleKind::Ptx,
                ..
            } => "external_ptx",
        }
    }

    pub(crate) fn proof_status(&self) -> &'static str {
        match &self.source {
            AsdModuleSource::BuiltinPtx(_) => "historical_evidence_bound",
            AsdModuleSource::External { .. } => "external_artifact_unverified",
        }
    }

    fn load_external(&self, path: &Path, kind: ExternalModuleKind) -> Result<CudaFunc> {
        let metadata = std::fs::metadata(path).map_err(|err| {
            Error::Msg(format!(
                "failed to stat external ASD CUDA module {}: {err}",
                path.display()
            ))
        })?;
        let modified_ns = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);

        let mut hasher = DefaultHasher::new();
        path.hash(&mut hasher);
        metadata.len().hash(&mut hasher);
        modified_ns.hash(&mut hasher);
        let cache_key = format!(
            "asd-external:{}:{:016x}",
            self.spec.implementation_id,
            hasher.finish()
        );

        if let Some(module) = self
            .device
            .custom_modules
            .read()
            .unwrap()
            .get(&cache_key)
            .cloned()
        {
            let func = module.load_function(self.spec.entry).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.device.stream.clone(),
            });
        }

        let image = match kind {
            ExternalModuleKind::Cubin => {
                let bytes = std::fs::read(path).map_err(|err| {
                    Error::Msg(format!(
                        "failed to read external ASD CUBIN {}: {err}",
                        path.display()
                    ))
                })?;
                Ptx::from_binary(bytes)
            }
            ExternalModuleKind::Ptx => {
                let source = std::fs::read_to_string(path).map_err(|err| {
                    Error::Msg(format!(
                        "failed to read external ASD PTX {}: {err}",
                        path.display()
                    ))
                })?;
                Ptx::from_src(source)
            }
        };

        let mut modules = self.device.custom_modules.write().unwrap();
        if let Some(module) = modules.get(&cache_key).cloned() {
            let func = module.load_function(self.spec.entry).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.device.stream.clone(),
            });
        }

        let module = self.device.context.load_module(image).w()?;
        modules.insert(cache_key, module.clone());
        drop(modules);

        let func = module.load_function(self.spec.entry).w()?;
        Ok(CudaFunc {
            func,
            stream: self.device.stream.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_provider_covers_the_frozen_sm61_catalogue() {
        let provider = BuiltinProvider;
        for spec in IMPLEMENTATIONS {
            assert!(
                provider.resolve(spec).unwrap().is_some(),
                "missing builtin provider entry for {}",
                spec.implementation_id
            );
        }
    }
}
