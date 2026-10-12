//! ASD V3 CUDA module resolution.
//!
//! External modules are resolved first from CANDLE_ASD_MODULE_DIR when set, then
//! from the architecture-scoped user ASD store under
//! `${CANDLE_ASD_HOME:-${XDG_DATA_HOME:-~/.local/share}/asd}/artifacts/smXX`.
//! Every executable implementation is manifest-driven: the manifest owns the
//! entry symbol, launch geometry, ABI, artifact hash and decision binding.
//! A sibling `foo.cu` is provenance/source only and is never executed by the runtime.

use super::{CudaDevice, CudaFunc};
use crate::cuda_backend::WrapErr;
use crate::{Error, Result};
use cudarc::driver::LaunchConfig;
use cudarc::nvrtc::Ptx;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::UNIX_EPOCH;

const MODULE_MANIFEST_HEADER: &str = "ASD-CUDA-MODULE-V1";
const MODULE_ABI_VERSION: u32 = 1;

#[derive(Clone, Debug)]
struct AsdKernelSpec {
    implementation_id: String,
    candidate_id: String,
    entry: String,
    output_count: usize,
    grid_x: u32,
    block_x: u32,
    shared_mem_bytes: u32,
    decision_identity_sha256: String,
    artifact_kind: String,
    artifact_sha256: String,
}

enum AsdRuntimeSlot {
    Unresolved,
    Unavailable,
    Resolved(Arc<ResolvedAsdImplementation>),
}

struct DynamicRuntimeEntry {
    spec: Arc<AsdKernelSpec>,
    slot: RwLock<AsdRuntimeSlot>,
}

pub(super) struct AsdRuntimeCache {
    dynamic: RwLock<HashMap<String, Arc<DynamicRuntimeEntry>>>,
    generation: AtomicU64,
}

#[derive(Clone)]
struct DynamicLastHit {
    cache_identity: usize,
    generation: u64,
    implementation_id: String,
    decision_identity_sha256: String,
    entry: Arc<DynamicRuntimeEntry>,
}

thread_local! {
    static DYNAMIC_LAST_HIT: RefCell<Option<DynamicLastHit>> = const { RefCell::new(None) };
}

impl AsdRuntimeCache {
    pub(super) fn new() -> Self {
        Self {
            dynamic: RwLock::new(HashMap::new()),
            generation: AtomicU64::new(1),
        }
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    pub(super) fn invalidate_plans(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    pub(super) fn clear_all(&self) {
        self.dynamic.write().unwrap().clear();
        self.invalidate_plans();
    }

    pub(super) fn clear(&self, implementation_id: &str) -> Result<()> {
        self.dynamic.write().unwrap().remove(implementation_id);
        self.invalidate_plans();
        Ok(())
    }

    pub(super) fn resolved_source(&self, implementation_id: &str) -> Result<Option<&'static str>> {
        let entry = self.dynamic.read().unwrap().get(implementation_id).cloned();
        let Some(entry) = entry else {
            return Ok(None);
        };
        let slot = entry.slot.read().unwrap();
        Ok(match &*slot {
            AsdRuntimeSlot::Resolved(resolved) => Some(resolved.provider_name),
            AsdRuntimeSlot::Unresolved | AsdRuntimeSlot::Unavailable => None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExternalModuleKind {
    Cubin,
    Ptx,
}

impl ExternalModuleKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cubin => "cubin",
            Self::Ptx => "ptx",
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct FileFingerprint {
    path: PathBuf,
    len: u64,
    modified_ns: u128,
}

fn modified_ns(metadata: &std::fs::Metadata) -> u128 {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

fn fingerprint(path: &Path, metadata: &std::fs::Metadata) -> FileFingerprint {
    FileFingerprint {
        path: path.to_path_buf(),
        len: metadata.len(),
        modified_ns: modified_ns(metadata),
    }
}

#[derive(Clone, Debug)]
struct CachedExternalArtifact {
    path: PathBuf,
    bytes: Arc<Vec<u8>>,
    sha256: String,
}

static ARTIFACT_CACHE: OnceLock<Mutex<HashMap<FileFingerprint, CachedExternalArtifact>>> =
    OnceLock::new();

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn load_cached_artifact(path: &Path) -> Result<CachedExternalArtifact> {
    let before = std::fs::metadata(path).map_err(|err| {
        Error::Msg(format!(
            "failed to stat external ASD CUDA module {}: {err}",
            path.display()
        ))
    })?;
    let key = fingerprint(path, &before);
    let cache = ARTIFACT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let cached_artifact = {
        let guard = cache.lock().unwrap();
        guard.get(&key).cloned()
    };
    if let Some(cached) = cached_artifact {
        return Ok(cached);
    }

    let bytes = std::fs::read(path).map_err(|err| {
        Error::Msg(format!(
            "failed to read external ASD CUDA module {}: {err}",
            path.display()
        ))
    })?;
    let after = std::fs::metadata(path).map_err(|err| {
        Error::Msg(format!(
            "failed to restat external ASD CUDA module {}: {err}",
            path.display()
        ))
    })?;
    if fingerprint(path, &after) != key {
        return Err(Error::Msg(format!(
            "external ASD CUDA module changed while being read: {}",
            path.display()
        )));
    }

    let cached = CachedExternalArtifact {
        path: path.to_path_buf(),
        sha256: sha256_hex(&bytes),
        bytes: Arc::new(bytes),
    };
    let mut cache = cache.lock().unwrap();
    cache.retain(|existing, _| existing.path != path);
    cache.insert(key, cached.clone());
    Ok(cached)
}

#[derive(Clone, Debug)]
struct ExternalModuleManifest {
    abi_version: u32,
    implementation_id: String,
    architecture: String,
    artifact_kind: String,
    entry: String,
    artifact_sha256: String,
    candidate_id: Option<String>,
    kernel_abi: Option<String>,
    output_count: Option<usize>,
    grid_x: Option<u32>,
    block_x: Option<u32>,
    shared_mem_bytes: Option<u32>,
    decision_identity_sha256: Option<String>,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_manifest(source: &str, path: &Path) -> Result<ExternalModuleManifest> {
    let mut lines = source.lines();
    if lines.next() != Some(MODULE_MANIFEST_HEADER) {
        return Err(Error::Msg(format!(
            "invalid ASD CUDA module manifest header in {}",
            path.display()
        )));
    }

    let mut fields = HashMap::<String, String>::new();
    for (index, raw) in lines.enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(Error::Msg(format!(
                "invalid ASD CUDA module manifest line {} in {}",
                index + 2,
                path.display()
            )));
        };
        let key = key.trim();
        let value = value.trim();
        if !matches!(
            key,
            "abi_version"
                | "implementation_id"
                | "architecture"
                | "artifact_kind"
                | "entry"
                | "artifact_sha256"
                | "candidate_id"
                | "kernel_abi"
                | "output_count"
                | "grid_x"
                | "block_x"
                | "shared_mem_bytes"
                | "decision_identity_sha256"
        ) {
            return Err(Error::Msg(format!(
                "unknown ASD CUDA module manifest key {key:?} in {}",
                path.display()
            )));
        }
        if value.is_empty() {
            return Err(Error::Msg(format!(
                "empty ASD CUDA module manifest value for {key} in {}",
                path.display()
            )));
        }
        if fields.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(Error::Msg(format!(
                "duplicate ASD CUDA module manifest key {key:?} in {}",
                path.display()
            )));
        }
    }

    let required = |key: &str| -> Result<String> {
        fields.get(key).cloned().ok_or_else(|| {
            Error::Msg(format!(
                "missing ASD CUDA module manifest key {key:?} in {}",
                path.display()
            ))
        })
    };

    let abi_version = required("abi_version")?.parse::<u32>().map_err(|err| {
        Error::Msg(format!(
            "invalid abi_version in ASD CUDA module manifest {}: {err}",
            path.display()
        ))
    })?;
    let artifact_sha256 = required("artifact_sha256")?;
    if !is_sha256(&artifact_sha256) {
        return Err(Error::Msg(format!(
            "artifact_sha256 must be 64 lowercase hex characters in {}",
            path.display()
        )));
    }

    let parse_optional_usize = |key: &str| -> Result<Option<usize>> {
        fields
            .get(key)
            .map(|value| {
                value.parse::<usize>().map_err(|err| {
                    Error::Msg(format!(
                        "invalid {key} in ASD CUDA module manifest {}: {err}",
                        path.display()
                    ))
                })
            })
            .transpose()
    };
    let parse_optional_u32 = |key: &str| -> Result<Option<u32>> {
        fields
            .get(key)
            .map(|value| {
                value.parse::<u32>().map_err(|err| {
                    Error::Msg(format!(
                        "invalid {key} in ASD CUDA module manifest {}: {err}",
                        path.display()
                    ))
                })
            })
            .transpose()
    };

    let decision_identity_sha256 = fields.get("decision_identity_sha256").cloned();
    if decision_identity_sha256
        .as_deref()
        .is_some_and(|value| !is_sha256(value))
    {
        return Err(Error::Msg(format!(
            "decision_identity_sha256 must be 64 lowercase hex characters in {}",
            path.display()
        )));
    }

    Ok(ExternalModuleManifest {
        abi_version,
        implementation_id: required("implementation_id")?,
        architecture: required("architecture")?,
        artifact_kind: required("artifact_kind")?,
        entry: required("entry")?,
        artifact_sha256,
        candidate_id: fields.get("candidate_id").cloned(),
        kernel_abi: fields.get("kernel_abi").cloned(),
        output_count: parse_optional_usize("output_count")?,
        grid_x: parse_optional_u32("grid_x")?,
        block_x: parse_optional_u32("block_x")?,
        shared_mem_bytes: parse_optional_u32("shared_mem_bytes")?,
        decision_identity_sha256,
    })
}

fn read_manifest(path: &Path) -> Result<ExternalModuleManifest> {
    let source = std::fs::read_to_string(path).map_err(|err| {
        Error::Msg(format!(
            "failed to read ASD CUDA module manifest {}: {err}",
            path.display()
        ))
    })?;
    parse_manifest(&source, path)
}

fn dynamic_spec_from_manifest(
    manifest: &ExternalModuleManifest,
    implementation_id: &str,
    decision_identity_sha256: &str,
    expected_architecture: &str,
    path: &Path,
) -> Result<AsdKernelSpec> {
    let required_string = |name: &str, value: &Option<String>| -> Result<String> {
        value.clone().ok_or_else(|| {
            Error::Msg(format!(
                "dynamic ASD implementation {implementation_id} requires {name} in {}",
                path.display()
            ))
        })
    };
    let required_usize = |name: &str, value: Option<usize>| -> Result<usize> {
        value.filter(|value| *value > 0).ok_or_else(|| {
            Error::Msg(format!(
                "dynamic ASD implementation {implementation_id} requires positive {name} in {}",
                path.display()
            ))
        })
    };
    let required_u32 = |name: &str, value: Option<u32>| -> Result<u32> {
        value.filter(|value| *value > 0).ok_or_else(|| {
            Error::Msg(format!(
                "dynamic ASD implementation {implementation_id} requires positive {name} in {}",
                path.display()
            ))
        })
    };

    if manifest.abi_version != MODULE_ABI_VERSION
        || manifest.implementation_id != implementation_id
        || manifest.architecture != expected_architecture
        || !matches!(manifest.artifact_kind.as_str(), "cubin" | "ptx")
    {
        return Err(Error::Msg(format!(
            "dynamic ASD implementation manifest identity mismatch in {}",
            path.display()
        )));
    }

    let kernel_abi = required_string("kernel_abi", &manifest.kernel_abi)?;
    if kernel_abi != "asd.xwo.f32.v1" {
        return Err(Error::Msg(format!(
            "dynamic ASD implementation {implementation_id} uses unsupported kernel_abi={kernel_abi:?}; expected asd.xwo.f32.v1"
        )));
    }

    let manifest_decision_identity = required_string(
        "decision_identity_sha256",
        &manifest.decision_identity_sha256,
    )?;
    if manifest_decision_identity != decision_identity_sha256 {
        return Err(Error::Msg(format!(
            "dynamic ASD implementation {implementation_id} decision identity mismatch: manifest={} selected={decision_identity_sha256}",
            manifest_decision_identity
        )));
    }

    Ok(AsdKernelSpec {
        implementation_id: implementation_id.to_owned(),
        candidate_id: required_string("candidate_id", &manifest.candidate_id)?,
        entry: manifest.entry.clone(),
        output_count: required_usize("output_count", manifest.output_count)?,
        grid_x: required_u32("grid_x", manifest.grid_x)?,
        block_x: required_u32("block_x", manifest.block_x)?,
        shared_mem_bytes: manifest.shared_mem_bytes.unwrap_or(0),
        decision_identity_sha256: manifest_decision_identity,
        artifact_kind: manifest.artifact_kind.clone(),
        artifact_sha256: manifest.artifact_sha256.clone(),
    })
}

#[derive(Debug)]
enum AsdModuleSource {
    External {
        artifact: CachedExternalArtifact,
        kind: ExternalModuleKind,
        manifest_verified: bool,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct ExternalCudaModuleProvider {
    root: PathBuf,
}

impl ExternalCudaModuleProvider {
    fn from_env(architecture: &str) -> Option<Self> {
        let root = std::env::var_os("CANDLE_ASD_MODULE_DIR")
            .filter(|root| !root.is_empty())
            .map(PathBuf::from)
            .or_else(|| candle_kernels::asd_paths::artifacts_dir(architecture))?;
        Some(Self { root })
    }

    fn candidate_path(&self, implementation_id: &str, extension: &str) -> PathBuf {
        self.root.join(format!("{implementation_id}.{extension}"))
    }

    fn dynamic_spec(
        &self,
        implementation_id: &str,
        decision_identity_sha256: &str,
        expected_architecture: &str,
    ) -> Result<Option<Arc<AsdKernelSpec>>> {
        let manifest_path = self.candidate_path(implementation_id, "manifest");
        if !manifest_path.is_file() {
            return Ok(None);
        }
        let manifest = read_manifest(&manifest_path)?;
        let spec = dynamic_spec_from_manifest(
            &manifest,
            implementation_id,
            decision_identity_sha256,
            expected_architecture,
            &manifest_path,
        )?;
        Ok(Some(Arc::new(spec)))
    }

    fn resolve_dynamic(&self, spec: &AsdKernelSpec) -> Result<Option<AsdModuleSource>> {
        let kind = match spec.artifact_kind.as_str() {
            "cubin" => ExternalModuleKind::Cubin,
            "ptx" => ExternalModuleKind::Ptx,
            other => {
                return Err(Error::Msg(format!(
                    "unsupported dynamic ASD artifact kind {other:?}"
                )))
            }
        };
        let path = self.candidate_path(spec.implementation_id.as_str(), kind.as_str());
        if !path.is_file() {
            return Ok(None);
        }
        let artifact = load_cached_artifact(&path)?;
        if artifact.sha256 != spec.artifact_sha256 {
            return Err(Error::Msg(format!(
                "dynamic ASD artifact SHA-256 mismatch for {}: manifest={} actual={}",
                path.display(),
                spec.artifact_sha256,
                artifact.sha256
            )));
        }
        Ok(Some(AsdModuleSource::External {
            artifact,
            kind,
            manifest_verified: true,
        }))
    }
}

pub(crate) struct AsdModuleRegistry<'a> {
    device: &'a CudaDevice,
}

struct ResolvedAsdImplementation {
    function: CudaFunc,
    provider_name: &'static str,
    proof_status: &'static str,
    artifact_sha256: Option<String>,
}

impl<'a> AsdModuleRegistry<'a> {
    pub(super) fn new(device: &'a CudaDevice) -> Self {
        Self { device }
    }

    fn runtime_architecture(&self) -> Result<String> {
        let (major, minor) = self
            .device
            .cuda_stream()
            .context()
            .compute_capability()
            .map_err(|err| {
                Error::Msg(format!(
                    "unable to read CUDA compute capability for ASD module resolution: {err:?}"
                ))
            })?;
        if major < 0 || minor < 0 {
            return Err(Error::Msg(format!(
                "invalid CUDA compute capability for ASD module resolution: {major}.{minor}"
            )));
        }
        Ok(format!("sm{}", major * 10 + minor))
    }

    fn materialize_dynamic(
        &self,
        spec: &AsdKernelSpec,
        source: AsdModuleSource,
    ) -> Result<ResolvedAsdImplementation> {
        let AsdModuleSource::External {
            artifact,
            kind,
            manifest_verified,
        } = source;
        let provider_name = match kind {
            ExternalModuleKind::Cubin => "external_cubin",
            ExternalModuleKind::Ptx => "external_ptx",
        };
        let proof_status = if manifest_verified {
            "external_artifact_verified"
        } else {
            "external_artifact_unverified"
        };
        let artifact_sha256 = Some(artifact.sha256.clone());
        let function = load_external_fields(self.device, &spec.entry, &artifact, kind)?;
        Ok(ResolvedAsdImplementation {
            function,
            provider_name,
            proof_status,
            artifact_sha256,
        })
    }

    fn resolve_dynamic_optional(
        &self,
        implementation_id: &str,
        decision_identity_sha256: &str,
    ) -> Result<Option<AsdCudaImplementation>> {
        let architecture = self.runtime_architecture()?;
        let provider = ExternalCudaModuleProvider::from_env(&architecture).ok_or_else(|| {
            Error::Msg(format!(
                "unable to resolve ASD artifact root for dynamic implementation {implementation_id}"
            ))
        })?;

        let cache_identity = Arc::as_ptr(&self.device.asd_runtime) as usize;
        let generation = self.device.asd_runtime.generation.load(Ordering::Acquire);
        let fast_entry = DYNAMIC_LAST_HIT.with(|cache| {
            cache
                .borrow()
                .as_ref()
                .filter(|hit| {
                    hit.cache_identity == cache_identity
                        && hit.generation == generation
                        && hit.implementation_id == implementation_id
                        && hit.decision_identity_sha256 == decision_identity_sha256
                })
                .map(|hit| hit.entry.clone())
        });

        let entry = if let Some(entry) = fast_entry {
            entry
        } else {
            let existing = self
                .device
                .asd_runtime
                .dynamic
                .read()
                .unwrap()
                .get(implementation_id)
                .cloned();

            let entry = if let Some(entry) = existing {
                if entry.spec.decision_identity_sha256 != decision_identity_sha256 {
                    return Err(Error::Msg(format!(
                        "dynamic ASD implementation {implementation_id} is cached for decision identity {} but selected decision is {decision_identity_sha256}",
                        entry.spec.decision_identity_sha256
                    )));
                }
                entry
            } else {
                let Some(spec) = provider.dynamic_spec(
                    implementation_id,
                    decision_identity_sha256,
                    &architecture,
                )?
                else {
                    return Ok(None);
                };
                let candidate = Arc::new(DynamicRuntimeEntry {
                    spec,
                    slot: RwLock::new(AsdRuntimeSlot::Unresolved),
                });
                let mut dynamic = self.device.asd_runtime.dynamic.write().unwrap();
                dynamic
                    .entry(implementation_id.to_owned())
                    .or_insert_with(|| candidate.clone())
                    .clone()
            };

            DYNAMIC_LAST_HIT.with(|cache| {
                *cache.borrow_mut() = Some(DynamicLastHit {
                    cache_identity,
                    generation,
                    implementation_id: implementation_id.to_owned(),
                    decision_identity_sha256: decision_identity_sha256.to_owned(),
                    entry: entry.clone(),
                });
            });
            entry
        };

        if entry.spec.decision_identity_sha256 != decision_identity_sha256 {
            return Err(Error::Msg(format!(
                "dynamic ASD implementation {implementation_id} is bound to decision identity {} but selected decision is {decision_identity_sha256}",
                entry.spec.decision_identity_sha256
            )));
        }

        {
            let slot = entry.slot.read().unwrap();
            match &*slot {
                AsdRuntimeSlot::Resolved(resolved) => {
                    return Ok(Some(AsdCudaImplementation {
                        spec: entry.spec.clone(),
                        resolved: resolved.clone(),
                    }))
                }
                AsdRuntimeSlot::Unavailable => return Ok(None),
                AsdRuntimeSlot::Unresolved => {}
            }
        }

        let mut slot = entry.slot.write().unwrap();
        match &*slot {
            AsdRuntimeSlot::Resolved(resolved) => {
                return Ok(Some(AsdCudaImplementation {
                    spec: entry.spec.clone(),
                    resolved: resolved.clone(),
                }))
            }
            AsdRuntimeSlot::Unavailable => return Ok(None),
            AsdRuntimeSlot::Unresolved => {}
        }

        let Some(source) = provider.resolve_dynamic(&entry.spec)? else {
            *slot = AsdRuntimeSlot::Unavailable;
            return Ok(None);
        };
        let resolved = Arc::new(self.materialize_dynamic(&entry.spec, source)?);
        *slot = AsdRuntimeSlot::Resolved(resolved.clone());

        Ok(Some(AsdCudaImplementation {
            spec: entry.spec.clone(),
            resolved,
        }))
    }

    pub(crate) fn resolve_optional_bound(
        &self,
        implementation_id: &str,
        decision_identity_sha256: &str,
    ) -> Result<Option<AsdCudaImplementation>> {
        self.resolve_dynamic_optional(implementation_id, decision_identity_sha256)
    }
}

#[derive(Clone)]
pub(crate) struct AsdCudaImplementation {
    spec: Arc<AsdKernelSpec>,
    resolved: Arc<ResolvedAsdImplementation>,
}

impl AsdCudaImplementation {
    pub(crate) fn function(&self) -> &CudaFunc {
        &self.resolved.function
    }

    pub(crate) fn output_count(&self) -> usize {
        self.spec.output_count
    }

    pub(crate) fn launch_config(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.spec.grid_x, 1, 1),
            block_dim: (self.spec.block_x, 1, 1),
            shared_mem_bytes: self.spec.shared_mem_bytes,
        }
    }

    pub(crate) fn candidate_id(&self) -> &str {
        &self.spec.candidate_id
    }

    pub(crate) fn provider_name(&self) -> &'static str {
        self.resolved.provider_name
    }

    pub(crate) fn proof_status(&self) -> &'static str {
        self.resolved.proof_status
    }

    pub(crate) fn artifact_sha256(&self) -> Option<&str> {
        self.resolved.artifact_sha256.as_deref()
    }
}

fn load_external_fields(
    device: &CudaDevice,
    entry: &str,
    artifact: &CachedExternalArtifact,
    kind: ExternalModuleKind,
) -> Result<CudaFunc> {
    // CUDA modules are immutable once loaded. Key the module cache by verified
    // artifact content rather than implementation id so multiple ASD decisions
    // that intentionally reference the same CUBIN/PTX share one loaded module.
    // The entry symbol is still resolved independently from that module.
    let cache_key = format!("asd-external:{}:{}", kind.as_str(), artifact.sha256);

    if let Some(module) = device
        .custom_modules
        .read()
        .unwrap()
        .get(&cache_key)
        .cloned()
    {
        let func = module.load_function(entry).w()?;
        return Ok(CudaFunc {
            func,
            stream: device.stream.clone(),
        });
    }

    let image = match kind {
        ExternalModuleKind::Cubin => Ptx::from_binary(artifact.bytes.as_ref().clone()),
        ExternalModuleKind::Ptx => {
            let source = String::from_utf8(artifact.bytes.as_ref().clone()).map_err(|err| {
                Error::Msg(format!(
                    "external ASD PTX is not UTF-8 ({}): {err}",
                    artifact.path.display()
                ))
            })?;
            Ptx::from_src(source)
        }
    };

    let mut modules = device.custom_modules.write().unwrap();
    if let Some(module) = modules.get(&cache_key).cloned() {
        let func = module.load_function(entry).w()?;
        return Ok(CudaFunc {
            func,
            stream: device.stream.clone(),
        });
    }

    let module = device.context.load_module(image).w()?;
    modules.insert(cache_key, module.clone());
    drop(modules);

    let func = module.load_function(entry).w()?;
    Ok(CudaFunc {
        func,
        stream: device.stream.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_decision_bound_dynamic_manifest() {
        let implementation_id = "candle.asd.dynamic-test";
        let decision_identity = "b".repeat(64);
        let architecture = "sm77".to_owned();
        let source = format!(
            "{MODULE_MANIFEST_HEADER}\nabi_version=1\nimplementation_id={implementation_id}\narchitecture={architecture}\nartifact_kind=cubin\nentry=dynamic_test_entry\nartifact_sha256={}\ncandidate_id=dynamic-test\nkernel_abi=asd.xwo.f32.v1\noutput_count=8192\ngrid_x=32\nblock_x=256\nshared_mem_bytes=0\ndecision_identity_sha256={decision_identity}\n",
            "a".repeat(64),
        );
        let path = Path::new("dynamic-test.manifest");
        let manifest = parse_manifest(&source, path).unwrap();
        let spec = dynamic_spec_from_manifest(
            &manifest,
            implementation_id,
            &decision_identity,
            &architecture,
            path,
        )
        .unwrap();
        assert_eq!(spec.implementation_id, implementation_id);
        assert_eq!(spec.entry, "dynamic_test_entry");
        assert_eq!(spec.output_count, 8192);
        assert_eq!(spec.grid_x, 32);
        assert_eq!(spec.block_x, 256);
        assert_eq!(spec.shared_mem_bytes, 0);
        assert_eq!(spec.decision_identity_sha256, decision_identity);
    }

    #[test]
    fn rejects_manifest_for_different_runtime_architecture() {
        let implementation_id = "thirdparty.generic-exact.test";
        let decision_identity = "b".repeat(64);
        let manifest_architecture = "sm77";
        let runtime_architecture = "sm88";
        let source = format!(
            "{MODULE_MANIFEST_HEADER}\nabi_version=1\nimplementation_id={implementation_id}\narchitecture={manifest_architecture}\nartifact_kind=cubin\nentry=generic_entry\nartifact_sha256={}\ncandidate_id=generic-test\nkernel_abi=asd.xwo.f32.v1\noutput_count=8192\ngrid_x=32\nblock_x=256\nshared_mem_bytes=0\ndecision_identity_sha256={decision_identity}\n",
            "a".repeat(64),
        );
        let path = Path::new("wrong-architecture.manifest");
        let manifest = parse_manifest(&source, path).unwrap();
        assert!(dynamic_spec_from_manifest(
            &manifest,
            implementation_id,
            &decision_identity,
            runtime_architecture,
            path,
        )
        .is_err());
    }

    #[test]
    fn rejects_manifest_without_launch_metadata() {
        let implementation_id = "candle.asd.incomplete";
        let decision_identity = "b".repeat(64);
        let architecture = "sm77".to_owned();
        let source = format!(
            "{MODULE_MANIFEST_HEADER}\nabi_version=1\nimplementation_id={implementation_id}\narchitecture={architecture}\nartifact_kind=cubin\nentry=incomplete_entry\nartifact_sha256={}\ndecision_identity_sha256={decision_identity}\n",
            "a".repeat(64),
        );
        let path = Path::new("incomplete.manifest");
        let manifest = parse_manifest(&source, path).unwrap();
        assert!(dynamic_spec_from_manifest(
            &manifest,
            implementation_id,
            &decision_identity,
            &architecture,
            path,
        )
        .is_err());
    }
}
