//! ASD V3 CUDA module resolution.
//!
//! External modules are opt-in through CANDLE_ASD_MODULE_DIR. For an implementation
//! id `foo`, the provider looks for `foo.cubin` first and then `foo.ptx`.
//! A sibling `foo.manifest` can bind the selected file to its SHA-256, ABI,
//! architecture, implementation id and entry symbol. Missing manifests remain
//! usable for tuner experiments but are reported as unverified.

use super::{CudaDevice, CudaFunc};
use crate::cuda_backend::WrapErr;
use crate::{Error, Result};
use cudarc::driver::LaunchConfig;
use cudarc::nvrtc::Ptx;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::UNIX_EPOCH;

const MODULE_MANIFEST_HEADER: &str = "ASD-CUDA-MODULE-V1";
const MODULE_ABI_VERSION: u32 = 1;
const MODULE_ARCH: &str = "sm61";

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
    if let Some(cached) = cache.lock().unwrap().get(&key).cloned() {
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

    Ok(ExternalModuleManifest {
        abi_version,
        implementation_id: required("implementation_id")?,
        architecture: required("architecture")?,
        artifact_kind: required("artifact_kind")?,
        entry: required("entry")?,
        artifact_sha256,
    })
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ManifestCacheKey {
    fingerprint: FileFingerprint,
    artifact_sha256: String,
}

static MANIFEST_CACHE: OnceLock<Mutex<HashMap<ManifestCacheKey, ExternalModuleManifest>>> =
    OnceLock::new();

fn validate_manifest(
    artifact: &CachedExternalArtifact,
    kind: ExternalModuleKind,
    spec: &'static AsdKernelSpec,
) -> Result<bool> {
    let manifest_path = artifact.path.with_extension("manifest");
    let metadata = match std::fs::metadata(&manifest_path) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => {
            return Err(Error::Msg(format!(
                "failed to stat ASD CUDA module manifest {}: {err}",
                manifest_path.display()
            )))
        }
    };

    let key = ManifestCacheKey {
        fingerprint: fingerprint(&manifest_path, &metadata),
        artifact_sha256: artifact.sha256.clone(),
    };
    let cache = MANIFEST_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let manifest = if let Some(manifest) = cache.lock().unwrap().get(&key).cloned() {
        manifest
    } else {
        let source = std::fs::read_to_string(&manifest_path).map_err(|err| {
            Error::Msg(format!(
                "failed to read ASD CUDA module manifest {}: {err}",
                manifest_path.display()
            ))
        })?;
        let after = std::fs::metadata(&manifest_path).map_err(|err| {
            Error::Msg(format!(
                "failed to restat ASD CUDA module manifest {}: {err}",
                manifest_path.display()
            ))
        })?;
        if fingerprint(&manifest_path, &after) != key.fingerprint {
            return Err(Error::Msg(format!(
                "ASD CUDA module manifest changed while being read: {}",
                manifest_path.display()
            )));
        }
        let manifest = parse_manifest(&source, &manifest_path)?;
        let mut cache = cache.lock().unwrap();
        cache.retain(|existing, _| existing.fingerprint.path != manifest_path);
        cache.insert(key, manifest.clone());
        manifest
    };

    if manifest.abi_version != MODULE_ABI_VERSION {
        return Err(Error::Msg(format!(
            "ASD CUDA module ABI mismatch for {}: manifest={} runtime={}",
            artifact.path.display(),
            manifest.abi_version,
            MODULE_ABI_VERSION
        )));
    }
    if manifest.implementation_id != spec.implementation_id {
        return Err(Error::Msg(format!(
            "ASD CUDA module implementation mismatch for {}: manifest={} expected={}",
            artifact.path.display(),
            manifest.implementation_id,
            spec.implementation_id
        )));
    }
    if manifest.architecture != MODULE_ARCH {
        return Err(Error::Msg(format!(
            "ASD CUDA module architecture mismatch for {}: manifest={} expected={}",
            artifact.path.display(),
            manifest.architecture,
            MODULE_ARCH
        )));
    }
    if manifest.artifact_kind != kind.as_str() {
        return Err(Error::Msg(format!(
            "ASD CUDA module kind mismatch for {}: manifest={} expected={}",
            artifact.path.display(),
            manifest.artifact_kind,
            kind.as_str()
        )));
    }
    if manifest.entry != spec.entry {
        return Err(Error::Msg(format!(
            "ASD CUDA module entry mismatch for {}: manifest={} expected={}",
            artifact.path.display(),
            manifest.entry,
            spec.entry
        )));
    }
    if manifest.artifact_sha256 != artifact.sha256 {
        return Err(Error::Msg(format!(
            "ASD CUDA module SHA-256 mismatch for {}: manifest={} actual={}",
            artifact.path.display(),
            manifest.artifact_sha256,
            artifact.sha256
        )));
    }

    Ok(true)
}

#[derive(Debug)]
enum AsdModuleSource {
    BuiltinPtx(&'static str),
    External {
        artifact: CachedExternalArtifact,
        kind: ExternalModuleKind,
        manifest_verified: bool,
    },
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
        self.root.join(format!("{implementation_id}.{extension}"))
    }

    fn source(
        &self,
        path: PathBuf,
        kind: ExternalModuleKind,
        spec: &'static AsdKernelSpec,
    ) -> Result<AsdModuleSource> {
        let artifact = load_cached_artifact(&path)?;
        let manifest_verified = validate_manifest(&artifact, kind, spec)?;
        Ok(AsdModuleSource::External {
            artifact,
            kind,
            manifest_verified,
        })
    }
}

impl AsdModuleProvider for ExternalCudaModuleProvider {
    fn resolve(&self, spec: &'static AsdKernelSpec) -> Result<Option<AsdModuleSource>> {
        let cubin = self.candidate_path(spec.implementation_id, "cubin");
        if cubin.is_file() {
            return self
                .source(cubin, ExternalModuleKind::Cubin, spec)
                .map(Some);
        }

        let ptx = self.candidate_path(spec.implementation_id, "ptx");
        if ptx.is_file() {
            return self.source(ptx, ExternalModuleKind::Ptx, spec).map(Some);
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
            AsdModuleSource::External { artifact, kind, .. } => {
                self.load_external(artifact, *kind)
            }
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
            AsdModuleSource::External {
                manifest_verified: true,
                ..
            } => "external_artifact_verified",
            AsdModuleSource::External {
                manifest_verified: false,
                ..
            } => "external_artifact_unverified",
        }
    }

    pub(crate) fn artifact_sha256(&self) -> Option<&str> {
        match &self.source {
            AsdModuleSource::BuiltinPtx(_) => None,
            AsdModuleSource::External { artifact, .. } => Some(&artifact.sha256),
        }
    }

    fn load_external(
        &self,
        artifact: &CachedExternalArtifact,
        kind: ExternalModuleKind,
    ) -> Result<CudaFunc> {
        let cache_key = format!(
            "asd-external:{}:{}",
            self.spec.implementation_id, artifact.sha256
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

    #[test]
    fn parses_strict_v1_manifest() {
        let source = format!(
            "{MODULE_MANIFEST_HEADER}\nabi_version=1\nimplementation_id={}\narchitecture=sm61\nartifact_kind=cubin\nentry={}\nartifact_sha256={}\n",
            IMPLEMENTATIONS[0].implementation_id,
            IMPLEMENTATIONS[0].entry,
            "a".repeat(64),
        );
        let parsed = parse_manifest(&source, Path::new("test.manifest")).unwrap();
        assert_eq!(parsed.abi_version, 1);
        assert_eq!(parsed.implementation_id, IMPLEMENTATIONS[0].implementation_id);
        assert_eq!(parsed.architecture, "sm61");
        assert_eq!(parsed.artifact_kind, "cubin");
        assert_eq!(parsed.entry, IMPLEMENTATIONS[0].entry);
        assert_eq!(parsed.artifact_sha256, "a".repeat(64));
    }
}
