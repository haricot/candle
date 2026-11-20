//! Opt-in, strict external module loading. Only trusted local packages.
//! Embedded modules remain available when the provider is not selected.
//! Reuse Candle's original ABI, context, stream, module cache and launch path.

use std::path::{Path, PathBuf};
#[cfg(test)]
use sha2::{Digest, Sha256};
use serde_json::Value;
use candle_kernels as kernels;
use cudarc::driver::CudaContext;
use crate::Result;
use super::WrapErr;

use candle_kernels::dkr_s31_contract as contract;
#[path = "dkr_s31_validation.rs"]
mod dkr_s31_validation;
use dkr_s31_validation::{verify_package, NativeSymbols};
#[cfg(test)]
use dkr_s31_validation::validate;
mod embedded {
    use super::*;
    include!(concat!(env!("OUT_DIR"), "/dkr_native_embedded.rs"));
}
use dkr_kernel_registry::{Registry, RegistryKey, VerifiedArtifact, NATIVE_ABI};

pub(super) struct ExternalModules {
    modules: Registry<VerifiedArtifact>,
    native_keys: std::collections::HashMap<&'static str, RegistryKey>,
    trace: bool,
    device_cc: u32,
    mmq_symbols: std::collections::HashMap<&'static str, std::collections::BTreeMap<String, String>>,
}

pub(super) fn family(index: usize) -> Option<&'static str> {
    [
        (kernels::AFFINE.index(), "affine"),
        (kernels::BINARY.index(), "binary"),
        (kernels::CAST.index(), "cast"),
        (kernels::CONV.index(), "conv"),
        (kernels::SORT.index(), "sort"),
        (kernels::FILL.index(), "fill"),
        (kernels::INDEXING.index(), "indexing"),
        (kernels::REDUCE.index(), "reduce"),
        (kernels::QUANTIZED.index(), "quantized"),
        (kernels::TERNARY.index(), "ternary"),
        (kernels::UNARY.index(), "unary"),
    ].into_iter().find_map(|(id, name)| (id == index).then_some(name))
}

fn read_text(path: &Path, limit: usize) -> Result<String> {
    use std::io::Read;
    let file=std::fs::File::open(path).map_err(|e| crate::Error::Msg(format!("Candle DKR {}: {e}",path.display())))?;
    let mut bytes=Vec::new();
    file.take((limit+1) as u64).read_to_end(&mut bytes).map_err(|e|crate::Error::Msg(e.to_string()))?;
    if bytes.len()>limit {crate::bail!("Candle DKR package size exceeded: {}",path.display());}
    String::from_utf8(bytes).map_err(|e|crate::Error::Msg(e.to_string()))
}


impl ExternalModules {
    pub(super) fn from_env(context: &CudaContext) -> Result<Option<Self>> {
        let source = std::env::var("MISTRALRS_DKR_ARTIFACT_SOURCE")
            .unwrap_or_else(|_| if embedded::ROOT.is_some() { "embedded".into() } else { "filesystem".into() });
        let Some(root) = std::env::var_os("CANDLE_DKR_S31_DIR").or_else(|| {
            (source == "embedded").then_some(embedded::ROOT).flatten().map(std::ffi::OsString::from)
        }) else {
            return Ok(None);
        };
        let (major, minor) = context.compute_capability().w()?;
        let device_cc = (major * 10 + minor) as u32;
        let root = PathBuf::from(root);
        if source == "embedded" {
            let compiled_root = embedded::ROOT.ok_or_else(|| crate::Error::Msg("native artifacts were not embedded".into()))?;
            if root != Path::new(compiled_root) {
                crate::bail!("native directory differs from the embedded build input");
            }
            let mut modules = Registry::default();
            let mut native_keys = std::collections::HashMap::new();
            let mut mmq_symbols = std::collections::HashMap::new();
            for (name, artifact, symbols) in embedded::modules() {
                artifact.admit_exact_target(device_cc).map_err(crate::Error::Msg)?;
                let key = RegistryKey::new(NATIVE_ABI, name, "native", artifact.key().target_cc());
                modules.insert(key.clone(), artifact).map_err(crate::Error::Msg)?;
                native_keys.insert(name, key);
                if let Some(symbols) = symbols { mmq_symbols.insert(name, symbols); }
            }
            let trace = std::env::var_os("CANDLE_DKR_TRACE").is_some();
            if trace {
                eprintln!("CANDLE_DKR_NATIVE_CATALOG source=embedded manifests={}", modules.len());
            }
            return Ok(Some(Self { modules, native_keys, mmq_symbols, trace, device_cc }));
        }
        if source != "filesystem" {
            crate::bail!("MISTRALRS_DKR_ARTIFACT_SOURCE must be embedded or filesystem");
        }
        if !root.is_absolute() || !root.is_dir() {
            crate::bail!("CANDLE_DKR_S31_DIR must be an existing absolute directory");
        }
        // Fail before inference if a selected module is absent or corrupt.
        let mut modules = Registry::default();
        let mut native_keys = std::collections::HashMap::new();
        let mut mmq_symbols = std::collections::HashMap::new();
        for &name in contract::FAMILIES {
            let (artifact, symbols) = Self::read(&root, name)?;
            artifact.admit_exact_target(device_cc).map_err(crate::Error::Msg)?;
            let key = RegistryKey::new(NATIVE_ABI, name, "native", artifact.key().target_cc());
            modules.insert(key.clone(), artifact).map_err(crate::Error::Msg)?;
            native_keys.insert(name, key);
            if let Some(symbols) = symbols { mmq_symbols.insert(name, symbols); }
        }
        if std::env::var_os("CANDLE_DKR_TRACE").is_some() {
            eprintln!("CANDLE_DKR_NATIVE_CATALOG source=filesystem manifests={}", modules.len());
        }
        Ok(Some(Self { modules, native_keys, mmq_symbols, trace: std::env::var_os("CANDLE_DKR_TRACE").is_some(), device_cc }))
    }

    fn read(root: &Path, name: &str) -> Result<(VerifiedArtifact, NativeSymbols)> {
        let manifest = read_text(&root.join(name).join("manifest.json"), 256*1024)?;
        let manifest: Value = serde_json::from_str(&manifest)
            .map_err(|e| crate::Error::Msg(format!("Candle DKR S3.1 manifest: {e}")))?;
        let ptx = read_text(&root.join(name).join(format!("{name}.ptx")), dkr_kernel_registry::MAX_PTX_BYTES)?;
        verify_package(&manifest, name, ptx).map_err(crate::Error::Msg)
    }

    fn artifact(&self, name: &str) -> Option<&VerifiedArtifact> {
        self.modules.get(self.native_keys.get(name)?)
    }

    pub(super) fn ptx(&self, index: usize) -> Option<&VerifiedArtifact> {
        family(index).and_then(|name| self.artifact(name))
    }

    pub(super) fn mmq_entry(&self, family: &str, key: &str) -> Option<(&VerifiedArtifact, &str)> {
        Some((self.artifact(family)?, self.mmq_symbols.get(family)?.get(key)?.as_str()))
    }

    pub(super) fn record_mmq_loaded(&self, family: &str) {
        if self.trace {
            eprintln!("CANDLE_DKR_MODULE_LOADED family={family} target=sm{} source=external", self.device_cc);
        }
    }

    pub(super) fn mmvq_ptx(&self) -> Option<&VerifiedArtifact> {
        self.artifact("mmvq_gguf")
    }

    pub(super) fn record_mmvq_loaded(&self) {
        if self.trace {
            eprintln!("CANDLE_DKR_MODULE_LOADED family=mmvq_gguf target=sm{} source=external", self.device_cc);
        }
    }

    pub(super) fn record_loaded(&self, index: usize) {
        if self.trace {
            if let Some(name) = family(index) {
                eprintln!("CANDLE_DKR_MODULE_LOADED family={name} target=sm{} source=external", self.device_cc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dkr_s31_manifest_contract() {
        let text = ".version 8.0\n.target sm_61\n.address_size 64\n.visible .entry sample() { ret; }\n";
        let mut m = serde_json::json!({
            "schema": contract::SCHEMA, "base_revision": contract::BASE,
            "target_cc": 61, "family": "unary", "flags": contract::FLAGS,
            "sha256": format!("{:x}", Sha256::digest(text.as_bytes())),
        });
        assert!(validate(&m, "unary", text).is_ok());
        let (artifact, symbols) = verify_package(&m, "unary", text.to_owned()).unwrap();
        assert_eq!(artifact.key().target_cc(), 61);
        assert!(artifact.admit_exact_target(61).is_ok());
        assert!(artifact.admit_exact_target(80).is_err());
        assert!(symbols.is_none());
        let mut wrong_target = m.clone();
        wrong_target["target_cc"] = serde_json::json!(80);
        assert!(verify_package(&wrong_target, "unary", text.to_owned()).is_err());
        assert!(validate(&m, "cast", text).is_err());
        assert!(validate(&m, "unary", &text.replace("sm_61", "sm_80")).is_err());
        m["sha256"] = serde_json::json!("bad");
        assert!(validate(&m, "unary", text).is_err());
        assert_eq!(family(kernels::UNARY.index()), Some("unary"));
        assert_eq!(family(kernels::REDUCE.index()), Some("reduce"));
        assert_eq!(family(kernels::INDEXING.index()), Some("indexing"));
        assert_eq!(family(kernels::QUANTIZED.index()), Some("quantized"));
        assert_eq!(family(kernels::CONV.index()), Some("conv"));
        assert_eq!(family(kernels::SORT.index()), Some("sort"));
        assert_eq!(contract::mmvq_entries().len(), 243);
    }
}
