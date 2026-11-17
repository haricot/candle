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
        if (major, minor) != (6, 1) {
            crate::bail!("Candle DKR S3.1 requires SM61, got {major}.{minor}");
        }
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
                let key = RegistryKey::new(NATIVE_ABI, name, "native", contract::CC);
                modules.insert(key.clone(), artifact).map_err(crate::Error::Msg)?;
                native_keys.insert(name, key);
                if let Some(symbols) = symbols { mmq_symbols.insert(name, symbols); }
            }
            let trace = std::env::var_os("CANDLE_DKR_TRACE").is_some();
            if trace {
                eprintln!("CANDLE_DKR_NATIVE_CATALOG source=embedded manifests={}", modules.len());
            }
            return Ok(Some(Self { modules, native_keys, mmq_symbols, trace }));
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
        for &name in contract::FAMILIES {
            let key = RegistryKey::new(NATIVE_ABI, name, "native", contract::CC);
            modules.insert(key.clone(), Self::read(&root, name)?).map_err(crate::Error::Msg)?;
            native_keys.insert(name,key);
        }
        let mut mmq_symbols = std::collections::HashMap::new();
        for (&name, key) in &native_keys {
            let ptx = modules.get(key).expect("registered native module").ptx();
            if name.starts_with("mmq_") || name.starts_with("moe_") {
                let symbols = if name.starts_with("moe_") { contract::moe_entries(ptx, name) }
                    else { contract::mmq_entries(ptx, name) }
                    .ok_or_else(|| crate::Error::Msg(format!("invalid MMQ entries: {name}")))?;
                mmq_symbols.insert(name, symbols);
            }
        }
        if std::env::var_os("CANDLE_DKR_TRACE").is_some() {
            eprintln!("CANDLE_DKR_NATIVE_CATALOG source=filesystem manifests={}", modules.len());
        }
        Ok(Some(Self { modules, native_keys, mmq_symbols, trace: std::env::var_os("CANDLE_DKR_TRACE").is_some() }))
    }

    fn read(root: &Path, name: &str) -> Result<VerifiedArtifact> {
        let manifest = read_text(&root.join(name).join("manifest.json"), 256*1024)?;
        let manifest: Value = serde_json::from_str(&manifest)
            .map_err(|e| crate::Error::Msg(format!("Candle DKR S3.1 manifest: {e}")))?;
        let ptx = read_text(&root.join(name).join(format!("{name}.ptx")), dkr_kernel_registry::MAX_PTX_BYTES)?;
        validate(&manifest, name, &ptx).map_err(crate::Error::Msg)?;
        VerifiedArtifact::new(ptx, contract::CC, manifest["sha256"].as_str().unwrap_or("")).map_err(crate::Error::Msg)
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
            eprintln!("CANDLE_DKR_MODULE_LOADED family={family} target=sm61 source=external");
        }
    }

    pub(super) fn mmvq_ptx(&self) -> Option<&VerifiedArtifact> {
        self.artifact("mmvq_gguf")
    }

    pub(super) fn record_mmvq_loaded(&self) {
        if self.trace {
            eprintln!("CANDLE_DKR_MODULE_LOADED family=mmvq_gguf target=sm61 source=external");
        }
    }

    pub(super) fn record_loaded(&self, index: usize) {
        if self.trace {
            if let Some(name) = family(index) {
                eprintln!("CANDLE_DKR_MODULE_LOADED family={name} target=sm61 source=external");
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
