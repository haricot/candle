//! Opt-in, strict external module loading. Only trusted local packages.
//! Embedded modules remain available when the provider is not selected.
//! Reuse Candle's original ABI, context, stream, module cache and launch path.

use std::path::{Path, PathBuf};
use sha2::{Digest, Sha256};
use serde_json::Value;
use candle_kernels as kernels;
use cudarc::driver::CudaContext;
use crate::Result;
use super::WrapErr;

use candle_kernels::dkr_s31_contract as contract;

pub(super) struct ExternalModules {
    modules: std::collections::HashMap<&'static str, String>,
    trace: bool,
    mmq_symbols: std::collections::HashMap<&'static str, std::collections::BTreeMap<String, String>>,
}

fn family(index: usize) -> Option<&'static str> {
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

fn read_text(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| {
        crate::Error::Msg(format!("Candle DKR S3.1 {}: {e}", path.display()))
    })
}

fn validate(manifest: &Value, name: &str, ptx: &str) -> Result<()> {
    let sha = format!("{:x}", Sha256::digest(ptx.as_bytes()));
    let flags: Vec<&str> = manifest["flags"].as_array()
        .map(|v| v.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    if manifest["schema"].as_str() != Some(contract::SCHEMA)
        || manifest["base_revision"].as_str() != Some(contract::BASE)
        || manifest["target_cc"].as_u64() != Some(u64::from(contract::CC))
        || manifest["family"].as_str() != Some(name)
        || manifest["sha256"].as_str() != Some(sha.as_str())
        || flags != contract::FLAGS
        || !contract::is_sm61_ptx(ptx)
        || (name == "mmvq_gguf" && !contract::has_mmvq_entries(ptx))
        || (matches!(name, "conv" | "sort") && !contract::has_conv_sort_entries(ptx, name))
        || (name.starts_with("mmq_") && contract::mmq_entries(ptx, name)
            .and_then(|entries| serde_json::to_value(entries).ok())
            .is_none_or(|entries| entries != manifest["mmq_entries"]))
    {
        crate::bail!("Candle DKR S3.1 invalid package for {name}");
    }
    Ok(())
}

impl ExternalModules {
    pub(super) fn from_env(context: &CudaContext) -> Result<Option<Self>> {
        let Some(root) = std::env::var_os("CANDLE_DKR_S31_DIR") else {
            return Ok(None);
        };
        let (major, minor) = context.compute_capability().w()?;
        if (major, minor) != (6, 1) {
            crate::bail!("Candle DKR S3.1 requires SM61, got {major}.{minor}");
        }
        let root = PathBuf::from(root);
        if !root.is_absolute() || !root.is_dir() {
            crate::bail!("CANDLE_DKR_S31_DIR must be an existing absolute directory");
        }
        // Fail before inference if a selected module is absent or corrupt.
        let mut modules = std::collections::HashMap::new();
        for &name in contract::FAMILIES {
            modules.insert(name, Self::read(&root, name)?);
        }
        let mut mmq_symbols = std::collections::HashMap::new();
        for (&name, ptx) in &modules {
            if name.starts_with("mmq_") {
                let symbols = contract::mmq_entries(ptx, name)
                    .ok_or_else(|| crate::Error::Msg(format!("invalid MMQ entries: {name}")))?;
                mmq_symbols.insert(name, symbols);
            }
        }
        Ok(Some(Self { modules, mmq_symbols, trace: std::env::var_os("CANDLE_DKR_TRACE").is_some() }))
    }

    fn read(root: &Path, name: &str) -> Result<String> {
        let manifest = read_text(&root.join(name).join("manifest.json"))?;
        let manifest: Value = serde_json::from_str(&manifest)
            .map_err(|e| crate::Error::Msg(format!("Candle DKR S3.1 manifest: {e}")))?;
        let ptx = read_text(&root.join(name).join(format!("{name}.ptx")))?;
        validate(&manifest, name, &ptx)?;
        Ok(ptx)
    }

    pub(super) fn ptx(&self, index: usize) -> Option<&str> {
        family(index).and_then(|name| self.modules.get(name)).map(String::as_str)
    }

    pub(super) fn mmq_entry(&self, family: &str, key: &str) -> Option<(&str, &str)> {
        Some((self.modules.get(family)?.as_str(), self.mmq_symbols.get(family)?.get(key)?.as_str()))
    }

    pub(super) fn record_mmq_loaded(&self, family: &str) {
        if self.trace {
            eprintln!("CANDLE_DKR_MODULE_LOADED family={family} target=sm61 source=external");
        }
    }

    pub(super) fn mmvq_ptx(&self) -> Option<&str> {
        self.modules.get("mmvq_gguf").map(String::as_str)
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
