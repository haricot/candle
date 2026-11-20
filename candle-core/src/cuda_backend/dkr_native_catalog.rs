//! Native catalogue inventory shared by the build script, runtime and producer.
//! An inventory describes a qualified host-launch ABI, not just loadable PTX.
use super::{contract, validation};
use dkr_kernel_registry::VerifiedArtifact;
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, io::Read, path::{Path, PathBuf}};

pub(super) const INDEX: &str = "native-catalog.json";
const SCHEMA: &str = "CANDLE-DKR-NATIVE-CATALOG-V1";
const LEGACY_RECIPE: &str = "s31-native-v1";
const MAX_MANIFEST: usize = 256 * 1024;

#[derive(Debug)]
pub(super) struct NativeCatalog {
    pub target_cc: u32,
    pub families: Vec<String>,
}

fn read_bounded(root: &Path, path: &Path, limit: usize) -> Result<String, String> {
    let resolved = path.canonicalize().map_err(|e| format!("{}: {e}", path.display()))?;
    if !resolved.starts_with(root) || !resolved.is_file() {
        return Err(format!("invalid native catalogue path: {}", path.display()));
    }
    let mut text = Vec::new();
    fs::File::open(&resolved).map_err(|e| e.to_string())?
        .take((limit + 1) as u64).read_to_end(&mut text).map_err(|e| e.to_string())?;
    if text.len() > limit {
        return Err(format!("native catalogue file exceeds budget: {}", path.display()));
    }
    String::from_utf8(text).map_err(|e| e.to_string())
}

impl NativeCatalog {
    /// Compatibility adapter for existing pre-index S3.1 packages.
    pub(super) fn legacy() -> Self {
        Self {
            target_cc: contract::CC,
            families: contract::FAMILIES.iter().map(|name| (*name).to_owned()).collect(),
        }
    }

    pub(super) fn index(&self) -> Value {
        json!({
            "schema": SCHEMA,
            "abi": dkr_kernel_registry::NATIVE_ABI,
            "launch_recipe": LEGACY_RECIPE,
            "target_cc": self.target_cc,
            "min_cc": self.target_cc,
            "max_cc": self.target_cc,
            "families": self.families,
        })
    }

    fn parse(value: &Value) -> Result<Self, String> {
        let object = value.as_object().ok_or("native catalogue must be an object")?;
        if object.len() != 7
            || value["schema"].as_str() != Some(SCHEMA)
            || value["abi"].as_str() != Some(dkr_kernel_registry::NATIVE_ABI)
            || value["launch_recipe"].as_str() != Some(LEGACY_RECIPE)
        {
            return Err("unsupported native catalogue schema/ABI/launch recipe".into());
        }
        let cc = value["target_cc"].as_u64().and_then(|cc| u32::try_from(cc).ok())
            .ok_or("invalid native catalogue target")?;
        // S3.1 still consumes a qualified Pascal host-launch recipe.
        // A future adapter must explicitly define its supported ranges and inventory.
        if cc != contract::CC || value["min_cc"].as_u64() != Some(u64::from(cc))
            || value["max_cc"].as_u64() != Some(u64::from(cc))
        {
            return Err("unsupported architecture for native launch recipe".into());
        }
        let list = value["families"].as_array().ok_or("missing native family inventory")?;
        let required: BTreeSet<&str> = contract::FAMILIES.iter().copied().collect();
        let mut seen = BTreeSet::new();
        let mut families = Vec::new();
        for entry in list {
            let family = entry.as_str().ok_or("invalid native family name")?;
            if !required.contains(family) || !seen.insert(family) {
                return Err(format!("unknown or duplicate native family: {family}"));
            }
            families.push(family.to_owned());
        }
        if seen != required {
            return Err("incomplete native family inventory; refusing static fallback".into());
        }
        Ok(Self { target_cc: cc, families })
    }

    /// Root must be canonical. Only an absent index enables the legacy adapter.
    pub(super) fn load(root: &Path) -> Result<Self, String> {
        let path = root.join(INDEX);
        match fs::symlink_metadata(&path) {
            Ok(_) => {
                let text = read_bounded(root, &path, MAX_MANIFEST)?;
                let value: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
                Self::parse(&value)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::legacy()),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    pub(super) fn paths(root: &Path, family: &str) -> (PathBuf, PathBuf) {
        (root.join(family).join("manifest.json"), root.join(family).join(format!("{family}.ptx")))
    }

    pub(super) fn read_package(&self, root: &Path, family: &str)
        -> Result<(VerifiedArtifact, validation::NativeSymbols), String>
    {
        if !self.families.iter().any(|name| name == family) {
            return Err(format!("family not declared by native catalogue: {family}"));
        }
        let (manifest, artifact) = Self::paths(root, family);
        let metadata: Value = serde_json::from_str(&read_bounded(root, &manifest, MAX_MANIFEST)?)
            .map_err(|e| e.to_string())?;
        let ptx = read_bounded(root, &artifact, dkr_kernel_registry::MAX_PTX_BYTES)?;
        let (verified, symbols) = validation::verify_package(&metadata, family, ptx)?;
        verified.admit_exact_target(self.target_cc)?;
        Ok((verified, symbols))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dkr_native_catalog_inventory() {
        let legacy = NativeCatalog::legacy();
        let mut index = legacy.index();
        index["families"].as_array_mut().unwrap().reverse();
        let parsed = NativeCatalog::parse(&index).unwrap();
        assert_eq!(parsed.target_cc, contract::CC);
        assert_eq!(parsed.families.first(), legacy.families.last());
        let mut missing = index.clone();
        missing["families"].as_array_mut().unwrap().pop();
        assert!(NativeCatalog::parse(&missing).is_err());
        let mut duplicate = index.clone();
        let first = duplicate["families"][0].clone();
        duplicate["families"].as_array_mut().unwrap().push(first);
        assert!(NativeCatalog::parse(&duplicate).is_err());
        let mut traversal = index.clone();
        traversal["families"][0] = json!("../unary");
        assert!(NativeCatalog::parse(&traversal).is_err());
        for (field, value) in [
            ("target_cc", json!(80)), ("max_cc", json!(120)),
            ("min_cc", json!(60)), ("launch_recipe", json!("unknown")),
            ("abi", json!("other-abi")), ("schema", json!("unknown")),
        ] {
            let mut bad = index.clone();
            bad[field] = value;
            assert!(NativeCatalog::parse(&bad).is_err(), "{field}");
        }
    }

    #[test]
    fn dkr_native_catalog_filesystem() {
        let root = std::env::temp_dir().join(format!("candle-dkr-index-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        struct Cleanup(PathBuf);
        impl Drop for Cleanup { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
        let _cleanup = Cleanup(root.clone());
        let root = root.canonicalize().unwrap();
        let legacy = NativeCatalog::load(&root).unwrap();
        fs::write(root.join(INDEX), legacy.index().to_string()).unwrap();
        let index = NativeCatalog::load(&root).unwrap();
        assert_eq!(legacy.families, index.families);
        assert!(index.read_package(&root, "unary").is_err());
        fs::write(root.join(INDEX), "{}").unwrap();
        assert!(NativeCatalog::load(&root).is_err());
    }
}
