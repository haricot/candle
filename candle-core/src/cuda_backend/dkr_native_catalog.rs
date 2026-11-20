//! Native catalogue inventory shared by the build script, runtime and producer.
//! An inventory describes a qualified host-launch ABI, not just loadable PTX.
use super::{contract, validation, fp8_recipe::{Fp8Recipe, S31_FP8}, fp4_recipe::{Fp4Recipe, S31_FP4}, bf16_recipe::{Bf16Recipe, S31_BF16}, routed_recipe::{RoutedRecipe,S31_ROUTED}};
use dkr_kernel_registry::VerifiedArtifact;
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, io::Read, path::{Path, PathBuf}};

pub(super) const INDEX: &str = "native-catalog.json";
const SCHEMA: &str = "CANDLE-DKR-NATIVE-CATALOG-V1";
const LEGACY_RECIPE: &str = "s31-native-v1";
const MAX_MANIFEST: usize = 256 * 1024;

/// Typed host-launch recipe admitted by the inventory. Do not execute
/// arbitrary geometry or argument ABI supplied by an untrusted manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeRecipe {
    S31NativeV1,
}

pub(super) struct MmqGeometry {
    pub matrix_key: String,
    pub quant_grid: (u32, u32, u32),
    pub quant_block: (u32, u32, u32),
    pub matrix_grid: (u32, u32, u32),
    pub matrix_block: (u32, u32, u32),
    pub matrix_shared: u32,
}

impl NativeRecipe {
    pub(super) fn from_id(id: &str) -> Result<Self, String> {
        match id {
            LEGACY_RECIPE => Ok(Self::S31NativeV1),
            _ => Err(format!("unsupported native launch recipe: {id}")),
        }
    }

    pub(super) fn routed(self)->RoutedRecipe { match self {Self::S31NativeV1=>S31_ROUTED} }

    pub(super) fn bf16(self) -> Bf16Recipe { match self { Self::S31NativeV1 => S31_BF16 } }

    pub(super) fn fp4(self) -> Fp4Recipe { match self { Self::S31NativeV1 => S31_FP4 } }

    pub(super) fn fp8(self) -> Fp8Recipe {
        match self { Self::S31NativeV1 => S31_FP8 }
    }

    pub(super) fn id(self) -> &'static str {
        match self { Self::S31NativeV1 => LEGACY_RECIPE }
    }

    pub(super) fn mmq_geometry(self, codec: &str, rows: u32, batch: u32,
        padded: u32, warp: i32, limit: i32) -> Result<MmqGeometry, String>
    {
        match self {
            Self::S31NativeV1 => {
                // Qualified S3.1 DP4A recipe: ordinary shared memory, no opt-in,
                // no Stream-K, no fixup buffer. Preserve the original geometry.
                if warp != 32 || limit <= 0 || padded == 0 || padded.div_ceil(512) > 65535 {
                    return Err("invalid S3.1 MMQ device limits or geometry".into());
                }
                let tile = contract::mmq_tile(codec, rows, batch, (limit as u32).min(49152))
                    .ok_or("no legal S3.1 MMQ tile")?;
                Ok(MmqGeometry {
                    matrix_key: format!("{}:{}", tile.x, u8::from(tile.checked)),
                    quant_grid: (batch, padded.div_ceil(512), 1),
                    quant_block: (128, 1, 1),
                    matrix_grid: tile.grid,
                    matrix_block: (32, 8, 1),
                    matrix_shared: tile.shared,
                })
            }
        }
    }
}

#[derive(Debug)]
pub(super) struct NativeCatalog {
    pub target_cc: u32,
    pub families: Vec<String>,
    pub recipe: NativeRecipe,
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
            recipe: NativeRecipe::S31NativeV1,
            families: contract::FAMILIES.iter().map(|name| (*name).to_owned()).collect(),
        }
    }

    pub(super) fn index(&self) -> Value {
        json!({
            "schema": SCHEMA,
            "abi": dkr_kernel_registry::NATIVE_ABI,
            "launch_recipe": self.recipe.id(),
            "target_cc": self.target_cc,
            "min_cc": self.target_cc,
            "max_cc": self.target_cc,
            "families": self.families,
            "fp8": fp8_descriptor(self.recipe.fp8()),
            "fp4": fp4_descriptor(self.recipe.fp4()),
            "bf16": bf16_descriptor(self.recipe.bf16()),
            "routed": routed_descriptor(self.recipe.routed()),
        })
    }

    fn parse(value: &Value) -> Result<Self, String> {
        let object = value.as_object().ok_or("native catalogue must be an object")?;
        if object.len() != 7 + (object.contains_key("fp8") as usize) + (object.contains_key("fp4") as usize) + (object.contains_key("bf16") as usize) + (object.contains_key("routed") as usize)
            || value["schema"].as_str() != Some(SCHEMA)
            || value["abi"].as_str() != Some(dkr_kernel_registry::NATIVE_ABI)
        {
            return Err("unsupported native catalogue schema/ABI/launch recipe".into());
        }
        let recipe = NativeRecipe::from_id(value["launch_recipe"].as_str().ok_or("missing native launch recipe")?)?;
        if let Some(fp8) = object.get("fp8") {
            if *fp8 != fp8_descriptor(recipe.fp8()) {
                return Err("FP8 descriptor differs from qualified launch ABI".into());
            }
        }
        if let Some(fp4) = object.get("fp4") {
            if *fp4 != fp4_descriptor(recipe.fp4()) { return Err("FP4 descriptor differs from qualified launch ABI".into()); }
        }
        if let Some(bf16) = object.get("bf16") {
            if *bf16 != bf16_descriptor(recipe.bf16()) {
                return Err("BF16 descriptor differs from qualified launch ABI".into());
            }
        }
        if let Some(routed) = object.get("routed") {
            if *routed != routed_descriptor(recipe.routed()) { return Err("MMVQ/MoE descriptor differs from qualified ABI".into()); }
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
        Ok(Self { target_cc: cc, families, recipe })
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
        if family == "cast" {
            verify_bf16_entries(self.recipe.bf16(), &verified)?;
        }
        if family == "quantized" {
            verify_fp8_entries(self.recipe.fp8(), &verified)?;
            verify_fp4_entries(self.recipe.fp4(), &verified)?;
        }
        Ok((verified, symbols))
    }
}

fn routed_descriptor(recipe:RoutedRecipe)->Value {
    json!({
        "max_batch": recipe.max_batch,
        "quant_threads": recipe.quant_threads,
        "warp": recipe.warp,
        "small_warps": recipe.small_warps,
        "large_warps": recipe.large_warps,
        "small_batch": recipe.small_batch,
        "row_padding": recipe.row_padding,
        "q8_block": recipe.q8_block,
        "q8_bytes": recipe.q8_bytes,
        "max_experts": recipe.max_experts,
        "simt_threads": recipe.simt_threads,
        "simt_tile_n": recipe.simt_tile_n,
        "simt_shared": recipe.simt_shared,
        "gguf_tile_n": recipe.gguf_tile_n,
        "gguf_shared_base": recipe.gguf_shared_base,
        "gguf_shared_limit": recipe.gguf_shared_limit,
        "mmvq_family": "mmvq_gguf",
        "simt_family": "moe_simt_f16",
        "gguf_family": "moe_gguf",
        "simt_input": "f16",
        "gguf_input": "f32",
        "gguf_abi": contract::MOE_GGUF_ABI,
    })
}

fn bf16_descriptor(recipe: Bf16Recipe) -> Value {
    json!({
        "matmul": recipe.route_id(),
        "cast_to_f32": recipe.cast_to_f32,
        "cast_from_f32": recipe.cast_from_f32,
    })
}

fn verify_bf16_entries(recipe: Bf16Recipe, artifact: &VerifiedArtifact) -> Result<(), String> {
    for symbol in recipe.cast_symbols() {
        if !artifact.has_entry(symbol) {
            return Err(format!("native BF16 conversion entry missing: {symbol}"));
        }
    }
    Ok(())
}

fn fp4_descriptor(recipe: Fp4Recipe) -> Value {
    json!({
        "scalar": recipe.scalar,
        "cast": recipe.cast,
        "quantize": recipe.quantize,
        "reference": recipe.reference,
        "tiled": recipe.tiled,
        "mx_vec": recipe.mx_vec,
        "mx_mat": recipe.mx_mat,
        "mx_moe": recipe.mx_moe,
        "auto_cc": recipe.auto_cc,
        "min_dp4a_cc": recipe.min_dp4a_cc,
        "auto_min_n": recipe.auto_min_n,
        "auto_min_k": recipe.auto_min_k,
        "auto_min_batch": recipe.auto_min_batch,
        "scalar_threads": recipe.scalar_threads,
        "cast_threads": recipe.cast_threads,
        "quant_threads": recipe.quant_threads,
        "warp": recipe.warp,
        "matrix_warps": recipe.matrix_warps,
        "tile_n": recipe.tile_n,
        "tile_m": recipe.tile_m,
    })
}

fn verify_fp4_entries(recipe: Fp4Recipe, artifact: &VerifiedArtifact) -> Result<(), String> {
    for symbol in recipe.symbols() {
        if !artifact.has_entry(symbol) { return Err(format!("native FP4 entry missing: {symbol}")); }
    }
    Ok(())
}

fn fp8_descriptor(recipe: Fp8Recipe) -> Value {
    json!({
        "simt_gemv": recipe.simt_gemv,
        "simt_prefill": recipe.simt_prefill,
        "dp4a_gemv": recipe.dp4a_gemv,
        "dp4a_prefill": recipe.dp4a_prefill,
        "quantize": recipe.quantize,
        "prefill_tokens": recipe.prefill_tokens,
        "rows_per_block": recipe.rows_per_block,
        "threads": recipe.threads,
        "k_alignment": recipe.k_alignment,
        "groups": recipe.groups,
    })
}

fn verify_fp8_entries(recipe: Fp8Recipe, artifact: &VerifiedArtifact) -> Result<(), String> {
    for symbol in recipe.symbols() {
        if !artifact.has_entry(symbol) {
            return Err(format!("native FP8 entry missing: {symbol}"));
        }
    }
    Ok(())
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
    fn dkr_native_catalog_routed_recipe() {
        let index=NativeCatalog::legacy().index();
        assert_eq!(NativeCatalog::parse(&index).unwrap().recipe.routed(),S31_ROUTED);
        let mut old=index.clone();
        old.as_object_mut().unwrap().remove("routed");
        assert_eq!(NativeCatalog::parse(&old).unwrap().recipe.routed(),S31_ROUTED);
        for (field,value) in [("max_batch",json!(16)),("gguf_shared_limit",json!(98304)),
            ("simt_input",json!("bf16")),("gguf_abi",json!("other"))] {
            let mut bad=index.clone(); bad["routed"][field]=value;
            assert!(NativeCatalog::parse(&bad).is_err(),"{field}");
        }
    }

    #[test]
    fn dkr_native_catalog_bf16_recipe() {
        use sha2::{Digest, Sha256};
        let legacy = NativeCatalog::legacy();
        let index = legacy.index();
        assert_eq!(NativeCatalog::parse(&index).unwrap().recipe.bf16(), S31_BF16);
        let mut old = index.clone();
        for field in ["bf16", "fp4", "fp8"] { old.as_object_mut().unwrap().remove(field); }
        assert_eq!(NativeCatalog::parse(&old).unwrap().recipe.bf16(), S31_BF16);
        for (field, replacement) in [
            ("matmul", json!("cublas-native-bf16")),
            ("cast_to_f32", json!("cast_bf16_f16")),
            ("cast_from_f32", json!("other")),
        ] {
            let mut bad = index.clone();
            bad["bf16"][field] = replacement;
            assert!(NativeCatalog::parse(&bad).is_err(), "{field}");
        }
        let make = |symbols: &[&str]| {
            let mut ptx = format!(".version 8.0\n.target sm_{}\n.address_size 64\n", legacy.target_cc);
            for symbol in symbols { ptx += &format!(".visible .entry {symbol}() {{ ret; }}\n"); }
            let digest = format!("{:x}", Sha256::digest(ptx.as_bytes()));
            VerifiedArtifact::new(ptx, legacy.target_cc, &digest).unwrap()
        };
        let symbols = S31_BF16.cast_symbols();
        assert!(verify_bf16_entries(S31_BF16, &make(&symbols)).is_ok());
        assert!(verify_bf16_entries(S31_BF16, &make(&symbols[..1])).is_err());
    }

    #[test]
    fn dkr_native_catalog_fp4_recipe() {
        use sha2::{Digest, Sha256};
        let legacy = NativeCatalog::legacy();
        let index = legacy.index();
        assert_eq!(NativeCatalog::parse(&index).unwrap().recipe.fp4(), S31_FP4);
        let mut old = index.clone();
        old.as_object_mut().unwrap().remove("fp4");
        old.as_object_mut().unwrap().remove("fp8");
        assert_eq!(NativeCatalog::parse(&old).unwrap().recipe.fp4(), S31_FP4);
        for (field, value) in [("auto_cc",json!(80)), ("tile_n",json!(8)), ("scalar",json!("other"))] {
            let mut bad = index.clone();
            bad["fp4"][field] = value;
            assert!(NativeCatalog::parse(&bad).is_err(), "{field}");
        }
        let symbols: Vec<_> = S31_FP4.symbols().collect();
        let make = |symbols: &[&str]| {
            let mut ptx = format!(".version 8.0\n.target sm_{}\n.address_size 64\n", legacy.target_cc);
            for symbol in symbols { ptx += &format!(".visible .entry {symbol}() {{ ret; }}\n"); }
            let digest = format!("{:x}", Sha256::digest(ptx.as_bytes()));
            VerifiedArtifact::new(ptx, legacy.target_cc, &digest).unwrap()
        };
        assert!(verify_fp4_entries(S31_FP4,&make(&symbols)).is_ok());
        assert!(verify_fp4_entries(S31_FP4,&make(&symbols[..symbols.len()-1])).is_err());
    }

    #[test]
    fn dkr_native_catalog_fp8_recipe() {
        use sha2::{Digest, Sha256};
        let legacy = NativeCatalog::legacy();
        let index = legacy.index();
        assert_eq!(NativeCatalog::parse(&index).unwrap().recipe.fp8(), S31_FP8);
        let mut old = index.clone();
        old.as_object_mut().unwrap().remove("fp8");
        assert_eq!(NativeCatalog::parse(&old).unwrap().recipe.fp8(), S31_FP8);
        for (field, replacement) in [("threads", json!(256)), ("simt_gemv", json!("other")),
            ("groups", json!([1, 2, 3])), ("prefill_tokens", json!(8))] {
            let mut bad = index.clone();
            bad["fp8"][field] = replacement;
            assert!(NativeCatalog::parse(&bad).is_err(), "{field}");
        }
        let make_artifact = |symbols: &[&str]| {
            let mut ptx = format!(".version 8.0\n.target sm_{}\n.address_size 64\n", legacy.target_cc);
            for symbol in symbols {
                ptx += &format!(".visible .entry {symbol}() {{ ret; }}\n");
            }
            let digest = format!("{:x}", Sha256::digest(ptx.as_bytes()));
            VerifiedArtifact::new(ptx, legacy.target_cc, &digest).unwrap()
        };
        let symbols = S31_FP8.symbols();
        assert!(verify_fp8_entries(S31_FP8, &make_artifact(&symbols)).is_ok());
        assert!(verify_fp8_entries(S31_FP8, &make_artifact(&symbols[..4])).is_err());
    }

    #[test]
    fn dkr_native_catalog_mmq_recipe() {
        let recipe = NativeRecipe::from_id(LEGACY_RECIPE).unwrap();
        assert_eq!(NativeCatalog::parse(&NativeCatalog::legacy().index()).unwrap().recipe, recipe);
        assert!(NativeRecipe::from_id("unknown").is_err());
        for codec in contract::MMVQ_CODECS {
            for rows in [1, 64, 65, 129] {
                for batch in [1, 9, 64, 65, 1024] {
                    let old = contract::mmq_tile(codec, rows, batch, 49152).unwrap();
                    let plan = recipe.mmq_geometry(codec, rows, batch, 1024, 32, 65536).unwrap();
                    assert_eq!(plan.matrix_key, format!("{}:{}", old.x, u8::from(old.checked)));
                    assert_eq!(plan.matrix_grid, old.grid);
                    assert_eq!(plan.matrix_shared, old.shared);
                    assert_eq!(plan.quant_grid, (batch, 2, 1));
                    assert_eq!(plan.quant_block, (128, 1, 1));
                    assert_eq!(plan.matrix_block, (32, 8, 1));
                }
            }
        }
        assert!(recipe.mmq_geometry("q4_0", 64, 9, 1024, 16, 49152).is_err());
        assert!(recipe.mmq_geometry("q4_0", 64, 9, 1024, 32, 1).is_err());
        assert!(recipe.mmq_geometry("q4_0", 64, 9, 0, 32, 49152).is_err());
        assert!(recipe.mmq_geometry("unknown", 64, 9, 1024, 32, 49152).is_err());
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
