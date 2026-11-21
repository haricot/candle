use std::{env, fs, path::PathBuf};
use sha2::{Digest, Sha256};

#[allow(dead_code)]
#[path = "../candle-kernels/dkr_s31_contract.rs"]
mod contract;
#[path = "src/cuda_backend/dkr_s31_validation.rs"]
mod validation;
#[path = "src/cuda_backend/dkr_native_catalog.rs"]
mod native_catalog;
#[allow(dead_code)]
#[path = "../candle-kernels/dkr_fp8_recipe.rs"]
mod fp8_recipe;
#[allow(dead_code)]
#[path = "../candle-kernels/dkr_fp4_recipe.rs"]
mod fp4_recipe;
#[allow(dead_code)]
#[path = "../candle-kernels/dkr_bf16_recipe.rs"]
mod bf16_recipe;
#[allow(dead_code)]
#[path = "../candle-kernels/dkr_routed_recipe.rs"]
mod routed_recipe;

const EMBED_ENV: &str = "DKR_EMBED_NATIVE_DIR";
const CUBIN_ENV: &str = "DKR_EMBED_CUBIN_DIR";
const CUBIN_SCHEMA: &str = "CANDLE-DKR-R124-S54B-CUBIN-V1";
const MAX_CUBIN_BYTES: usize = 32 * 1024 * 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-env-changed={EMBED_ENV}");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA_DKR_EMBEDDED");
    let strict = env::var_os("CARGO_FEATURE_CUDA_DKR_EMBEDDED").is_some();
    let cubin_enabled = env::var_os("CARGO_FEATURE_CUDA_DKR_CUBIN_EMBEDDED").is_some();
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA_DKR_CUBIN_EMBEDDED");
    println!("cargo:rerun-if-env-changed={CUBIN_ENV}");
    if cubin_enabled && !strict {
        return Err("cuda-dkr-cubin-embedded requires cuda-dkr-embedded".into());
    }
    if strict {
        if env::var_os(EMBED_ENV).is_none() {
            return Err("cuda-dkr-embedded requires DKR_EMBED_NATIVE_DIR at build time".into());
        }
        for feature in ["CUDA_DKR_S31", "CUDA_LEGACY_BF16", "CUDA_LEGACY_FP8", "CUDA_LEGACY_FP4", "CUTILE"] {
            if env::var_os(format!("CARGO_FEATURE_{feature}")).is_some() {
                return Err(format!("cuda-dkr-embedded forbids feature {feature}").into());
            }
        }
    }
    println!("cargo:rerun-if-changed=src/cuda_backend/dkr_s31_validation.rs");
    println!("cargo:rerun-if-changed=src/cuda_backend/dkr_native_catalog.rs");
    println!("cargo:rerun-if-changed=../candle-kernels/dkr_s31_contract.rs");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut rows = Vec::new();
    let mut cubin_rows = Vec::new();
    let cubin_input: Option<(PathBuf, serde_json::Value)> = if cubin_enabled {
        let p = PathBuf::from(env::var_os(CUBIN_ENV)
            .ok_or("cuda-dkr-cubin-embedded requires DKR_EMBED_CUBIN_DIR")?);
        if !p.is_absolute() { return Err("DKR_EMBED_CUBIN_DIR must be absolute".into()); }
        let dir = p.canonicalize()?;
        if !dir.is_dir() { return Err("CUBIN input must be a directory".into()); }
        let mf = dir.join("manifest.json");
        println!("cargo:rerun-if-changed={}", mf.display());
        let bytes = fs::read(&mf)?;
        if bytes.len() > 256 * 1024 { return Err("CUBIN manifest too large".into()); }
        let v: serde_json::Value = serde_json::from_slice(&bytes)?;
        if v["schema"].as_str() != Some(CUBIN_SCHEMA)
            || v["target_cc"].as_u64() != Some(u64::from(contract::CC))
            || v["families"].as_object().map(|x| x.len()) != Some(contract::FAMILIES.len())
            || v["ptxas_version"].as_str().is_none_or(|s| !s.contains("release 12."))
        {
            return Err("CUBIN manifest schema, SM61 target, family count, or CUDA 12 toolchain invalid".into());
        }
        Some((dir, v))
    } else { None };
    let mut root_literal = "None".to_owned();
    let mut recipe_literal = "None".to_owned();
    let mut fp8_literal = "None".to_owned();
    let mut fp4_literal = "None".to_owned();
    let mut bf16_literal = "None".to_owned();
    let mut routed_literal = "None".to_owned();
    println!("cargo:rerun-if-changed=../candle-kernels/dkr_routed_recipe.rs");
    println!("cargo:rerun-if-changed=../candle-kernels/dkr_bf16_recipe.rs");
    println!("cargo:rerun-if-changed=../candle-kernels/dkr_fp4_recipe.rs");
    println!("cargo:rerun-if-changed=../candle-kernels/dkr_fp8_recipe.rs");
    if let Some(root) = env::var_os(EMBED_ENV) {
        let root = PathBuf::from(root);
        if strict && !root.is_absolute() {
            return Err("DKR_EMBED_NATIVE_DIR must be absolute in cuda-dkr-embedded mode".into());
        }
        let root = root.canonicalize()?;
        assert!(
            env::var_os("CARGO_FEATURE_CUDA_DKR_CATALOG").is_some(),
            "{EMBED_ENV} requires cuda-dkr-catalog (or compatibility cuda-dkr-s31)"
        );
        root_literal = format!("Some({:?})", root.to_str().ok_or("non-UTF8 embedded path")?);
        // Watch the directory as well, so adding an index invalidates a legacy build.
        println!("cargo:rerun-if-changed={}", root.display());
        let catalog = native_catalog::NativeCatalog::load(&root)?;
        recipe_literal = format!("Some({:?})", catalog.recipe.id());
        fp8_literal = format!("Some({:?})", catalog.recipe.fp8());
        fp4_literal = format!("Some({:?})", catalog.recipe.fp4());
        routed_literal = format!("Some({:?})", catalog.recipe.routed());
        let bf16 = catalog.recipe.bf16();
        bf16_literal = format!(
            "Some(Bf16Recipe {{ matmul: bf16_recipe::Bf16MatmulRoute::{:?}, cast_to_f32: {:?}, cast_from_f32: {:?} }})",
            bf16.matmul, bf16.cast_to_f32, bf16.cast_from_f32
        );
        for name in &catalog.families {
            let (manifest, artifact) = native_catalog::NativeCatalog::paths(&root, name);
            println!("cargo:rerun-if-changed={}", manifest.display());
            println!("cargo:rerun-if-changed={}", artifact.display());
            let (verified, symbols) = catalog.read_package(&root, name)?;
            let copy = out.join(format!("dkr-native-{name}.ptx"));
            fs::write(&copy, verified.ptx())?;
            let entries = verified
                .entry_names()
                .map(|s| format!("{s:?}.into()"))
                .collect::<Vec<_>>()
                .join(",");
            let symbols = symbols
                .map(|pairs| {
                    let pairs = pairs
                        .iter()
                        .map(|(k, v)| format!("({k:?}.into(),{v:?}.into())"))
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("Some([{pairs}].into_iter().collect())")
                })
                .unwrap_or("None".into());
            rows.push(format!(
                "({name:?}, VerifiedArtifact::from_build_verified(include_str!({copy:?}), {}, {:?}, \
                 vec![{entries}]), {symbols})", verified.key().target_cc(), verified.key().sha256()
            ));
            if let Some((cubin_root, cubin_index)) = &cubin_input {
                let rec = &cubin_index["families"][name.as_str()];
                if rec["target_cc"].as_u64() != Some(u64::from(verified.key().target_cc()))
                    || rec["ptx_sha256"].as_str() != Some(verified.key().sha256())
                {
                    return Err(format!("CUBIN {name} target/PTX provenance mismatch").into());
                }
                let image_path = cubin_root.join("cubins").join(format!("{name}.cubin"));
                let resolved = image_path.canonicalize()?;
                if !resolved.starts_with(cubin_root) || !resolved.is_file() {
                    return Err(format!("CUBIN outside qualified input directory: {name}").into());
                }
                println!("cargo:rerun-if-changed={}", resolved.display());
                let data = fs::read(&resolved)?;
                if data.len() < 64 || data.len() > MAX_CUBIN_BYTES || !data.starts_with(b"\x7fELF") {
                    return Err(format!("CUBIN {name} is not an acceptable ELF image").into());
                }
                let image_sha = format!("{:x}", Sha256::digest(&data));
                if rec["cubin_sha256"].as_str() != Some(image_sha.as_str())
                    || rec["cubin_bytes"].as_u64() != Some(data.len() as u64)
                {
                    return Err(format!("CUBIN {name} digest/size mismatch").into());
                }
                let dest = out.join(format!("dkr-native-{name}.cubin"));
                fs::write(&dest, data)?;
                cubin_rows.push(format!(
                    "({name:?}, include_bytes!({dest:?}) as &'static [u8], {image_sha:?}, {}, {:?})",
                    verified.key().target_cc(), verified.key().sha256()
                ));
            }
        }
    }
    if cubin_enabled {
        assert_eq!(cubin_rows.len(), rows.len(), "incomplete CUBIN inventory");
        println!("cargo:warning=CANDLE_DKR_CUBIN_EMBEDDED=1 packages={}", cubin_rows.len());
    }
    if strict {
        assert_eq!(rows.len(), contract::FAMILIES.len(), "incomplete embedded catalogue");
        println!("cargo:warning=CANDLE_DKR_EMBEDDED_REQUIRED=1 packages={}", rows.len());
    }
    fs::write(
        out.join("dkr_native_embedded.rs"),
        format!(
            "pub const ROOT: Option<&str> = {root_literal};\n\
         pub const RECIPE: Option<&str> = {recipe_literal};\n\
         pub const FP8_RECIPE: Option<Fp8Recipe> = {fp8_literal};\n\
         pub const FP4_RECIPE: Option<Fp4Recipe> = {fp4_literal};\n\
         pub const BF16_RECIPE: Option<Bf16Recipe> = {bf16_literal};\n\
         pub const ROUTED_RECIPE: Option<RoutedRecipe> = {routed_literal};\n\
         pub fn modules() -> Vec<(&'static str, VerifiedArtifact, \
         Option<std::collections::BTreeMap<String,String>>)> {{ vec![{}] }}\n\
         pub fn cubins() -> Vec<(&'static str, &'static [u8], &'static str, u32, &'static str)> {{ vec![{}] }}\n",
            rows.join(",\n"), cubin_rows.join(",\n")
        ),
    )?;
    Ok(())
}
