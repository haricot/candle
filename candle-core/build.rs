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
#[path = "src/cuda_backend/dkr_s55b_build.rs"]
mod neutral_recipe_build;

const EMBED_ENV: &str = "DKR_EMBED_NATIVE_DIR";
const CUBIN_ENV: &str = "DKR_EMBED_CUBIN_DIR";
const MMQ_ENV: &str = "DKR_EMBED_S55_MMQ_DIR";
const RECIPES_ENV: &str = "DKR_EMBED_S55B_RECIPES_DIR";
const CUDAFORGE_ENV: &str = "DKR_EMBED_S55C_CUDAFORGE_DIR";
const CUDAFORGE_SCHEMA: &str = "CANDLE-DKR-R124-S55C-CUDAFORGE-CATALOG-V1";
const MMQ_SCHEMA: &str = "CANDLE-DKR-R124-S55A-MMQ-LAUNCH-V1";
const CUBIN_SCHEMA: &str = "CANDLE-DKR-R124-S54B-CUBIN-V1";
const MAX_CUBIN_BYTES: usize = 32 * 1024 * 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-env-changed={EMBED_ENV}");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA_DKR_EMBEDDED");
    let strict = env::var_os("CARGO_FEATURE_CUDA_DKR_EMBEDDED").is_some();
    let cubin_enabled = env::var_os("CARGO_FEATURE_CUDA_DKR_CUBIN_EMBEDDED").is_some();
    let neutral_mmq = env::var_os("CARGO_FEATURE_CUDA_DKR_NEUTRAL_MMQ").is_some();
    let neutral_recipes = env::var_os("CARGO_FEATURE_CUDA_DKR_NEUTRAL_RECIPES").is_some();
    let cudaforge_catalogue = env::var_os("CARGO_FEATURE_CUDA_DKR_CUDAFORGE_CATALOG").is_some();
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA_DKR_CUDAFORGE_CATALOG");
    println!("cargo:rerun-if-env-changed={CUDAFORGE_ENV}");
    if cudaforge_catalogue && (!neutral_recipes || !cubin_enabled) {
        return Err("Cudaforge catalogue requires verified neutral recipes and embedded CUBIN".into());
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA_DKR_NEUTRAL_RECIPES");
    println!("cargo:rerun-if-env-changed={RECIPES_ENV}");
    println!("cargo:rerun-if-changed=src/cuda_backend/dkr_s55b_build.rs");
    if neutral_recipes && !neutral_mmq {
        return Err("cuda-dkr-neutral-recipes requires cuda-dkr-neutral-mmq".into());
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA_DKR_NEUTRAL_MMQ");
    println!("cargo:rerun-if-env-changed={MMQ_ENV}");
    if neutral_mmq && !strict {
        return Err("cuda-dkr-neutral-mmq requires cuda-dkr-embedded".into());
    }
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
                    || rec["ptxas_version"].as_str() != cubin_index["ptxas_version"].as_str()
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
    // S5.5a: table-only producer contract. The runtime contains no SM61
    // shared-memory equation or tile candidate list under this feature.
    if neutral_mmq {
        let original = PathBuf::from(env::var_os(EMBED_ENV)
            .ok_or("missing embedded native catalogue")?).canonicalize()?;
        let source_index = original.join(native_catalog::INDEX);
        // S5.1 accepts an absent index as a sealed legacy catalogue.
        // Preserve this distinction instead of generating an index inside
        // the immutable native packages.
        let index_sha = if source_index.is_file() {
            Some(format!("{:x}", Sha256::digest(fs::read(&source_index)?)))
        } else {
            None
        };
        let index_mode = if index_sha.is_some() { "present" } else { "legacy-absent" };
        let meta_root = PathBuf::from(env::var_os(MMQ_ENV)
            .ok_or("S5.5a requires DKR_EMBED_S55_MMQ_DIR at build time")?);
        if !meta_root.is_absolute() {
            return Err("S5.5a MMQ recipe directory must be absolute".into());
        }
        let meta_root = meta_root.canonicalize()?;
        let metadata = meta_root.join("manifest.json");
        println!("cargo:rerun-if-changed={}", metadata.display());
        let data = fs::read(&metadata)?;
        if data.len() > 256 * 1024 {
            return Err("S5.5a recipe manifest too large".into());
        }
        let v: serde_json::Value = serde_json::from_slice(&data)?;
        let index_matches = match (index_sha.as_deref(), v.get("source_index_sha256")) {
            (Some(expected), Some(serde_json::Value::String(actual))) => actual == expected,
            (None, Some(serde_json::Value::Null)) => true,
            _ => false,
        };
        if v["schema"].as_str() != Some(MMQ_SCHEMA)
            || !index_matches
            || v["source_index_mode"].as_str() != Some(index_mode)
            || v["target_cc"].as_u64() != Some(u64::from(contract::CC))
        {
            return Err("S5.5a schema/source-index-state/target mismatch".into());
        }
        let meta_codecs = v["codecs"].as_object().ok_or("missing MMQ codecs")?;
        if meta_codecs.len() != contract::MMVQ_CODECS.len() {
            return Err("S5.5a incomplete/extra MMQ codec family".into());
        }
        let number = |name: &str| -> Result<u32, Box<dyn std::error::Error>> {
            let n = v[name].as_u64().ok_or("MMQ global field is not an unsigned integer")?;
            Ok(u32::try_from(n)?)
        };
        let row_tile = number("row_tile")?;
        let padding = number("quant_padding")?;
        let warp = number("warp")?;
        let max_shared = number("max_shared_bytes")?;
        if row_tile == 0 || row_tile > 1024
            || padding == 0 || padding > 65536
            || warp == 0 || warp > 64 || max_shared == 0 || max_shared > 65536
        {
            return Err("S5.5a out-of-range general MMQ geometry".into());
        }
        let parse_block = |name: &str| -> Result<(u32,u32,u32), Box<dyn std::error::Error>> {
            let arr = v[name].as_array().ok_or("MMQ block must be a 3-vector")?;
            if arr.len() != 3 { return Err("MMQ block must have 3 dimensions".into()); }
            let get = |i: usize| -> Result<u32, Box<dyn std::error::Error>> {
                Ok(u32::try_from(arr[i].as_u64().ok_or("invalid MMQ block entry")?)?)
            };
            let block = (get(0)?,get(1)?,get(2)?);
            let product = u64::from(block.0)*u64::from(block.1)*u64::from(block.2);
            if product == 0 || product > 1024 {
                return Err("MMQ block has illegal CUDA thread count".into());
            }
            Ok(block)
        };
        let qb = parse_block("quant_block")?;
        let mb = parse_block("matrix_block")?;
        let mut generated = Vec::new();
        for &codec in contract::MMVQ_CODECS {
            let record = meta_codecs.get(codec).ok_or("S5.5a missing declared codec")?;
            let family = format!("mmq_instance_{codec}");
            if record["family"].as_str() != Some(family.as_str()) {
                return Err(format!("S5.5a wrong family for {codec}").into());
            }
            let ptx = original.join(&family).join(format!("{family}.ptx"));
            let real_sha = format!("{:x}",Sha256::digest(fs::read(&ptx)?));
            if record["ptx_sha256"].as_str() != Some(real_sha.as_str()) {
                return Err(format!("S5.5a PTX provenance mismatch for {family}").into());
            }
            let source_manifest: serde_json::Value = serde_json::from_slice(
                &fs::read(original.join(&family).join("manifest.json"))?)?;
            let exported = source_manifest["mmq_entries"].as_object()
                .ok_or("qualified native MMQ template map missing")?;
            if exported.len() != 16 {
                return Err(format!("S5.5a native MMQ symbol map incomplete: {family}").into());
            }
            let tiles = record["tiles"].as_array().ok_or("MMQ tile list missing")?;
            if tiles.len() != 8 {
                return Err(format!("S5.5a invalid tile count for {family}").into());
            }
            let mut generated_tiles = Vec::new();
            let mut prev_width = 0;
            for tile in tiles {
                let width = u32::try_from(tile["width"].as_u64()
                    .ok_or("invalid MMQ tile width")?)?;
                let shared = u32::try_from(tile["shared_bytes"].as_u64()
                    .ok_or("invalid MMQ shared memory")?)?;
                if width <= prev_width || width > 1024 || shared == 0 || shared > max_shared {
                    return Err(format!("S5.5a illegal MMQ tile for {family}").into());
                }
                for checked in [0,1] {
                    let key = format!("{width}:{checked}");
                    if exported.get(&key).and_then(|s| s.as_str()).is_none() {
                        return Err(format!("S5.5a missing actual symbol key {family}/{key}").into());
                    }
                }
                generated_tiles.push(format!(
                    "dkr_kernel_registry::mmq_recipe::MmqTile {{ width:{width}, shared_bytes:{shared} }}"
                ));
                prev_width = width;
            }
            generated.push(format!(
                "dkr_kernel_registry::mmq_recipe::MmqCodec {{ name:{codec:?}, tiles:&[{}] }}",
                generated_tiles.join(",")
            ));
        }
        let generated = format!(
            "pub(super) const RECIPE: dkr_kernel_registry::mmq_recipe::MmqRecipe = \
             dkr_kernel_registry::mmq_recipe::MmqRecipe {{ \
             row_tile:{row_tile}, quant_padding:{padding}, warp:{warp}, \
             max_shared_bytes:{max_shared}, \
             quant_block:({},{},{}), matrix_block:({},{},{}), \
             codecs:&[{}] }};\n",
            qb.0,qb.1,qb.2,mb.0,mb.1,mb.2,generated.join(",")
        );
        fs::write(out.join("dkr_neutral_mmq.rs"),generated)?;
        println!("cargo:warning=DKR_R124_S55A_EMBEDDED_MMQ=10");
    }
    if neutral_recipes {
        let root = PathBuf::from(env::var_os(EMBED_ENV)
            .ok_or("S5.5b requires sealed DKR_EMBED_NATIVE_DIR")?).canonicalize()?;
        let descriptor = PathBuf::from(env::var_os(RECIPES_ENV)
            .ok_or("S5.5b requires DKR_EMBED_S55B_RECIPES_DIR")?);
        let mmq_dir = PathBuf::from(env::var_os(MMQ_ENV)
            .ok_or("S5.5b requires DKR_EMBED_S55_MMQ_DIR")?);
        neutral_recipe_build::generate(&out,&root,&descriptor,&mmq_dir,contract::CC)?;
    }
    if cudaforge_catalogue {
        // Strict, cross-attested derived manifests: no runtime producer paths,
        // no text policy inferred from GPU type, no mutation of sealed inputs.
        let dir = PathBuf::from(env::var_os(CUDAFORGE_ENV)
            .ok_or("S5.5c requires DKR_EMBED_S55C_CUDAFORGE_DIR")?);
        if !dir.is_absolute() {
            return Err("S5.5c catalogue path must be absolute".into());
        }
        let dir = dir.canonicalize()?;
        let bundle_path = dir.join("catalogue.json");
        println!("cargo:rerun-if-changed={}",bundle_path.display());
        let bytes = fs::read(&bundle_path)?;
        if bytes.len()>256*1024 {return Err("S5.5c bundle manifest too large".into());}
        let bundle:serde_json::Value=serde_json::from_slice(&bytes)?;
        let cc=contract::CC;
        if bundle["schema"].as_str()!=Some(CUDAFORGE_SCHEMA)
            || bundle["producer"].as_str()!=Some("cudaforge")
            || bundle["producer_command"].as_str()!=Some("dkr-cudaforge-catalog")
            || bundle["target_cc"].as_u64()!=Some(u64::from(cc))
            || bundle["family_count"].as_u64()!=Some(25)
            || bundle["recipe_count"].as_u64()!=Some(4)
            || bundle["mmq_codec_count"].as_u64()!=Some(10)
        {
            return Err("S5.5c catalogue schema/producer/target cardinality mismatch".into());
        }
        let source_recipe = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tools/dkr-s31-builder/recipes/sm61.json");
        println!("cargo:rerun-if-changed={}",source_recipe.display());
        let spec_sha=format!("{:x}",Sha256::digest(fs::read(&source_recipe)?));
        if bundle["recipe_spec_sha256"].as_str()!=Some(spec_sha.as_str()) {
            return Err("S5.5c source recipe spec digest does not match producer".into());
        }
        let mmq = dir.join("mmq").canonicalize()?;
        let recipes = dir.join("recipes").canonicalize()?;
        let runtime_mmq = PathBuf::from(env::var_os(MMQ_ENV)
            .ok_or("S5.5c missing S5.5a recipe input")?).canonicalize()?;
        let runtime_recipes = PathBuf::from(env::var_os(RECIPES_ENV)
            .ok_or("S5.5c missing S5.5b recipe input")?).canonicalize()?;
        if mmq!=runtime_mmq || recipes!=runtime_recipes {
            return Err("S5.5c runtime recipe paths do not match attested Cudaforge outputs".into());
        }
        let mmq_bytes=fs::read(mmq.join("manifest.json"))?;
        let recipes_bytes=fs::read(recipes.join("manifest.json"))?;
        let cubins = PathBuf::from(env::var_os(CUBIN_ENV)
            .ok_or("S5.5c CUBIN inventory input missing")?).canonicalize()?;
        let cubin_bytes=fs::read(cubins.join("manifest.json"))?;
        let mmq_sha=format!("{:x}",Sha256::digest(&mmq_bytes));
        let recipes_sha=format!("{:x}",Sha256::digest(&recipes_bytes));
        let cubin_sha=format!("{:x}",Sha256::digest(&cubin_bytes));
        if bundle["mmq_manifest_sha256"].as_str()!=Some(mmq_sha.as_str())
            || bundle["recipes_manifest_sha256"].as_str()!=Some(recipes_sha.as_str())
            || bundle["cubin_manifest_sha256"].as_str()!=Some(cubin_sha.as_str()) {
            return Err("S5.5c derived recipe/CUBIN manifest digests inconsistent".into());
        }
        let mmq_v:serde_json::Value=serde_json::from_slice(&mmq_bytes)?;
        let recipes_v:serde_json::Value=serde_json::from_slice(&recipes_bytes)?;
        if bundle["source_seal_sha256"]!=mmq_v["source_seal_sha256"]
            || bundle["source_seal_sha256"]!=recipes_v["source_seal_sha256"]
            || bundle["source_index_sha256"]!=mmq_v["source_index_sha256"]
            || bundle["source_index_sha256"]!=recipes_v["source_index_sha256"]
            || bundle["source_index_mode"]!=mmq_v["source_index_mode"]
            || bundle["source_index_mode"]!=recipes_v["source_index_mode"]
            || mmq_v["target_cc"]!=bundle["target_cc"]
            || recipes_v["target_cc"]!=bundle["target_cc"]
        {
            return Err("S5.5c receipt/PTX/index/target provenance mismatch".into());
        }
        // The two sub-manifests also pass all existing S5.5a/S5.5b
        // build verifiers: 10×8 MMQ tiles, PTX digests and entrypoint ABI.
        fs::write(out.join("dkr_cudaforge_catalogue.rs"), format!(
            "pub(super) const PRODUCER:&str=\"cudaforge\";\n\
             pub(super) const TARGET_CC:u32={cc};\n\
             pub(super) const ATTESTATION_SHA256:&str={:?};\n",
            format!("{:x}",Sha256::digest(&bytes))
        ))?;
        println!("cargo:warning=DKR_R124_S55C_CUDAFORGE_ADMITTED=1 families=25 recipes=4");
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
