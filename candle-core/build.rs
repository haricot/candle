use std::{env, fs, path::PathBuf};

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

const EMBED_ENV: &str = "DKR_EMBED_NATIVE_DIR";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-env-changed={EMBED_ENV}");
    println!("cargo:rerun-if-changed=src/cuda_backend/dkr_s31_validation.rs");
    println!("cargo:rerun-if-changed=src/cuda_backend/dkr_native_catalog.rs");
    println!("cargo:rerun-if-changed=../candle-kernels/dkr_s31_contract.rs");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut rows = Vec::new();
    let mut root_literal = "None".to_owned();
    let mut recipe_literal = "None".to_owned();
    let mut fp8_literal = "None".to_owned();
    let mut fp4_literal = "None".to_owned();
    println!("cargo:rerun-if-changed=../candle-kernels/dkr_fp4_recipe.rs");
    println!("cargo:rerun-if-changed=../candle-kernels/dkr_fp8_recipe.rs");
    if let Some(root) = env::var_os(EMBED_ENV) {
        let root = PathBuf::from(root).canonicalize()?;
        assert!(
            env::var_os("CARGO_FEATURE_CUDA_DKR_S31").is_some(),
            "{EMBED_ENV} requires cuda-dkr-s31"
        );
        root_literal = format!("Some({:?})", root.to_str().ok_or("non-UTF8 embedded path")?);
        // Watch the directory as well, so adding an index invalidates a legacy build.
        println!("cargo:rerun-if-changed={}", root.display());
        let catalog = native_catalog::NativeCatalog::load(&root)?;
        recipe_literal = format!("Some({:?})", catalog.recipe.id());
        fp8_literal = format!("Some({:?})", catalog.recipe.fp8());
        fp4_literal = format!("Some({:?})", catalog.recipe.fp4());
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
        }
    }
    fs::write(
        out.join("dkr_native_embedded.rs"),
        format!(
            "pub const ROOT: Option<&str> = {root_literal};\n\
         pub const RECIPE: Option<&str> = {recipe_literal};\n\
         pub const FP8_RECIPE: Option<Fp8Recipe> = {fp8_literal};\n\
         pub const FP4_RECIPE: Option<Fp4Recipe> = {fp4_literal};\n\
         pub fn modules() -> Vec<(&'static str, VerifiedArtifact, \
         Option<std::collections::BTreeMap<String,String>>)> {{ vec![{}] }}\n",
            rows.join(",\n")
        ),
    )?;
    Ok(())
}
