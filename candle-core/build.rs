use dkr_kernel_registry::VerifiedArtifact;
use std::{env, fs, path::PathBuf};

#[allow(dead_code)]
#[path = "../candle-kernels/dkr_s31_contract.rs"]
mod contract;
#[path = "src/cuda_backend/dkr_s31_validation.rs"]
mod validation;

const EMBED_ENV: &str = "DKR_EMBED_NATIVE_DIR";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-env-changed={EMBED_ENV}");
    println!("cargo:rerun-if-changed=src/cuda_backend/dkr_s31_validation.rs");
    println!("cargo:rerun-if-changed=../candle-kernels/dkr_s31_contract.rs");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut rows = Vec::new();
    let mut root_literal = "None".to_owned();
    if let Some(root) = env::var_os(EMBED_ENV) {
        let root = PathBuf::from(root).canonicalize()?;
        assert!(
            env::var_os("CARGO_FEATURE_CUDA_DKR_S31").is_some(),
            "{EMBED_ENV} requires cuda-dkr-s31"
        );
        root_literal = format!("Some({:?})", root.to_str().ok_or("non-UTF8 embedded path")?);
        for name in contract::FAMILIES {
            let manifest = root.join(name).join("manifest.json");
            let artifact = root.join(name).join(format!("{name}.ptx"));
            println!("cargo:rerun-if-changed={}", manifest.display());
            println!("cargo:rerun-if-changed={}", artifact.display());
            assert!(
                fs::metadata(&manifest)?.len() <= 256 * 1024,
                "native manifest exceeds budget"
            );
            assert!(
                fs::metadata(&artifact)?.len() <= dkr_kernel_registry::MAX_PTX_BYTES as u64,
                "native PTX exceeds budget"
            );
            let metadata: serde_json::Value = serde_json::from_slice(&fs::read(manifest)?)?;
            let ptx = fs::read_to_string(artifact)?;
            validation::validate(&metadata, name, &ptx)?;
            let verified =
                VerifiedArtifact::new(ptx, contract::CC, metadata["sha256"].as_str().unwrap())?;
            let copy = out.join(format!("dkr-native-{name}.ptx"));
            fs::write(&copy, verified.ptx())?;
            let entries = verified
                .entry_names()
                .map(|s| format!("{s:?}.into()"))
                .collect::<Vec<_>>()
                .join(",");
            let symbols = if name.starts_with("moe_") {
                Some(contract::moe_entries(verified.ptx(), name).ok_or("invalid MoE entries")?)
            } else if name.starts_with("mmq_") {
                Some(contract::mmq_entries(verified.ptx(), name).ok_or("invalid MMQ entries")?)
            } else {
                None
            };
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
                 vec![{entries}]), {symbols})", contract::CC, verified.key().sha256()
            ));
        }
    }
    fs::write(
        out.join("dkr_native_embedded.rs"),
        format!(
            "pub const ROOT: Option<&str> = {root_literal};\n\
         pub fn modules() -> Vec<(&'static str, VerifiedArtifact, \
         Option<std::collections::BTreeMap<String,String>>)> {{ vec![{}] }}\n",
            rows.join(",\n")
        ),
    )?;
    Ok(())
}
