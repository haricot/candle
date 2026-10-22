use anyhow::{bail, ensure, Context, Result};
use candle::{DType, Device};
use candle_nn::VarBuilder;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

pub const FIRERED_SHA: &str = "9d67fa8cc3c0d54b3a3f83b2f6905d98a2740693635296bd2fde5679f3ab2f11";
pub const SMART_SHA: &str = "ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be";
pub const CMVN_SHA: &str = "c87f6f13edf0f0ec7535ddfc9cc3387d9268cb234b70182d566c5e2edf3ca473";

#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    model: String,
    source_sha256: String,
    #[serde(default)]
    cmvn_source_sha256: Option<String>,
    files: BTreeMap<String, String>,
}

pub fn load(root: &Path, model: &str, device: &Device) -> Result<VarBuilder<'static>> {
    let manifest: Manifest = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
    ensure!(
        manifest.schema_version == 1 && manifest.model == model,
        "Model package contract mismatch"
    );
    let expected = match model {
        "fireredvad-stream" => FIRERED_SHA,
        "smart-turn-v3.2-fp32" => SMART_SHA,
        _ => bail!("Unknown native model {model}"),
    };
    ensure!(
        manifest.source_sha256 == expected,
        "Unexpected upstream model identity"
    );
    if model == "fireredvad-stream" {
        ensure!(
            manifest.cmvn_source_sha256.as_deref() == Some(CMVN_SHA),
            "Unexpected CMVN identity"
        );
    }
    ensure!(
        manifest.files.len() == 2
            && manifest.files.contains_key("model.safetensors")
            && manifest.files.contains_key("config.json"),
        "Incomplete model package"
    );
    for (file, expected) in &manifest.files {
        let bytes = fs::read(root.join(file)).with_context(|| format!("Missing {file}"))?;
        ensure!(
            format!("{:x}", Sha256::digest(&bytes)) == *expected,
            "Model file digest mismatch: {file}"
        );
    }
    // Own the weights rather than mmap them: the source files cannot be changed
    // under a live model, and no unsafe mmap contract is exposed to the caller.
    Ok(VarBuilder::from_buffered_safetensors(
        fs::read(root.join("model.safetensors"))?,
        DType::F32,
        device,
    )?)
}
