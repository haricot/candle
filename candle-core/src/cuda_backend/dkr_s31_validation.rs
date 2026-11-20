use super::contract;
use serde_json::Value;
use sha2::{Digest, Sha256};
use dkr_kernel_registry::VerifiedArtifact;
use std::collections::BTreeMap;

pub(super) fn validate(manifest: &Value, name: &str, ptx: &str) -> std::result::Result<(), String> {
    let sha = format!("{:x}", Sha256::digest(ptx.as_bytes()));
    let flags: Vec<&str> = manifest["flags"]
        .as_array()
        .map(|v| v.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if manifest["schema"].as_str() != Some(contract::SCHEMA)
        || manifest["base_revision"].as_str() != Some(contract::BASE)
        || manifest["target_cc"].as_u64() != Some(u64::from(contract::CC))
        || manifest["family"].as_str() != Some(name)
        || manifest["sha256"].as_str() != Some(sha.as_str())
        || flags != contract::FLAGS
        || !contract::is_sm61_ptx(ptx)
        || (name == "mmvq_gguf" && !contract::has_mmvq_entries(ptx))
        || (matches!(name, "conv" | "sort") && !contract::has_conv_sort_entries(ptx, name))
        || (name.starts_with("moe_")
            && contract::moe_entries(ptx, name)
                .and_then(|entries| serde_json::to_value(entries).ok())
                .is_none_or(|entries| entries != manifest["moe_entries"]))
        || (name.starts_with("mmq_")
            && contract::mmq_entries(ptx, name)
                .and_then(|entries| serde_json::to_value(entries).ok())
                .is_none_or(|entries| entries != manifest["mmq_entries"]))
    {
        return Err(format!("Candle DKR S3.1 invalid package for {name}"));
    }
    Ok(())
}

// The S3.1 validator is a legacy package adapter, not a device-selection
// policy. Do not relax its qualified source, symbols or flags when adding
// a new native manifest schema.
pub(super) type NativeSymbols = Option<BTreeMap<String, String>>;

/// One verification entrypoint for build-time embedding and filesystem loading.
pub(super) fn verify_package(
    manifest: &Value, name: &str, ptx: String,
) -> Result<(VerifiedArtifact, NativeSymbols), String> {
    validate(manifest, name, &ptx)?;
    let target = manifest["target_cc"].as_u64()
        .and_then(|cc| u32::try_from(cc).ok())
        .ok_or("invalid native package target")?;
    let digest = manifest["sha256"].as_str().ok_or("missing native package digest")?;
    let artifact = VerifiedArtifact::new(ptx, target, digest)?;
    // The legacy validator has already checked these exact maps against the PTX.
    let field = if name.starts_with("moe_") {
        Some("moe_entries")
    } else if name.starts_with("mmq_") {
        Some("mmq_entries")
    } else {
        None
    };
    let symbols = field.map(|key| serde_json::from_value::<BTreeMap<String, String>>(manifest[key].clone()))
        .transpose().map_err(|e| format!("invalid native entry map: {e}"))?;
    if let Some(map) = &symbols {
        if map.values().any(|symbol| !artifact.has_entry(symbol)) {
            return Err("native entry map references an absent PTX symbol".into());
        }
    }
    Ok((artifact, symbols))
}
