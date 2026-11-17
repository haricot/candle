use super::contract;
use serde_json::Value;
use sha2::{Digest, Sha256};

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
