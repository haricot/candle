//! Opt-in execution of ONE A/B/A-validated MotionBricks bundle imported by Flow ASD.
//! Neither a generic Candle dispatch nor per-signature benchmark evidence.
use anyhow::{bail, Context, Result};
use candle_core::Device;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use crate::motionbricks_runtime::MotionBricksConv1dPolicy;

const SCHEMA: &str = "flow-frontier.wham.motionbricks.asd-bundle.v1";
const PROFILE: &str = "fixed16-no-text-all-root-constraints";
const BUNDLE: &str = "sm61_k1_gemm_k3_l32_direct";

fn sha_file(path: &Path, limit: u64) -> Result<String> {
    let meta = fs::symlink_metadata(path).with_context(|| format!("ASD file metadata {}", path.display()))?;
    if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > limit {
        bail!("ASD file absent, symlink or too large: {}", path.display());
    }
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn required<'a>(doc: &'a Value, field: &str) -> Result<&'a str> {
    doc.get(field).and_then(Value::as_str).context(format!("ASD missing string {field}"))
}

fn check_measurements(doc: &Value) -> Result<()> {
    let aba = doc.get("aba").and_then(Value::as_object).context("ASD missing A/B/A")?;
    if aba.len() != 5 { bail!("ASD requires five measured A/B/A components"); }
    for name in ["root.decoder_only", "vq.decoder_only", "root.full", "vq.full", "runtime.full"] {
        let row = aba.get(name).context(format!("ASD missing {name}"))?;
        let number = |key: &str| -> Result<f64> {
            let x = row.get(key).and_then(Value::as_f64).context(format!("ASD missing {name}.{key}"))?;
            if !x.is_finite() || x <= 0.0 { bail!("ASD invalid {name}.{key}"); }
            Ok(x)
        };
        let (a1, b, a2) = (number("a1_us")?, number("b_us")?, number("a2_us")?);
        if b >= a1 || b >= a2 { bail!("ASD cannot split/nonwinning A/B/A policy {name}"); }
        let mid = number("baseline_midpoint_us")?;
        if (mid - (a1 + a2) / 2.0).abs() > 0.003 { bail!("ASD incoherent midpoint {name}"); }
        let drift = row.get("control_drift_pct").and_then(Value::as_f64).context("ASD missing drift")?;
        if !drift.is_finite() || !(0.0..=5.0).contains(&drift) {
            bail!("ASD unstable controls {name}");
        }
        if name == "runtime.full" && drift > 1.0 { bail!("ASD unstable runtime controls"); }
    }
    Ok(())
}

fn check_original_log(log: &str, doc: &Value) -> Result<()> {
    for marker in [
        "MOTIONBRICKS_ROOT_NUMERICAL_GATE=PASS", "MOTIONBRICKS_POSE_NUMERICAL_GATE=PASS",
        "MOTIONBRICKS_VQVAE_DECODER_GATE=PASS", "V0_3_GATE=PASS",
        "V0_3_R2_TORCH_ORACLE_GATE=PASS",
        "MOTIONBRICKS_V0_3_2_R2_DISPATCH_NUMERICAL_GATE=PASS",
        "MOTIONBRICKS_V0_3_2_R2_DECODER_ABA_GATE=PASS",
        "MOTIONBRICKS_V0_3_2_R2_RUNTIME_ABA_GATE=PASS",
        "MOTIONBRICKS_V0_3_2_R2_PROMOTION_GATE=PASS", "exit_code=0",
        "dispatch_inventory_conv_instances=42", "dispatch_inventory_k1_hits=16",
        "dispatch_inventory_k3_l32_direct_hits=4", "candidate_policy=sm61_k1_gemm_k3_l32_direct",
        "validated_dispatch_sm=61", "validated_cudnn_key=9.1.0.2", "dtype=f32",
    ] {
        if log.lines().filter(|line| *line == marker).count() != 1 {
            bail!("ASD missing/ambiguous original evidence: {marker}");
        }
    }
    for name in ["root.decoder_only", "vq.decoder_only", "root.full", "vq.full", "runtime.full"] {
        let prefix = format!("aba_summary={name} ");
        let lines = log.lines().filter(|line| line.starts_with(&prefix)).collect::<Vec<_>>();
        if lines.len() != 1 { bail!("ASD source log missing/duplicate A/B/A {name}"); }
        let tokens = lines[0].split_whitespace().filter_map(|part| part.split_once('='))
            .collect::<std::collections::HashMap<_, _>>();
        if tokens.get("candidate_beats_both") != Some(&"true") {
            bail!("ASD source did not beat both A controls: {name}");
        }
        for (source_key, policy_key) in [("A1_median_us", "a1_us"),
                                         ("B_median_us", "b_us"),
                                         ("A2_median_us", "a2_us")] {
            let original = tokens.get(source_key)
                .context(format!("ASD source missing {name}.{source_key}"))?
                .parse::<f64>()?;
            let stored = doc["aba"][name][policy_key].as_f64()
                .context(format!("ASD policy missing {name}.{policy_key}"))?;
            if !original.is_finite() || (original - stored).abs() > 0.000_001 {
                bail!("ASD policy A/B/A does not match original log {name}.{source_key}");
            }
        }
    }
    Ok(())
}

/// Fail on corrupt/unbound evidence; FALLBACK on missing hardware-key match.
/// Hardware/cuDNN versions are caller-declared: NOT a runtime hardware attestation.
/// This function can only choose the complete bundle; no independent route claim.
pub fn resolve_motionbricks_asd_policy(
    policy_file: &Path,
    weights: &Path,
    reference: &Path,
    device: &Device,
    declared_sm: Option<u32>,
    declared_cudnn: Option<&str>,
) -> Result<MotionBricksConv1dPolicy> {
    let policy_sha = sha_file(policy_file, 128 * 1024)?;
    let manifest = policy_file.parent().context("ASD policy has no parent")?.join("MANIFEST.sha256");
    let manifest_sha = sha_file(&manifest, 4 * 1024)?;
    let manifest_text = fs::read_to_string(&manifest)?;
    let expected_line = format!("{policy_sha}  motionbricks-asd-policy.json\n");
    if policy_file.file_name().and_then(|x| x.to_str()) != Some("motionbricks-asd-policy.json")
        || manifest_text != expected_line {
        bail!("ASD bundle manifest does not seal the exact policy filename/content");
    }
    let doc: Value = serde_json::from_slice(&fs::read(policy_file)?)?;
    for (key, expected) in [
        ("schema", SCHEMA), ("model", "haricot-motionbricks-v0.3.4"),
        ("profile", PROFILE), ("scope", "motionbricks_local_only"),
        ("dtype", "F32"), ("decision", BUNDLE),
        ("decision_kind", "atomic_measured_bundle"),
        ("fallback", "baseline_candle_conv1d"),
    ] {
        if required(&doc, key)? != expected { bail!("ASD unsupported {key}"); }
    }
    if doc["required_sm"] != 61 || doc["required_cudnn"] != "9.1.0.2"
        || doc["observed_calls"] != 42 || doc["unique_signatures"] != 15
        || doc["bundle_route_counts"]["k1_explicit_gemm"] != 16
        || doc["bundle_route_counts"]["k3_l32_cudnn_direct"] != 4
        || doc["independent_route_speedups_validated"] != false
        || doc["individual_routes_dispatch_eligible"] != false
        || doc["candle_core_changes"] != 0 || doc["global_candle_dispatch_changes"] != 0
        || doc["numerical"]["candle_vs_candidate"] != "PASS"
        || doc["numerical"]["candidate_vs_torch"] != "PASS"
        || doc["numerical"]["prior_cpu_cuda_intermediate_strict_parity"] != "FAIL_RETAINED"
        || doc["numerical"]["atol"] != 0.0005 || doc["numerical"]["rmse_tol"] != 0.0001 {
        bail!("ASD bundle shape/route/numeric scope altered");
    }
    check_measurements(&doc)?;
    let p = &doc["provenance"];
    let receipt_path = Path::new(required(p, "shape_overlay_receipt_path")?);
    if sha_file(receipt_path, 16 * 1024 * 1024)? != required(p, "shape_overlay_receipt_sha256")? {
        bail!("ASD Candle overlay receipt differs from catalog/import");
    }
    let catalog_path = Path::new(required(p, "shape_catalog_path")?);
    if sha_file(catalog_path, 4 * 1024 * 1024)? != required(p, "shape_catalog_sha256")? {
        bail!("ASD original model shape catalog changed/missing");
    }
    let aba_path = Path::new(required(p, "aba_log_path")?);
    if sha_file(aba_path, 4 * 1024 * 1024)? != required(p, "aba_log_sha256")? {
        bail!("ASD A/B/A source log changed/missing");
    }
    let log = fs::read_to_string(aba_path)?;
    check_original_log(&log, &doc)?;
    if sha_file(weights, 2 * 1024 * 1024 * 1024)? != required(p, "weights_sha256")?
        || sha_file(reference, 2 * 1024 * 1024 * 1024)? != required(p, "torch_reference_sha256")? {
        bail!("ASD weights/reference differ from measured import");
    }
    let selected = matches!(device, Device::Cuda(_)) && declared_sm == Some(61)
        && declared_cudnn == Some("9.1.0.2");
    println!("FLOW_MOTIONBRICKS_ASD_POLICY={} scope=atomic_bundle policy_sha256={} manifest_sha256={} hardware_identity=CALLER_DECLARED_NOT_ATTESTED individual_route_speedups=NOT_MEASURED",
             if selected { "SELECTED" } else { "FALLBACK" }, policy_sha, manifest_sha);
    Ok(if selected { MotionBricksConv1dPolicy::Sm61Validated }
       else { MotionBricksConv1dPolicy::Baseline })
}
