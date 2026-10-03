use candle_core::{Error, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

const INPUT_HEADER_V1: &str = "ASD-CUDA-PERFORMANCE-EVIDENCE-V1";
const INPUT_HEADER_V2: &str = "ASD-CUDA-PERFORMANCE-EVIDENCE-V2";
const OUTPUT_HEADER: &str = "ASD-CUDA-PERFORMANCE-QUALIFICATION-V1";
const PROTOCOL_ID: &str = "time_equivalent_settled_alternating_v1";

#[derive(Debug)]
struct Evidence {
    path: PathBuf,
    sha256: String,
    header: String,
    fields: BTreeMap<String, String>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn read_evidence(path: &Path) -> Result<Evidence> {
    let bytes = std::fs::read(path)
        .map_err(|err| Error::Msg(format!("failed to read {}: {err}", path.display())))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|err| Error::Msg(format!("evidence is not UTF-8 {}: {err}", path.display())))?;

    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| Error::Msg(format!("empty performance evidence {}", path.display())))?;
    if header != INPUT_HEADER_V1 && header != INPUT_HEADER_V2 {
        return Err(Error::Msg(format!(
            "invalid performance evidence header {header:?} in {}",
            path.display()
        )));
    }

    let mut fields = BTreeMap::new();
    for (index, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(Error::Msg(format!(
                "invalid line {} in {}",
                index + 2,
                path.display()
            )));
        };
        if key.is_empty() || value.is_empty() {
            return Err(Error::Msg(format!(
                "empty key/value on line {} in {}",
                index + 2,
                path.display()
            )));
        }
        if fields.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(Error::Msg(format!(
                "duplicate field {key:?} in {}",
                path.display()
            )));
        }
    }

    Ok(Evidence {
        path: path.to_path_buf(),
        sha256: sha256_hex(&bytes),
        header: header.to_owned(),
        fields,
    })
}

fn required<'a>(evidence: &'a Evidence, key: &str) -> Result<&'a str> {
    evidence.fields.get(key).map(String::as_str).ok_or_else(|| {
        Error::Msg(format!(
            "missing field {key:?} in {}",
            evidence.path.display()
        ))
    })
}

fn same_field(run1: &Evidence, run2: &Evidence, key: &str) -> Result<String> {
    let value1 = required(run1, key)?;
    let value2 = required(run2, key)?;
    if value1 != value2 {
        return Err(Error::Msg(format!(
            "qualification mismatch for {key}: {} has {value1:?}, {} has {value2:?}",
            run1.path.display(),
            run2.path.display()
        )));
    }
    Ok(value1.to_owned())
}

fn require_value(evidence: &Evidence, key: &str, expected: &str) -> Result<()> {
    let actual = required(evidence, key)?;
    if actual != expected {
        return Err(Error::Msg(format!(
            "{} has {key}={actual:?}, expected {expected:?}",
            evidence.path.display()
        )));
    }
    Ok(())
}

fn parse_args() -> Result<([PathBuf; 2], PathBuf)> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut evidence = Vec::new();
    let mut out = None;
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--evidence" => {
                i += 1;
                evidence
                    .push(PathBuf::from(args.get(i).ok_or_else(|| {
                        Error::Msg("--evidence requires a path".into())
                    })?));
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| Error::Msg("--out requires a path".into()))?,
                ));
            }
            other => return Err(Error::Msg(format!("unknown argument {other:?}"))),
        }
        i += 1;
    }

    let evidence: [PathBuf; 2] = evidence.try_into().map_err(|paths: Vec<PathBuf>| {
        Error::Msg(format!(
            "exactly two --evidence files are required, got {}",
            paths.len()
        ))
    })?;
    let out = out.ok_or_else(|| Error::Msg("--out is required".into()))?;
    Ok((evidence, out))
}

fn main() -> Result<()> {
    let (paths, out) = parse_args()?;
    let run1 = read_evidence(&paths[0])?;
    let run2 = read_evidence(&paths[1])?;

    if run1.sha256 == run2.sha256 {
        return Err(Error::Msg(
            "the two qualification evidences must be distinct".into(),
        ));
    }

    if run1.header != run2.header {
        return Err(Error::Msg(format!(
            "qualification evidence schema mismatch: {} vs {}",
            run1.header, run2.header
        )));
    }
    if run1.header == INPUT_HEADER_V2 {
        require_value(&run1, "telemetry_observe_only", "true")?;
        require_value(&run2, "telemetry_observe_only", "true")?;
    }

    for run in [&run1, &run2] {
        require_value(run, "status", "PASS")?;
        require_value(run, "parity_pass", "true")?;
        require_value(run, "drift_pass", "true")?;
        require_value(run, "warmup_policy", "time_equivalent_per_backend")?;
        require_value(
            run,
            "settling_policy",
            "time_equivalent_pair_non_authoritative",
        )?;
        require_value(run, "sequence", "a1,b1,a2,b2,a3,b3")?;
    }

    let implementation_id = same_field(&run1, &run2, "implementation_id")?;
    let architecture = same_field(&run1, &run2, "architecture")?;
    let gpu_uuid = same_field(&run1, &run2, "gpu_uuid")?;
    let artifact_a_sha256 = same_field(&run1, &run2, "artifact_a_sha256")?;
    let artifact_b_sha256 = same_field(&run1, &run2, "artifact_b_sha256")?;
    let parity_evidence_sha256 = same_field(&run1, &run2, "parity_evidence_sha256")?;
    let warmup_ms = same_field(&run1, &run2, "warmup_ms")?;
    let settling_ms_per_artifact = same_field(&run1, &run2, "settling_ms_per_artifact")?;
    let timed_samples = same_field(&run1, &run2, "timed_samples")?;
    let launches_per_sample = same_field(&run1, &run2, "launches_per_sample")?;
    let min_speedup_x = same_field(&run1, &run2, "min_speedup_x")?;
    let decision = same_field(&run1, &run2, "decision")?;

    let performance_evidence_schema = run1.header.clone();

    if decision != "PROMOTE_CANDIDATE" && decision != "REJECT_CANDIDATE" {
        return Err(Error::Msg(format!(
            "unsupported qualified decision {decision:?}"
        )));
    }

    let run1_speedup_x = required(&run1, "speedup_x")?;
    let run2_speedup_x = required(&run2, "speedup_x")?;
    let run1_a_us = required(&run1, "a_consensus_us")?;
    let run1_b_us = required(&run1, "b_consensus_us")?;
    let run2_a_us = required(&run2, "a_consensus_us")?;
    let run2_b_us = required(&run2, "b_consensus_us")?;

    let telemetry_lines = if performance_evidence_schema == INPUT_HEADER_V2 {
        format!(
            "run1_telemetry_status={}\nrun1_gpu_temp_c_max_observed={}\nrun1_sm_clock_mhz_min_observed={}\nrun1_sm_clock_mhz_max_observed={}\nrun1_thermal_state_end={}\nrun1_throttle_state_end={}\nrun2_telemetry_status={}\nrun2_gpu_temp_c_max_observed={}\nrun2_sm_clock_mhz_min_observed={}\nrun2_sm_clock_mhz_max_observed={}\nrun2_thermal_state_end={}\nrun2_throttle_state_end={}\n",
            required(&run1, "telemetry_status")?,
            required(&run1, "gpu_temp_c_max_observed")?,
            required(&run1, "sm_clock_mhz_min_observed")?,
            required(&run1, "sm_clock_mhz_max_observed")?,
            required(&run1, "telemetry_end_thermal_state")?,
            required(&run1, "telemetry_end_throttle_state")?,
            required(&run2, "telemetry_status")?,
            required(&run2, "gpu_temp_c_max_observed")?,
            required(&run2, "sm_clock_mhz_min_observed")?,
            required(&run2, "sm_clock_mhz_max_observed")?,
            required(&run2, "telemetry_end_thermal_state")?,
            required(&run2, "telemetry_end_throttle_state")?,
        )
    } else {
        String::new()
    };

    let closure = format!(
        "{OUTPUT_HEADER}\n\
protocol={PROTOCOL_ID}\n\
performance_evidence_schema={performance_evidence_schema}\n\
{telemetry_lines}implementation_id={implementation_id}\n\
architecture={architecture}\n\
gpu_uuid={gpu_uuid}\n\
artifact_a_sha256={artifact_a_sha256}\n\
artifact_b_sha256={artifact_b_sha256}\n\
parity_evidence_sha256={parity_evidence_sha256}\n\
warmup_ms={warmup_ms}\n\
settling_ms_per_artifact={settling_ms_per_artifact}\n\
timed_samples={timed_samples}\n\
launches_per_sample={launches_per_sample}\n\
min_speedup_x={min_speedup_x}\n\
replications=2\n\
run1_evidence_sha256={}\n\
run2_evidence_sha256={}\n\
run1_a_consensus_us={run1_a_us}\n\
run1_b_consensus_us={run1_b_us}\n\
run1_speedup_x={run1_speedup_x}\n\
run2_a_consensus_us={run2_a_us}\n\
run2_b_consensus_us={run2_b_us}\n\
run2_speedup_x={run2_speedup_x}\n\
status=PASS\n\
decision={decision}\n",
        run1.sha256, run2.sha256
    );
    let closure_sha256 = sha256_hex(closure.as_bytes());

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&out)
        .map_err(|err| {
            Error::Msg(format!(
                "refusing to overwrite qualification closure {}: {err}",
                out.display()
            ))
        })?;
    file.write_all(closure.as_bytes()).map_err(|err| {
        Error::Msg(format!(
            "failed to write qualification closure {}: {err}",
            out.display()
        ))
    })?;

    println!("=== ASD V3 PERFORMANCE QUALIFICATION CLOSURE ===");
    println!("protocol={PROTOCOL_ID}");
    println!("performance_evidence_schema={performance_evidence_schema}");
    println!("implementation_id={implementation_id}");
    println!("gpu_uuid={gpu_uuid}");
    println!("artifact_a_sha256={artifact_a_sha256}");
    println!("artifact_b_sha256={artifact_b_sha256}");
    println!("run1_evidence_sha256={}", run1.sha256);
    println!("run2_evidence_sha256={}", run2.sha256);
    println!("run1_speedup_x={run1_speedup_x}");
    println!("run2_speedup_x={run2_speedup_x}");
    println!("STATUS=PASS");
    println!("DECISION={decision}");
    println!("qualification_closure_sha256={closure_sha256}");
    println!("qualification_closure_file={}", out.display());

    Ok(())
}
