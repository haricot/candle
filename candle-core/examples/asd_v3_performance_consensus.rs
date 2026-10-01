use candle_core::{Error, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const HEADER: &str = "ASD-CUDA-PERFORMANCE-EVIDENCE-V1";
const ABI_VERSION: u32 = 1;
const RUN_COUNT: usize = 3;

#[derive(Debug)]
struct Evidence {
    path: PathBuf,
    file_sha256: String,
    fields: BTreeMap<String, String>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn read_evidence(path: &Path) -> Result<Evidence> {
    let bytes = std::fs::read(path).map_err(|err| {
        Error::Msg(format!("failed to read evidence {}: {err}", path.display()))
    })?;
    let text = std::str::from_utf8(&bytes).map_err(|err| {
        Error::Msg(format!("evidence is not UTF-8 {}: {err}", path.display()))
    })?;
    let mut lines = text.lines();
    if lines.next() != Some(HEADER) {
        return Err(Error::Msg(format!(
            "invalid performance evidence header in {}",
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
        if key.is_empty() || value.is_empty() || fields.insert(key.into(), value.into()).is_some() {
            return Err(Error::Msg(format!(
                "duplicate/empty field on line {} in {}",
                index + 2,
                path.display()
            )));
        }
    }

    Ok(Evidence {
        path: path.to_path_buf(),
        file_sha256: sha256_hex(&bytes),
        fields,
    })
}

fn required<'a>(e: &'a Evidence, key: &str) -> Result<&'a str> {
    e.fields.get(key).map(String::as_str).ok_or_else(|| {
        Error::Msg(format!("missing field {key:?} in {}", e.path.display()))
    })
}

fn parse_f64(e: &Evidence, key: &str) -> Result<f64> {
    required(e, key)?.parse::<f64>().map_err(|err| {
        Error::Msg(format!(
            "invalid floating field {key:?} in {}: {err}",
            e.path.display()
        ))
    })
}

fn same_field(runs: &[Evidence], key: &str) -> Result<String> {
    let first = required(&runs[0], key)?.to_owned();
    for run in &runs[1..] {
        let value = required(run, key)?;
        if value != first {
            return Err(Error::Msg(format!(
                "consensus mismatch for {key}: {} has {value:?}, expected {first:?}",
                run.path.display()
            )));
        }
    }
    Ok(first)
}

fn median3(mut values: [f64; 3]) -> f64 {
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[1]
}

fn parse_args() -> Result<([PathBuf; RUN_COUNT], Option<PathBuf>, Option<PathBuf>)> {
    let mut evidence = Vec::new();
    let mut consensus_out = None;
    let mut promotion_out = None;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--evidence" => {
                i += 1;
                let value = args.get(i).ok_or_else(|| {
                    Error::Msg("--evidence requires a path".into())
                })?;
                evidence.push(PathBuf::from(value));
            }
            "--consensus-out" => {
                i += 1;
                consensus_out = Some(PathBuf::from(args.get(i).ok_or_else(|| {
                    Error::Msg("--consensus-out requires a path".into())
                })?));
            }
            "--promotion-out" => {
                i += 1;
                promotion_out = Some(PathBuf::from(args.get(i).ok_or_else(|| {
                    Error::Msg("--promotion-out requires a path".into())
                })?));
            }
            other => return Err(Error::Msg(format!("unknown argument {other:?}"))),
        }
        i += 1;
    }

    let paths: [PathBuf; RUN_COUNT] = evidence.try_into().map_err(|e: Vec<PathBuf>| {
        Error::Msg(format!(
            "exactly {RUN_COUNT} --evidence files are required, got {}",
            e.len()
        ))
    })?;
    Ok((paths, consensus_out, promotion_out))
}

fn main() -> Result<()> {
    let (paths, consensus_out, promotion_out) = parse_args()?;
    let runs = paths
        .iter()
        .map(|path| read_evidence(path))
        .collect::<Result<Vec<_>>>()?;

    let implementation_id = same_field(&runs, "implementation_id")?;
    let architecture = same_field(&runs, "architecture")?;
    let gpu_uuid = same_field(&runs, "gpu_uuid")?;
    let artifact_a_sha256 = same_field(&runs, "artifact_a_sha256")?;
    let artifact_b_sha256 = same_field(&runs, "artifact_b_sha256")?;
    let parity_evidence_sha256 = same_field(&runs, "parity_evidence_sha256")?;
    let min_speedup_x = same_field(&runs, "min_speedup_x")?;

    if architecture != "sm61" {
        return Err(Error::Msg(format!(
            "unsupported consensus architecture {architecture:?}"
        )));
    }

    let speedups = [
        parse_f64(&runs[0], "speedup_x")?,
        parse_f64(&runs[1], "speedup_x")?,
        parse_f64(&runs[2], "speedup_x")?,
    ];
    let consensus_speedup_x = median3(speedups);

    let all_harness_pass = runs
        .iter()
        .all(|run| required(run, "status").ok() == Some("PASS"));
    let all_candidate_promote = runs
        .iter()
        .all(|run| required(run, "decision").ok() == Some("PROMOTE_CANDIDATE"));
    let all_parity = runs
        .iter()
        .all(|run| required(run, "parity_pass").ok() == Some("true"));
    let all_drift = runs
        .iter()
        .all(|run| required(run, "drift_pass").ok() == Some("true"));
    let all_p90 = runs
        .iter()
        .all(|run| required(run, "p90_non_regression").ok() == Some("true"));
    let all_speedup = runs
        .iter()
        .all(|run| required(run, "speedup_pass").ok() == Some("true"));

    for (idx, run) in runs.iter().enumerate() {
        println!(
            "RUN run={} status={} decision={} parity={} drift={} speedup={} p90_non_regression={} speedup_x={}",
            idx + 1,
            required(run, "status")?,
            required(run, "decision")?,
            required(run, "parity_pass")?,
            required(run, "drift_pass")?,
            required(run, "speedup_pass")?,
            required(run, "p90_non_regression")?,
            required(run, "speedup_x")?,
        );
    }

    let consensus_valid = all_harness_pass && all_parity && all_drift;
    let promote = consensus_valid
        && all_candidate_promote
        && all_p90
        && all_speedup;
    let decision = if promote {
        "PROMOTE_CANDIDATE"
    } else {
        "REJECT_CANDIDATE"
    };
    let status = if consensus_valid { "PASS" } else { "HOLD" };

    let consensus = format!(
        "ASD-CUDA-CONSENSUS-V1\nabi_version={ABI_VERSION}\nimplementation_id={implementation_id}\narchitecture={architecture}\ngpu_uuid={gpu_uuid}\nartifact_a_sha256={artifact_a_sha256}\nartifact_b_sha256={artifact_b_sha256}\nparity_evidence_sha256={parity_evidence_sha256}\nrun1_evidence_sha256={}\nrun2_evidence_sha256={}\nrun3_evidence_sha256={}\nrun1_speedup_x={:.6}\nrun2_speedup_x={:.6}\nrun3_speedup_x={:.6}\nconsensus_speedup_x={consensus_speedup_x:.6}\nmin_speedup_x={min_speedup_x}\nall_harness_pass={all_harness_pass}\nall_parity={all_parity}\nall_drift={all_drift}\nall_p90_non_regression={all_p90}\nall_speedup_pass={all_speedup}\nstatus={status}\ndecision={decision}\n",
        runs[0].file_sha256,
        runs[1].file_sha256,
        runs[2].file_sha256,
        speedups[0],
        speedups[1],
        speedups[2],
    );
    let consensus_sha256 = sha256_hex(consensus.as_bytes());

    println!("=== ASD V3 THREE-RUN CONSENSUS ===");
    println!("implementation_id={implementation_id}");
    println!("gpu_uuid={gpu_uuid}");
    println!("artifact_a_sha256={artifact_a_sha256}");
    println!("artifact_b_sha256={artifact_b_sha256}");
    println!(
        "speedups={:.6},{:.6},{:.6}",
        speedups[0], speedups[1], speedups[2]
    );
    println!("consensus_speedup_x={consensus_speedup_x:.6}");
    println!(
        "CONSENSUS_GATES all_harness_pass={} all_parity={} all_drift={} all_p90_non_regression={} all_speedup_pass={}",
        all_harness_pass, all_parity, all_drift, all_p90, all_speedup
    );
    println!("STATUS={status}");
    println!("DECISION={decision}");
    println!("consensus_evidence_sha256={consensus_sha256}");

    if let Some(path) = consensus_out {
        std::fs::write(&path, &consensus).map_err(|err| {
            Error::Msg(format!(
                "failed to write consensus evidence {}: {err}",
                path.display()
            ))
        })?;
        println!("consensus_evidence_file={}", path.display());
    }

    if promote {
        let promotion = format!(
            "ASD-CUDA-PROMOTION-V1\nabi_version={ABI_VERSION}\nimplementation_id={implementation_id}\narchitecture={architecture}\ngpu_uuid={gpu_uuid}\nartifact_sha256={artifact_b_sha256}\nparity_evidence_sha256={parity_evidence_sha256}\nperformance_consensus_sha256={consensus_sha256}\nstate=promoted\n"
        );
        let promotion_sha256 = sha256_hex(promotion.as_bytes());
        println!("promotion_record_sha256={promotion_sha256}");
        if let Some(path) = promotion_out {
            std::fs::write(&path, promotion).map_err(|err| {
                Error::Msg(format!(
                    "failed to write promotion record {}: {err}",
                    path.display()
                ))
            })?;
            println!("promotion_record_file={}", path.display());
        }
    } else if promotion_out.is_some() {
        println!("promotion_record_file=not_written");
    }

    Ok(())
}
