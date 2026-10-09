use candle_core::{Error, Result};
use sha2::{Digest, Sha256};
use std::io::Write as _;
use std::path::{Path, PathBuf};

const IMPLEMENTATION_ID: &str = "candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256";
const ABI_VERSION: u32 = 1;
const ARCHITECTURE: &str = "sm61";
const ENTRY: &str = "flow_v0322_ct1d_s32_g2_u1_b256";

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn parse_args() -> Result<(PathBuf, PathBuf, Option<PathBuf>)> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut cubin = None;
    let mut out_dir = None;
    let mut receipt_out = None;
    let mut i = 0usize;

    while i < args.len() {
        match args[i].as_str() {
            "--candidate-cubin" => {
                i += 1;
                cubin = Some(PathBuf::from(args.get(i).ok_or_else(|| {
                    Error::Msg("--candidate-cubin requires a path".into())
                })?));
            }
            "--out-dir" => {
                i += 1;
                out_dir =
                    Some(PathBuf::from(args.get(i).ok_or_else(|| {
                        Error::Msg("--out-dir requires a path".into())
                    })?));
            }
            "--receipt-out" => {
                i += 1;
                receipt_out =
                    Some(PathBuf::from(args.get(i).ok_or_else(|| {
                        Error::Msg("--receipt-out requires a path".into())
                    })?));
            }
            other => return Err(Error::Msg(format!("unknown argument {other:?}"))),
        }
        i += 1;
    }

    let cubin = cubin.ok_or_else(|| Error::Msg("--candidate-cubin is required".into()))?;
    let out_dir = out_dir.ok_or_else(|| Error::Msg("--out-dir is required".into()))?;
    Ok((cubin, out_dir, receipt_out))
}

fn create_new_file(path: &Path, bytes: &[u8], what: &str) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| {
            Error::Msg(format!(
                "refusing to overwrite {what} {}: {err}",
                path.display()
            ))
        })?;
    file.write_all(bytes)
        .map_err(|err| Error::Msg(format!("failed to write {what} {}: {err}", path.display())))
}

fn main() -> Result<()> {
    let (candidate_cubin, out_dir, receipt_out) = parse_args()?;
    if !candidate_cubin.is_file() {
        return Err(Error::Msg(format!(
            "candidate CUBIN does not exist: {}",
            candidate_cubin.display()
        )));
    }

    let cubin = std::fs::read(&candidate_cubin).map_err(|err| {
        Error::Msg(format!(
            "failed to read candidate CUBIN {}: {err}",
            candidate_cubin.display()
        ))
    })?;
    if cubin.is_empty() {
        return Err(Error::Msg("candidate CUBIN is empty".into()));
    }

    let artifact_sha256 = sha256_hex(&cubin);
    std::fs::create_dir_all(&out_dir).map_err(|err| {
        Error::Msg(format!(
            "failed to create candidate directory {}: {err}",
            out_dir.display()
        ))
    })?;

    let staged_cubin = out_dir.join(format!("{IMPLEMENTATION_ID}.cubin"));
    let manifest_path = out_dir.join(format!("{IMPLEMENTATION_ID}.manifest"));

    create_new_file(&staged_cubin, &cubin, "staged CUBIN")?;

    let manifest = format!(
        "ASD-CUDA-MODULE-V1\n\
abi_version={ABI_VERSION}\n\
implementation_id={IMPLEMENTATION_ID}\n\
architecture={ARCHITECTURE}\n\
artifact_kind=cubin\n\
entry={ENTRY}\n\
artifact_sha256={artifact_sha256}\n"
    );
    if let Err(err) = create_new_file(&manifest_path, manifest.as_bytes(), "manifest") {
        let _ = std::fs::remove_file(&staged_cubin);
        return Err(err);
    }

    let receipt = format!(
        "ASD-CUDA-TUNER-CANDIDATE-V1\n\
implementation_id={IMPLEMENTATION_ID}\n\
abi_version={ABI_VERSION}\n\
architecture={ARCHITECTURE}\n\
artifact_kind=cubin\n\
entry={ENTRY}\n\
artifact_sha256={artifact_sha256}\n"
    );
    let receipt_sha256 = sha256_hex(receipt.as_bytes());

    if let Some(path) = receipt_out {
        if let Err(err) = create_new_file(&path, receipt.as_bytes(), "candidate receipt") {
            let _ = std::fs::remove_file(&manifest_path);
            let _ = std::fs::remove_file(&staged_cubin);
            return Err(err);
        }
        println!("candidate_receipt_file={}", path.display());
    }

    println!("=== ASD V3 FLOW-ADAPTIVE CANDIDATE STAGING ===");
    println!("implementation_id={IMPLEMENTATION_ID}");
    println!("architecture={ARCHITECTURE}");
    println!("entry={ENTRY}");
    println!("artifact_sha256={artifact_sha256}");
    println!("staged_cubin={}", staged_cubin.display());
    println!("manifest={}", manifest_path.display());
    println!("candidate_receipt_sha256={receipt_sha256}");
    println!("STATUS=STAGED");
    Ok(())
}
