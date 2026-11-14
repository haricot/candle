//! Standalone cudaforge builder: no candle-core dependency or CUDA static archives.
use std::{fs, path::{Path, PathBuf}};
use sha2::{Digest, Sha256};

#[path = "../../../candle-kernels/dkr_s31_contract.rs"]
mod contract;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 || args[0] != "--output" {
        return Err("usage: candle-dkr-s31-builder --output ABSOLUTE_DIRECTORY".into());
    }
    let output = PathBuf::from(&args[1]);
    if !output.is_absolute() {
        return Err("--output must be absolute".into());
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize()?;
    let source = root.join("candle-kernels/src");
    for name in contract::FAMILIES {
        let dir = output.join(name);
        fs::create_dir_all(&dir)?;
        let cu = source.join(format!("{name}.cu"));
        let mut builder = cudaforge::KernelBuilder::new()
            .source_files([&cu]).out_dir(&dir).compute_cap(contract::CC as usize);
        for flag in contract::FLAGS {
            builder = builder.arg(*flag);
        }
        builder.build_ptx()?;
        let ptx = fs::read_to_string(dir.join(format!("{name}.ptx")))?;
        if !contract::is_sm61_ptx(&ptx) {
            return Err(format!("invalid SM61 PTX for {name}").into());
        }
        let manifest = serde_json::json!({
            "schema": contract::SCHEMA, "base_revision": contract::BASE,
            "target_cc": contract::CC, "family": name, "flags": contract::FLAGS,
            "sha256": format!("{:x}", Sha256::digest(ptx.as_bytes())),
        });
        fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&manifest)? + "\n")?;
    }
    println!("CANDLE_DKR_S31_PACKAGES=STAGED target=sm61 families=6");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("CANDLE_DKR_S31_PACKAGES=FAIL {e}");
        std::process::exit(1);
    }
}
