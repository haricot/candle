//! Standalone cudaforge builder: no candle-core dependency or CUDA static archives.
use std::{fs, path::{Path, PathBuf}};
use sha2::{Digest, Sha256};

#[path = "../../../candle-kernels/dkr_s31_contract.rs"]
mod contract;
#[path = "../../../candle-core/src/cuda_backend/dkr_s31_validation.rs"]
mod validation;
#[path = "../../../candle-core/src/cuda_backend/dkr_native_catalog.rs"]
mod native_catalog;
#[allow(dead_code)]
#[path = "../../../candle-kernels/dkr_fp8_recipe.rs"]
mod fp8_recipe;
#[allow(dead_code)]
#[path = "../../../candle-kernels/dkr_fp4_recipe.rs"]
mod fp4_recipe;
#[allow(dead_code)]
#[path = "../../../candle-kernels/dkr_bf16_recipe.rs"]
mod bf16_recipe;
#[allow(dead_code)]
#[path = "../../../candle-kernels/dkr_routed_recipe.rs"]
mod routed_recipe;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 || (args[0] != "--output" && args[0] != "--index-existing") {
        return Err("usage: candle-dkr-s31-builder (--output | --index-existing) ABSOLUTE_DIRECTORY".into());
    }
    let output = PathBuf::from(&args[1]);
    if !output.is_absolute() {
        return Err("--output must be absolute".into());
    }
    if args[0] == "--index-existing" {
        return write_index(&output);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize()?;
    let source = root.join("candle-kernels/src");
    for name in contract::FAMILIES {
        let dir = output.join(name);
        fs::create_dir_all(&dir)?;
        let cu = if name.starts_with("mmq_") {
            source.join("mmq_gguf").join(format!("{name}.cu"))
        } else if name.starts_with("moe_") {
            source.join("moe").join(format!("{name}.cu"))
        } else {
            source.join(format!("{name}.cu"))
        };
        let mut builder = cudaforge::KernelBuilder::new()
            .source_files([&cu]).out_dir(&dir).compute_cap(contract::CC as usize);
        for flag in contract::FLAGS {
            builder = builder.arg(*flag);
        }
        builder.build_ptx()?;
        let ptx = fs::read_to_string(dir.join(format!("{name}.ptx")))?;
        if !contract::is_sm61_ptx(&ptx) || (*name == "mmvq_gguf" && !contract::has_mmvq_entries(&ptx)) {
            return Err(format!("invalid SM61 PTX for {name}").into());
        }
        if matches!(*name, "conv" | "sort") && !contract::has_conv_sort_entries(&ptx, name) {
            return Err(format!("missing or duplicate SM61 {name} entry").into());
        }
        let mmq_entries = if name.starts_with("mmq_") {
            Some(contract::mmq_entries(&ptx, name).ok_or_else(|| format!("missing or ambiguous SM61 MMQ entries for {name}"))?)
        } else { None };
        let moe_entries = if name.starts_with("moe_") {
            Some(contract::moe_entries(&ptx,name).ok_or_else(|| format!("invalid MoE native ABI for {name}"))?)
        } else { None };
        let manifest = serde_json::json!({
            "schema": contract::SCHEMA, "base_revision": contract::BASE,
            "target_cc": contract::CC, "family": name, "flags": contract::FLAGS,
            "sha256": format!("{:x}", Sha256::digest(ptx.as_bytes())),
            "mmq_entries": mmq_entries,
            "moe_entries": moe_entries,
        });
        fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&manifest)? + "\n")?;
    }
    write_index(&output)?;
    println!("CANDLE_DKR_S31_PACKAGES=STAGED target=sm61 families={}", contract::FAMILIES.len());
    Ok(())
}

// Reuse the loader verification before publishing an inventory. No CUDA
// compiler is invoked by --index-existing; all PTX bytes remain unchanged.
fn write_index(output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let root = output.canonicalize()?;
    let catalog = native_catalog::NativeCatalog::legacy();
    for family in &catalog.families {
        catalog.read_package(&root, family)?;
    }
    let path = root.join(native_catalog::INDEX);
    let temp = root.join(format!(".native-catalog-{}.tmp", std::process::id()));
    {
        use std::io::Write;
        let mut file = fs::OpenOptions::new().write(true).create_new(true).open(&temp)?;
        if let Err(e) = file.write_all((serde_json::to_string_pretty(&catalog.index())? + "\n").as_bytes()) {
            let _ = fs::remove_file(&temp);
            return Err(e.into());
        }
    }
    if let Err(e) = fs::rename(&temp, &path) {
        let _ = fs::remove_file(&temp);
        return Err(e.into());
    }
    println!("CANDLE_DKR_NATIVE_INDEX=STAGED target=sm{} families={}", catalog.target_cc, catalog.families.len());
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("CANDLE_DKR_S31_PACKAGES=FAIL {e}");
        std::process::exit(1);
    }
}
