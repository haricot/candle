//! R5 operator probe: execute the actual native RelPositionalEncoding on CPU.
//! Input fixtures are separate from full encoder inference; no weights are read.
use anyhow::{bail, ensure, Context, Result};
use candle::{DType, Device, Tensor};
use clap::Parser;
use parakeet_candle::parakeet::attention::RelPositionalEncoding;
use serde::Deserialize;
use serde_json::json;
use std::{fs, path::{Path, PathBuf}};

#[derive(Parser)]
struct Args {
    #[arg(long)] cases: PathBuf,
    #[arg(long)] output: PathBuf,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    length: usize,
    d_model: usize,
    offset: usize,
    max_len: usize,
    scale_input: bool,
}

fn dump(folder: &Path, name: &str, x: &Tensor) -> Result<()> {
    let x = x.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let values = x.flatten_all()?.to_vec1::<f32>()?;
    ensure!(values.iter().all(|v| v.is_finite()), "R5 nonfinite output");
    let mut bytes = Vec::with_capacity(values.len()*4);
    for value in values { bytes.extend_from_slice(&value.to_le_bytes()); }
    fs::write(folder.join(format!("{name}.f32le")), bytes)?;
    fs::write(folder.join(format!("{name}.json")), serde_json::to_vec_pretty(
        &json!({"shape":x.dims(),"layout":"B,T,D","dtype":"f32le"}))?)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let manifest: serde_json::Value = serde_json::from_slice(&fs::read(&args.cases)?)?;
    ensure!(manifest["signature"] == "hdb2c3-r5-positional-f32-order-v0.7.5", "R5 signature mismatch");
    let cases: Vec<Case> = serde_json::from_value(manifest["cases"].clone())?;
    ensure!(cases.len() == 17, "R5 expected all 17 cases");
    let root = args.cases.parent().context("R5 missing case directory")?;
    fs::create_dir_all(&args.output)?;
    let mut seen = std::collections::HashSet::new();
    for case in cases {
        ensure!(!case.id.is_empty() && case.id.chars().all(|c| c.is_ascii_alphanumeric() || c=='_'),
            "R5 unsafe case id");
        ensure!(seen.insert(case.id.clone()), "R5 duplicate case id");
        ensure!(case.length>0 && case.d_model>0 && case.d_model%2==0 && case.max_len>0,
            "R5 invalid dimensions");
        let bytes = fs::read(root.join(&case.id).join("input.f32le"))?;
        ensure!(bytes.len()==case.length*case.d_model*4, "R5 input size mismatch");
        let values: Vec<f32> = bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
        if values.iter().any(|v| !v.is_finite()) { bail!("R5 nonfinite input"); }
        let x = Tensor::from_vec(values, (1,case.length,case.d_model), &Device::Cpu)?;
        let mut encoding = RelPositionalEncoding::new(case.d_model,case.max_len,case.scale_input,&Device::Cpu)?;
        let (y,pos) = encoding.forward(&x,case.offset)?;
        ensure!(pos.dims()==[1,2*(case.length+case.offset)-1,case.d_model].as_slice(), "R5 relative position shape mismatch");
        let out = args.output.join(&case.id);
        fs::create_dir_all(&out)?;
        dump(&out,"input",&x)?;
        dump(&out,"output",&y)?;
        dump(&out,"pos_emb",&pos)?;
    }
    fs::write(args.output.join("manifest.json"), serde_json::to_vec_pretty(&json!({
        "signature":manifest["signature"], "cases":manifest["cases"],
        "device":"Cpu", "implementation":"native RelPositionalEncoding; no reference table loading"
    }))?)?;
    println!("HDB2C3_R5_NATIVE_OPERATOR=17_REAL_CANDLE_CPU_CASES");
    Ok(())
}
