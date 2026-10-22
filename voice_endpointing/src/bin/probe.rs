//! Offline trace/qualification executable. Production never needs these traces.
use anyhow::{ensure, Result};
use candle::{Device, Tensor};
use candle_voice_endpointing::{
    firered::{FireRedVad, Postprocessor},
    frontend::read_wav,
    smart_turn::SmartTurn,
};
use clap::Parser;
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    models: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    shared_vad: Option<PathBuf>,
    #[arg(long)]
    shared_smart: Option<PathBuf>,
    #[arg(long, default_value_t = 1600)]
    chunk_samples: usize,
}
fn save(path: PathBuf, data: &[f32]) -> Result<()> {
    let bytes: Vec<_> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(path, bytes)?;
    Ok(())
}
fn read(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    ensure!(bytes.len() % 4 == 0, "Invalid tensor bytes");
    let v: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(v.iter().all(|x| x.is_finite()), "Nonfinite shared features");
    Ok(v)
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(a.chunk_samples > 0, "Chunk size must be positive");
    fs::create_dir(&a.output)?;
    let pcm = read_wav(&a.input)?;
    let mut vad = FireRedVad::load(&a.models.join("firered"))?;
    let mut features = Vec::new();
    let mut probs = Vec::new();
    let mut events = Vec::new();
    let mut post = Postprocessor::default();
    let shared = a.shared_vad.as_ref().map(|p| read(p)).transpose()?;
    for chunk in pcm.chunks(a.chunk_samples) {
        let native = vad.features(chunk);
        if native.is_empty() {
            continue;
        }
        let offset = features.len();
        let n = native.len();
        features.extend(&native);
        let input = if let Some(shared) = &shared {
            ensure!(shared.len() >= offset + n, "Shared VAD shape mismatch");
            &shared[offset..offset + n]
        } else {
            &native
        };
        let x = Tensor::from_slice(input, (1, n / 80, 80), &Device::Cpu)?;
        let (p, c) = vad.model.forward(&x, &vad.cache)?;
        vad.cache = c;
        let p = p.flatten_all()?.to_vec1::<f32>()?;
        events.extend(post.feed(&p)?);
        probs.extend(p);
    }
    if let Some(shared) = &shared {
        ensure!(
            shared.len() == features.len(),
            "Shared VAD frame count mismatch"
        );
    }
    save(a.output.join("vad.features.f32le"), &features)?;
    save(a.output.join("vad.probs.f32le"), &probs)?;
    save(
        a.output.join("vad.cache.f32le"),
        &vad.cache.flatten_all()?.to_vec1::<f32>()?,
    )?;
    let smart = SmartTurn::load(&a.models.join("smart-turn"))?;
    let feat = smart.features(&pcm)?;
    save(a.output.join("smart.features.f32le"), &feat)?;
    let shared = a.shared_smart.as_ref().map(|p| read(p)).transpose()?;
    let inp = shared.as_ref().unwrap_or(&feat);
    ensure!(
        inp.len() == 64000,
        "Shared Smart Turn feature shape mismatch"
    );
    let output = smart.forward(&Tensor::from_slice(
        inp.as_slice(),
        (1, 80, 800),
        &Device::Cpu,
    )?)?;
    save(
        a.output.join("smart.encoder.f32le"),
        &output.encoder.flatten_all()?.to_vec1::<f32>()?,
    )?;
    save(
        a.output.join("smart.pooled.f32le"),
        &output.pooled.flatten_all()?.to_vec1::<f32>()?,
    )?;
    let result = json!({"runtime":"RUST_CANDLE_NATIVE","device":"Cpu","samples":pcm.len(),"vad_frames":probs.len(),"vad_events":events,
        "smart_probability":output.probability,"smart_complete":output.probability>=0.5,"shared_inputs":a.shared_smart.is_some() || a.shared_vad.is_some()});
    fs::write(
        a.output.join("result.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{result}");
    Ok(())
}
