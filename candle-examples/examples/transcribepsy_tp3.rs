//! TP3 real-model parity driver. Deliberately kept outside candle-transformers:
//! no WAV I/O, tokenizer or reference NeMo dependency in the reusable model.
use anyhow::{bail, Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::transcribepsy::pipeline::{
    DecodeLimits, Model, ModelConfig,
};
use clap::Parser;
use serde_json::{json, Value};
use std::{fs, path::{Path, PathBuf}};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    weights: PathBuf,
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    wav: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn pcm16_mono(path: &Path) -> Result<Vec<f32>> {
    let wav = fs::read(path)?;
    if wav.len() < 44 || &wav[..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        bail!("TP3 accepts only RIFF PCM WAV");
    }
    let mut i = 12usize;
    let mut rate = None;
    let mut samples = None;
    while i + 8 <= wav.len() {
        let len = u32::from_le_bytes(wav[i+4..i+8].try_into()?) as usize;
        let end = (i+8).checked_add(len).context("WAV chunk overflow")?;
        if end > wav.len() { bail!("truncated WAV chunk") }
        if &wav[i..i+4] == b"fmt " {
            if len < 16 { bail!("short WAV fmt") }
            let fmt = &wav[i+8..end];
            let tag = u16::from_le_bytes(fmt[0..2].try_into()?);
            let chans = u16::from_le_bytes(fmt[2..4].try_into()?);
            let hz = u32::from_le_bytes(fmt[4..8].try_into()?);
            let bits = u16::from_le_bytes(fmt[14..16].try_into()?);
            if (tag, chans, hz, bits) != (1, 1, 16000, 16) {
                bail!("TP3 needs uncompressed 16-kHz mono PCM16");
            }
            rate = Some(hz);
        } else if &wav[i..i+4] == b"data" {
            if len % 2 != 0 { bail!("PCM16 data must be even-sized") }
            if samples.is_some() { bail!("multiple PCM data chunks") }
            samples = Some(wav[i+8..end].chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0],b[1]]) as f32 / 32768.)
                .collect::<Vec<_>>());
        }
        i = end + (len & 1);
    }
    let pcm = samples.context("WAV missing data")?;
    if rate != Some(16000) || pcm.is_empty() || pcm.len() > 16000 * 15 {
        bail!("TP3 WAV must be 0-15s, mono, 16kHz");
    }
    Ok(pcm)
}

fn save_f32(path: &Path, tensor: &Tensor) -> Result<Vec<usize>> {
    let t = tensor.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.contiguous()?;
    let shape = t.dims().to_vec();
    let values = t.flatten_all()?.to_vec1::<f32>()?;
    let mut bytes = Vec::with_capacity(4 * values.len());
    for v in values {
        if !v.is_finite() { bail!("nonfinite TP3 activation") }
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(shape)
}

fn execute(args: Args) -> Result<()> {
    fs::create_dir_all(&args.output)?;
    let config: ModelConfig = serde_json::from_slice(&fs::read(&args.config)?)
        .context("invalid QVAC native model config")?;
    config.validate()?;
    let pcm = pcm16_mono(&args.wav)?;
    let vb = unsafe {
        // We open a trusted, pinned, SHA-256-verified QVAC model only.
        // The caller verifies its hash *before* starting this executable.
        VarBuilder::from_mmaped_safetensors(&[args.weights], DType::F32, &Device::Cpu)?
    };
    let model = Model::load(config, vb).context("native QVAC weights incompatible with TP2 model")?;
    let mel = model.mel(&pcm, 16000)?;
    let mel_shape = save_f32(&args.output.join("mel.f32le"), &mel.values)?;
    let (encoded, len) = model.encode_features(&mel.values, &[mel.valid_frames])?;
    let encoder_shape = save_f32(&args.output.join("encoder.f32le"), &encoded)?;
    let trace = model.transcribe_mono(&pcm, 16000, DecodeLimits::default())?;
    let decisions: Vec<Value> = trace.decisions.iter().map(|d| json!({
        "frame": d.frame_before, "token": d.token,
        "duration_index": d.duration_index, "duration": d.duration,
        "advance": d.frame_after - d.frame_before,
        "forced_advance": d.forced_advance,
    })).collect();
    let report = json!({
        "schema": "TP3-CANDLE-QVAC-F32-V1",
        "device": "cpu", "sample_rate": 16000,
        "mel": { "shape": mel_shape, "valid": mel.valid_frames },
        "encoder": { "shape": encoder_shape, "valid": len[0] },
        "tokens": trace.tokens, "decisions": decisions,
        "text": "DECODE_WITH_ORIGINAL_QVAC_SENTENCEPIECE",
    });
    fs::write(args.output.join("trace.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("TP3_CANDLE=REAL_FORWARD_COMPLETE mel_valid={} encoded_valid={} decisions={}",
        mel.valid_frames, len[0], trace.decisions.len());
    Ok(())
}

fn main() -> Result<()> {
    execute(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refuses_non_wav_and_invalid_format() {
        let p = std::env::temp_dir().join(format!("tp3-{}.wav", std::process::id()));
        fs::write(&p, b"NOT_A_WAV").unwrap();
        assert!(pcm16_mono(&p).is_err());
        fs::remove_file(p).unwrap();
    }
}
