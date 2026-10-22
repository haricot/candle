//! Resident R9 ASR + resident native FireRedVAD/Smart Turn in one Rust process.
use anyhow::{Context, Result, bail, ensure};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_voice_endpointing::runner::{self, Args};
use clap::Parser;
use parakeet_candle::parakeet::{
    DecodingConfig, ParakeetModel, from_config_value, get_logmel, load_audio,
};
use serde_json::{Value, json};
use std::fs;

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.asr_binary.is_none(),
        "hdb1_native holds R9 resident; do not pass --asr-binary"
    );
    ensure!(
        std::env::var("HDB2C6_MASK_MODE").as_deref() == Ok("both"),
        "Set HDB2C6_MASK_MODE=both for the qualified R9 encoder"
    );
    let model_dir = args
        .asr_model_dir
        .as_ref()
        .context("--asr-model-dir is required")?;
    let dev = match args.asr_device.as_str() {
        "cpu" => Device::Cpu,
        "cuda-cudnn" => Device::new_cuda(0)?,
        _ => bail!("Unsupported explicit ASR device"),
    };
    let cfg: Value = serde_json::from_slice(&fs::read(model_dir.join("config.json"))?)?;
    let weights = model_dir.join("model.safetensors");
    // SAFETY: the checkpoint must remain immutable while this model is alive,
    // as required by the already-qualified R9 loader.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &dev)? };
    let mut cfg = cfg;
    cfg["hdb2c9_tokenizer"] = serde_json::from_str(r###"@TOKENIZER@"###)?;
    let mut model = from_config_value(cfg, vb)?;
    let ParakeetModel::Tdt(m) = &mut model else {
        bail!("Native half-duplex requires the qualified TDT model");
    };
    println!("HDB_NATIVE_ASR=R9_RESIDENT_LOADED_ONCE device={dev:?}");
    runner::run(args, |wav, folder| {
        let audio = load_audio(wav, m.preprocessor_config.sample_rate)?;
        let mel = get_logmel(&audio, &m.preprocessor_config, &dev)?;
        let valid = parakeet_candle::parakeet::audio::hdb2c_valid_frames(
            audio.len(),
            &m.preprocessor_config,
        )?;
        let input_lengths = Tensor::from_vec(vec![valid as i64], (1,), &dev)?;
        let (encoded, lengths) = m.encoder.forward(&mel, Some(&input_lengths))?;
        let (tokens, _) = m.decode(
            &encoded,
            Some(&lengths),
            None,
            None,
            &DecodingConfig::default(),
        )?;
        let tokens = &tokens[0];
        let ids: Vec<usize> = tokens.iter().map(|t| t.id).collect();
        let frames: Vec<usize> = tokens
            .iter()
            .map(|t| (t.start / m.time_ratio()).round() as usize)
            .collect();
        let durations: Vec<usize> = tokens
            .iter()
            .map(|t| (t.duration / m.time_ratio()).round() as usize)
            .collect();
        let text = m.hdb2c9_text(&ids)?;
        let result = json!({"scope":"NATIVE_PCM_FRONTEND_ENCODER_TDT","device":format!("{dev:?}"),
            "ids":ids,"frames":frames,"durations":durations,"text":text,
            "encoder_valid_frames":lengths.to_vec1::<i64>()?[0],"mel_valid_frames":valid,
            "execution":"R9_RESIDENT_MODEL_REUSED_DECODER_STATE_RESET_PER_TURN"});
        fs::write(
            folder.join("result.json"),
            serde_json::to_vec_pretty(&result)?,
        )?;
        fs::write(folder.join("transcription.txt"), format!("{text}\n"))?;
        Ok(result)
    })?;
    Ok(())
}
