//! Native Nemotron 3.5 ASR: unchanged HF Safetensors, PCM WAV and Rust tokenization.
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::nemotron3_5_asr::{
    Config, DecodeTrace, Model, ProcessorConfig, MODEL_ID, MODEL_REVISION,
};
use clap::{Parser, ValueEnum};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{io::Write, path::PathBuf, time::Instant};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Streaming,
    Offline,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Cpu,
    Cuda,
    CudaCudnn,
}

#[derive(Parser, Debug)]
#[command(about = "Native Nemotron 3.5 ASR, mono PCM16 WAV at 16 kHz")]
struct Args {
    /// Directory containing the official config, processor, tokenizer and model.safetensors.
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long, value_enum, default_value = "streaming")]
    mode: Mode,
    /// Published language prompt, e.g. fr-FR, en-US or auto.
    #[arg(long, default_value = "fr-FR")]
    language: String,
    #[arg(long)]
    lookahead: Option<usize>,
    #[arg(long, value_enum, default_value = "cpu")]
    device: Backend,
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    #[arg(long, default_value = "f32", value_parser = ["f32", "f16", "bf16"])]
    dtype: String,
    /// PCM delivery packet size; independent of the model's encoder block size.
    #[arg(long, default_value_t = 20)]
    pcm_chunk_ms: usize,
    /// Save result.json in this directory (also printed to stdout).
    #[arg(long)]
    output: Option<PathBuf>,
    /// Also save F32 Mel and encoder activations for optional reference qualification.
    #[arg(long, requires = "output")]
    dump_activations: bool,
    /// Require identical tokens, text and frame decisions to a previous result.json.
    #[arg(long)]
    expect_result: Option<PathBuf>,
}

fn read_wav(path: &std::path::Path) -> Result<(Vec<f32>, usize)> {
    let mut wav = hound::WavReader::open(path).context("opening PCM WAV")?;
    let spec = wav.spec();
    if spec.channels != 1
        || spec.sample_rate != 16_000
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        bail!("expected mono PCM16 WAV at 16000 Hz; resample explicitly before transcription")
    }
    let pcm: Vec<_> = wav
        .samples::<i16>()
        .map(|s| s.map(|s| s as f32 / 32768.0))
        .collect::<std::result::Result<_, _>>()?;
    if pcm.len() < 160 {
        bail!("audio must contain at least 160 samples")
    }
    Ok((pcm, spec.sample_rate as usize))
}

fn dump(path: &std::path::Path, tensor: &Tensor) -> Result<serde_json::Value> {
    let values = tensor
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if values.iter().any(|v| !v.is_finite()) {
        bail!("non-finite activation in {}", path.display())
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in values {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;
    Ok(json!({"shape": tensor.dims(), "dtype": "f32le"}))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.pcm_chunk_ms == 0 || args.pcm_chunk_ms > 60_000 {
        bail!("pcm-chunk-ms must be between 1 and 60000")
    }
    let device = match args.device {
        Backend::Cpu => Device::Cpu,
        Backend::Cuda => {
            Device::new_cuda(args.cuda_device).context("CUDA requested; no CPU fallback")?
        }
        Backend::CudaCudnn => {
            if !cfg!(feature = "cudnn") {
                bail!("cuda-cudnn requires building with --features cuda,cudnn")
            }
            Device::new_cuda(args.cuda_device).context("CUDA/cuDNN requested; no CPU fallback")?
        }
    };
    let dtype = match args.dtype.as_str() {
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        _ => DType::F32,
    };
    if device.is_cpu() && dtype != DType::F32 {
        bail!("CPU qualification uses f32; select --dtype f32")
    }
    let config: Config =
        serde_json::from_slice(&std::fs::read(args.model_dir.join("config.json"))?)?;
    let processor: ProcessorConfig = serde_json::from_slice(&std::fs::read(
        args.model_dir.join("processor_config.json"),
    )?)?;
    let lookahead = args
        .lookahead
        .unwrap_or(config.encoder_config.default_num_lookahead_tokens);
    config.chunk_plan(&processor, lookahead)?;
    processor.prompt_id(&args.language)?;
    let tokenizer = tokenizers::Tokenizer::from_file(args.model_dir.join("tokenizer.json"))
        .map_err(anyhow::Error::msg)?;
    let (pcm, rate) = read_wav(&args.input)?;
    let wav_sha256 = format!("{:x}", Sha256::digest(std::fs::read(&args.input)?));
    let start = Instant::now();
    // The file must not be modified while its memory mapping is in use.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[args.model_dir.join("model.safetensors")],
            dtype,
            &device,
        )?
    };
    let model = Model::load(config, processor, vb)?;
    let load_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut trace = DecodeTrace::default();
    let mut encoded_chunks = Vec::new();
    let mut mel = None;
    let mut cache_frames = 0;
    let encoder_frames;
    if args.dump_activations {
        mel = Some(model.mel(&pcm, rate)?);
    }
    match args.mode {
        Mode::Offline => {
            let features = match mel.take() {
                Some(m) => m,
                None => model.mel(&pcm, rate)?,
            };
            let (encoded, valid) =
                model.encode_features(&features.values, features.valid_frames, lookahead)?;
            trace = model.decode_encoded(&encoded, valid, &args.language)?;
            encoder_frames = valid;
            if args.dump_activations {
                encoded_chunks.push(encoded);
                mel = Some(features);
            }
        }
        Mode::Streaming => {
            let mut state = model.new_stream(&args.language, lookahead)?;
            let mut observe = |encoded: &Tensor| {
                if args.dump_activations {
                    encoded_chunks.push(encoded.to_device(&Device::Cpu)?);
                }
                Ok(())
            };
            for packet in pcm.chunks(rate * args.pcm_chunk_ms / 1000) {
                trace.append(model.push_pcm_with_encoder(
                    &mut state,
                    packet,
                    rate,
                    &mut observe,
                )?);
            }
            trace.append(model.finish_with_encoder(&mut state, &mut observe)?);
            encoder_frames = state.encoder_frames();
            cache_frames = state.attention_cache_frames();
        }
    }
    let inference_seconds = start.elapsed().as_secs_f64();
    let text = tokenizer
        .decode(&trace.tokens, true)
        .map_err(anyhow::Error::msg)?
        .trim()
        .to_owned();
    let mut result = json!({
        "model": MODEL_ID, "expected_model_revision": MODEL_REVISION,
        "device": format!("{device:?}"), "dtype": args.dtype,
        "mode": match args.mode {Mode::Offline => "offline", Mode::Streaming => "streaming"},
        "language": args.language, "lookahead": lookahead,
        "sample_rate": rate, "samples": pcm.len(), "mel_frames": pcm.len() / 160,
        "wav_sha256": wav_sha256,
        "encoder_frames": encoder_frames, "attention_cache_frames": cache_frames,
        "tokens": trace.tokens, "decisions": trace.decisions, "text": text,
        "load_seconds": load_seconds, "inference_seconds": inference_seconds,
        "rtf": inference_seconds / (pcm.len() as f64 / rate as f64),
    });
    let mut differences = Vec::new();
    if let Some(expected) = &args.expect_result {
        let expected: serde_json::Value = serde_json::from_slice(&std::fs::read(expected)?)?;
        for key in [
            "wav_sha256",
            "mode",
            "language",
            "lookahead",
            "samples",
            "sample_rate",
            "mel_frames",
            "encoder_frames",
            "text",
            "tokens",
            "decisions",
        ] {
            if expected.get(key).is_none() || expected[key] != result[key] {
                differences.push(key);
            }
        }
        result["exact_expected_result"] = json!(differences.is_empty());
        result["different_fields"] = json!(differences);
    }
    if let Some(output) = &args.output {
        std::fs::create_dir_all(output)?;
        if let Some(mel) = mel {
            let encoded = Tensor::cat(&encoded_chunks, 1)?;
            let stages = json!({
                "mel": dump(&output.join("mel.f32le"), &mel.values)?,
                "encoder": dump(&output.join("encoder.f32le"), &encoded)?,
                "mel_valid_frames": mel.valid_frames,
                "encoder_valid_frames": encoder_frames,
            });
            std::fs::write(
                output.join("activations.json"),
                serde_json::to_vec_pretty(&stages)?,
            )?;
            result["activations"] = stages;
        }
        std::fs::write(
            output.join("result.json"),
            serde_json::to_vec_pretty(&result)?,
        )?;
    }
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !differences.is_empty() {
        bail!("qualification failed: different fields {differences:?}")
    }
    Ok(())
}
