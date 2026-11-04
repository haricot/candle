//! One native executable for the composed Candle Extended runtime.
use std::ffi::OsString;

mod acoustic;
#[allow(dead_code)]
#[path = "../embedding-gemma2/main.rs"]
mod embedding;
#[allow(dead_code)]
#[path = "../nemotron_asr.rs"]
mod nemotron;
#[allow(dead_code)]
#[path = "../pockettts-sm61/main.rs"]
mod pockettts;
#[allow(dead_code)]
#[path = "../quantized-qwen3/main.rs"]
mod qwen3;
#[allow(dead_code)]
#[path = "../transcribepsy_tp3.rs"]
mod transcribepsy;

const HELP: &str = "Candle Extended: one native CPU/CUDA executable

Usage: candle-extended <COMMAND> [OPTIONS]

Commands:
  qwen3                Generate text with the native Qwen3 GGUF model
  embedding-gemma2     Generate native EmbeddingGemma 2 text embeddings
  nemotron-asr         Transcribe PCM WAV with Nemotron 3.5 ASR
  transcribepsy        Run the native TranscribePsy audio pipeline
  firered-vad          Run FireRedVAD on normalized feature tensors
  smart-turn           Run Smart Turn on Whisper feature tensors
  pockettts-check      Check the native PocketTTS FlowLM/Mimi operators
  capabilities         Print this executable's build capabilities as JSON

Use candle-extended <COMMAND> --help for the command's options.
CUDA is selected explicitly by each command; model files remain external.
";

fn main() -> anyhow::Result<()> {
    let args: Vec<OsString> = std::env::args_os().collect();
    let command = args.get(1).and_then(|x| x.to_str()).unwrap_or("--help");
    let mut forwarded = vec![OsString::from(format!("candle-extended {command}"))];
    forwarded.extend(args.iter().skip(2).cloned());
    match command {
        "--help" | "-h" | "help" => print!("{HELP}"),
        "--version" | "-V" => println!("candle-extended {}", env!("CARGO_PKG_VERSION")),
        "capabilities" => println!(
            "{}",
            serde_json::json!({
                "program": "candle-extended",
                "version": env!("CARGO_PKG_VERSION"),
                "candidate": option_env!("CANDLE_EXTENDED_SOURCE_SHA"),
                "cpu_profile": option_env!("CANDLE_EXTENDED_CPU_PROFILE"),
                "cuda": cfg!(feature = "cuda"),
                "cudnn": cfg!(feature = "cudnn"),
                "legacy_bf16": cfg!(feature = "cuda-legacy-bf16"),
                "legacy_fp8": cfg!(feature = "cuda-legacy-fp8"),
                "legacy_fp4": cfg!(feature = "cuda-legacy-fp4"),
                "avx2": cfg!(target_feature = "avx2"),
                "fma": cfg!(target_feature = "fma"),
                "f16c": cfg!(target_feature = "f16c"),
                "commands": ["qwen3", "embedding-gemma2", "nemotron-asr", "transcribepsy",
                             "firered-vad", "smart-turn", "pockettts-check"]
            })
        ),
        "qwen3" => qwen3::run_from(forwarded)?,
        "embedding-gemma2" => embedding::run_from(forwarded)?,
        "nemotron-asr" => nemotron::run_from(forwarded)?,
        "transcribepsy" => transcribepsy::run_from(forwarded)?,
        "firered-vad" => acoustic::run_from(forwarded, false)?,
        "smart-turn" => acoustic::run_from(forwarded, true)?,
        "pockettts-check" => {
            if forwarded.iter().skip(1).any(|a| a == "--help" || a == "-h") {
                println!("Usage: candle-extended pockettts-check [--device cpu|cuda]\nNative operator check; no full-model speech synthesis is performed.");
            } else {
                pockettts::run_from(forwarded)?;
            }
        }
        other => anyhow::bail!("unknown command {other:?}; use candle-extended --help"),
    }
    Ok(())
}
