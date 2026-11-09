use anyhow::{Context, Result};
use candle::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::{fireredvad, smart_turn};
use clap::{Parser, ValueEnum};
use std::{ffi::OsString, path::PathBuf};

#[derive(Clone, Debug, ValueEnum)]
enum Backend {
    Cpu,
    Cuda,
}

#[derive(Parser)]
#[command(
    about = "Native acoustic inference from a safetensors file containing a normalized 'features' tensor"
)]
struct Args {
    #[arg(long)]
    weights: PathBuf,
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    features: PathBuf,
    #[arg(long, value_enum, default_value = "cpu")]
    device: Backend,
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    #[arg(long)]
    output: Option<PathBuf>,
}

pub fn run_from(argv: Vec<OsString>, turn: bool) -> Result<()> {
    let args = Args::parse_from(argv);
    let device = match args.device {
        Backend::Cpu => Device::Cpu,
        Backend::Cuda => {
            Device::new_cuda(args.cuda_device).context("CUDA requested; no CPU fallback")?
        }
    };
    let tensors = candle::safetensors::load(&args.features, &device)?;
    let features = tensors
        .get("features")
        .context("missing 'features' tensor")?
        .to_dtype(DType::F32)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[args.weights], DType::F32, &device)? };
    let config = std::fs::read(args.config)?;
    let values = if turn {
        let cfg: smart_turn::Config = serde_json::from_slice(&config)?;
        smart_turn::Model::new(&cfg, vb)?
            .forward(&features)?
            .flatten_all()?
            .to_vec1::<f32>()?
    } else {
        let cfg: fireredvad::Config = serde_json::from_slice(&config)?;
        let model = fireredvad::Model::new(&cfg, vb)?;
        model
            .forward(&features, &model.zero_cache()?)?
            .0
            .flatten_all()?
            .to_vec1::<f32>()?
    };
    anyhow::ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite model probabilities"
    );
    let data = serde_json::to_string_pretty(&serde_json::json!({
        "model": if turn { "smart-turn" } else { "firered-vad" },
        "backend": format!("{:?}", args.device), "probabilities": values
    }))?;
    if let Some(path) = args.output {
        std::fs::write(path, &data)?;
    }
    println!("{data}");
    Ok(())
}
