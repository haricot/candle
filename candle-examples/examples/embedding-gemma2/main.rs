//! Standalone Rust text embeddings for google/embeddinggemma-2.
//! No Python, pip, venv, Torch, or external embedding service required.
//! CPU F32 is the reproducible reference. GPU F32 is experimental.
use std::{collections::BTreeSet, path::{Path, PathBuf}};
use anyhow::{Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::embedding_gemma2::{Config, Model};
use clap::Parser;
use tokenizers::Tokenizer;

#[derive(Debug, Parser)]
#[command(about = "EmbeddingGemma 2, text-only native Candle, no Python")]
struct Args {
    /// Text passed to tokenizer. Add the task instruction when appropriate.
    #[arg(long)]
    text: String,
    /// Matryoshka output dimension.
    #[arg(long, default_value_t = 256)]
    dims: usize,
    /// Local model directory containing config.json, tokenizer.json and safetensors.
    #[arg(long)]
    local_dir: Option<PathBuf>,
    /// Hugging Face model repository, used when --local-dir is absent.
    #[arg(long, default_value = "google/embeddinggemma-2")]
    model_id: String,
    #[arg(long, default_value = "main")]
    revision: String,
    /// Use CUDA device 0 in F32 (experimental, requires --features cuda).
    #[arg(long)]
    cuda: bool,
    /// Print full vector as JSON for parity testing.
    #[arg(long)]
    json: bool,
}

fn local_weights(root: &Path) -> Result<Vec<PathBuf>> {
    let index_path = root.join("model.safetensors.index.json");
    if index_path.exists() {
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(index_path)?)?;
        let map = value.get("weight_map").and_then(|x| x.as_object())
            .context("safetensors index has no weight_map")?;
        let mut files = BTreeSet::new();
        for file in map.values() {
            files.insert(file.as_str().context("non-string weight map entry")?.to_string());
        }
        let paths: Vec<_> = files.into_iter().map(|name| root.join(name)).collect();
        for path in &paths {
            anyhow::ensure!(path.is_file(), "missing weight shard: {}", path.display());
        }
        Ok(paths)
    } else {
        let path = root.join("model.safetensors");
        anyhow::ensure!(path.is_file(), "missing weights in {}", root.display());
        Ok(vec![path])
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!([128, 256, 512, 768].contains(&args.dims), "dims must be 128,256,512,768");
    let (config_path, tokenizer_path, weights) = if let Some(local) = &args.local_dir {
        (local.join("config.json"), local.join("tokenizer.json"), local_weights(local)?)
    } else {
        let api = candle_examples::hub::Api::new()?;
        let repo = api.model(args.model_id.clone()).with_revision(args.revision.clone());
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights = match repo.get("model.safetensors.index.json") {
            Ok(index) => {
                let value: serde_json::Value = serde_json::from_slice(&std::fs::read(index)?)?;
                let map = value.get("weight_map").and_then(|x| x.as_object())
                    .context("HF safetensors index has no weight_map")?;
                let names: BTreeSet<_> = map.values()
                    .map(|x| x.as_str().context("non-string weight map entry"))
                    .collect::<Result<_>>()?;
                names.into_iter().map(|name| repo.get(name).map_err(Into::into))
                    .collect::<Result<Vec<_>>>()?
            }
            Err(_) => vec![repo.get("model.safetensors")?],
        };
        (config_path, tokenizer_path, weights)
    };

    let device = if args.cuda { Device::new_cuda(0)? } else { Device::Cpu };
    let config: Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    let ids = tokenizer.encode(args.text.clone(), true)
        .map_err(anyhow::Error::msg)?.get_ids().to_vec();
    anyhow::ensure!(!ids.is_empty() && ids.len() <= 8192, "token count out of bounds");
    let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;

    // F32 is mandatory for Pascal GPUs. Never silently fall back to F16.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, DType::F32, &device)? };
    let model = Model::load(&config, vb)?;
    let embedding = model.embed(&input, args.dims)?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    let vector = &embedding[0];
    if args.json {
        println!("{}", serde_json::json!({
            "model": args.model_id,
            "dtype": "f32",
            "dimensions": args.dims,
            "tokens": ids.len(),
            "embedding": vector
        }));
    } else {
        println!("model={} dtype=f32 dims={} tokens={}", args.model_id, vector.len(), ids.len());
        println!("first_16={:?}", &vector[..vector.len().min(16)]);
        println!("l2={:.8}", vector.iter().map(|x| x * x).sum::<f32>().sqrt());
    }
    Ok(())
}
