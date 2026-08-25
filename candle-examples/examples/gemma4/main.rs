#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::gemma4::{
    config::{Gemma4AssistantConfig, Gemma4Config, Gemma4TextConfig},
    text::TextModel,
    AssistantModel, Model,
};

use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::{
    speculative::{SpeculativeDecoder, SpeculativeModel},
    LogitsProcessor, Sampling,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[allow(clippy::large_enum_variant)]
enum ModelKind {
    TextOnly(TextModel),
    Multimodal(Model),
    Speculative(SpeculativeDecoder),
}

struct TextGeneration {
    model: ModelKind,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: ModelKind,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = {
            let temperature = temp.unwrap_or(0.);
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (top_k, top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(seed, sampling)
        };

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        if let ModelKind::Speculative(decoder) = &mut self.model {
            let start_gen = std::time::Instant::now();
            let eos_token_id = self.tokenizer.get_token("</s>");
            let tokens = decoder.generate(&tokens, sample_len, eos_token_id, &self.device)?;
            let dt = start_gen.elapsed();
            for &t in tokens.iter() {
                if let Some(t) = self.tokenizer.next_token(t)? {
                    print!("{t}")
                }
            }
            if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
                print!("{rest}");
            }
            std::io::stdout().flush()?;
            println!(
                "\n{} tokens total ({:.2} token/s)",
                tokens.len(),
                tokens.len() as f64 / dt.as_secs_f64(),
            );
            return Ok(());
        }
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = match &mut self.model {
                ModelKind::TextOnly(m) => m.forward(&input, start_pos)?,
                ModelKind::Multimodal(m) => m.forward(&input, start_pos)?,
                ModelKind::Speculative(_) => {
                    anyhow::bail!("Speculative generation should use run_speculative")
                }
            };
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(long)]
    assistant_model_id: Option<String>,

    #[arg(long)]
    assistant_weight_files: Option<String>,

    /// Load the multimodal model (vision + audio encoders).
    #[arg(long)]
    multimodal: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Use the slower dmmv cuda kernel.
    #[arg(long)]
    force_dmmv: bool,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    #[cfg(feature = "cuda")]
    candle::quantized::cuda::set_force_dmmv(args.force_dmmv);

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = args
        .model_id
        .clone()
        .unwrap_or_else(|| "google/gemma-4-E4B-it".to_string());
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision.clone(),
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            match candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json") {
                Ok(files) => files,
                Err(_) => vec![repo.get("model.safetensors")?],
            }
        }
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

    let model = if args.multimodal {
        let config: Gemma4Config = match args.config_file {
            Some(config_file) => serde_json::from_slice(&std::fs::read(config_file)?)?,
            None => {
                let config_file = repo.get("config.json")?;
                serde_json::from_slice(&std::fs::read(config_file)?)?
            }
        };
        let model = Model::new(&config, vb)?;
        ModelKind::Multimodal(model)
    } else {
        let mut config: Gemma4TextConfig = match args.config_file {
            Some(config_file) => serde_json::from_slice(&std::fs::read(config_file)?)?,
            None => {
                let config_file = repo.get("config.json")?;
                // For text-only, try to parse the text_config sub-object
                let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(config_file)?)?;
                if let Some(text_cfg) = raw.get("text_config") {
                    serde_json::from_value(text_cfg.clone())?
                } else {
                    serde_json::from_value(raw)?
                }
            }
        };
        config.use_flash_attn = args.use_flash_attn;
        let model = TextModel::new(&config, vb)?;
        ModelKind::TextOnly(model)
    };

    println!("loaded the model in {:?}", start.elapsed());

    let model = if let Some(assistant_id) = args.assistant_model_id {
        let assistant_repo = api.repo(Repo::with_revision(
            assistant_id,
            RepoType::Model,
            args.revision.clone(),
        ));
        let assistant_filenames = match args.assistant_weight_files {
            Some(files) => files
                .split(',')
                .map(std::path::PathBuf::from)
                .collect::<Vec<_>>(),
            None => {
                match candle_examples::hub_load_safetensors(
                    &assistant_repo,
                    "model.safetensors.index.json",
                ) {
                    Ok(files) => files,
                    Err(_) => vec![assistant_repo.get("model.safetensors")?],
                }
            }
        };
        let assistant_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&assistant_filenames, dtype, &device)? };
        let assistant_config: Gemma4AssistantConfig =
            serde_json::from_slice(&std::fs::read(assistant_repo.get("config.json")?)?)?;
        let assistant_model = AssistantModel::new(&assistant_config, assistant_vb)?;

        let target_model: Box<dyn SpeculativeModel> = match model {
            ModelKind::TextOnly(m) => Box::new(m),
            ModelKind::Multimodal(m) => Box::new(m),
            _ => anyhow::bail!("nested speculative models are not supported"),
        };

        let logits_processor = {
            let temperature = args.temperature.unwrap_or(0.);
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (args.top_k, args.top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(args.seed, sampling)
        };

        ModelKind::Speculative(SpeculativeDecoder::new(
            target_model,
            Box::new(assistant_model),
            logits_processor,
            4, // Default max draft tokens
        ))
    } else {
        model
    };

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}
