#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::qwen2::{Config as ConfigBase, ModelForCausalLM as ModelBase};
use candle_transformers::models::qwen2_moe::{Config as ConfigMoe, Model as ModelMoe};
use candle_transformers::models::qwen3::{Config as Config3, ModelForCausalLM as Model3};
use candle_transformers::models::qwen3_5::{Config as Config3_5, ModelForCausalLM as Model3_5};
use candle_transformers::models::qwen3_moe::{Config as ConfigMoe3, ModelForCausalLM as ModelMoe3};

use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

enum Model {
    Base(ModelBase),
    Moe(ModelMoe),
    Base3(Model3),
    Moe3(ModelMoe3),
    Base3_5(Model3_5),
}

impl Model {
    fn forward(&mut self, xs: &Tensor, s: usize) -> candle::Result<Tensor> {
        match self {
            Self::Moe(ref mut m) => m.forward(xs, s),
            Self::Base(ref mut m) => m.forward(xs, s),
            Self::Base3(ref mut m) => m.forward(xs, s),
            Self::Moe3(ref mut m) => m.forward(xs, s),
            Self::Base3_5(ref mut m) => m.forward(xs, s),
        }
    }
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let temperature = temp.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match (top_k, top_p) {
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (None, None) => Sampling::All { temperature },
            },
        };
        let logits_processor = LogitsProcessor::from_sampling(seed, sampling);
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
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let eos_token2 = match self.tokenizer.get_token("<|im_end|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|im_end|> token"),
        };
        let prompt_len = tokens.len();
        let start_prompt_processing = std::time::Instant::now();
        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self
            .model
            .forward(&input, 0)?
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let prompt_dt = start_prompt_processing.elapsed();
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
        let mut next_token = self.logits_processor.sample(&logits)?;
        tokens.push(next_token);
        generated_tokens += 1;
        if next_token != eos_token && next_token != eos_token2 {
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }

        let start_gen = std::time::Instant::now();
        while generated_tokens < sample_len && next_token != eos_token && next_token != eos_token2 {
            let start_pos = tokens.len().saturating_sub(1);
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
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
            next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == eos_token2 {
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
            "\n{prompt_len} prompt tokens processed ({:.2} token/s)",
            prompt_len as f64 / prompt_dt.as_secs_f64(),
        );
        println!(
            "{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
enum WhichModel {
    #[value(name = "0.5b")]
    W0_5b,
    #[value(name = "1.8b")]
    W1_8b,
    #[value(name = "4b")]
    W4b,
    #[value(name = "7b")]
    W7b,
    #[value(name = "14b")]
    W14b,
    #[value(name = "72b")]
    W72b,
    #[value(name = "moe-a2.7b")]
    MoeA27b,
    #[value(name = "2-0.5b")]
    W2_0_5b,
    #[value(name = "2-1.5b")]
    W2_1_5b,
    #[value(name = "2-7b")]
    W2_7b,
    #[value(name = "2-72b")]
    W2_72b,
    #[value(name = "3-0.6b")]
    W3_0_6b,
    #[value(name = "3-1.7b")]
    W3_1_7b,
    #[value(name = "3-4b")]
    W3_4b,
    #[value(name = "3-8b")]
    W3_8b,
    #[value(name = "3-moe-a3b")]
    W3MoeA3b,
    #[value(name = "3.5-0.8b")]
    W3_5_0_8b,
    #[value(name = "3.5-2b")]
    W3_5_2b,
    #[value(name = "3.5-4b")]
    W3_5_4b,
    #[value(name = "3.5-9b")]
    W3_5_9b,
    #[value(name = "3.5-27b")]
    W3_5_27b,
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

    /// Restrict sampling to the k most likely next tokens.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Data type for model weights and activations: auto, f16, bf16, or f32.
    #[arg(long, default_value = "auto")]
    dtype: String,

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
    weight_path: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long, default_value = "0.5b")]
    model: WhichModel,

    /// Skip chat template formatting (use raw prompt, like base model)
    #[arg(long)]
    no_chat_template: bool,

    /// Enable thinking/reasoning mode (allows model to show its reasoning process)
    #[arg(long)]
    thinking: bool,
}

impl Args {
    fn should_use_chat_template(&self) -> bool {
        matches!(
            self.model,
            WhichModel::W3_0_6b
                | WhichModel::W3_1_7b
                | WhichModel::W3_4b
                | WhichModel::W3_8b
                | WhichModel::W3MoeA3b
                | WhichModel::W3_5_0_8b
                | WhichModel::W3_5_2b
                | WhichModel::W3_5_4b
                | WhichModel::W3_5_9b
                | WhichModel::W3_5_27b
        ) && !self.no_chat_template
    }

    fn is_qwen3(&self) -> bool {
        matches!(
            self.model,
            WhichModel::W3_0_6b
                | WhichModel::W3_1_7b
                | WhichModel::W3_4b
                | WhichModel::W3_8b
                | WhichModel::W3MoeA3b
        )
    }

    fn supports_thinking_switch(&self) -> bool {
        self.is_qwen3()
    }

    fn sampling_settings(&self) -> SamplingSettings {
        let greedy = matches!(self.temperature, Some(v) if v < 1e-7);
        if greedy || !self.is_qwen3() {
            return SamplingSettings {
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                used_qwen3_defaults: false,
            };
        }

        let (default_temp, default_top_p, default_top_k) = if self.thinking {
            (Some(0.6), Some(0.95), Some(20))
        } else {
            (Some(0.7), Some(0.8), Some(20))
        };

        SamplingSettings {
            temperature: self.temperature.or(default_temp),
            top_p: self.top_p.or(default_top_p),
            top_k: self.top_k.or(default_top_k),
            used_qwen3_defaults: self.temperature.is_none()
                || self.top_p.is_none()
                || self.top_k.is_none(),
        }
    }
}

fn format_prompt(
    prompt: &str,
    use_chat_template: bool,
    thinking: bool,
    supports_thinking_switch: bool,
) -> String {
    if !use_chat_template {
        return prompt.to_string();
    }

    if !supports_thinking_switch || thinking {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    } else {
        format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
        )
    }
}

struct SamplingSettings {
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    used_qwen3_defaults: bool,
}

impl SamplingSettings {
    fn is_greedy(&self) -> bool {
        match self.temperature {
            None => true,
            Some(v) => v < 1e-7,
        }
    }
}

fn parse_dtype(dtype: &str, device: &Device) -> Result<DType> {
    match dtype {
        "auto" => {
            if device.is_cuda() || device.is_metal() {
                Ok(DType::BF16)
            } else {
                Ok(DType::F32)
            }
        }
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        "f32" => Ok(DType::F32),
        other => anyhow::bail!("unsupported dtype '{other}', use auto, f16, bf16, or f32"),
    }
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
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
        "repeat-penalty: {:.2} repeat-last-n: {}",
        args.repeat_penalty, args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let use_chat_template = args.should_use_chat_template();
    let supports_thinking_switch = args.supports_thinking_switch();
    let thinking = args.thinking;
    let sampling = args.sampling_settings();
    println!(
        "sampling: temp={} top-p={} top-k={}",
        sampling
            .temperature
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "greedy".to_string()),
        sampling
            .top_p
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "-".to_string()),
        sampling
            .top_k
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string()),
    );
    if sampling.used_qwen3_defaults {
        println!(
            "using Qwen3 recommended sampling defaults for {} mode",
            if thinking { "thinking" } else { "non-thinking" }
        );
    }
    if args.is_qwen3() && sampling.is_greedy() {
        eprintln!(
            "warning: Qwen3 degrades with greedy decoding, use --temperature 0.6 --top-p 0.95 --top-k 20 in thinking mode"
        );
    }
    let model_id = match args.model_id {
        Some(model_id) => model_id,
        None => {
            let (version, size) = match args.model {
                WhichModel::W2_0_5b => ("2", "0.5B"),
                WhichModel::W2_1_5b => ("2", "1.5B"),
                WhichModel::W2_7b => ("2", "7B"),
                WhichModel::W2_72b => ("2", "72B"),
                WhichModel::W0_5b => ("1.5", "0.5B"),
                WhichModel::W1_8b => ("1.5", "1.8B"),
                WhichModel::W4b => ("1.5", "4B"),
                WhichModel::W7b => ("1.5", "7B"),
                WhichModel::W14b => ("1.5", "14B"),
                WhichModel::W72b => ("1.5", "72B"),
                WhichModel::MoeA27b => ("1.5", "MoE-A2.7B"),
                WhichModel::W3_0_6b => ("3", "0.6B"),
                WhichModel::W3_1_7b => ("3", "1.7B"),
                WhichModel::W3_4b => ("3", "4B"),
                WhichModel::W3_8b => ("3", "8B"),
                WhichModel::W3MoeA3b => ("3", "30B-A3B"),
                WhichModel::W3_5_0_8b => ("3.5", "0.8B"),
                WhichModel::W3_5_2b => ("3.5", "2B"),
                WhichModel::W3_5_4b => ("3.5", "4B"),
                WhichModel::W3_5_9b => ("3.5", "9B"),
                WhichModel::W3_5_27b => ("3.5", "27B"),
            };
            format!("Qwen/Qwen{version}-{size}")
        }
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));

    let tokenizer_filename = match (args.weight_path.as_ref(), args.tokenizer_file.as_ref()) {
        (Some(_), Some(file)) => std::path::PathBuf::from(file),
        (None, Some(file)) => std::path::PathBuf::from(file),
        (Some(path), None) => std::path::Path::new(path).join("tokenizer.json"),
        (None, None) => repo.get("tokenizer.json")?,
    };
    let config_file = match &args.weight_path {
        Some(path) => std::path::Path::new(path).join("config.json"),
        _ => repo.get("config.json")?,
    };

    let filenames = match args.weight_path {
        Some(path) => {
            if std::path::Path::new(&path)
                .join("model.safetensors.index.json")
                .exists()
            {
                candle_examples::hub_load_local_safetensors(path, "model.safetensors.index.json")?
            } else {
                vec!["model.safetensors".into()]
            }
        }
        None => match args.model {
            WhichModel::W0_5b
            | WhichModel::W2_0_5b
            | WhichModel::W2_1_5b
            | WhichModel::W1_8b
            | WhichModel::W3_0_6b => {
                vec![repo.get("model.safetensors")?]
            }
            WhichModel::W4b
            | WhichModel::W7b
            | WhichModel::W2_7b
            | WhichModel::W14b
            | WhichModel::W72b
            | WhichModel::W2_72b
            | WhichModel::MoeA27b
            | WhichModel::W3_1_7b
            | WhichModel::W3_4b
            | WhichModel::W3_8b
            | WhichModel::W3MoeA3b
            | WhichModel::W3_5_0_8b
            | WhichModel::W3_5_2b
            | WhichModel::W3_5_4b
            | WhichModel::W3_5_9b
            | WhichModel::W3_5_27b => {
                candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
            }
        },
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let device = candle_examples::device(args.cpu)?;
    let dtype = parse_dtype(&args.dtype, &device)?;
    println!("dtype: {:?}", dtype);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = match args.model {
        WhichModel::MoeA27b => {
            let config: ConfigMoe = serde_json::from_slice(&std::fs::read(config_file)?)?;
            Model::Moe(ModelMoe::new(&config, vb)?)
        }
        WhichModel::W3_0_6b | WhichModel::W3_1_7b | WhichModel::W3_4b | WhichModel::W3_8b => {
            let config: Config3 = serde_json::from_slice(&std::fs::read(config_file)?)?;
            Model::Base3(Model3::new(&config, vb)?)
        }
        WhichModel::W3MoeA3b => {
            let config: ConfigMoe3 = serde_json::from_slice(&std::fs::read(config_file)?)?;
            Model::Moe3(ModelMoe3::new(&config, vb)?)
        }
        WhichModel::W3_5_0_8b
        | WhichModel::W3_5_2b
        | WhichModel::W3_5_4b
        | WhichModel::W3_5_9b
        | WhichModel::W3_5_27b => {
            let config: Config3_5 = serde_json::from_slice(&std::fs::read(config_file)?)?;
            Model::Base3_5(Model3_5::new(&config, vb)?)
        }
        _ => {
            let config: ConfigBase = serde_json::from_slice(&std::fs::read(config_file)?)?;
            Model::Base(ModelBase::new(&config, vb)?)
        }
    };

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        sampling.temperature,
        sampling.top_p,
        sampling.top_k,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    let prompt = format_prompt(
        &args.prompt,
        use_chat_template,
        thinking,
        supports_thinking_switch,
    );
    pipeline.run(&prompt, args.sample_len)?;
    Ok(())
}
