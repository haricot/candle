use candle::{DType, Device, Tensor};
use candle_transformers::generation::dflash::DFlashGenerator;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen3_5::{Config as Qwen3_5Config, ModelForCausalLM};
use candle_transformers::models::qwen3_dflash::{DFlashConfig, DFlashDraftModel};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    let api = Api::new()?;
    let repo = api.model("Qwen/Qwen3.5-7B".to_string());
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(candle::Error::msg)?;

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let dtype = DType::BF16;

    // Load Target Model
    let config_filename = repo.get("config.json")?;
    let config: Qwen3_5Config = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
    let filenames = vec![repo.get("model.safetensors")?]; // Simplification for 7B
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let mut target = ModelForCausalLM::new(&config, vb)?;

    // Load Draft Model
    let draft_repo = api.model("z-lab/Qwen3.5-7B-DFlash".to_string());
    // Qwen3.5 DFlash draft models use Qwen3Config architecture (standard transformer)
    let draft_config_filename = draft_repo.get("config.json")?;
    let draft_config: candle_transformers::models::qwen3_5::Config = serde_json::from_reader(std::fs::File::open(draft_config_filename)?)?;
    let dflash_config_filename = draft_repo.get("dflash_config.json")?;
    let dflash_config: DFlashConfig = serde_json::from_reader(std::fs::File::open(dflash_config_filename)?)?;
    let draft_filenames = vec![draft_repo.get("model.safetensors")?];
    let draft_vb = unsafe { VarBuilder::from_mmaped_safetensors(&draft_filenames, dtype, &device)? };
    let mut draft = DFlashDraftModel::new(&draft_config, &dflash_config, draft_vb)?;

    let prompt = "Explain the importance of speculative decoding.";
    let tokens = tokenizer.encode(prompt, true).map_err(candle::Error::msg)?;
    let input_ids = Tensor::new(tokens.get_ids(), &device)?.unsqueeze(0)?;

    let logits_processor = LogitsProcessor::new(42, Some(0.0), None);
    let mut generator = DFlashGenerator::new(&mut draft, &mut target, logits_processor);

    let output_ids = generator.generate(&input_ids, 100, &[tokenizer.token_to_id("<|endoftext|>").unwrap_or(151643)])?;

    println!("Generated text: {}", tokenizer.decode(&output_ids, true).map_err(candle::Error::msg)?);

    Ok(())
}
