//! Typed configuration for the official Transformers Safetensors layout.

use candle::Result;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_mel_bins: usize,
    pub conv_kernel_size: usize,
    pub subsampling_factor: usize,
    pub subsampling_conv_channels: usize,
    pub subsampling_conv_kernel_size: usize,
    pub subsampling_conv_stride: usize,
    pub sliding_window: usize,
    pub default_num_lookahead_tokens: usize,
    pub supported_num_lookahead_tokens: Vec<usize>,
    pub attention_bias: bool,
    pub convolution_bias: bool,
    pub scale_input: bool,
    pub hidden_act: String,
    #[serde(default = "default_positions")]
    pub max_position_embeddings: usize,
}

fn default_positions() -> usize {
    5000
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub encoder_config: EncoderConfig,
    pub vocab_size: usize,
    pub blank_token_id: usize,
    pub pad_token_id: usize,
    pub decoder_hidden_size: usize,
    pub num_decoder_layers: usize,
    pub num_prompts: usize,
    pub prompt_intermediate_size: usize,
    pub default_prompt_id: usize,
    pub max_symbols_per_step: usize,
    pub hidden_act: String,
    #[serde(default)]
    pub durations: Vec<usize>,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        let e = &self.encoder_config;
        if self.model_type != "nemotron3_5_asr"
            || e.model_type != "nemotron_asr_streaming_encoder"
            || self.hidden_act != "relu"
            || e.hidden_act != "silu"
            || !self.durations.is_empty()
        {
            candle::bail!("expected Nemotron 3.5 ASR RNN-T configuration")
        }
        if self.vocab_size < 2
            || self.blank_token_id >= self.vocab_size
            || self.pad_token_id >= self.vocab_size
            || self.decoder_hidden_size == 0
            || self.num_decoder_layers == 0
            || self.num_prompts == 0
            || self.default_prompt_id >= self.num_prompts
            || self.prompt_intermediate_size == 0
            || self.max_symbols_per_step == 0
            || e.hidden_size == 0
            || e.hidden_size % 2 != 0
            || e.intermediate_size == 0
            || e.num_hidden_layers == 0
            || e.num_attention_heads == 0
            || e.hidden_size % e.num_attention_heads != 0
            || e.num_key_value_heads != e.num_attention_heads
            || e.num_mel_bins == 0
            || e.conv_kernel_size < 3
            || e.conv_kernel_size % 2 == 0
            || e.subsampling_factor < 2
            || !e.subsampling_factor.is_power_of_two()
            || e.subsampling_conv_channels == 0
            || e.subsampling_conv_kernel_size != 3
            || e.subsampling_conv_stride != 2
            || e.sliding_window == 0
            || e.max_position_embeddings == 0
            || e.supported_num_lookahead_tokens.is_empty()
            || !e
                .supported_num_lookahead_tokens
                .contains(&e.default_num_lookahead_tokens)
            || e.hidden_size.checked_add(self.num_prompts).is_none()
        {
            candle::bail!("invalid or unsupported Nemotron ASR dimensions/context")
        }
        Ok(())
    }

    /// Nominal audio block sizes, not a measurement of inference latency.
    /// Includes the distinct first-block size required by the reference protocol.
    pub fn chunk_plan(&self, processor: &ProcessorConfig, lookahead: usize) -> Result<ChunkPlan> {
        processor.validate(self)?;
        let e = &self.encoder_config;
        if !e.supported_num_lookahead_tokens.contains(&lookahead) {
            candle::bail!("lookahead {lookahead} is not supported by the pinned checkpoint")
        }
        let encoder_frames = lookahead
            .checked_add(1)
            .ok_or_else(|| candle::Error::Msg("ASR chunk overflow".into()))?;
        let mel_frames = encoder_frames
            .checked_mul(e.subsampling_factor)
            .ok_or_else(|| candle::Error::Msg("ASR chunk overflow".into()))?;
        let audio_samples = mel_frames
            .checked_mul(processor.feature_extractor.hop_length)
            .ok_or_else(|| candle::Error::Msg("ASR chunk overflow".into()))?;
        Ok(ChunkPlan {
            lookahead_tokens: lookahead,
            left_context_tokens: e.sliding_window - 1,
            encoder_frames,
            mel_frames,
            audio_samples,
            sample_rate: processor.feature_extractor.sampling_rate,
            first_mel_frames: 1 + e.subsampling_factor * lookahead,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct FeatureExtractorConfig {
    pub feature_extractor_type: String,
    pub feature_size: usize,
    pub sampling_rate: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_fft: usize,
    pub preemphasis: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProcessorConfig {
    pub processor_class: String,
    pub feature_extractor: FeatureExtractorConfig,
    pub num_prompts: usize,
    pub prompt_dictionary: HashMap<String, usize>,
    pub default_num_lookahead_tokens: usize,
    pub supported_num_lookahead_tokens: Vec<usize>,
}

impl ProcessorConfig {
    pub fn validate(&self, model: &Config) -> Result<()> {
        model.validate()?;
        let f = &self.feature_extractor;
        let e = &model.encoder_config;
        if self.processor_class != "Nemotron3_5AsrProcessor"
            || f.feature_extractor_type != "NemotronAsrStreamingFeatureExtractor"
            || self.num_prompts != model.num_prompts
            || f.feature_size != e.num_mel_bins
            || f.sampling_rate == 0
            || f.hop_length == 0
            || f.win_length == 0
            || f.n_fft < f.win_length
            || !f.n_fft.is_power_of_two()
            || !f.preemphasis.is_finite()
            || self.prompt_dictionary.is_empty()
            || self
                .prompt_dictionary
                .values()
                .any(|&p| p >= self.num_prompts)
            || self.prompt_dictionary.get("auto") != Some(&model.default_prompt_id)
            || self.default_num_lookahead_tokens != e.default_num_lookahead_tokens
            || self.supported_num_lookahead_tokens != e.supported_num_lookahead_tokens
        {
            candle::bail!("Nemotron model/processor configuration mismatch")
        }
        Ok(())
    }

    /// Resolve the published prompt slot. A slot is not an accuracy guarantee
    /// for every language in the dictionary; use the model card's language tiers.
    pub fn prompt_id(&self, language: &str) -> Result<usize> {
        match self.prompt_dictionary.get(language).copied() {
            Some(id) if id < self.num_prompts => Ok(id),
            _ => candle::bail!("unknown or invalid Nemotron language prompt {language:?}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPlan {
    pub lookahead_tokens: usize,
    pub left_context_tokens: usize,
    pub encoder_frames: usize,
    pub mel_frames: usize,
    pub audio_samples: usize,
    pub sample_rate: usize,
    pub first_mel_frames: usize,
}

#[cfg(test)]
pub(super) fn official_config() -> (Config, ProcessorConfig) {
    (
        serde_json::from_str(include_str!("fixtures/model-config.json")).unwrap(),
        serde_json::from_str(include_str!("fixtures/processor-config.json")).unwrap(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_configs_resolve_french_and_auto_without_english_fallback() -> Result<()> {
        let (model, processor) = official_config();
        processor.validate(&model)?;
        assert_eq!(processor.prompt_id("fr-FR")?, 8);
        assert_eq!(processor.prompt_id("fr")?, 8);
        assert_eq!(processor.prompt_id("fr-CA")?, 100);
        assert_eq!(processor.prompt_id("auto")?, 101);
        assert!(processor.prompt_id("unknown").is_err());
        assert_eq!(model.blank_token_id, 13087);
        assert_eq!(model.num_decoder_layers, 2);
        Ok(())
    }

    #[test]
    fn block_sizes_follow_the_checkpoint_supported_contexts() -> Result<()> {
        let (model, processor) = official_config();
        for (r, ms) in [(0, 80), (3, 320), (6, 560), (13, 1120)] {
            let plan = model.chunk_plan(&processor, r)?;
            assert_eq!(plan.left_context_tokens, 56);
            assert_eq!(plan.audio_samples * 1000 / plan.sample_rate, ms);
        }
        // The model card mentions 160 ms, but this pinned config omits r=1.
        assert!(model.chunk_plan(&processor, 1).is_err());
        Ok(())
    }

    #[test]
    fn rejects_tdt_and_mismatched_processor_metadata() {
        let (mut model, mut processor) = official_config();
        model.durations = vec![0, 1, 2, 3, 4];
        assert!(model.validate().is_err());
        model.durations.clear();
        processor.prompt_dictionary.insert("auto".into(), 0);
        assert!(processor.validate(&model).is_err());
        processor.prompt_dictionary.insert("auto".into(), 101);
        model.encoder_config.num_attention_heads = 0;
        assert!(model.validate().is_err());
    }
}
