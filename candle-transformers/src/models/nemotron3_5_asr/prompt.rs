//! Language conditioning from the published Transformers checkpoint.
//!
//! Reference: huggingface/transformers, Nemotron3_5AsrPromptProjector and
//! Nemotron3_5AsrForRNNT.get_audio_features. No converted NeMo tensor aliases.

use candle::{Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use super::Config;

#[derive(Debug, Clone)]
pub struct LanguageConditioner {
    first: Linear,
    second: Linear,
    encoder_projector: Linear,
    hidden_size: usize,
    num_prompts: usize,
}

impl LanguageConditioner {
    /// `vb` points at the official HF model.safetensors root. All projection
    /// weights and biases are required; no missing tensor is replaced with zero.
    pub fn load(config: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let hidden = config.encoder_config.hidden_size;
        Ok(Self {
            first: candle_nn::linear(
                hidden + config.num_prompts,
                config.prompt_intermediate_size,
                vb.pp("prompt_projector").pp("linear_1"),
            )?,
            second: candle_nn::linear(
                config.prompt_intermediate_size,
                hidden,
                vb.pp("prompt_projector").pp("linear_2"),
            )?,
            encoder_projector: candle_nn::linear(
                hidden,
                config.decoder_hidden_size,
                vb.pp("encoder_projector"),
            )?,
            hidden_size: hidden,
            num_prompts: config.num_prompts,
        })
    }

    /// Input [batch, frames, encoder_hidden], one prompt per batch element.
    /// Output [batch, frames, decoder_hidden], ready for the RNN-T joint.
    /// This does not run an encoder or perform speech recognition.
    pub fn forward(&self, encoder_output: &Tensor, prompt_ids: &[usize]) -> Result<Tensor> {
        let (batch, frames, hidden) = encoder_output.dims3()?;
        if batch == 0
            || frames == 0
            || hidden != self.hidden_size
            || prompt_ids.len() != batch
            || prompt_ids.iter().any(|&id| id >= self.num_prompts)
        {
            candle::bail!("invalid Nemotron encoder output or language prompt IDs")
        }
        let count = batch
            .checked_mul(self.num_prompts)
            .ok_or_else(|| candle::Error::Msg("language prompt size overflow".into()))?;
        let mut values = vec![0f32; count];
        for (b, &id) in prompt_ids.iter().enumerate() {
            values[b * self.num_prompts + id] = 1.0;
        }
        let one_hot = Tensor::from_vec(
            values,
            (batch, 1, self.num_prompts),
            encoder_output.device(),
        )?
        .to_dtype(encoder_output.dtype())?
        .broadcast_as((batch, frames, self.num_prompts))?;
        let fused = Tensor::cat(&[encoder_output, &one_hot], D::Minus1)?;
        self.encoder_projector
            .forward(&self.second.forward(&self.first.forward(&fused)?.relu()?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use std::collections::HashMap;

    fn tiny() -> Result<(Config, HashMap<String, Tensor>)> {
        let (mut cfg, _) = super::super::config::official_config();
        cfg.encoder_config.hidden_size = 2;
        cfg.encoder_config.num_attention_heads = 1;
        cfg.encoder_config.num_key_value_heads = 1;
        cfg.decoder_hidden_size = 2;
        cfg.prompt_intermediate_size = 2;
        cfg.num_prompts = 3;
        cfg.default_prompt_id = 2;
        let mut tensors = HashMap::new();
        for (key, values, dims) in [
            (
                "prompt_projector.linear_1.weight",
                vec![1., 0., 1., 2., 3., 0., 1., 4., 5., 6.],
                vec![2, 5],
            ),
            ("prompt_projector.linear_1.bias", vec![0., 0.], vec![2]),
            (
                "prompt_projector.linear_2.weight",
                vec![1., 0., 0., 1.],
                vec![2, 2],
            ),
            ("prompt_projector.linear_2.bias", vec![0., 0.], vec![2]),
            ("encoder_projector.weight", vec![1., 0., 0., 1.], vec![2, 2]),
            ("encoder_projector.bias", vec![0., 0.], vec![2]),
        ] {
            tensors.insert(
                key.to_owned(),
                Tensor::from_vec(values, dims, &Device::Cpu)?,
            );
        }
        Ok((cfg, tensors))
    }

    #[test]
    fn language_prompt_is_concatenated_per_batch_and_broadcast_over_time() -> Result<()> {
        let (cfg, tensors) = tiny()?;
        let model = LanguageConditioner::load(
            &cfg,
            VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
        )?;
        let input = Tensor::from_vec(
            vec![1f32, 2., 3., 4., -10., -20., 1., 1.],
            (2, 2, 2),
            &Device::Cpu,
        )?;
        let actual = model
            .forward(&input, &[0, 2])?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, vec![2., 6., 4., 8., 0., 0., 4., 7.]);
        assert!(model.forward(&input, &[0]).is_err());
        assert!(model.forward(&input, &[0, 3]).is_err());
        Ok(())
    }

    #[test]
    fn missing_native_bias_is_rejected() -> Result<()> {
        let (cfg, mut tensors) = tiny()?;
        tensors.remove("prompt_projector.linear_1.bias");
        assert!(LanguageConditioner::load(
            &cfg,
            VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
        )
        .is_err());
        Ok(())
    }
}
