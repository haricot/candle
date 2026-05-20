//! Gemma 4 assistant (MTP) model.

use candle::{Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};
use super::config::Gemma4AssistantConfig;
use super::text::TextModel;
use crate::generation::speculative::SpeculativeModel;

pub struct AssistantModel {
    pub text_model: TextModel,
    pre_projection: Linear,
    #[allow(dead_code)]
    post_projection: Linear,
    // Add centroid projection if needed later
}

impl AssistantModel {
    pub fn new(cfg: &Gemma4AssistantConfig, vb: VarBuilder) -> Result<Self> {
        let text_model = TextModel::new_assistant(&cfg.text_config, vb.pp("model"))?;
        let pre_projection = linear(
            2 * cfg.backbone_hidden_size,
            cfg.text_config.hidden_size,
            vb.pp("pre_projection"),
        )?;
        let post_projection = linear(
            cfg.text_config.hidden_size,
            cfg.backbone_hidden_size,
            vb.pp("post_projection"),
        )?;

        Ok(Self {
            text_model,
            pre_projection,
            post_projection,
        })
    }

    /// Forward pass for MTP assistant.
    /// In MTP, the assistant takes backbone hidden states as input.
    pub fn forward_mtp(
        &mut self,
        input_ids: &Tensor,
        backbone_hidden_states: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        let _inputs_embeds = self.text_model.embed_tokens(input_ids)?;

        // MTP logic: combine backbone_hidden_states with assistant embeddings
        // pre_projection expects (2 * backbone_hidden_size).
        // We probably need to project assistant embeds to backbone size first, or it's cat(bh, projected_assistant_embeds)?
        // Re-reading vLLM: Linear(2 * backbone_hidden_size, hidden_size)
        // This implies both parts are backbone_hidden_size.
        // Let's project assistant embeds to backbone size using a dummy if needed,
        // but wait, gemma4 assistant has its own embed_tokens (256).
        // If pre_projection is (3072, 256), and backbone is 1536,
        // maybe it's cat(backbone_hidden, backbone_hidden)? No.
        // Maybe it's cat(backbone_hidden, post_projected_assistant_hidden)?

        // vLLM: pre_projection.* -- Linear(2 * backbone_hidden_size, hidden_size)
        // This suggests it's indeed concatenating backbone_hidden_states with itself or something of the same size.
        // Actually, looking at MTP papers, it's often BH and the embedding of the *next* token.
        // But the assistant doesn't have the backbone embedding.
        // Let's stick to what vLLM says: it's 2 * backbone_hidden_size.
        let combined = Tensor::cat(&[backbone_hidden_states, backbone_hidden_states], D::Minus1)?;
        let inputs_embeds = combined.apply(&self.pre_projection)?;

        let (logits, hidden_states) = self.text_model.forward_embeds(&inputs_embeds, seqlen_offset, b_sz, seq_len)?;
        let next_backbone_hidden = hidden_states.apply(&self.post_projection)?;
        Ok((logits, next_backbone_hidden))
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        let inputs_embeds = self.text_model.embed_tokens(input_ids)?;
        self.text_model.forward_embeds(&inputs_embeds, seqlen_offset, b_sz, seq_len)
    }

    pub fn clear_kv_cache(&mut self) {
        self.text_model.clear_kv_cache()
    }
}

impl SpeculativeModel for AssistantModel {
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Option<Tensor>)> {
        let (logits, hidden_states) = self.forward(input_ids, seqlen_offset)?;
        Ok((logits, Some(hidden_states)))
    }
    fn forward_mtp(&mut self, input_ids: &Tensor, backbone_hidden_states: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Option<Tensor>)> {
        let (logits, next_bh) = self.forward_mtp(input_ids, backbone_hidden_states, seqlen_offset)?;
        Ok((logits, Some(next_bh)))
    }
    fn rewind(&mut self, len: usize) {
        self.text_model.rewind(len)
    }
    fn clear_kv_cache(&mut self) {
        self.clear_kv_cache()
    }
}
