use candle::{Result, Tensor, Device};
use crate::generation::LogitsProcessor;

pub trait SpeculativeModel {
    /// Forward pass for a single token, returning logits and optionally hidden states.
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Option<Tensor>)>;

    /// Parallel forward pass for multiple candidate tokens.
    fn forward_batch(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Option<Tensor>)> {
        // Default implementation might be inefficient if it appends tokens one by one
        // Specialized implementations should use a batch forward pass.
        self.forward(input_ids, seqlen_offset)
    }

    /// Specialized forward for MTP assistants, taking backbone hidden states.
    fn forward_mtp(&mut self, input_ids: &Tensor, backbone_hidden_states: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Option<Tensor>)> {
        let _ = backbone_hidden_states;
        self.forward(input_ids, seqlen_offset)
    }

    /// Roll back the model state to a previous position.
    fn rewind(&mut self, len: usize);

    /// Reset KV cache.
    fn clear_kv_cache(&mut self);
}

pub struct SpeculativeDecoder {
    target_model: Box<dyn SpeculativeModel>,
    draft_model: Box<dyn SpeculativeModel>,
    logits_processor: LogitsProcessor,
    max_draft_tokens: usize,
}

impl SpeculativeDecoder {
    pub fn new(
        target_model: Box<dyn SpeculativeModel>,
        draft_model: Box<dyn SpeculativeModel>,
        logits_processor: LogitsProcessor,
        max_draft_tokens: usize,
    ) -> Self {
        Self {
            target_model,
            draft_model,
            logits_processor,
            max_draft_tokens,
        }
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        sample_len: usize,
        eos_token_id: Option<u32>,
        device: &Device,
    ) -> Result<Vec<u32>> {
        let mut tokens = prompt_tokens.to_vec();

        // Initial prefill
        let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
        let (_, mut backbone_hidden_states) = self.target_model.forward(&input, 0)?;
        let _ = self.draft_model.forward(&input, 0)?;

        let mut generated_count = 0;
        while generated_count < sample_len {
            let start_pos = tokens.len();
            let mut draft_tokens = Vec::with_capacity(self.max_draft_tokens);
            let mut current_backbone_states = backbone_hidden_states.clone();

            // 1. Generate draft tokens
            for i in 0..self.max_draft_tokens {
                let last_token = if i == 0 { *tokens.last().unwrap() } else { *draft_tokens.last().unwrap() };
                let input = Tensor::new(&[last_token], device)?.unsqueeze(0)?;

                let (logits, next_states) = if let Some(bh) = &current_backbone_states {
                     let (l, h) = self.draft_model.forward_mtp(&input, bh, start_pos + i)?;
                     (l, h)
                } else {
                    let (l, h) = self.draft_model.forward(&input, start_pos + i)?;
                    (l, h)
                };

                // MTP models return projected backbone states for the next step
                if next_states.is_some() {
                    current_backbone_states = next_states;
                }

                let logits = logits.squeeze(0)?.squeeze(0)?;
                let next_token = self.logits_processor.sample(&logits)?;
                draft_tokens.push(next_token);
                if Some(next_token) == eos_token_id {
                    break;
                }
            }

            // 2. Verify draft tokens in parallel with target model
            let verify_input = Tensor::new(draft_tokens.as_slice(), device)?.unsqueeze(0)?;
            let (logits, hidden_states) = self.target_model.forward_batch(&verify_input, start_pos)?;
            let logits = logits.squeeze(0)?; // (num_draft_tokens, vocab_size)

            let mut accepted_count = 0;
            let mut last_correct_token = 0;

            for i in 0..draft_tokens.len() {
                let target_logits = logits.get(i)?;
                let sampled_token = self.logits_processor.sample(&target_logits)?;

                if sampled_token == draft_tokens[i] {
                    accepted_count += 1;
                    tokens.push(sampled_token);
                    if Some(sampled_token) == eos_token_id {
                        return Ok(tokens);
                    }
                } else {
                    last_correct_token = sampled_token;
                    break;
                }
            }

            if accepted_count < draft_tokens.len() {
                // Rejected at some point, add the corrected token
                tokens.push(last_correct_token);
                // Rewind KV caches
                self.target_model.rewind(tokens.len());
                self.draft_model.rewind(tokens.len());

                // Update backbone_hidden_states for next round
                if let Some(h) = hidden_states {
                    backbone_hidden_states = Some(h.narrow(1, accepted_count, 1)?);
                }

                if Some(last_correct_token) == eos_token_id {
                    return Ok(tokens);
                }
            } else {
                // All accepted, target model already predicted one more token in the last logit
                let extra_logits = logits.get(draft_tokens.len() - 1)?;
                let extra_token = self.logits_processor.sample(&extra_logits)?;
                tokens.push(extra_token);

                if let Some(h) = hidden_states {
                    backbone_hidden_states = Some(h.narrow(1, draft_tokens.len() - 1, 1)?);
                }

                if Some(extra_token) == eos_token_id {
                    return Ok(tokens);
                }
            }

            generated_count = tokens.len() - prompt_tokens.len();
        }

        Ok(tokens)
    }
}
