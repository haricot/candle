use candle::{Result, Tensor, D};
use crate::models::qwen3_dflash::DFlashDraftModel;
use crate::models::qwen3_5::ModelForCausalLM;
use crate::generation::LogitsProcessor;

pub struct DFlashGenerator<'a> {
    draft: &'a mut DFlashDraftModel,
    target: &'a mut ModelForCausalLM,
    logits_processor: LogitsProcessor,
}

impl<'a> DFlashGenerator<'a> {
    pub fn new(
        draft: &'a mut DFlashDraftModel,
        target: &'a mut ModelForCausalLM,
        logits_processor: LogitsProcessor,
    ) -> Self {
        Self {
            draft,
            target,
            logits_processor,
        }
    }

    fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        self.logits_processor.sample(logits)
    }

    fn extract_context_feature(
        hidden_states: &[Tensor],
        layer_ids: &[usize],
    ) -> Result<Tensor> {
        let offset = 1;
        let selected: Vec<Tensor> = layer_ids
            .iter()
            .map(|&id| hidden_states[id + offset].clone())
            .collect();
        Tensor::cat(&selected, D::Minus1)
    }

    pub fn generate(
        &mut self,
        input_ids: &Tensor,
        max_new_tokens: usize,
        stop_token_ids: &[u32],
    ) -> Result<Vec<u32>> {
        let dev = input_ids.device();
        let mut output_ids = input_ids.to_vec1::<u32>()?;
        let mut current_offset = 0;

        let block_size = self.draft.block_size;

        // 1. Initial prefill of the target model
        let (logits, hidden_states) = self.target.forward_all(input_ids, 0, block_size > 1)?;
        let next_token = self.sample(&logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(1)?)?;
        output_ids.push(next_token);
        let prefill_len = input_ids.dim(1)?;
        current_offset += prefill_len;

        let mut target_hidden = if block_size > 1 {
            Some(Self::extract_context_feature(
                hidden_states.as_ref().unwrap(),
                &self.draft.target_layer_ids,
            )?)
        } else {
            None
        };

        while output_ids.len() < prefill_len + max_new_tokens {
            // 2. Draft blocks
            let mut block_output_ids = vec![output_ids.last().copied().unwrap()];
            if block_size > 1 {
                use candle::Module;
                // Initial draft noise embedding from the last accepted token
                let last_token_tensor = Tensor::new(&[block_output_ids[0]], dev)?.unsqueeze(0)?;
                let mut noise_embedding = self.target.base_model.embed_tokens.forward(&last_token_tensor)?;

                for i in 1..block_size {
                    let draft_out = self.draft.forward(target_hidden.as_ref().unwrap(), &noise_embedding, current_offset + i - 1)?;
                    let draft_logits = draft_out.apply(&self.target.lm_head)?;
                    let next_draft_token = self.sample(&draft_logits.squeeze(0)?.squeeze(0)?)?;
                    block_output_ids.push(next_draft_token);

                    // Update noise embedding for next draft step
                    let next_draft_token_tensor = Tensor::new(&[next_draft_token], dev)?.unsqueeze(0)?;
                    noise_embedding = self.target.base_model.embed_tokens.forward(&next_draft_token_tensor)?;
                }
            }

            // 3. Verify with target model
            let block_tensor = Tensor::new(block_output_ids.as_slice(), dev)?.unsqueeze(0)?;
            let (target_logits, target_hidden_states) = self.target.forward_all(&block_tensor, current_offset, block_size > 1)?;

            let mut accepted_count = 0;
            let mut posterior_tokens = Vec::new();
            for i in 0..block_output_ids.len() {
                let logits_i = target_logits.narrow(1, i, 1)?.squeeze(1)?.squeeze(0)?;
                let posterior_token = self.sample(&logits_i)?;
                posterior_tokens.push(posterior_token);

                if i < block_output_ids.len() - 1 && block_output_ids[i+1] == posterior_token {
                    accepted_count += 1;
                } else {
                    break;
                }
            }

            // Add accepted tokens and the one corrective token from target
            for i in 1..=accepted_count {
                output_ids.push(block_output_ids[i]);
            }
            output_ids.push(posterior_tokens[accepted_count]);

            let total_added = accepted_count + 1;
            current_offset += total_added;

            // 4. Update target_hidden for next block
            if block_size > 1 {
                target_hidden = Some(Self::extract_context_feature(
                    &target_hidden_states.unwrap(),
                    &self.draft.target_layer_ids,
                )?.narrow(1, 0, total_added)?);
            }

            if stop_token_ids.contains(output_ids.last().unwrap()) {
                break;
            }

            self.draft.clear_kv_cache();
        }

        Ok(output_ids)
    }
}
