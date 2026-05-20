use candle::{Device, Result, Tensor, DType};
use candle_transformers::generation::speculative::{SpeculativeDecoder, SpeculativeModel};
use candle_transformers::generation::{LogitsProcessor, Sampling};

struct DummyModel {
    vocab_size: usize,
}

impl SpeculativeModel for DummyModel {
    fn forward(&mut self, input_ids: &Tensor, _seqlen_offset: usize) -> Result<(Tensor, Option<Tensor>)> {
        let b_size = input_ids.dim(0)?;
        let seq_len = input_ids.dim(1)?;
        // Return zeros as logits
        let logits = Tensor::zeros((b_size, seq_len, self.vocab_size), DType::F32, input_ids.device())?;
        Ok((logits, None))
    }
    fn rewind(&mut self, _len: usize) {}
    fn clear_kv_cache(&mut self) {}
}

#[test]
fn test_speculative_decoder_logic() -> Result<()> {
    let device = Device::Cpu;
    let vocab_size = 100;
    let target_model = Box::new(DummyModel { vocab_size });
    let draft_model = Box::new(DummyModel { vocab_size });
    let logits_processor = LogitsProcessor::from_sampling(42, Sampling::ArgMax);

    let mut decoder = SpeculativeDecoder::new(target_model, draft_model, logits_processor, 4);

    let prompt = vec![1, 2, 3];
    let result = decoder.generate(&prompt, 10, None, &device)?;

    assert!(result.len() > prompt.len());
    Ok(())
}
