//! Greedy RNN-T with persistent predictor state and explicit blank/frame decisions.
use super::Config;
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::rnn::LSTMState;
use candle_nn::{Embedding, LSTMConfig, Linear, Module, VarBuilder, LSTM, RNN};
use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Decision {
    pub frame: usize,
    pub token: u32,
    pub advance: usize,
    pub forced_advance: bool,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DecodeTrace {
    pub tokens: Vec<u32>,
    pub decisions: Vec<Decision>,
}
impl DecodeTrace {
    pub fn append(&mut self, mut other: Self) {
        self.tokens.append(&mut other.tokens);
        self.decisions.append(&mut other.decisions);
    }
}

#[derive(Debug, Clone, Default)]
pub struct DecoderState {
    layers: Vec<LSTMState>,
    prediction: Option<Tensor>,
    frames_seen: usize,
}
impl DecoderState {
    pub fn frames_seen(&self) -> usize {
        self.frames_seen
    }
}

#[derive(Debug, Clone)]
pub struct Rnnt {
    embedding: Embedding,
    layers: Vec<LSTM>,
    predictor_projection: Linear,
    head: Linear,
    blank: u32,
    max_symbols: usize,
    hidden: usize,
}
impl Rnnt {
    /// `vb` is rooted at the official HF checkpoint, not `decoder`.
    pub fn load(config: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let h = config.decoder_hidden_size;
        let d = vb.pp("decoder");
        let mut layers = Vec::new();
        for i in 0..config.num_decoder_layers {
            layers.push(candle_nn::lstm(
                h,
                h,
                LSTMConfig {
                    layer_idx: i,
                    ..Default::default()
                },
                d.pp("lstm"),
            )?);
        }
        Ok(Self {
            embedding: candle_nn::embedding(config.vocab_size, h, d.pp("embedding"))?,
            layers,
            predictor_projection: candle_nn::linear(h, h, d.pp("decoder_projector"))?,
            head: candle_nn::linear(h, config.vocab_size, vb.pp("joint").pp("head"))?,
            blank: config.blank_token_id as u32,
            max_symbols: config.max_symbols_per_step,
            hidden: h,
        })
    }
    fn predict(&self, token: u32, state: &mut DecoderState) -> Result<()> {
        let ids = Tensor::from_vec(vec![token], (1,), self.embedding.embeddings().device())?;
        let mut x = self.embedding.forward(&ids)?;
        let mut layers = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let old = if state.layers.is_empty() {
                layer.zero_state(1)?
            } else {
                state.layers[i].clone()
            };
            let next = layer.step(&x, &old)?;
            x = next.h().clone();
            layers.push(next);
        }
        state.prediction = Some(self.predictor_projection.forward(&x)?);
        state.layers = layers;
        Ok(())
    }
    pub fn decode(
        &self,
        projected_encoder: &Tensor,
        valid: usize,
        state: &mut DecoderState,
    ) -> Result<DecodeTrace> {
        let (batch, frames, hidden) = projected_encoder.dims3()?;
        if batch != 1 || hidden != self.hidden || valid > frames || valid == 0 {
            candle::bail!("invalid single-stream RNN-T encoder output")
        }
        let mut next = state.clone();
        if next.prediction.is_none() {
            self.predict(self.blank, &mut next)?;
        }
        let mut trace = DecodeTrace::default();
        for frame in 0..valid {
            let encoded = projected_encoder.i((0, frame))?.unsqueeze(0)?;
            for emitted in 0..self.max_symbols {
                let logits = self
                    .head
                    .forward(&(&encoded + next.prediction.as_ref().unwrap())?.relu()?)?;
                let logits = logits
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let token = first_argmax(&logits)? as u32;
                let blank = token == self.blank;
                let forced = !blank && emitted + 1 == self.max_symbols;
                trace.decisions.push(Decision {
                    frame: next.frames_seen,
                    token,
                    advance: usize::from(blank || forced),
                    forced_advance: forced,
                });
                if !blank {
                    trace.tokens.push(token);
                    self.predict(token, &mut next)?;
                }
                if blank || forced {
                    break;
                }
            }
            next.frames_seen = next
                .frames_seen
                .checked_add(1)
                .ok_or_else(|| candle::Error::Msg("RNN-T frame overflow".into()))?;
        }
        *state = next;
        Ok(trace)
    }
}

fn first_argmax(values: &[f32]) -> Result<usize> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        candle::bail!("nonfinite or empty Nemotron RNN-T logits")
    }
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn greedy_ties_and_nonfinite_scores() -> Result<()> {
        assert_eq!(first_argmax(&[1., 2., 2.])?, 1);
        assert!(first_argmax(&[f32::NAN, 1.]).is_err());
        assert!(first_argmax(&[f32::INFINITY]).is_err());
        Ok(())
    }

    #[test]
    fn symbol_limit_advances_and_predictor_survives_block_boundaries() -> Result<()> {
        let (mut config, _) = super::super::config::official_config();
        config.decoder_hidden_size = 4;
        config.vocab_size = 3;
        config.blank_token_id = 2;
        config.max_symbols_per_step = 2;
        let mut vars = candle_nn::VarMap::new();
        let decoder = Rnnt::load(
            &config,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
        )?;
        vars.set_one(
            "joint.head.weight",
            Tensor::zeros((3, 4), DType::F32, &Device::Cpu)?,
        )?;
        vars.set_one(
            "joint.head.bias",
            Tensor::new(&[1f32, 0., 0.], &Device::Cpu)?,
        )?;
        let input = Tensor::zeros((1, 5, 4), DType::F32, &Device::Cpu)?;
        let full = decoder.decode(&input, 5, &mut DecoderState::default())?;
        let mut state = DecoderState::default();
        let mut chunks = decoder.decode(&input.narrow(1, 0, 2)?, 2, &mut state)?;
        chunks.append(decoder.decode(&input.narrow(1, 2, 3)?, 3, &mut state)?);
        assert_eq!(full.decisions, chunks.decisions);
        assert_eq!(state.frames_seen(), 5);
        assert_eq!(chunks.tokens, vec![0; 10]);
        assert_eq!(
            chunks.decisions.iter().filter(|s| s.forced_advance).count(),
            5
        );
        assert!(decoder.decode(&input, 6, &mut state).is_err());
        assert_eq!(state.frames_seen(), 5);
        Ok(())
    }
}
