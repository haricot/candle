//! Greedy token/duration decisions for a TDT output head.
//!
//! Pure decision logic: no microphone, tokenizer, filesystem, or benchmark
//! dependencies. The caller owns the encoder and predictor state.

use candle::{Result, Tensor};

use super::Config;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice {
    /// None denotes the blank token; a blank never commits predictor state.
    pub token: Option<usize>,
    pub duration: usize,
    pub duration_index: usize,
    pub frame_before: usize,
    pub frame_after: usize,
    pub forced_advance: bool,
}

#[derive(Debug, Clone, Default)]
pub struct GreedyState {
    frame: usize,
    symbols_at_frame: usize,
    last_token: Option<usize>,
}

impl GreedyState {
    pub fn frame(&self) -> usize {
        self.frame
    }

    pub fn last_token(&self) -> Option<usize> {
        self.last_token
    }
}

#[derive(Debug, Clone)]
pub struct Greedy {
    blank_id: usize,
    durations: Vec<usize>,
    max_symbols: Option<usize>,
}

impl Greedy {
    pub fn from_config(config: &Config) -> Result<Self> {
        config.validate()?;
        let max_symbols = config
            .decoding
            .greedy
            .as_ref()
            .and_then(|c| c.max_symbols)
            .and_then(|n| (n > 0).then_some(n as usize));
        Ok(Self {
            blank_id: config.decoder.vocab_size,
            durations: config.decoding.durations.clone(),
            max_symbols,
        })
    }

    /// Split [vocab, blank, duration logits], choosing the first maximum
    /// under ties (as in the reference argmax). Reject nonfinite values.
    pub fn advance(&self, state: &mut GreedyState, logits: &[f32]) -> Result<Choice> {
        let split = self.blank_id + 1;
        if logits.len() != split + self.durations.len() {
            candle::bail!(
                "TDT output length mismatch: got {}, expected {}",
                logits.len(),
                split + self.durations.len()
            )
        }
        let token_id = first_argmax(&logits[..split])?;
        let duration_index = first_argmax(&logits[split..])?;
        let duration = self.durations[duration_index];
        let token = (token_id != self.blank_id).then_some(token_id);
        let frame_before = state.frame;
        let mut next = state.clone();
        if let Some(token) = token {
            next.last_token = Some(token);
            next.symbols_at_frame = next.symbols_at_frame
                .checked_add(1)
                .ok_or_else(|| candle::Error::Msg("TDT symbol counter overflow".into()))?;
        }
        let mut forced_advance = false;
        if duration != 0 {
            next.frame = next
                .frame
                .checked_add(duration)
                .ok_or_else(|| candle::Error::Msg("TDT frame counter overflow".into()))?;
            next.symbols_at_frame = 0;
        } else if self
            .max_symbols
            .is_some_and(|limit| next.symbols_at_frame >= limit)
        {
            next.frame = next
                .frame
                .checked_add(1)
                .ok_or_else(|| candle::Error::Msg("TDT frame counter overflow".into()))?;
            next.symbols_at_frame = 0;
            forced_advance = true;
        }
        let frame_after = next.frame;
        *state = next;
        Ok(Choice {
            token,
            duration,
            duration_index,
            frame_before,
            frame_after,
            forced_advance,
        })
    }

    /// The matching Candle joint tensor must be a vector of logits.
    pub fn advance_tensor(&self, state: &mut GreedyState, logits: &Tensor) -> Result<Choice> {
        let data = logits.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;
        self.advance(state, &data)
    }
}

fn first_argmax(values: &[f32]) -> Result<usize> {
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        candle::bail!("TDT logits must be nonempty finite floats")
    }
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::transcribepsy::{
        DecoderConfig, GreedyConfig, JointConfig, JointNetworkConfig, PredictionNetworkConfig,
        TdtDecodingConfig,
    };

    fn config(max_symbols: Option<i64>) -> Config {
        Config {
            decoder: DecoderConfig {
                blank_as_pad: true,
                vocab_size: 2,
                prednet: PredictionNetworkConfig {
                    pred_hidden: 4,
                    pred_rnn_layers: 1,
                    rnn_hidden_size: None,
                },
            },
            joint: JointConfig {
                num_classes: 2,
                num_extra_outputs: 3,
                jointnet: JointNetworkConfig {
                    joint_hidden: 4,
                    activation: "relu".to_owned(),
                    encoder_hidden: 4,
                    pred_hidden: 4,
                },
            },
            decoding: TdtDecodingConfig {
                model_type: "tdt".to_owned(),
                durations: vec![0, 2, 4],
                greedy: Some(GreedyConfig { max_symbols }),
            },
        }
    }

    #[test]
    fn blank_does_not_commit_token_or_lstm_state() -> Result<()> {
        let decoder = Greedy::from_config(&config(Some(2)))?;
        let mut state = GreedyState::default();
        let event = decoder.advance(&mut state, &[0., 0., 8., 0., 7., 0.])?;
        assert_eq!(event.token, None);
        assert_eq!(state.last_token(), None);
        assert_eq!(event.duration, 2);
        assert_eq!(event.frame_after, 2);
        assert!(!event.forced_advance);
        Ok(())
    }

    #[test]
    fn tdt_zero_duration_and_forced_advance() -> Result<()> {
        let decoder = Greedy::from_config(&config(Some(2)))?;
        let mut state = GreedyState::default();
        let e1 = decoder.advance(&mut state, &[9., 0., 0., 9., 0., 0.])?;
        assert_eq!(e1.token, Some(0));
        assert_eq!(e1.frame_after, 0);
        assert!(!e1.forced_advance);
        let e2 = decoder.advance(&mut state, &[0., 9., 0., 9., 0., 0.])?;
        assert_eq!(e2.token, Some(1));
        assert_eq!(e2.frame_after, 1);
        assert!(e2.forced_advance);
        assert_eq!(state.last_token(), Some(1));
        Ok(())
    }

    #[test]
    fn invalid_logits_fail_closed_without_state_change() -> Result<()> {
        let decoder = Greedy::from_config(&config(Some(2)))?;
        let mut state = GreedyState::default();
        assert!(decoder.advance(&mut state, &[0.0]).is_err());
        assert!(decoder.advance(&mut state, &[f32::NAN, 0., 0., 0., 0., 0.]).is_err());
        assert_eq!(state.frame(), 0);
        Ok(())
    }
}
