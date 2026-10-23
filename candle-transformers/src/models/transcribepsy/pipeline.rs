//! Assembly of the native QVAC frontend, FastConformer, predictor and TDT.
//! External applications provide the exact tokenizer (e.g. SentencePiece):
//! this model crate only returns deterministic token IDs and decision traces.
//! No audio capture, file I/O, fallback weights or automatic resampling.

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use super::{encoder, frontend, tdt, Config as TransducerConfig, Joint, Predictor};

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub preprocessor: frontend::Config,
    pub encoder: encoder::Config,
    #[serde(flatten)]
    pub transducer: TransducerConfig,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        self.transducer.validate()?;
        self.preprocessor.validate()?;
        self.encoder.validate()?;
        if self.encoder.feat_in != self.preprocessor.features
            || self.encoder.d_model != self.transducer.joint.jointnet.encoder_hidden
        {
            candle::bail!("TranscribePsy frontend/encoder/joint dimensions mismatch")
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeLimits {
    /// Hard cap per encoder frame even when TDT max_symbols is absent.
    pub decisions_per_frame: usize,
    /// Hard cap on the full emitted sequence.
    pub max_tokens: usize,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            decisions_per_frame: 64,
            max_tokens: 16384,
        }
    }
}

impl DecodeLimits {
    fn validate(self) -> Result<()> {
        if self.decisions_per_frame == 0 || self.max_tokens == 0 || self.decisions_per_frame > 4096
        {
            candle::bail!("invalid TranscribePsy TDT decode limits")
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DecodeTrace {
    pub input_mel_frames: Option<usize>,
    pub encoder_frames: usize,
    pub tokens: Vec<usize>,
    pub decisions: Vec<tdt::Choice>,
}

impl DecodeTrace {
    /// Tokenizer dependency stays outside the model layer. The supplied
    /// callback must use the original QVAC tokenizer, not string heuristics.
    pub fn decode_text<F>(&self, decoder: F) -> Result<String>
    where
        F: Fn(&[usize]) -> Result<String>,
    {
        decoder(&self.tokens)
    }
}

/// Predictor/joint decoder can also be tested independently with fixed
/// encoder activations, without loading a waveform or FastConformer weights.
#[derive(Debug, Clone)]
pub struct Transducer {
    predictor: Predictor,
    joint: Joint,
    greedy: tdt::Greedy,
}

impl Transducer {
    /// `vb` must point to the model.safetensors root.
    pub fn load(cfg: &TransducerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            predictor: Predictor::load(&cfg.decoder, vb.pp("decoder"))?,
            joint: Joint::load(&cfg.joint, vb.pp("joint"))?,
            greedy: tdt::Greedy::from_config(cfg)?,
        })
    }

    /// `encoded` is [1, frames, encoder_hidden] in batch-first order.
    pub fn decode(
        &self,
        encoded: &Tensor,
        valid_frames: usize,
        limits: DecodeLimits,
    ) -> Result<DecodeTrace> {
        limits.validate()?;
        let (batch, total_frames, _) = encoded.dims3()?;
        if batch != 1 || valid_frames == 0 || valid_frames > total_frames {
            candle::bail!("TranscribePsy TDT expects one valid encoded utterance")
        }
        let bound = valid_frames
            .checked_mul(limits.decisions_per_frame)
            .ok_or_else(|| candle::Error::Msg("TDT decision budget overflow".into()))?;
        let mut tokens = Vec::new();
        let mut decisions = Vec::new();
        let mut policy = tdt::GreedyState::default();
        let mut active_frame = 0usize;
        let mut at_frame = 0usize;
        // The blank-as-pad initial predictor input is zero, not a fabricated
        // vocabulary token. The resulting LSTM state is the initial state.
        let (mut prediction, mut prediction_state) = self.predictor.step(None, None)?;
        while policy.frame() < valid_frames {
            if decisions.len() >= bound {
                candle::bail!("TranscribePsy TDT exceeded decision budget")
            }
            if policy.frame() != active_frame {
                active_frame = policy.frame();
                at_frame = 0;
            }
            if at_frame >= limits.decisions_per_frame {
                candle::bail!("TranscribePsy TDT exceeded per-frame decision budget")
            }
            at_frame += 1;
            let frame = encoded.i((0, policy.frame()))?.unsqueeze(0)?;
            let logits = self
                .joint
                .step(&frame, &prediction)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?;
            let choice = self.greedy.advance(&mut policy, &logits)?;
            if choice.frame_after == choice.frame_before && choice.token.is_none() {
                candle::bail!("TDT blank failed to advance the encoder frame")
            }
            if let Some(token) = choice.token {
                if tokens.len() >= limits.max_tokens {
                    candle::bail!("TranscribePsy TDT exceeded token budget")
                }
                let (next_prediction, next_state) =
                    self.predictor.step(Some(token), Some(&prediction_state))?;
                prediction = next_prediction;
                prediction_state = next_state;
                tokens.push(token);
            }
            decisions.push(choice);
        }
        Ok(DecodeTrace {
            input_mel_frames: None,
            encoder_frames: valid_frames,
            tokens,
            decisions,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    preprocessor: frontend::LogMel,
    encoder: encoder::FastConformer,
    transducer: Transducer,
}

impl Model {
    /// All tensors are loaded under the *original* QVAC checkpoint paths:
    /// preprocessor, encoder, decoder, joint.
    pub fn load(cfg: ModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            preprocessor: frontend::LogMel::load(cfg.preprocessor, vb.pp("preprocessor"))?,
            encoder: encoder::FastConformer::load(cfg.encoder, vb.pp("encoder"))?,
            transducer: Transducer::load(&cfg.transducer, vb)?,
        })
    }

    /// Expose Mel for one-clip numerical parity without adding HDB2 scripts.
    pub fn mel(&self, pcm: &[f32], sample_rate: usize) -> Result<frontend::Features> {
        self.preprocessor.process_mono(pcm, sample_rate)
    }

    /// Encode log-Mel directly, e.g. a public WAV parity fixture.
    pub fn encode_features(
        &self,
        features: &Tensor,
        lengths: &[usize],
    ) -> Result<(Tensor, Vec<usize>)> {
        self.encoder.forward(features, lengths)
    }

    /// Decode caller-provided encoder activations independently of the
    /// frontend and encoder. The caller supplies the valid frame count.
    pub fn decode_encoded(
        &self,
        encoded: &Tensor,
        valid_frames: usize,
        limits: DecodeLimits,
    ) -> Result<DecodeTrace> {
        self.transducer.decode(encoded, valid_frames, limits)
    }

    pub fn transcribe_mono(
        &self,
        pcm: &[f32],
        sample_rate: usize,
        limits: DecodeLimits,
    ) -> Result<DecodeTrace> {
        let mel = self.mel(pcm, sample_rate)?;
        let (encoded, lengths) = self.encode_features(&mel.values, &[mel.valid_frames])?;
        let mut trace = self.decode_encoded(&encoded, lengths[0], limits)?;
        trace.input_mel_frames = Some(mel.valid_frames);
        Ok(trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};
    use std::collections::HashMap;

    fn transducer_config() -> TransducerConfig {
        use super::super::*;
        TransducerConfig {
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
                num_extra_outputs: 2,
                jointnet: JointNetworkConfig {
                    joint_hidden: 4,
                    activation: "relu".into(),
                    encoder_hidden: 4,
                    pred_hidden: 4,
                },
            },
            decoding: TdtDecodingConfig {
                model_type: "tdt".into(),
                durations: vec![0, 2],
                greedy: Some(GreedyConfig {
                    max_symbols: Some(8),
                }),
            },
        }
    }

    #[test]
    fn native_transducer_blank_zero_moves_without_tokens() -> Result<()> {
        let cfg = transducer_config();
        let mut map = HashMap::new();
        let d = Device::Cpu;
        let mut zero = |key: &str, shape: Vec<usize>| -> Result<()> {
            map.insert(key.to_owned(), Tensor::zeros(shape, DType::F32, &d)?);
            Ok(())
        };
        zero("decoder.prediction.embed.weight", vec![3, 4])?;
        for key in ["weight_ih_l0", "weight_hh_l0"] {
            zero(
                &format!("decoder.prediction.dec_rnn.lstm.{key}"),
                vec![16, 4],
            )?;
        }
        for key in ["bias_ih_l0", "bias_hh_l0"] {
            zero(&format!("decoder.prediction.dec_rnn.lstm.{key}"), vec![16])?;
        }
        for path in ["enc", "pred"] {
            zero(&format!("joint.{path}.weight"), vec![4, 4])?;
            zero(&format!("joint.{path}.bias"), vec![4])?;
        }
        zero("joint.joint_net.2.weight", vec![5, 4])?;
        map.insert(
            "joint.joint_net.2.bias".into(),
            Tensor::from_vec(vec![0f32, 0., 10., 9., 0.], (5,), &d)?,
        );
        let transducer = Transducer::load(&cfg, VarBuilder::from_tensors(map, DType::F32, &d))?;
        let encoded = Tensor::zeros((1, 3, 4), DType::F32, &d)?;
        let trace = transducer.decode(&encoded, 3, DecodeLimits::default())?;
        assert!(trace.tokens.is_empty());
        assert_eq!(trace.decisions.len(), 3);
        for (i, choice) in trace.decisions.iter().enumerate() {
            assert_eq!(choice.token, None);
            assert_eq!(choice.duration, 0);
            assert_eq!(choice.frame_before, i);
            assert_eq!(choice.frame_after, i + 1);
        }
        assert_eq!(trace.decode_text(|ids| Ok(format!("{ids:?}")))?, "[]");
        Ok(())
    }

    #[test]
    fn model_config_rejects_cross_component_dimension_drift() -> Result<()> {
        let data = serde_json::json!({
            "preprocessor": {
                "sample_rate":16, "window_size":0.25, "window_stride":0.25,
                "n_fft":8, "features":4, "normalize":"NA"
            },
            "encoder": {
                "feat_in":4, "n_layers":1, "d_model":4, "n_heads":2,
                "ff_expansion_factor":2, "subsampling_factor":1,
                "subsampling":"dw_striding", "subsampling_conv_channels":2,
                "self_attention_model":"rel_pos", "conv_kernel_size":3,
                "pos_emb_max_len":16
            },
            "decoder": {
                "blank_as_pad":true, "vocab_size":2,
                "prednet":{"pred_hidden":4,"pred_rnn_layers":1}
            },
            "joint": {
                "num_classes":2, "num_extra_outputs":2,
                "jointnet":{"joint_hidden":4,"activation":"relu",
                            "encoder_hidden":4,"pred_hidden":4}
            },
            "decoding":{"model_type":"tdt","durations":[0,2]}
        });
        let cfg: ModelConfig =
            serde_json::from_value(data.clone()).map_err(|e| candle::Error::Msg(e.to_string()))?;
        cfg.validate()?;
        let mut drift = data;
        drift["encoder"]["feat_in"] = serde_json::json!(5);
        let invalid: ModelConfig =
            serde_json::from_value(drift).map_err(|e| candle::Error::Msg(e.to_string()))?;
        assert!(invalid.validate().is_err());
        Ok(())
    }

    #[test]
    fn decode_limits_are_explicit() {
        assert!(DecodeLimits::default().validate().is_ok());
        assert!(DecodeLimits {
            decisions_per_frame: 0,
            max_tokens: 2
        }
        .validate()
        .is_err());
    }
}
