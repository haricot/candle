//! QVAC TranscribePsy FastConformer/TDT model components.
//!
//! This is a deliberately small, reusable Candle model module. It loads the
//! official native NeMo predictor and joint weights without conversion aliases.
//! Native log-Mel, FastConformer, predictor, joint and TDT wiring is present.
//! TP3-R1 qualifies CPU F32 parity with NeMo on one pinned public utterance.
//! General ASR quality, CUDA and streaming require separate qualification.

use candle::{D, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;

pub mod encoder;
pub mod frontend;
pub mod pipeline;
pub mod tdt;

#[derive(Debug, Clone, Deserialize)]
pub struct PredictionNetworkConfig {
    pub pred_hidden: usize,
    pub pred_rnn_layers: usize,
    #[serde(default)]
    pub rnn_hidden_size: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub blank_as_pad: bool,
    pub vocab_size: usize,
    pub prednet: PredictionNetworkConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JointNetworkConfig {
    pub joint_hidden: usize,
    pub activation: String,
    pub encoder_hidden: usize,
    pub pred_hidden: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JointConfig {
    pub num_classes: usize,
    #[serde(default)]
    pub num_extra_outputs: usize,
    pub jointnet: JointNetworkConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TdtDecodingConfig {
    pub model_type: String,
    pub durations: Vec<usize>,
    #[serde(default)]
    pub greedy: Option<GreedyConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GreedyConfig {
    #[serde(default)]
    pub max_symbols: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub decoder: DecoderConfig,
    pub joint: JointConfig,
    pub decoding: TdtDecodingConfig,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        let decoder = &self.decoder;
        if !decoder.blank_as_pad
            || decoder.vocab_size == 0
            || decoder.prednet.pred_hidden == 0
            || decoder.prednet.pred_rnn_layers == 0
        {
            candle::bail!("TranscribePsy expects a nonempty blank-as-pad predictor")
        }
        let joint = &self.joint;
        if joint.num_classes != decoder.vocab_size
            || joint.num_extra_outputs != self.decoding.durations.len()
            || joint.jointnet.encoder_hidden == 0
            || joint.jointnet.joint_hidden == 0
        {
            candle::bail!("TranscribePsy joint/decoder/duration dimensions mismatch")
        }
        let hidden = decoder
            .prednet
            .rnn_hidden_size
            .filter(|&size| size > 0)
            .map(|size| size as usize)
            .unwrap_or(decoder.prednet.pred_hidden);
        if joint.jointnet.pred_hidden != hidden {
            candle::bail!("TranscribePsy joint.pred_hidden must match predictor LSTM output")
        }
        if self.decoding.model_type != "tdt" || self.decoding.durations.is_empty() {
            candle::bail!("TranscribePsy requires nonempty TDT durations")
        }
        match joint.jointnet.activation.as_str() {
            "relu" | "sigmoid" | "tanh" => Ok(()),
            name => candle::bail!("unsupported TranscribePsy joint activation {name:?}"),
        }
    }
}

#[derive(Debug, Clone)]
struct LstmCell {
    weight_input: Tensor,
    weight_hidden: Tensor,
    bias_input: Tensor,
    bias_hidden: Tensor,
}

impl LstmCell {
    fn load(vb: VarBuilder<'_>, layer: usize, input: usize, hidden: usize) -> Result<Self> {
        // QVAC's original model.safetensors stores these keys exactly.
        // All four are mandatory: missing native biases must never become zero.
        let weight_input = vb.get((4 * hidden, input), &format!("weight_ih_l{layer}"))?;
        let weight_hidden = vb.get((4 * hidden, hidden), &format!("weight_hh_l{layer}"))?;
        let bias_input = vb.get((4 * hidden,), &format!("bias_ih_l{layer}"))?;
        let bias_hidden = vb.get((4 * hidden,), &format!("bias_hh_l{layer}"))?;
        Ok(Self {
            weight_input,
            weight_hidden,
            bias_input,
            bias_hidden,
        })
    }

    fn step(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
        let gates = x
            .matmul(&self.weight_input.t()?)?
            .broadcast_add(&self.bias_input)?
            .broadcast_add(&h.matmul(&self.weight_hidden.t()?)?)?
            .broadcast_add(&self.bias_hidden)?;
        let chunks = gates.chunk(4, D::Minus1)?;
        let input = candle_nn::ops::sigmoid(&chunks[0])?;
        let forget = candle_nn::ops::sigmoid(&chunks[1])?;
        let update = chunks[2].tanh()?;
        let output = candle_nn::ops::sigmoid(&chunks[3])?;
        let next_c = ((&forget * c)? + (&input * &update)?)?;
        let next_h = (&output * next_c.tanh()?)?;
        Ok((next_h, next_c))
    }
}

/// LSTM state for one autoregressive transcription hypothesis (batch size 1).
#[derive(Debug, Clone)]
pub struct PredictorState {
    layers: Vec<(Tensor, Tensor)>,
}

impl PredictorState {
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

/// Native QVAC LSTM predictor; independent of the FastConformer encoder.
#[derive(Debug, Clone)]
pub struct Predictor {
    embedding: Embedding,
    layers: Vec<LstmCell>,
    input_dim: usize,
    hidden_dim: usize,
}

impl Predictor {
    pub fn load(config: &DecoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        if !config.blank_as_pad || config.prednet.pred_rnn_layers == 0 {
            candle::bail!("TranscribePsy requires blank_as_pad and >=1 LSTM layer")
        }
        let input_dim = config.prednet.pred_hidden;
        let hidden_dim = config
            .prednet
            .rnn_hidden_size
            .filter(|&size| size > 0)
            .map(|size| size as usize)
            .unwrap_or(input_dim);
        if input_dim == 0 || config.vocab_size == 0 {
            candle::bail!("invalid TranscribePsy predictor size")
        }
        let embedding = candle_nn::embedding(
            config.vocab_size + 1,
            input_dim,
            vb.pp("prediction").pp("embed"),
        )?;
        let native = vb.pp("prediction").pp("dec_rnn").pp("lstm");
        let mut layers = Vec::with_capacity(config.prednet.pred_rnn_layers);
        for layer in 0..config.prednet.pred_rnn_layers {
            let input = if layer == 0 { input_dim } else { hidden_dim };
            layers.push(LstmCell::load(native.clone(), layer, input, hidden_dim)?);
        }
        Ok(Self {
            embedding,
            layers,
            input_dim,
            hidden_dim,
        })
    }

    /// Calculate a single predictor emission. The returned state must only
    /// be committed by a TDT decoder when the decision is NOT the blank token.
    pub fn step(
        &self,
        token: Option<usize>,
        previous: Option<&PredictorState>,
    ) -> Result<(Tensor, PredictorState)> {
        if let Some(state) = previous {
            if state.layers.len() != self.layers.len() {
                candle::bail!("TranscribePsy predictor state layer count mismatch")
            }
        }
        let device = self.embedding.embeddings().device();
        let mut x = if let Some(token) = token {
            let token = u32::try_from(token)
                .map_err(|_| candle::Error::Msg("predictor token ID exceeds u32".into()))?;
            let input = Tensor::from_vec(vec![token], (1,), device)?;
            self.embedding.forward(&input)?
        } else {
            Tensor::zeros((1, self.input_dim), self.embedding.embeddings().dtype(), device)?
        };
        let mut next_state = Vec::with_capacity(self.layers.len());
        for (idx, layer) in self.layers.iter().enumerate() {
            let (h, c) = if let Some(state) = previous {
                (&state.layers[idx].0, &state.layers[idx].1)
            } else {
                // Fresh hypothesis: one row of zeros for each LSTM state.
                let zero_h = Tensor::zeros((1, self.hidden_dim), x.dtype(), device)?;
                let zero_c = Tensor::zeros((1, self.hidden_dim), x.dtype(), device)?;
                let (next_h, next_c) = layer.step(&x, &zero_h, &zero_c)?;
                x = next_h.clone();
                next_state.push((next_h, next_c));
                continue;
            };
            let (next_h, next_c) = layer.step(&x, h, c)?;
            x = next_h.clone();
            next_state.push((next_h, next_c));
        }
        Ok((x, PredictorState { layers: next_state }))
    }
}

/// Joint head combining one encoder frame and one predictor state.
#[derive(Debug, Clone)]
pub struct Joint {
    encoder: Linear,
    predictor: Linear,
    output: Linear,
    activation: String,
}

impl Joint {
    pub fn load(config: &JointConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let c = &config.jointnet;
        if !matches!(c.activation.as_str(), "relu" | "sigmoid" | "tanh") {
            candle::bail!("unsupported TranscribePsy activation {:?}", c.activation)
        }
        let encoder = candle_nn::linear(c.encoder_hidden, c.joint_hidden, vb.pp("enc"))?;
        let predictor = candle_nn::linear(c.pred_hidden, c.joint_hidden, vb.pp("pred"))?;
        let output = candle_nn::linear(
            c.joint_hidden,
            config.num_classes + 1 + config.num_extra_outputs,
            vb.pp("joint_net").pp(2),
        )?;
        Ok(Self {
            encoder,
            predictor,
            output,
            activation: c.activation.clone(),
        })
    }

    /// Inputs are [1, encoder_hidden] and [1, predictor_hidden]; output is
    /// [1, vocab_size + 1 + number_of_durations].
    pub fn step(&self, frame: &Tensor, predicted: &Tensor) -> Result<Tensor> {
        let joined = self
            .encoder
            .forward(frame)?
            .broadcast_add(&self.predictor.forward(predicted)?)?;
        let joined = match self.activation.as_str() {
            "relu" => joined.relu()?,
            "sigmoid" => candle_nn::ops::sigmoid(&joined)?,
            "tanh" => joined.tanh()?,
            _ => candle::bail!("invalid TranscribePsy joint activation"),
        };
        self.output.forward(&joined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use std::collections::HashMap;

    fn config() -> Config {
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
                num_extra_outputs: 2,
                jointnet: JointNetworkConfig {
                    joint_hidden: 4,
                    activation: "relu".to_owned(),
                    encoder_hidden: 4,
                    pred_hidden: 4,
                },
            },
            decoding: TdtDecodingConfig {
                model_type: "tdt".to_owned(),
                durations: vec![0, 2],
                greedy: None,
            },
        }
    }

    #[test]
    fn load_native_qvac_predictor_without_flattened_aliases() -> Result<()> {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "prediction.embed.weight".to_owned(),
            Tensor::zeros((3, 4), DType::F32, &device)?,
        );
        for (name, shape) in [
            ("weight_ih_l0", (16, 4)),
            ("weight_hh_l0", (16, 4)),
        ] {
            tensors.insert(
                format!("prediction.dec_rnn.lstm.{name}"),
                Tensor::zeros(shape, DType::F32, &device)?,
            );
        }
        for name in ["bias_ih_l0", "bias_hh_l0"] {
            tensors.insert(
                format!("prediction.dec_rnn.lstm.{name}"),
                Tensor::zeros((16,), DType::F32, &device)?,
            );
        }
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, &device);
        let predictor = Predictor::load(&config().decoder, vb)?;
        let (output, state) = predictor.step(None, None)?;
        assert_eq!(output.dims(), &[1, 4]);
        assert_eq!(state.layer_count(), 1);
        let (output2, _) = predictor.step(Some(1), Some(&state))?;
        assert_eq!(output2.dims(), &[1, 4]);

        tensors.remove("prediction.dec_rnn.lstm.bias_hh_l0");
        let missing = Predictor::load(
            &config().decoder,
            VarBuilder::from_tensors(tensors, DType::F32, &device),
        );
        assert!(missing.is_err(), "QVAC native LSTM bias is mandatory");
        Ok(())
    }

    #[test]
    fn config_checks_tdt_dimensions() {
        let mut cfg = config();
        assert!(cfg.validate().is_ok());
        cfg.joint.num_extra_outputs = 3;
        assert!(cfg.validate().is_err());
    }
}
