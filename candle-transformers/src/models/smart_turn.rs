//! Smart Turn v3.2 FP32 acoustic end-of-turn classifier.
//!
//! This model loads the trained Whisper Tiny encoder positions, attention
//! pooling and classifier weights. It consumes Whisper features, not PCM, and
//! leaves thresholds, silence holds and conversation policy to the caller.
//! Reference: <https://github.com/pipecat-ai/smart-turn> (BSD-2-Clause).
//! The original notice is retained in `SMART-TURN-LICENSE` next to this module.

use candle::{DType, Module, Result, Tensor, D};
use candle_nn::{conv1d, linear, linear_no_bias, Conv1d, Conv1dConfig, Linear, VarBuilder};

pub mod audio;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub num_mel_bins: usize,
    pub num_frames: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub pooling_hidden_size: usize,
    pub classifier_hidden_size: usize,
    pub classifier_bottleneck_size: usize,
    pub layer_norm_eps: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_mel_bins: 80,
            num_frames: 800,
            hidden_size: 384,
            num_attention_heads: 6,
            num_hidden_layers: 4,
            intermediate_size: 1536,
            pooling_hidden_size: 256,
            classifier_hidden_size: 256,
            classifier_bottleneck_size: 64,
            layer_norm_eps: 1e-5,
        }
    }
}

// Keep the qualified centered FP64 statistics, followed by FP32 output. The
// generic E[x^2]-E[x]^2 normalization loses accuracy on low-variance padding.
struct StableNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}
impl StableNorm {
    fn load(n: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(n, "weight")?.to_dtype(DType::F64)?,
            bias: vb.get(n, "bias")?.to_dtype(DType::F64)?,
            eps,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(DType::F64)?;
        let centered = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
        let std = centered
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, self.eps)?
            .sqrt()?;
        centered
            .broadcast_div(&std)?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?
            .to_dtype(DType::F32)
    }
}

struct EncoderBlock {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    att_norm: StableNorm,
    final_norm: StableNorm,
    fc1: Linear,
    fc2: Linear,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl EncoderBlock {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let att = vb.pp("self_attn");
        Ok(Self {
            q: linear(cfg.hidden_size, cfg.hidden_size, att.pp("q_proj"))?,
            k: linear_no_bias(cfg.hidden_size, cfg.hidden_size, att.pp("k_proj"))?,
            v: linear(cfg.hidden_size, cfg.hidden_size, att.pp("v_proj"))?,
            out: linear(cfg.hidden_size, cfg.hidden_size, att.pp("out_proj"))?,
            att_norm: StableNorm::load(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("self_attn_layer_norm"),
            )?,
            final_norm: StableNorm::load(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("final_layer_norm"),
            )?,
            fc1: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?,
            fc2: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            hidden_size: cfg.hidden_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.att_norm.forward(x)?;
        let (batch, frames, _) = x.dims3()?;
        let shape = (batch, frames, self.num_heads, self.head_dim);
        // Preserve the qualified FP32 split scaling for the published model.
        let scale = if self.head_dim == 64 {
            0.3535533905932738
        } else {
            (self.head_dim as f64).powf(-0.25)
        };
        let q = self
            .q
            .forward(&h)?
            .reshape(shape)?
            .transpose(1, 2)?
            .affine(scale, 0.0)?
            .contiguous()?;
        let k = self
            .k
            .forward(&h)?
            .reshape(shape)?
            .transpose(1, 2)?
            .affine(scale, 0.0)?
            .contiguous()?;
        let v = self
            .v
            .forward(&h)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let attention = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let attention =
            attention
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch, frames, self.hidden_size))?;
        let x = (x + self.out.forward(&attention)?)?;
        let mlp = self.fc2.forward(
            &self
                .fc1
                .forward(&self.final_norm.forward(&x)?)?
                .gelu_erf()?,
        )?;
        &x + mlp
    }
}

pub struct Model {
    conv1: Conv1d,
    conv2: Conv1d,
    positions: Tensor,
    blocks: Vec<EncoderBlock>,
    norm: StableNorm,
    pool1: Linear,
    pool2: Linear,
    class1: Linear,
    class_norm: StableNorm,
    class2: Linear,
    class3: Linear,
    config: Config,
}

/// Detailed output for diagnostics; all values stay on the selected device.
pub struct Output {
    pub probability: Tensor,
    pub encoder: Tensor,
    pub pooled: Tensor,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        if vb.dtype() != DType::F32 {
            candle::bail!("Smart Turn v3.2 currently requires FP32 weights")
        }
        if cfg.num_mel_bins == 0
            || cfg.num_frames == 0
            || cfg.num_frames % 2 != 0
            || cfg.hidden_size == 0
            || cfg.num_attention_heads == 0
            || cfg.hidden_size % cfg.num_attention_heads != 0
            || cfg.num_hidden_layers == 0
            || cfg.intermediate_size == 0
            || cfg.pooling_hidden_size == 0
            || cfg.classifier_hidden_size == 0
            || cfg.classifier_bottleneck_size == 0
            || !cfg.layer_norm_eps.is_finite()
            || cfg.layer_norm_eps <= 0.0
        {
            candle::bail!("Invalid Smart Turn configuration")
        }
        let enc = vb.pp("encoder");
        Ok(Self {
            conv1: conv1d(
                cfg.num_mel_bins,
                cfg.hidden_size,
                3,
                Conv1dConfig {
                    padding: 1,
                    ..Default::default()
                },
                enc.pp("conv1"),
            )?,
            conv2: conv1d(
                cfg.hidden_size,
                cfg.hidden_size,
                3,
                Conv1dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
                enc.pp("conv2"),
            )?,
            positions: enc.get(
                (cfg.num_frames / 2, cfg.hidden_size),
                "embed_positions.weight",
            )?,
            blocks: (0..cfg.num_hidden_layers)
                .map(|i| EncoderBlock::load(cfg, enc.pp(format!("layers.{i}"))))
                .collect::<Result<_>>()?,
            norm: StableNorm::load(cfg.hidden_size, cfg.layer_norm_eps, enc.pp("layer_norm"))?,
            pool1: linear(
                cfg.hidden_size,
                cfg.pooling_hidden_size,
                vb.pp("pool_attention.0"),
            )?,
            pool2: linear(cfg.pooling_hidden_size, 1, vb.pp("pool_attention.2"))?,
            class1: linear(
                cfg.hidden_size,
                cfg.classifier_hidden_size,
                vb.pp("classifier.0"),
            )?,
            class_norm: StableNorm::load(
                cfg.classifier_hidden_size,
                cfg.layer_norm_eps,
                vb.pp("classifier.1"),
            )?,
            class2: linear(
                cfg.classifier_hidden_size,
                cfg.classifier_bottleneck_size,
                vb.pp("classifier.4"),
            )?,
            class3: linear(cfg.classifier_bottleneck_size, 1, vb.pp("classifier.6"))?,
            config: cfg.clone(),
        })
    }

    /// Input `[batch, mel_bins, frames]`; output probabilities `[batch, 1]`.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        Ok(self.forward_detailed(features)?.probability)
    }

    pub fn forward_detailed(&self, features: &Tensor) -> Result<Output> {
        let (batch, mel, frames) = features.dims3()?;
        if batch == 0 || mel != self.config.num_mel_bins || frames != self.config.num_frames {
            candle::bail!("Smart Turn feature shape mismatch")
        }
        let mut x = self
            .conv2
            .forward(&self.conv1.forward(features)?.gelu_erf()?)?
            .gelu_erf()?
            .transpose(1, 2)?
            .contiguous()?
            .broadcast_add(&self.positions)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let encoder = self.norm.forward(&x)?;
        let weights = self.pool2.forward(&self.pool1.forward(&encoder)?.tanh()?)?;
        let weights = candle_nn::ops::softmax(&weights, 1)?;
        let pooled = encoder.broadcast_mul(&weights)?.sum(1)?;
        let x = self
            .class_norm
            .forward(&self.class1.forward(&pooled)?)?
            .gelu_erf()?;
        let x = self.class2.forward(&x)?.gelu_erf()?;
        let probability = candle_nn::ops::sigmoid(&self.class3.forward(&x)?)?;
        Ok(Output {
            probability,
            encoder,
            pooled,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_is_stable_for_large_mean_small_variance() -> Result<()> {
        let device = candle::Device::Cpu;
        let norm = StableNorm {
            weight: Tensor::ones(4, DType::F64, &device)?,
            bias: Tensor::zeros(4, DType::F64, &device)?,
            eps: 1e-5,
        };
        let input = Tensor::new(&[[10000.0f32, 10000.1, 9999.9, 10000.0]], &device)?;
        let result = norm.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(result.iter().all(|x| x.is_finite()));
        assert!(result.iter().sum::<f32>().abs() < 1e-6);
        let variance = result.iter().map(|x| x * x).sum::<f32>() / 4.0;
        assert!((variance - 1.0).abs() < 0.003);
        Ok(())
    }
}
