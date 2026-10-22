//! Pipecat Smart Turn v3.2, official FP32 weights. This is NOT the INT8 graph.
//! The trained positional embeddings and the endpointing head are both loaded.
use crate::{frontend::WhisperFeatures, weights};
use anyhow::{ensure, Result};
use candle::{DType, Device, Module, Tensor, D};
use candle_nn::{conv1d, linear, linear_no_bias, Conv1d, Conv1dConfig, Linear, VarBuilder};
use std::path::Path;

// Centered FP64 statistics avoid cancellation in E[x^2]-E[x]^2 for low-variance
// padded frames. This is local to this new model, not a change to Candle/R9.
struct StableNorm {
    weight: Tensor,
    bias: Tensor,
}
impl StableNorm {
    fn load(n: usize, vb: VarBuilder) -> candle::Result<Self> {
        Ok(Self {
            weight: vb.get(n, "weight")?.to_dtype(DType::F64)?,
            bias: vb.get(n, "bias")?.to_dtype(DType::F64)?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let x = x.to_dtype(DType::F64)?;
        let centered = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
        let std = centered
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, 1e-5)?
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
}

impl EncoderBlock {
    fn load(vb: VarBuilder) -> candle::Result<Self> {
        let att = vb.pp("self_attn");
        Ok(Self {
            q: linear(384, 384, att.pp("q_proj"))?,
            k: linear_no_bias(384, 384, att.pp("k_proj"))?,
            v: linear(384, 384, att.pp("v_proj"))?,
            out: linear(384, 384, att.pp("out_proj"))?,
            att_norm: StableNorm::load(384, vb.pp("self_attn_layer_norm"))?,
            final_norm: StableNorm::load(384, vb.pp("final_layer_norm"))?,
            fc1: linear(384, 1536, vb.pp("fc1"))?,
            fc2: linear(1536, 384, vb.pp("fc2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let h = self.att_norm.forward(x)?;
        let shape = (1, 400, 6, 64);
        // PyTorch's exported scaled_dot_product_attention splits the scale
        // across Q and K. Retain the same FP32 multiplication order.
        let scale = 0.3535533905932738;
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
        let attention = attention
            .transpose(1, 2)?
            .contiguous()?
            .reshape((1, 400, 384))?;
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

pub struct SmartTurn {
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
    features: WhisperFeatures,
}

pub struct SmartOutput {
    pub probability: f32,
    pub encoder: Tensor,
    pub pooled: Tensor,
}

impl SmartTurn {
    pub fn load(root: &Path) -> Result<Self> {
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        ensure!(
            cfg["architecture"] == "smart-turn-v3.2-whisper-tiny-400" && cfg["dtype"] == "float32",
            "Unsupported Smart Turn configuration"
        );
        let vb = weights::load(root, "smart-turn-v3.2-fp32", &Device::Cpu)?;
        let enc = vb.pp("encoder");
        Ok(Self {
            conv1: conv1d(
                80,
                384,
                3,
                Conv1dConfig {
                    padding: 1,
                    ..Default::default()
                },
                enc.pp("conv1"),
            )?,
            conv2: conv1d(
                384,
                384,
                3,
                Conv1dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
                enc.pp("conv2"),
            )?,
            positions: enc.get((400, 384), "embed_positions.weight")?,
            blocks: (0..4)
                .map(|i| EncoderBlock::load(enc.pp(format!("layers.{i}"))))
                .collect::<candle::Result<_>>()?,
            norm: StableNorm::load(384, enc.pp("layer_norm"))?,
            pool1: linear(384, 256, vb.pp("pool_attention.0"))?,
            pool2: linear(256, 1, vb.pp("pool_attention.2"))?,
            class1: linear(384, 256, vb.pp("classifier.0"))?,
            class_norm: StableNorm::load(256, vb.pp("classifier.1"))?,
            class2: linear(256, 64, vb.pp("classifier.4"))?,
            class3: linear(64, 1, vb.pp("classifier.6"))?,
            features: WhisperFeatures::default(),
        })
    }
    pub fn features(&self, pcm: &[i16]) -> Result<Vec<f32>> {
        self.features.compute(pcm)
    }
    pub fn forward(&self, features: &Tensor) -> Result<SmartOutput> {
        ensure!(
            features.dims() == [1, 80, 800],
            "Smart Turn requires [1,80,800] features"
        );
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
        let probability = candle_nn::ops::sigmoid(&self.class3.forward(&x)?)?
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        ensure!(
            probability.is_finite() && (0.0..=1.0).contains(&probability),
            "Invalid Smart Turn probability"
        );
        Ok(SmartOutput {
            probability,
            encoder,
            pooled,
        })
    }
    pub fn predict(&self, pcm: &[i16]) -> Result<f32> {
        let features = Tensor::from_vec(self.features(pcm)?, (1, 80, 800), &Device::Cpu)?;
        Ok(self.forward(&features)?.probability)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_is_stable_for_large_mean_small_variance() {
        let device = Device::Cpu;
        let norm = StableNorm {
            weight: Tensor::ones(4, DType::F64, &device).unwrap(),
            bias: Tensor::zeros(4, DType::F64, &device).unwrap(),
        };
        let input = Tensor::new(&[[10000.0f32, 10000.1, 9999.9, 10000.0]], &device).unwrap();
        let result = norm
            .forward(&input)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(result.iter().all(|x| x.is_finite()));
        assert!(result.iter().sum::<f32>().abs() < 1e-6);
        let variance = result.iter().map(|x| x * x).sum::<f32>() / 4.0;
        assert!((variance - 1.0).abs() < 0.003);
    }
}
