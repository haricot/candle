//! EmbeddingGemma 2 text tower for Candle.
//!
//! Initial CPU F32 implementation of the bidirectional EmbeddingGemma2Model
//! architecture. Unlike Gemma 4's decoder, it has NO causal KV cache.
//! Image/audio towers are intentionally not loaded.
//!
//! The checkpoint layout and forward equations follow Google's
//! EmbeddingGemma2Model and the independent Hanzo implementation.
//! Model accuracy must be established against a pinned Transformers reference
//! before the backend is used to construct durable memory indexes.

use std::{collections::HashMap, sync::Arc};

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{embedding, linear_b, linear_no_bias, Activation, Embedding, Linear, VarBuilder};
use serde::Deserialize;

const MAX_TOKENS: usize = 8192;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct LayerShape {
    pub head_dim: usize,
    pub num_key_value_heads: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeSpec {
    pub rope_theta: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeSpecs {
    pub sliding_attention: RopeSpec,
    pub full_attention: RopeSpec,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_activation: Activation,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    pub hidden_size_per_layer_input: usize,
    pub embedding_dim: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
    #[serde(default)]
    pub per_layer_config: Option<HashMap<String, LayerShape>>,
    pub rope_parameters: RopeSpecs,
    #[serde(default)]
    pub attention_bias: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
}

impl TextConfig {
    fn layer_type(&self, index: usize) -> LayerType {
        match self.layer_types.as_ref() {
            Some(items) => items[index],
            None if (index + 1) % 6 == 0 || index + 1 == self.num_hidden_layers => {
                LayerType::FullAttention
            }
            None => LayerType::SlidingAttention,
        }
    }

    fn layer_shape(&self, index: usize) -> Result<LayerShape> {
        let fallback = if self.layer_type(index) == LayerType::FullAttention {
            LayerShape { head_dim: 512, num_key_value_heads: 1 }
        } else {
            LayerShape { head_dim: self.head_dim, num_key_value_heads: self.num_key_value_heads }
        };
        if let Some(per_layer) = &self.per_layer_config {
            for (key, shape) in per_layer {
                if key.parse::<usize>().ok() == Some(index) {
                    return Ok(*shape);
                }
            }
        }
        Ok(fallback)
    }

    pub fn validate(&self) -> Result<()> {
        if self.num_hidden_layers == 0 || self.hidden_size == 0
            || self.hidden_size_per_layer_input == 0
            || self.embedding_dim == 0 || self.sliding_window == 0
        {
            candle::bail!("EmbeddingGemma2 configuration contains zero dimensions")
        }
        if let Some(types) = &self.layer_types {
            if types.len() != self.num_hidden_layers {
                candle::bail!("EmbeddingGemma2 layer_types length is not num_hidden_layers")
            }
        }
        for layer in 0..self.num_hidden_layers {
            let shape = self.layer_shape(layer)?;
            if shape.head_dim % 2 != 0 || shape.num_key_value_heads == 0
                || self.num_attention_heads % shape.num_key_value_heads != 0
            {
                candle::bail!("EmbeddingGemma2 invalid attention shape at layer {layer}")
            }
        }
        Ok(())
    }
}

/// Gemma-4-style RMSNorm: checkpoint weight is a direct scale, NOT (1 + weight).
struct Norm {
    weight: Tensor,
    epsilon: f64,
}

impl Norm {
    fn load(width: usize, epsilon: f64, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self { weight: vb.get(width, "weight")?, epsilon })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let fp = xs.to_dtype(DType::F32)?;
        let denom = (fp.sqr()?.mean_keepdim(D::Minus1)? + self.epsilon)?.sqrt()?;
        fp.broadcast_div(&denom)?
            .to_dtype(dtype)?
            .broadcast_mul(&self.weight)
    }
}

/// Unweighted RMSNorm used for the attention value stream.
fn norm_v(xs: &Tensor, epsilon: f64) -> Result<Tensor> {
    let dtype = xs.dtype();
    let fp = xs.to_dtype(DType::F32)?;
    let denom = (fp.sqr()?.mean_keepdim(D::Minus1)? + epsilon)?.sqrt()?;
    fp.broadcast_div(&denom)?.to_dtype(dtype)
}

struct RoPE {
    sin: Tensor,
    cos: Tensor,
}

impl RoPE {
    fn new(dim: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv: Vec<f32> = (0..dim).step_by(2)
            .map(|i| theta.powf(-(i as f64 / dim as f64)) as f32)
            .collect();
        let inv = Tensor::from_vec(inv, (1, dim / 2), device)?;
        let pos = Tensor::arange(0u32, MAX_TOKENS as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_TOKENS, 1))?;
        let angles = pos.matmul(&inv)?;
        Ok(Self { sin: angles.sin()?, cos: angles.cos()? })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq = q.dim(2)?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Norm,
    k_norm: Norm,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    epsilon: f64,
    kind: LayerType,
    rope: Arc<RoPE>,
}

impl Attention {
    fn load(cfg: &TextConfig, index: usize, rope: Arc<RoPE>, vb: VarBuilder<'_>) -> Result<Self> {
        let shape = cfg.layer_shape(index)?;
        let hidden = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let kv_heads = shape.num_key_value_heads;
        let dim = shape.head_dim;
        let bias = cfg.attention_bias;
        Ok(Self {
            q: linear_b(hidden, heads * dim, bias, vb.pp("q_proj"))?,
            k: linear_b(hidden, kv_heads * dim, bias, vb.pp("k_proj"))?,
            v: linear_b(hidden, kv_heads * dim, bias, vb.pp("v_proj"))?,
            o: linear_b(heads * dim, hidden, bias, vb.pp("o_proj"))?,
            q_norm: Norm::load(dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: Norm::load(dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            heads, kv_heads, dim, epsilon: cfg.rms_norm_eps,
            kind: cfg.layer_type(index), rope,
        })
    }

    fn forward(&self, xs: &Tensor, sliding_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let heads = |tensor: Tensor, count: usize| -> Result<Tensor> {
            tensor.reshape((batch, seq, count, self.dim))?.transpose(1, 2)
        };
        let q = self.q_norm.forward(&heads(self.q.forward(xs)?, self.heads)?)?;
        let k = self.k_norm.forward(&heads(self.k.forward(xs)?, self.kv_heads)?)?;
        let v = norm_v(&heads(self.v.forward(xs)?, self.kv_heads)?, self.epsilon)?;
        let (q, k) = self.rope.apply(&q, &k)?;
        let groups = self.heads / self.kv_heads;
        let k = crate::utils::repeat_kv(k, groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, groups)?.contiguous()?;
        // EmbeddingGemma2 uses attention scale 1.0 (not 1/sqrt(head_dim)).
        let mut scores = q.matmul(&k.transpose(2, 3)?)?;
        if self.kind == LayerType::SlidingAttention {
            if let Some(mask) = sliding_mask { scores = scores.broadcast_add(mask)?; }
        }
        let scores = candle_nn::ops::softmax_last_dim(&scores)?;
        let result = scores.matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, seq, self.heads * self.dim))?;
        self.o.forward(&result)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
    activation: Activation,
}

impl Mlp {
    fn load(cfg: &TextConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            gate: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            activation: cfg.hidden_activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        (self.activation.forward(&self.gate.forward(xs)?)? * self.up.forward(xs)?)?
            .apply(&self.down)
    }
}

struct Layer {
    attention: Attention,
    mlp: Mlp,
    input_norm: Norm,
    post_attn_norm: Norm,
    pre_ff_norm: Norm,
    post_ff_norm: Norm,
    ple_input: Linear,
    ple_input_norm: Arc<Norm>,
    ple_gate: Linear,
    ple_project: Linear,
    ple_post_norm: Norm,
    scalar: Tensor,
    activation: Activation,
}

impl Layer {
    fn load(cfg: &TextConfig, i: usize, rope: Arc<RoPE>,
        ple_input: Linear, ple_input_norm: Arc<Norm>, vb: VarBuilder<'_>) -> Result<Self> {
        let n = |name| Norm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp(name));
        let ple = vb.pp("ple_block");
        let p = cfg.hidden_size_per_layer_input;
        Ok(Self {
            attention: Attention::load(cfg, i, rope, vb.pp("self_attn"))?,
            mlp: Mlp::load(cfg, vb.pp("mlp"))?,
            input_norm: n("input_layernorm")?,
            post_attn_norm: n("post_attention_layernorm")?,
            pre_ff_norm: n("pre_feedforward_layernorm")?,
            post_ff_norm: n("post_feedforward_layernorm")?,
            ple_input, ple_input_norm,
            ple_gate: linear_no_bias(cfg.hidden_size, p, ple.pp("per_layer_input_gate"))?,
            ple_project: linear_no_bias(p, cfg.hidden_size, ple.pp("per_layer_projection"))?,
            ple_post_norm: Norm::load(cfg.hidden_size, cfg.rms_norm_eps,
                ple.pp("post_per_layer_input_norm"))?,
            scalar: vb.get(1, "layer_scalar")?,
            activation: cfg.hidden_activation,
        })
    }

    fn forward(&self, xs: &Tensor, embeds: &Tensor,
        ple_scale: f64, sliding_mask: Option<&Tensor>) -> Result<Tensor> {
        let ple = (self.ple_input.forward(embeds)? * ple_scale)?
            .apply_norm(&self.ple_input_norm)?;
        let attn = self.attention.forward(&self.input_norm.forward(xs)?, sliding_mask)?;
        let xs = (xs + self.post_attn_norm.forward(&attn)?)?;
        let mlp = self.mlp.forward(&self.pre_ff_norm.forward(&xs)?)?;
        let xs = (&xs + self.post_ff_norm.forward(&mlp)?)?;
        let gated = (self.activation.forward(&self.ple_gate.forward(&xs)?)? * ple)?;
        let ple = self.ple_post_norm.forward(&self.ple_project.forward(&gated)?)?;
        (xs + ple)?.broadcast_mul(&self.scalar)
    }
}

/// Native, text-only EmbeddingGemma 2 encoder; one input sequence per request.
pub struct Model {
    embeddings: Embedding,
    layers: Vec<Layer>,
    norm: Norm,
    projection: Linear,
    sliding_window: usize,
    hidden_scale: f64,
    ple_scale: f64,
    embedding_dim: usize,
}

impl Model {
    /// The VarBuilder must address the top level of the HF safetensors checkpoint.
    /// Text tensors live under `language_model.*`, not `model.*`.
    pub fn load(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let t = &cfg.text_config;
        t.validate()?;
        if vb.dtype() == DType::F16 {
            candle::bail!("EmbeddingGemma 2 does not support F16; use F32 (Pascal) or BF16")
        }
        let vb = vb.pp("language_model");
        let embeddings = embedding(t.vocab_size, t.hidden_size, vb.pp("embed_tokens"))?;
        let ple_vb = vb.pp("ple");
        let p = t.hidden_size_per_layer_input;
        let ple_weights = ple_vb.get(
            (t.num_hidden_layers * p, t.hidden_size),
            "per_layer_model_projection.weight",
        )?;
        let ple_input_norm = Arc::new(Norm::load(p, t.rms_norm_eps,
            ple_vb.pp("per_layer_projection_norm"))?);

        let mut ropes: HashMap<(usize, bool), Arc<RoPE>> = HashMap::new();
        let mut layers = Vec::with_capacity(t.num_hidden_layers);
        for i in 0..t.num_hidden_layers {
            let kind = t.layer_type(i);
            let shape = t.layer_shape(i)?;
            let sliding = kind == LayerType::SlidingAttention;
            let key = (shape.head_dim, sliding);
            let rope = if let Some(rope) = ropes.get(&key) {
                rope.clone()
            } else {
                let theta = if sliding {
                    t.rope_parameters.sliding_attention.rope_theta
                } else {
                    t.rope_parameters.full_attention.rope_theta
                };
                let rope = Arc::new(RoPE::new(shape.head_dim, theta, vb.device())?);
                ropes.insert(key, rope.clone());
                rope
            };
            let ple_input = Linear::new(ple_weights.narrow(0, i * p, p)?, None);
            layers.push(Layer::load(t, i, rope, ple_input, ple_input_norm.clone(),
                vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self {
            embeddings, layers,
            norm: Norm::load(t.hidden_size, t.rms_norm_eps, vb.pp("norm"))?,
            projection: linear_no_bias(t.hidden_size, t.embedding_dim,
                vb.pp("embedding_projection"))?,
            sliding_window: t.sliding_window,
            hidden_scale: (t.hidden_size as f64).sqrt(),
            ple_scale: (t.hidden_size as f64).powf(-0.5),
            embedding_dim: t.embedding_dim,
        })
    }

    /// Returns per-token embeddings (1, tokens, embedding_dim).
    pub fn token_embeddings(&self, ids: &Tensor) -> Result<Tensor> {
        let (batch, seq) = ids.dims2()?;
        if batch != 1 || seq == 0 || seq > MAX_TOKENS {
            candle::bail!("EmbeddingGemma 2 expects one sequence of 1..=8192 tokens")
        }
        let embeds = (self.embeddings.forward(ids)? * self.hidden_scale)?;
        let mask = if seq > self.sliding_window + 1 {
            let data: Vec<f32> = (0..seq).flat_map(|i| (0..seq).map(move |j| {
                if i.abs_diff(j) > self.sliding_window { f32::NEG_INFINITY } else { 0. }
            })).collect();
            Some(Tensor::from_vec(data, (1, 1, seq, seq), ids.device())?)
        } else { None };
        let mut xs = embeds.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, &embeds, self.ple_scale, mask.as_ref())?;
        }
        self.projection.forward(&self.norm.forward(&xs)?)
    }

    /// Mean pooling and L2 normalization *after* Matryoshka truncation.
    /// This is the same order as the Sentence Transformers reference.
    pub fn embed(&self, ids: &Tensor, dims: usize) -> Result<Tensor> {
        if ![128, 256, 512, 768].contains(&dims) || dims > self.embedding_dim {
            candle::bail!("EmbeddingGemma 2 requested unsupported output dimension {dims}")
        }
        let token_states = self.token_embeddings(ids)?;
        let len = ids.dim(1)?;
        let mean = (token_states.sum(1)? / len as f64)?.narrow(1, 0, dims)?;
        let norm = (mean.sqr()?.sum_keepdim(1)? + 1e-12)?.sqrt()?;
        mean.broadcast_div(&norm)
    }
}

// Kept separate to make the PLE normalization explicit in the layer equation.
trait ApplyNorm {
    fn apply_norm(&self, norm: &Norm) -> Result<Tensor>;
}
impl ApplyNorm for Tensor {
    fn apply_norm(&self, norm: &Norm) -> Result<Tensor> { norm.forward(self) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_google_layer_shapes() {
        let cfg: Config = serde_json::from_str(r#"{
          "text_config": {
            "vocab_size": 262144, "hidden_size": 512, "intermediate_size": 2048,
            "num_hidden_layers": 24, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-6,
            "sliding_window": 512, "hidden_size_per_layer_input": 512,
            "embedding_dim": 768,
            "per_layer_config": {
                "05": {"head_dim":512,"num_key_value_heads":1},
                "11": {"head_dim":512,"num_key_value_heads":1},
                "17": {"head_dim":512,"num_key_value_heads":1},
                "23": {"head_dim":512,"num_key_value_heads":1}
            },
            "rope_parameters": {
                "sliding_attention": {"rope_theta":10000},
                "full_attention": {"rope_theta":1000000}
            }
          }
        }"#).unwrap();
        assert_eq!(cfg.text_config.layer_shape(4).unwrap().head_dim, 256);
        assert_eq!(cfg.text_config.layer_shape(5).unwrap().head_dim, 512);
        assert_eq!(cfg.text_config.layer_shape(23).unwrap().num_key_value_heads, 1);
        cfg.text_config.validate().unwrap();
    }
}
