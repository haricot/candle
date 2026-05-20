use crate::models::with_tracing::{linear_b, linear_no_bias, Linear};
use candle::{Module, Result, Tensor};
use candle_nn::{kv_cache::ConcatKvCache, VarBuilder};
use std::sync::Arc;

use super::qwen3_5::{Config as Qwen3_5Config, Qwen3_5TextRotaryEmbedding};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DFlashConfig {
    pub target_layer_ids: Vec<usize>,
    pub block_size: usize,
    pub mask_token_id: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Qwen3_5DFlashAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: crate::models::qwen3_5::Qwen3_5RmsNorm,
    k_norm: crate::models::qwen3_5::Qwen3_5RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<Qwen3_5TextRotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl Qwen3_5DFlashAttention {
    pub fn new(
        cfg: &Qwen3_5Config,
        rotary_emb: Arc<Qwen3_5TextRotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.text_config.num_attention_heads;
        let num_kv_heads = cfg.text_config.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = linear_b(
            cfg.text_config.hidden_size,
            num_heads * head_dim,
            cfg.text_config.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.text_config.hidden_size,
            num_kv_heads * head_dim,
            cfg.text_config.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.text_config.hidden_size,
            num_kv_heads * head_dim,
            cfg.text_config.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.text_config.hidden_size,
            cfg.text_config.attention_bias,
            vb.pp("o_proj"),
        )?;

        let q_norm = crate::models::qwen3_5::Qwen3_5RmsNorm::new(head_dim, cfg.text_config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = crate::models::qwen3_5::Qwen3_5RmsNorm::new(head_dim, cfg.text_config.rms_norm_eps, vb.pp("k_norm"))?;

        let hidden_size = head_dim * cfg.text_config.num_attention_heads;
        let kv_cache = ConcatKvCache::new(2);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        target_hidden: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, q_len, _) = hidden_states.dims3()?;
        let ctx_len = target_hidden.dim(1)?;

        let q = self.q_proj.forward(hidden_states)?;
        let q = q
            .reshape((b, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.q_norm.forward(&q.flatten(0, 2)?)?
            .reshape((b, self.num_heads, q_len, self.head_dim))?;

        let k_ctx = self.k_proj.forward(target_hidden)?;
        let k_noise = self.k_proj.forward(hidden_states)?;
        let v_ctx = self.v_proj.forward(target_hidden)?;
        let v_noise = self.v_proj.forward(hidden_states)?;

        let k = Tensor::cat(&[k_ctx, k_noise], 1)?
            .reshape((b, ctx_len + q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = Tensor::cat(&[v_ctx, v_noise], 1)?
            .reshape((b, ctx_len + q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let k = self.k_norm.forward(&k.flatten(0, 2)?)?
            .reshape((b, self.num_kv_heads, ctx_len + q_len, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        let (k, v) = self.kv_cache.append(&k, &v)?;

        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        ctx.transpose(1, 2)?
            .reshape((b, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct DFlashDecoderLayer {
    self_attn: Qwen3_5DFlashAttention,
    mlp: crate::models::qwen3_5::Qwen3_5MLP,
    input_layernorm: crate::models::qwen3_5::Qwen3_5RmsNorm,
    post_attention_layernorm: crate::models::qwen3_5::Qwen3_5RmsNorm,
}

impl DFlashDecoderLayer {
    fn new(cfg: &Qwen3_5Config, rotary: Arc<Qwen3_5TextRotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen3_5DFlashAttention::new(cfg, rotary, vb.pp("self_attn"))?;
        let mlp = crate::models::qwen3_5::Qwen3_5MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = crate::models::qwen3_5::Qwen3_5RmsNorm::new(cfg.text_config.hidden_size, cfg.text_config.rms_norm_eps, vb.pp("input_layernorm"))?;
        let ln2 = crate::models::qwen3_5::Qwen3_5RmsNorm::new(
            cfg.text_config.hidden_size,
            cfg.text_config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm: ln2,
        })
    }

    fn forward(&mut self, hidden_states: &Tensor, target_hidden: &Tensor, offset: usize) -> Result<Tensor> {
        let residual = hidden_states;
        let h = self.input_layernorm.forward(hidden_states)?;
        let h = self.self_attn.forward(&h, target_hidden, offset)?;
        let hidden_states = (residual + h)?;

        let residual = &hidden_states;
        let h = self.post_attention_layernorm.forward(&hidden_states)?;
        let h = self.mlp.forward(&h)?;
        residual + h
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct DFlashDraftModel {
    layers: Vec<DFlashDecoderLayer>,
    norm: crate::models::qwen3_5::Qwen3_5RmsNorm,
    fc: Linear,
    hidden_norm: crate::models::qwen3_5::Qwen3_5RmsNorm,
    pub target_layer_ids: Vec<usize>,
    pub block_size: usize,
    pub mask_token_id: Option<u32>,
}

impl DFlashDraftModel {
    pub fn new(cfg: &Qwen3_5Config, dflash_cfg: &DFlashConfig, vb: VarBuilder) -> Result<Self> {
        let rotary_emb = Arc::new(Qwen3_5TextRotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.text_config.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for i in 0..cfg.text_config.num_hidden_layers {
            layers.push(DFlashDecoderLayer::new(cfg, rotary_emb.clone(), vb_l.pp(i))?);
        }

        let fc = linear_no_bias(
            dflash_cfg.target_layer_ids.len() * cfg.text_config.hidden_size,
            cfg.text_config.hidden_size,
            vb.pp("fc"),
        )?;
        let hidden_norm = crate::models::qwen3_5::Qwen3_5RmsNorm::new(cfg.text_config.hidden_size, cfg.text_config.rms_norm_eps, vb.pp("hidden_norm"))?;
        let norm = crate::models::qwen3_5::Qwen3_5RmsNorm::new(cfg.text_config.hidden_size, cfg.text_config.rms_norm_eps, vb.pp("norm"))?;

        Ok(Self {
            layers,
            norm,
            fc,
            hidden_norm,
            target_layer_ids: dflash_cfg.target_layer_ids.clone(),
            block_size: dflash_cfg.block_size,
            mask_token_id: dflash_cfg.mask_token_id,
        })
    }

    pub fn forward(
        &mut self,
        target_hidden: &Tensor,
        noise_embedding: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        let target_hidden = target_hidden.apply(&self.fc)?.apply(&self.hidden_norm)?;
        let mut h = noise_embedding.clone();

        for layer in &mut self.layers {
            h = layer.forward(&h, &target_hidden, offset)?;
        }
        self.norm.forward(&h)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}
