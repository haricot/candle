//! NeMo-style FastConformer encoder for QVAC TranscribePsy.
//!
//! Full-utterance inference only. Inputs are log-Mel features [batch, frames,
//! features]. No microphone, model conversion, caching, or decoder dependency.
//! Tensor names and shapes are loaded directly from the official checkpoint.

use candle::{D, DType, Device, Result, Tensor};
use candle_nn::{
    BatchNorm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, ModuleT,
    VarBuilder,
};
use serde::Deserialize;

fn yes() -> bool { true }
fn batch_norm_name() -> String { "batch_norm".into() }

/// NeMo's `encoder` section; deliberately no conversion-only fields.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub feat_in: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_expansion_factor: usize,
    pub subsampling_factor: usize,
    pub subsampling: String,
    pub subsampling_conv_channels: usize,
    pub self_attention_model: String,
    pub conv_kernel_size: usize,
    pub pos_emb_max_len: usize,
    #[serde(default = "batch_norm_name")]
    pub conv_norm_type: String,
    #[serde(default)]
    pub conv_context_size: Option<serde_json::Value>,
    #[serde(default)]
    pub att_context_size: Option<Vec<i64>>,
    #[serde(default = "yes")]
    pub use_bias: bool,
    #[serde(default)]
    pub xscaling: bool,
    #[serde(default)]
    pub causal_downsampling: bool,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.feat_in == 0 || self.n_layers == 0 || self.d_model == 0
            || self.d_model % 2 != 0 || self.n_heads == 0
            || self.d_model % self.n_heads != 0 || self.ff_expansion_factor == 0
            || self.subsampling_conv_channels == 0 || self.pos_emb_max_len == 0
            || self.conv_kernel_size == 0
        {
            candle::bail!("invalid TranscribePsy FastConformer dimensions")
        }
        if self.subsampling_factor == 0 || !self.subsampling_factor.is_power_of_two()
            || (self.subsampling_factor > 1
                && (self.subsampling != "dw_striding" || self.causal_downsampling))
        {
            candle::bail!("unsupported FastConformer subsampling")
        }
        if !matches!(self.self_attention_model.as_str(), "rel_pos" | "rel_pos_local_attn") {
            candle::bail!("unsupported FastConformer attention type")
        }
        if !matches!(self.conv_norm_type.as_str(), "batch_norm" | "layer_norm") {
            candle::bail!("unsupported FastConformer conv_norm_type")
        }
        if let Some(v) = &self.att_context_size {
            if v.len() != 2 || v.iter().any(|&n| n < -1) {
                candle::bail!("invalid FastConformer attention context")
            }
        }
        self.conv_padding()?;
        Ok(())
    }

    fn conv_padding(&self) -> Result<(usize, usize)> {
        let k = self.conv_kernel_size;
        let default = ((k - 1) / 2, k / 2);
        match &self.conv_context_size {
            None | Some(serde_json::Value::Null) => Ok(default),
            Some(serde_json::Value::String(s)) if s == "default" => Ok(default),
            Some(serde_json::Value::String(s)) if s == "causal" => Ok((k - 1, 0)),
            Some(serde_json::Value::Array(v)) if v.len() == 2 => {
                let l = v[0].as_u64().and_then(|n| usize::try_from(n).ok());
                let r = v[1].as_u64().and_then(|n| usize::try_from(n).ok());
                match (l, r) {
                    (Some(l), Some(r)) if l.checked_add(r) == Some(k - 1) => Ok((l, r)),
                    _ => candle::bail!("invalid FastConformer convolution context"),
                }
            }
            _ => candle::bail!("unsupported FastConformer convolution context"),
        }
    }
}

fn norm(size: usize, vb: VarBuilder<'_>) -> Result<LayerNorm> {
    candle_nn::layer_norm(
        size,
        candle_nn::LayerNormConfig { eps: 1e-5, remove_mean: true, affine: true },
        vb,
    )
}

fn maybe_bias_linear(input: usize, output: usize, vb: VarBuilder<'_>) -> Result<Linear> {
    let weight = vb.get((output, input), "weight")?;
    let bias = if vb.contains_tensor("bias") {
        Some(vb.get((output,), "bias")?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

fn configured_linear(input: usize, output: usize, bias: bool, vb: VarBuilder<'_>) -> Result<Linear> {
    if bias { candle_nn::linear(input, output, vb) }
    else { candle_nn::linear_no_bias(input, output, vb) }
}

#[derive(Debug, Clone)]
struct FeedForward {
    first: Linear,
    second: Linear,
}

impl FeedForward {
    fn load(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let width = cfg.d_model.checked_mul(cfg.ff_expansion_factor)
            .ok_or_else(|| candle::Error::Msg("FastConformer FFN width overflow".into()))?;
        Ok(Self {
            first: maybe_bias_linear(cfg.d_model, width, vb.pp("linear1"))?,
            second: maybe_bias_linear(width, cfg.d_model, vb.pp("linear2"))?,
        })
    }
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.second.forward(&self.first.forward(input)?.silu()?)
    }
}

fn conv1d_native(vb: VarBuilder<'_>, in_ch: usize, out_ch: usize, k: usize,
                 groups: usize) -> Result<Conv1d> {
    let w = vb.get((out_ch, in_ch / groups, k), "weight")?;
    let b = if vb.contains_tensor("bias") { Some(vb.get((out_ch,), "bias")?) } else { None };
    Ok(Conv1d::new(w, b, Conv1dConfig {
        padding: 0, stride: 1, dilation: 1, groups, cudnn_fwd_algo: None,
    }))
}

#[derive(Debug, Clone)]
enum ConvNormalization {
    Layer(LayerNorm),
    Batch(BatchNorm),
}

#[derive(Debug, Clone)]
struct ConformerConvolution {
    expand: Conv1d,
    depthwise: Conv1d,
    project: Conv1d,
    normalize: ConvNormalization,
    pad_left: usize,
    pad_right: usize,
}

impl ConformerConvolution {
    fn load(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let d = cfg.d_model;
        let (pad_left, pad_right) = cfg.conv_padding()?;
        let normalize = match cfg.conv_norm_type.as_str() {
            "layer_norm" => ConvNormalization::Layer(norm(d, vb.pp("batch_norm"))?),
            "batch_norm" => ConvNormalization::Batch(
                candle_nn::batch_norm(d, 1e-5, vb.pp("batch_norm"))?),
            _ => candle::bail!("invalid convolution normalization"),
        };
        Ok(Self {
            expand: conv1d_native(vb.pp("pointwise_conv1"), d, d * 2, 1, 1)?,
            depthwise: conv1d_native(vb.pp("depthwise_conv"), d, d, cfg.conv_kernel_size, d)?,
            project: conv1d_native(vb.pp("pointwise_conv2"), d, d, 1, 1)?,
            normalize,
            pad_left,
            pad_right,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let expanded = self.expand.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?;
        let d = expanded.dims3()?.2 / 2;
        let a = expanded.narrow(2, 0, d)?;
        let gate = candle_nn::ops::sigmoid(&expanded.narrow(2, d, d)?)?;
        let mut hidden = (&a * &gate)?;
        if self.pad_left != 0 || self.pad_right != 0 {
            hidden = hidden.pad_with_zeros(D::Minus2, self.pad_left, self.pad_right)?;
        }
        let hidden = self.depthwise.forward(&hidden.transpose(1, 2)?)?;
        let hidden = match &self.normalize {
            ConvNormalization::Layer(ln) => ln.forward(&hidden.transpose(1, 2)?)?.transpose(1, 2)?,
            ConvNormalization::Batch(bn) => bn.forward_t(&hidden, false)?,
        };
        self.project.forward(&hidden.silu()?)?.transpose(1, 2)
    }
}

#[derive(Debug, Clone)]
struct RelativeAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    position: Linear,
    output: Linear,
    bias_u: Tensor,
    bias_v: Tensor,
    heads: usize,
    head_dim: usize,
}

impl RelativeAttention {
    fn load(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let d = cfg.d_model;
        Ok(Self {
            query: configured_linear(d, d, cfg.use_bias, vb.pp("linear_q"))?,
            key: configured_linear(d, d, cfg.use_bias, vb.pp("linear_k"))?,
            value: configured_linear(d, d, cfg.use_bias, vb.pp("linear_v"))?,
            position: candle_nn::linear_no_bias(d, d, vb.pp("linear_pos"))?,
            output: configured_linear(d, d, cfg.use_bias, vb.pp("linear_out"))?,
            bias_u: vb.get((cfg.n_heads, d / cfg.n_heads), "pos_bias_u")?,
            bias_v: vb.get((cfg.n_heads, d / cfg.n_heads), "pos_bias_v")?,
            heads: cfg.n_heads,
            head_dim: d / cfg.n_heads,
        })
    }

    fn forward(&self, x: &Tensor, position: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let shape = (b, t, self.heads, self.head_dim);
        let q = self.query.forward(x)?.reshape(shape)?;
        let k = self.key.forward(x)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let v = self.value.forward(x)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let p = self.position.forward(position)?
            .reshape((1, 2 * t - 1, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .broadcast_as((b, self.heads, 2 * t - 1, self.head_dim))?
            .contiguous()?;
        let u = q.broadcast_add(&self.bias_u.reshape((1, 1, self.heads, self.head_dim))?)?
            .transpose(1, 2)?.contiguous()?;
        let w = q.broadcast_add(&self.bias_v.reshape((1, 1, self.heads, self.head_dim))?)?
            .transpose(1, 2)?.contiguous()?;
        let content = u.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let relative = w.matmul(&p.transpose(2, 3)?.contiguous()?)?;
        // Transformer-XL's relative shift: align the zero-relative index with
        // every query, then keep exactly the causal/noncausal key columns.
        let relative = relative.pad_with_zeros(D::Minus1, 1, 0)?;
        let relative = relative.reshape((b, self.heads, 2 * t, t))?;
        let relative = relative.narrow(2, 1, 2 * t - 1)?
            .reshape((b, self.heads, t, 2 * t - 1))?
            .narrow(3, 0, t)?;
        let logits = ((&content + &relative)? * (self.head_dim as f64).powf(-0.5))?;
        let logits = logits.broadcast_add(mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&logits)?;
        self.output.forward(
            &weights.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((b, t, d))?
        )
    }
}

fn position_table(t: usize, dimension: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    if t == 0 { candle::bail!("empty FastConformer sequence") }
    let count = 2 * t - 1;
    let mut values = vec![0f32; count * dimension];
    for row in 0..count {
        let delta = (t - 1) as f32 - row as f32;
        for i in 0..dimension / 2 {
            let a = delta / 10000f32.powf((2 * i) as f32 / dimension as f32);
            values[row * dimension + i * 2] = a.sin();
            values[row * dimension + i * 2 + 1] = a.cos();
        }
    }
    Tensor::from_vec(values, (1, count, dimension), device)?.to_dtype(dtype)
}

/// One mask per batch element; a padded query keeps key 0 to avoid a fully
/// masked softmax row. Padded query outputs are not externally observable.
fn attention_mask(lengths: &[usize], t: usize, context: Option<&[i64]>,
                  dtype: DType, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; lengths.len() * t * t];
    for (b, &valid) in lengths.iter().enumerate() {
        if valid == 0 || valid > t { candle::bail!("invalid FastConformer input length") }
        for q in 0..t {
            for k in 0..t {
                let out_of_range = if q >= valid { k != 0 } else { k >= valid };
                let outside_context = if let Some(c) = context {
                    (c[0] >= 0 && k < q.saturating_sub(c[0] as usize))
                        || (c[1] >= 0 && k > q.saturating_add(c[1] as usize))
                } else { false };
                if out_of_range || (q < valid && outside_context) {
                    data[(b * t + q) * t + k] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Tensor::from_vec(data, (lengths.len(), 1, t, t), device)?.to_dtype(dtype)
}

#[derive(Debug, Clone)]
struct Block {
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_att: LayerNorm,
    attention: RelativeAttention,
    norm_conv: LayerNorm,
    convolution: ConformerConvolution,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_output: LayerNorm,
}

impl Block {
    fn load(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let d = cfg.d_model;
        Ok(Self {
            norm_ff1: norm(d, vb.pp("norm_feed_forward1"))?,
            ff1: FeedForward::load(cfg, vb.pp("feed_forward1"))?,
            norm_att: norm(d, vb.pp("norm_self_att"))?,
            attention: RelativeAttention::load(cfg, vb.pp("self_attn"))?,
            norm_conv: norm(d, vb.pp("norm_conv"))?,
            convolution: ConformerConvolution::load(cfg, vb.pp("conv"))?,
            norm_ff2: norm(d, vb.pp("norm_feed_forward2"))?,
            ff2: FeedForward::load(cfg, vb.pp("feed_forward2"))?,
            norm_output: norm(d, vb.pp("norm_out"))?,
        })
    }

    fn forward(&self, input: &Tensor, pos: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let ff1 = self.ff1.forward(&self.norm_ff1.forward(input)?)?;
        let x = (input + &(&ff1 * 0.5)?)?;
        let att = self.attention.forward(&self.norm_att.forward(&x)?, pos, mask)?;
        let x = (&x + &att)?;
        let conv = self.convolution.forward(&self.norm_conv.forward(&x)?)?;
        let x = (&x + &conv)?;
        let ff2 = self.ff2.forward(&self.norm_ff2.forward(&x)?)?;
        self.norm_output.forward(&(&x + &(&ff2 * 0.5)?)?)
    }
}

fn conv2d_native(vb: VarBuilder<'_>, input: usize, output: usize,
                 kernel: usize, groups: usize, stride: usize) -> Result<Conv2d> {
    let weight = vb.get((output, input / groups, kernel, kernel), "weight")?;
    // NeMo serializes subsampling convolution biases, unlike some block convs.
    let bias = vb.get((output,), "bias")?;
    Ok(Conv2d::new(weight, Some(bias), Conv2dConfig {
        padding: if kernel == 3 { 1 } else { 0 },
        stride, dilation: 1, groups, cudnn_fwd_algo: None,
    }))
}

#[derive(Debug, Clone)]
struct Subsample {
    convs: Vec<(Conv2d, bool)>,
    projection: Linear,
}

impl Subsample {
    fn load(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let stages = cfg.subsampling_factor.trailing_zeros() as usize;
        let c = cfg.subsampling_conv_channels;
        let mut convs = Vec::with_capacity(2 * stages - 1);
        convs.push((conv2d_native(vb.pp("conv").pp(0), 1, c, 3, 1, 2)?, true));
        for stage in 1..stages {
            let depth_index = 3 * stage - 1;
            let point_index = depth_index + 1;
            convs.push((conv2d_native(vb.pp("conv").pp(depth_index), c, c, 3, c, 2)?, true));
            convs.push((conv2d_native(vb.pp("conv").pp(point_index), c, c, 1, 1, 1)?, false));
        }
        let mut frequency = cfg.feat_in;
        for _ in 0..stages { frequency = frequency.div_ceil(2); }
        Ok(Self {
            convs,
            projection: candle_nn::linear(c * frequency, cfg.d_model, vb.pp("out"))?,
        })
    }

    fn forward(&self, features: &Tensor, input_lengths: &[usize]) -> Result<(Tensor, Vec<usize>)> {
        let mut lengths = input_lengths.to_vec();
        let mut x = features.unsqueeze(1)?;
        for (conv, activate) in &self.convs {
            x = conv.forward(&x)?;
            if *activate {
                x = x.relu()?;
                for len in &mut lengths { *len = len.div_ceil(2); }
            }
        }
        let (b, t, c, f) = x.transpose(1, 2)?.dims4()?;
        let x = x.transpose(1, 2)?.reshape((b, t, c * f))?;
        Ok((self.projection.forward(&x)?, lengths))
    }
}

#[derive(Debug, Clone)]
enum Front {
    IdentityProjection(Linear),
    Downsample(Subsample),
}

/// Full-utterance FastConformer with QVAC-native tensor names.
#[derive(Debug, Clone)]
pub struct FastConformer {
    cfg: Config,
    front: Front,
    layers: Vec<Block>,
}

impl FastConformer {
    /// `vb` must already point at the `encoder` prefix, not checkpoint root.
    pub fn load(cfg: Config, vb: VarBuilder<'_>) -> Result<Self> {
        cfg.validate()?;
        let front = if cfg.subsampling_factor == 1 {
            Front::IdentityProjection(candle_nn::linear(cfg.feat_in, cfg.d_model, vb.pp("pre_encode"))?)
        } else {
            Front::Downsample(Subsample::load(&cfg, vb.pp("pre_encode"))?)
        };
        let layers = (0..cfg.n_layers).map(|i| Block::load(&cfg, vb.pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { cfg, front, layers })
    }

    /// Return [batch, reduced_frames, d_model] and per-sample frame lengths.
    /// The input is log-Mel [batch, frames, feat_in], `lengths` in input frames.
    pub fn forward(&self, features: &Tensor, lengths: &[usize]) -> Result<(Tensor, Vec<usize>)> {
        let (batch, frames, freq) = features.dims3()?;
        if features.dtype() != DType::F32 && features.dtype() != DType::F16 {
            candle::bail!("FastConformer needs f32/f16 log-Mel tensors")
        }
        if freq != self.cfg.feat_in || batch == 0 || frames == 0 || lengths.len() != batch {
            candle::bail!("invalid FastConformer log-Mel layout or lengths")
        }
        if lengths.iter().any(|&n| n == 0 || n > frames) {
            candle::bail!("invalid FastConformer nonzero frame lengths")
        }
        let (mut x, out_lengths) = match &self.front {
            Front::IdentityProjection(p) => (p.forward(features)?, lengths.to_vec()),
            Front::Downsample(down) => down.forward(features, lengths)?,
        };
        let (_, t, _) = x.dims3()?;
        let mask = attention_mask(
            &out_lengths, t,
            if self.cfg.self_attention_model == "rel_pos_local_attn" {
                self.cfg.att_context_size.as_deref()
            } else { None }, x.dtype(), x.device()
        )?;
        if self.cfg.xscaling { x = (&x * (self.cfg.d_model as f64).sqrt())?; }
        let pe = position_table(t, self.cfg.d_model, x.dtype(), x.device())?;
        for block in &self.layers { x = block.forward(&x, &pe, &mask)?; }
        Ok((x, out_lengths))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn sample_config() -> Config {
        Config {
            feat_in: 4, n_layers: 1, d_model: 4, n_heads: 2, ff_expansion_factor: 2,
            subsampling_factor: 1, subsampling: "dw_striding".into(),
            subsampling_conv_channels: 2, self_attention_model: "rel_pos_local_attn".into(),
            conv_kernel_size: 3, pos_emb_max_len: 16,
            conv_norm_type: "layer_norm".into(), conv_context_size: None,
            att_context_size: Some(vec![2, 1]), use_bias: true, xscaling: false,
            causal_downsampling: false,
        }
    }

    fn tensor_map(c: &Config) -> Result<HashMap<String, Tensor>> {
        let mut map = HashMap::new();
        let device = Device::Cpu;
        let d = c.d_model;
        let prefix = "encoder.layers.0.";
        let mut insert = |name: String, shape: Vec<usize>, unit: bool| -> Result<()> {
            let len: usize = shape.iter().product();
            let values = vec![if unit { 1f32 } else { 0f32 }; len];
            let t = Tensor::from_vec(values, shape, &device)?;
            map.insert(name, t);
            Ok(())
        };
        if c.subsampling_factor == 1 {
            insert("encoder.pre_encode.weight".into(), vec![d, c.feat_in], true)?;
            insert("encoder.pre_encode.bias".into(), vec![d], false)?;
        }
        for name in ["norm_feed_forward1", "norm_self_att", "norm_conv",
                     "norm_feed_forward2", "norm_out", "conv.batch_norm"] {
            insert(format!("{prefix}{name}.weight"), vec![d], true)?;
            insert(format!("{prefix}{name}.bias"), vec![d], false)?;
        }
        let w = d * c.ff_expansion_factor;
        for ff in ["feed_forward1", "feed_forward2"] {
            insert(format!("{prefix}{ff}.linear1.weight"), vec![w, d], false)?;
            insert(format!("{prefix}{ff}.linear2.weight"), vec![d, w], false)?;
        }
        for key in ["linear_q", "linear_k", "linear_v", "linear_out"] {
            insert(format!("{prefix}self_attn.{key}.weight"), vec![d, d], false)?;
            insert(format!("{prefix}self_attn.{key}.bias"), vec![d], false)?;
        }
        insert(format!("{prefix}self_attn.linear_pos.weight"), vec![d, d], false)?;
        insert(format!("{prefix}self_attn.pos_bias_u"), vec![c.n_heads, d / c.n_heads], false)?;
        insert(format!("{prefix}self_attn.pos_bias_v"), vec![c.n_heads, d / c.n_heads], false)?;
        for (name, shape) in [
            ("pointwise_conv1", vec![2 * d, d, 1]),
            ("depthwise_conv", vec![d, 1, c.conv_kernel_size]),
            ("pointwise_conv2", vec![d, d, 1]),
        ] {
            insert(format!("{prefix}conv.{name}.weight"), shape, false)?;
        }
        Ok(map)
    }

    #[test]
    fn rejects_unsupported_native_configs() {
        let mut cfg = sample_config();
        assert!(cfg.validate().is_ok());
        cfg.conv_norm_type = "instance_norm".into();
        assert!(cfg.validate().is_err());
        cfg = sample_config();
        cfg.conv_context_size = Some(serde_json::json!([0, 0]));
        assert!(cfg.validate().is_err());
        cfg = sample_config();
        cfg.n_heads = 3;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn native_encoder_layout_and_masked_padding() -> Result<()> {
        let cfg = sample_config();
        let mut tensors = tensor_map(&cfg)?;
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, &Device::Cpu).pp("encoder");
        let encoder = FastConformer::load(cfg.clone(), vb)?;
        let data = Tensor::from_vec(vec![0.3f32; 2 * 5 * 4], (2, 5, 4), &Device::Cpu)?;
        let (out, lengths) = encoder.forward(&data, &[5, 3])?;
        assert_eq!(out.dims(), &[2, 5, 4]);
        assert_eq!(lengths, vec![5, 3]);
        assert!(out.flatten_all()?.to_vec1::<f32>()?.iter().all(|x| x.is_finite()));
        tensors.remove("encoder.layers.0.self_attn.pos_bias_v");
        let missing = FastConformer::load(
            cfg, VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).pp("encoder")
        );
        assert!(missing.is_err(), "relative positional bias is mandatory");
        Ok(())
    }

    #[test]
    fn subsampling_has_native_depthwise_indexes_and_ceil_lengths() -> Result<()> {
        let mut cfg = sample_config();
        cfg.subsampling_factor = 4;
        cfg.feat_in = 6;
        let mut map = HashMap::new();
        let mut add = |suffix: &str, shape: Vec<usize>| -> Result<()> {
            map.insert(format!("encoder.pre_encode.{suffix}"),
                Tensor::zeros(shape, DType::F32, &Device::Cpu)?);
            Ok(())
        };
        add("conv.0.weight", vec![2, 1, 3, 3])?;
        add("conv.0.bias", vec![2])?;
        add("conv.2.weight", vec![2, 1, 3, 3])?;
        add("conv.2.bias", vec![2])?;
        add("conv.3.weight", vec![2, 2, 1, 1])?;
        add("conv.3.bias", vec![2])?;
        add("out.weight", vec![4, 4])?; // [d_model, 2 channels * ceil(6/4)]
        add("out.bias", vec![4])?;
        let down = Subsample::load(&cfg,
            VarBuilder::from_tensors(map.clone(), DType::F32, &Device::Cpu).pp("encoder").pp("pre_encode"))?;
        let data = Tensor::zeros((2, 9, 6), DType::F32, &Device::Cpu)?;
        let (out, lengths) = down.forward(&data, &[9, 7])?;
        assert_eq!(out.dims(), &[2, 3, 4]);
        assert_eq!(lengths, vec![3, 2]);
        map.remove("encoder.pre_encode.conv.3.bias");
        let missing = Subsample::load(&cfg,
            VarBuilder::from_tensors(map, DType::F32, &Device::Cpu).pp("encoder").pp("pre_encode"));
        assert!(missing.is_err(), "QVAC subsampler native bias is mandatory");
        Ok(())
    }
}
