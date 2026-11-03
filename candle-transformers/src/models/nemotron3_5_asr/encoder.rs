//! Cache-aware FastConformer using the official HF checkpoint tensor names.
//! Single-utterance inference; batch dimension is explicitly one.
use super::{Config, EncoderConfig};
use candle::{DType, Device, Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder,
};

fn linear(input: usize, output: usize, bias: bool, vb: VarBuilder<'_>) -> Result<Linear> {
    if bias {
        candle_nn::linear(input, output, vb)
    } else {
        candle_nn::linear_no_bias(input, output, vb)
    }
}
fn norm(width: usize, vb: VarBuilder<'_>) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        vb.get(width, "weight")?,
        vb.get(width, "bias")?,
        1e-5,
    ))
}
fn pointwise(input: usize, output: usize, bias: bool, vb: VarBuilder<'_>) -> Result<Linear> {
    let weight = vb.get((output, input, 1), "weight")?.squeeze(2)?;
    let bias = if bias {
        Some(vb.get(output, "bias")?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

#[derive(Debug, Clone)]
struct CausalConv2d {
    conv: Conv2d,
}
impl CausalConv2d {
    fn load(input: usize, output: usize, groups: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            conv: Conv2d::new(
                vb.get((output, input / groups, 3, 3), "weight")?,
                Some(vb.get(output, "bias")?),
                Conv2dConfig {
                    padding: 0,
                    stride: 2,
                    dilation: 1,
                    groups,
                    cudnn_fwd_algo: None,
                },
            ),
        })
    }
    fn forward(&self, x: &Tensor, cache: Option<&mut Option<Tensor>>) -> Result<Tensor> {
        let x = x.pad_with_zeros(3, 2, 1)?;
        let input = if let Some(cache) = cache {
            let (b, c, t, f) = x.dims4()?;
            let old = match cache.as_ref() {
                Some(old) => old.clone(),
                None => Tensor::zeros((b, c, 2, f), x.dtype(), x.device())?,
            };
            // Subsequent chunks prepend k-stride=1 cached row; only the
            // first call has the full k-1=2 initial zero rows.
            *cache = Some(x.narrow(2, t - 1, 1)?.contiguous()?);
            Tensor::cat(&[&old, &x], 2)?
        } else {
            x.pad_with_zeros(2, 2, 1)?
        };
        self.conv.forward(&input)
    }
}

#[derive(Debug, Clone)]
struct Subsampling {
    stem: CausalConv2d,
    stages: Vec<(CausalConv2d, Conv2d)>,
    projection: Linear,
}
fn mask_time_2d(x: &Tensor, valid: Option<usize>) -> Result<Tensor> {
    if let Some(valid) = valid {
        let t = x.dim(2)?;
        let mask = Tensor::from_vec(
            (0..t)
                .map(|i| if i < valid { 1f32 } else { 0f32 })
                .collect(),
            (1, 1, t, 1),
            x.device(),
        )?
        .to_dtype(x.dtype())?;
        x.broadcast_mul(&mask)
    } else {
        Ok(x.clone())
    }
}
impl Subsampling {
    fn load(c: &EncoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let channels = c.subsampling_conv_channels;
        let depth = c.subsampling_factor.trailing_zeros() as usize;
        let stem = CausalConv2d::load(1, channels, 1, vb.pp("conv_in"))?;
        let mut stages = Vec::new();
        for i in 0..depth - 1 {
            let layer = vb.pp("layers").pp(i);
            let point = Conv2d::new(
                layer
                    .pp("pointwise_conv")
                    .get((channels, channels, 1, 1), "weight")?,
                Some(layer.pp("pointwise_conv").get(channels, "bias")?),
                Conv2dConfig::default(),
            );
            stages.push((
                CausalConv2d::load(channels, channels, channels, layer.pp("depthwise_conv"))?,
                point,
            ));
        }
        let mut freq = c.num_mel_bins;
        for _ in 0..depth {
            freq = freq / 2 + 1;
        }
        let projection = candle_nn::linear(channels * freq, c.hidden_size, vb.pp("linear"))?;
        Ok(Self {
            stem,
            stages,
            projection,
        })
    }
    fn forward(
        &self,
        features: &Tensor,
        valid: Option<usize>,
        mut caches: Option<&mut Vec<Option<Tensor>>>,
    ) -> Result<(Tensor, Option<usize>)> {
        let stream = caches.is_some();
        let mut length = valid;
        let mut x = mask_time_2d(&features.unsqueeze(1)?, length)?;
        x = self.stem.forward(&x, caches.as_mut().map(|c| &mut c[0]))?;
        length = length.map(|n| if stream { n / 2 } else { n / 2 + 1 });
        x = mask_time_2d(&x, length)?.relu()?;
        for (i, (depth, point)) in self.stages.iter().enumerate() {
            x = depth.forward(&x, caches.as_mut().map(|c| &mut c[i + 1]))?;
            length = length.map(|n| if stream { n / 2 } else { n / 2 + 1 });
            x = point.forward(&x)?;
            x = mask_time_2d(&x, length)?.relu()?;
        }
        let (b, channels, t, freq) = x.dims4()?;
        let x = x
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, channels * freq))?;
        Ok((self.projection.forward(&x)?, length))
    }
}

#[derive(Debug, Clone, Default)]
struct LayerCache {
    key: Option<Tensor>,
    value: Option<Tensor>,
    convolution: Option<Tensor>,
}
#[derive(Debug, Clone)]
pub struct EncoderState {
    subsampling: Vec<Option<Tensor>>,
    layers: Vec<LayerCache>,
    frames_seen: usize,
}
impl EncoderState {
    pub fn frames_seen(&self) -> usize {
        self.frames_seen
    }
    pub fn attention_cache_frames(&self) -> usize {
        self.layers
            .first()
            .and_then(|s| s.key.as_ref())
            .map_or(0, |k| k.dims()[2])
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    first: Linear,
    second: Linear,
}
impl FeedForward {
    fn load(c: &EncoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            first: linear(
                c.hidden_size,
                c.intermediate_size,
                c.attention_bias,
                vb.pp("linear1"),
            )?,
            second: linear(
                c.intermediate_size,
                c.hidden_size,
                c.attention_bias,
                vb.pp("linear2"),
            )?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.second.forward(&self.first.forward(x)?.silu()?)
    }
}

#[derive(Debug, Clone)]
struct Convolution {
    expand: Linear,
    depth: Conv1d,
    project: Linear,
    norm: LayerNorm,
    width: usize,
    history: usize,
}
impl Convolution {
    fn load(c: &EncoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let h = c.hidden_size;
        let depth = Conv1d::new(
            vb.pp("depthwise_conv")
                .get((h, 1, c.conv_kernel_size), "weight")?,
            if c.convolution_bias {
                Some(vb.pp("depthwise_conv").get(h, "bias")?)
            } else {
                None
            },
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: h,
                cudnn_fwd_algo: None,
            },
        );
        Ok(Self {
            expand: pointwise(h, 2 * h, c.convolution_bias, vb.pp("pointwise_conv1"))?,
            depth,
            project: pointwise(h, h, c.convolution_bias, vb.pp("pointwise_conv2"))?,
            norm: norm(h, vb.pp("norm"))?,
            width: h,
            history: c.conv_kernel_size - 1,
        })
    }
    fn forward(&self, x: &Tensor, valid: usize, cache: Option<&mut LayerCache>) -> Result<Tensor> {
        let expanded = self.expand.forward(x)?;
        let hidden = (&expanded.narrow(2, 0, self.width)?
            * candle_nn::ops::sigmoid(&expanded.narrow(2, self.width, self.width)?)?)?;
        let (b, t, _) = hidden.dims3()?;
        let valid_mask = Tensor::from_vec(
            (0..t)
                .map(|i| if i < valid { 1f32 } else { 0f32 })
                .collect(),
            (1, t, 1),
            hidden.device(),
        )?
        .to_dtype(hidden.dtype())?;
        let hidden = hidden
            .broadcast_mul(&valid_mask)?
            .transpose(1, 2)?
            .contiguous()?;
        let hidden = if let Some(cache) = cache {
            let old = match &cache.convolution {
                Some(old) => old.clone(),
                None => Tensor::zeros(
                    (b, self.width, self.history),
                    hidden.dtype(),
                    hidden.device(),
                )?,
            };
            let joined = Tensor::cat(&[&old, &hidden], 2)?;
            cache.convolution = Some(joined.narrow(2, t, self.history)?.contiguous()?);
            joined
        } else {
            hidden.pad_with_zeros(2, self.history, 0)?
        };
        let hidden = self.depth.forward(&hidden)?.transpose(1, 2)?.contiguous()?;
        self.project.forward(&self.norm.forward(&hidden)?.silu()?)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    relative: Linear,
    output: Linear,
    bias_u: Tensor,
    bias_v: Tensor,
    heads: usize,
    head_dim: usize,
    history: usize,
}
impl Attention {
    fn load(c: &EncoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let h = c.hidden_size;
        Ok(Self {
            q: linear(h, h, c.attention_bias, vb.pp("q_proj"))?,
            k: linear(h, h, c.attention_bias, vb.pp("k_proj"))?,
            v: linear(h, h, c.attention_bias, vb.pp("v_proj"))?,
            output: linear(h, h, c.attention_bias, vb.pp("o_proj"))?,
            relative: candle_nn::linear_no_bias(h, h, vb.pp("relative_k_proj"))?,
            bias_u: vb.get((c.num_attention_heads, h / c.num_attention_heads), "bias_u")?,
            bias_v: vb.get((c.num_attention_heads, h / c.num_attention_heads), "bias_v")?,
            heads: c.num_attention_heads,
            head_dim: h / c.num_attention_heads,
            history: c.sliding_window - 1,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        position: &Tensor,
        mask: &Tensor,
        cache: Option<&mut LayerCache>,
    ) -> Result<Tensor> {
        let (b, t, h) = x.dims3()?;
        let shape = (b, t, self.heads, self.head_dim);
        let q = self
            .q
            .forward(x)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let mut k = self
            .k
            .forward(x)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = self
            .v
            .forward(x)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        if let Some(cache) = cache {
            if let Some(old) = &cache.key {
                k = Tensor::cat(&[old, &k], 2)?;
            }
            if let Some(old) = &cache.value {
                v = Tensor::cat(&[old, &v], 2)?;
            }
            let total = k.dim(2)?;
            let keep = self.history.min(total);
            cache.key = Some(k.narrow(2, total - keep, keep)?.contiguous()?);
            cache.value = Some(v.narrow(2, total - keep, keep)?.contiguous()?);
        }
        let keys = k.dim(2)?;
        let p_len = position.dim(1)?;
        let p = self
            .relative
            .forward(position)?
            .reshape((b, p_len, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let u = q.broadcast_add(&self.bias_u.reshape((1, self.heads, 1, self.head_dim))?)?;
        let w = q.broadcast_add(&self.bias_v.reshape((1, self.heads, 1, self.head_dim))?)?;
        let content = u.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let relative = w.matmul(&p.transpose(2, 3)?.contiguous()?)?;
        let relative = relative
            .pad_with_zeros(3, 1, 0)?
            .reshape((b, self.heads, p_len + 1, t))?
            .narrow(2, 1, p_len)?
            .reshape((b, self.heads, t, p_len))?
            .narrow(3, 0, keys)?;
        let scores = ((content + relative)? * (self.head_dim as f64).powf(-0.5))?;
        let allow = mask.broadcast_as(scores.shape())?;
        let fill = Tensor::full(-10000f32, scores.shape(), x.device())?.to_dtype(x.dtype())?;
        let scores = allow.where_cond(&scores, &fill)?;
        let weights =
            candle_nn::ops::softmax_last_dim(&scores)?.mul(&allow.to_dtype(x.dtype())?)?;
        let context = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, h))?;
        self.output.forward(&context)
    }
}

fn positions(total: usize, width: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; (2 * total - 1) * width];
    for row in 0..2 * total - 1 {
        let delta = (total - 1) as f32 - row as f32;
        for i in 0..width / 2 {
            let frequency = 1f32 / 10000f32.powf((2 * i) as f32 / width as f32);
            let angle = delta * frequency;
            data[row * width + 2 * i] = angle.sin();
            data[row * width + 2 * i + 1] = angle.cos();
        }
    }
    Tensor::from_vec(data, (1, 2 * total - 1, width), device)?.to_dtype(dtype)
}

fn chunk_mask(
    t: usize,
    seen: usize,
    cached: usize,
    valid: usize,
    left: usize,
    right: usize,
    device: &Device,
) -> Result<Tensor> {
    let block = right + 1;
    let left_chunks = left / block;
    let keys = cached + t;
    let mut values = Vec::with_capacity(t * keys);
    for q in 0..t {
        let q_block = (seen + q) / block;
        for k in 0..keys {
            let k_block = (seen - cached + k) / block;
            let allowed = q < valid
                && (k < cached || k - cached < valid)
                && k_block <= q_block
                && q_block - k_block <= left_chunks;
            values.push(u8::from(allowed));
        }
    }
    Tensor::from_vec(values, (1, 1, t, keys), device)
}

#[derive(Debug, Clone)]
struct Block {
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_att: LayerNorm,
    att: Attention,
    norm_conv: LayerNorm,
    conv: Convolution,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_out: LayerNorm,
}
impl Block {
    fn load(c: &EncoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let h = c.hidden_size;
        Ok(Self {
            norm_ff1: norm(h, vb.pp("norm_feed_forward1"))?,
            ff1: FeedForward::load(c, vb.pp("feed_forward1"))?,
            norm_att: norm(h, vb.pp("norm_self_att"))?,
            att: Attention::load(c, vb.pp("self_attn"))?,
            norm_conv: norm(h, vb.pp("norm_conv"))?,
            conv: Convolution::load(c, vb.pp("conv"))?,
            norm_ff2: norm(h, vb.pp("norm_feed_forward2"))?,
            ff2: FeedForward::load(c, vb.pp("feed_forward2"))?,
            norm_out: norm(h, vb.pp("norm_out"))?,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        pos: &Tensor,
        mask: &Tensor,
        valid: usize,
        mut cache: Option<&mut LayerCache>,
    ) -> Result<Tensor> {
        let x = (x + (self.ff1.forward(&self.norm_ff1.forward(x)?)? * 0.5)?)?;
        let attention =
            self.att
                .forward(&self.norm_att.forward(&x)?, pos, mask, cache.as_deref_mut())?;
        let x = (&x + attention)?;
        let conv = self
            .conv
            .forward(&self.norm_conv.forward(&x)?, valid, cache)?;
        let x = (&x + conv)?;
        self.norm_out
            .forward(&(&x + (self.ff2.forward(&self.norm_ff2.forward(&x)?)? * 0.5)?)?)
    }
}

#[derive(Debug, Clone)]
pub struct Encoder {
    cfg: EncoderConfig,
    front: Subsampling,
    layers: Vec<Block>,
}
impl Encoder {
    /// `vb` is rooted at `encoder` in the official HF checkpoint.
    pub fn load(config: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let c = &config.encoder_config;
        Ok(Self {
            cfg: c.clone(),
            front: Subsampling::load(c, vb.pp("subsampling"))?,
            layers: (0..c.num_hidden_layers)
                .map(|i| Block::load(c, vb.pp("layers").pp(i)))
                .collect::<Result<Vec<_>>>()?,
        })
    }
    pub fn new_state(&self) -> EncoderState {
        EncoderState {
            subsampling: vec![None; self.cfg.subsampling_factor.trailing_zeros() as usize],
            layers: vec![LayerCache::default(); self.layers.len()],
            frames_seen: 0,
        }
    }
    fn check(&self, x: &Tensor, lookahead: usize) -> Result<()> {
        let (b, t, f) = x.dims3()?;
        if b != 1
            || t == 0
            || f != self.cfg.num_mel_bins
            || !matches!(x.dtype(), DType::F32 | DType::F16 | DType::BF16)
            || !self.cfg.supported_num_lookahead_tokens.contains(&lookahead)
        {
            candle::bail!("invalid Nemotron encoder input/context (one utterance required)")
        }
        Ok(())
    }
    fn blocks(
        &self,
        mut x: Tensor,
        valid: usize,
        right: usize,
        state: Option<&mut EncoderState>,
    ) -> Result<Tensor> {
        let t = x.dim(1)?;
        let seen = state.as_ref().map_or(0, |s| s.frames_seen);
        let cached = state.as_ref().map_or(0, |s| s.attention_cache_frames());
        if t + cached > self.cfg.max_position_embeddings {
            candle::bail!("Nemotron position limit exceeded")
        }
        if self.cfg.scale_input {
            x = (&x * (self.cfg.hidden_size as f64).sqrt())?;
        }
        let pos = positions(t + cached, self.cfg.hidden_size, x.dtype(), x.device())?;
        let mask = chunk_mask(
            t,
            seen,
            cached,
            valid,
            self.cfg.sliding_window - 1,
            right,
            x.device(),
        )?;
        if let Some(state) = state {
            for (block, layer) in self.layers.iter().zip(&mut state.layers) {
                x = block.forward(&x, &pos, &mask, valid, Some(layer))?;
            }
            state.frames_seen = seen
                .checked_add(t)
                .ok_or_else(|| candle::Error::Msg("encoder frame overflow".into()))?;
        } else {
            for block in &self.layers {
                x = block.forward(&x, &pos, &mask, valid, None)?;
            }
        }
        Ok(x)
    }
    pub fn forward(
        &self,
        features: &Tensor,
        valid_frames: usize,
        lookahead: usize,
    ) -> Result<(Tensor, usize)> {
        self.check(features, lookahead)?;
        if valid_frames == 0 || valid_frames > features.dim(1)? {
            candle::bail!("invalid Mel valid length")
        }
        let (x, valid) = self.front.forward(features, Some(valid_frames), None)?;
        let valid = valid.unwrap();
        Ok((self.blocks(x, valid, lookahead, None)?, valid))
    }
    pub fn forward_stream(
        &self,
        features: &Tensor,
        lookahead: usize,
        state: &mut EncoderState,
    ) -> Result<Tensor> {
        self.check(features, lookahead)?;
        let required = if state.frames_seen == 0 {
            1 + self.cfg.subsampling_factor * lookahead
        } else {
            self.cfg.subsampling_factor * (lookahead + 1)
        };
        if features.dim(1)? != required || state.layers.len() != self.layers.len() {
            candle::bail!("incorrect streaming Mel block length or encoder state")
        }
        let mut next = state.clone();
        let (x, _) = self
            .front
            .forward(features, None, Some(&mut next.subsampling))?;
        let t = x.dim(1)?;
        let output = self.blocks(x, t, lookahead, Some(&mut next))?;
        *state = next;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunk_attention_preserves_global_alignment_after_cache_eviction() -> Result<()> {
        let m = chunk_mask(4, 64, 56, 4, 56, 3, &Device::Cpu)?;
        let v = m.flatten_all()?.to_vec1::<u8>()?;
        assert!(v.iter().all(|&x| x == 1));
        let m = chunk_mask(8, 0, 0, 7, 56, 3, &Device::Cpu)?;
        let v = m.flatten_all()?.to_vec1::<u8>()?;
        assert_eq!(&v[..8], &[1, 1, 1, 1, 0, 0, 0, 0]);
        assert!(v[7 * 8..].iter().all(|&x| x == 0));
        Ok(())
    }
}
