use candle_core::{conv::CudnnFwdAlgo, D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    conv1d, embedding, layer_norm, linear, Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear,
    Module, VarBuilder,
};

pub const MB_NUM_TOKENS: usize = 16;
pub const MB_FRAMES_PER_TOKEN: usize = 4;
pub const MB_FRAMES: usize = MB_NUM_TOKENS * MB_FRAMES_PER_TOKEN;
pub const MB_POSE_HEADS: usize = 8;
pub const MB_CODES_PER_HEAD: usize = 10;
pub const MB_POSE_DIM: usize = 304;
pub const MB_POSE_DECODER_DIM: usize = 413;
pub const MB_GLOBAL_ROOT_DIM: usize = 5;
pub const MB_LOCAL_ROOT_DIM: usize = 4;

#[derive(Clone, Debug)]
pub struct Conv1dFrontierCase {
    pub source: String,
    pub input: Tensor,
    pub conv: Conv1d,
}

impl Conv1dFrontierCase {
    pub fn signature(&self) -> Result<String> {
        let (b, cin, len) = self.input.dims3()?;
        let (cout, wcin, kernel) = self.conv.weight().dims3()?;
        if cin != wcin {
            candle_core::bail!("conv frontier case channel mismatch input={cin} weight={wcin}")
        }
        let cfg = self.conv.config();
        Ok(format!(
            "b{b}_cin{cin}_cout{cout}_k{kernel}_l{len}_d{}_p{}_s{}",
            cfg.dilation, cfg.padding, cfg.stride
        ))
    }
}

fn leaky_relu(xs: &Tensor) -> Result<Tensor> {
    let pos = xs.relu()?;
    let neg = xs.neg()?.relu()?;
    let slope = Tensor::new(0.01f32, xs.device())?;
    pos.sub(&neg.broadcast_mul(&slope)?)
}

fn blend(mask: &Tensor, when_true: &Tensor, when_false: &Tensor) -> Result<Tensor> {
    let one = mask.ones_like()?;
    let inv = one.sub(mask)?;
    let a = when_true.broadcast_mul(mask)?;
    let b = when_false.broadcast_mul(&inv)?;
    a.add(&b)
}

fn upsample_nearest_2x(xs: &Tensor) -> Result<Tensor> {
    let (b, c, t) = xs.dims3()?;
    xs.unsqueeze(3)?
        .broadcast_as((b, c, t, 2))?
        .contiguous()?
        .reshape((b, c, t * 2))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionBricksConv1dPolicy {
    /// v0.3.1 numerical baseline: every standard Conv1d uses Candle/cuDNN's
    /// existing default path.
    Baseline,
    /// v0.3.2-r2 policy validated on Pascal sm61: exact 512x512 K=1 decoder
    /// shapes use GEMM and the exact K=3/L=32/d=1 shape uses cuDNN Direct.
    Sm61Validated,
}

impl MotionBricksConv1dPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline_candle_conv1d",
            Self::Sm61Validated => "sm61_k1_gemm_k3_l32_direct",
        }
    }
}

/// A 1x1 Conv1d is exactly a position-wise matrix multiplication.
fn conv1d_k1_gemm(conv: &Conv1d, xs: &Tensor) -> Result<Tensor> {
    let (b, cin, len) = xs.dims3()?;
    let (cout, wcin, kernel) = conv.weight().dims3()?;
    if kernel != 1 || cin != wcin {
        candle_core::bail!(
            "K1 GEMM dispatch requires weight [cout,cin,1], got input cin={cin} weight=[{cout},{wcin},{kernel}]"
        )
    }
    let x = xs
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b * len, cin))?;
    let wt = conv
        .weight()
        .reshape((cout, cin))?
        .transpose(0, 1)?
        .contiguous()?;
    let mut y = x.matmul(&wt)?;
    if let Some(bias) = conv.bias() {
        y = y.broadcast_add(&bias.reshape((1, cout))?)?;
    }
    y.reshape((b, len, cout))?
        .transpose(1, 2)?
        .contiguous()
}

fn conv1d_is_promoted_k1(conv: &Conv1d, xs: &Tensor) -> Result<bool> {
    let (b, cin, len) = xs.dims3()?;
    let (cout, wcin, kernel) = conv.weight().dims3()?;
    let cfg = conv.config();
    Ok(matches!(xs.device(), Device::Cuda(_))
        && xs.dtype() == DType::F32
        && conv.weight().dtype() == DType::F32
        && xs.is_contiguous()
        && conv.weight().is_contiguous()
        && b == 1
        && cin == 512
        && wcin == 512
        && cout == 512
        && kernel == 1
        && (len == 16 || len == 32)
        && cfg.groups == 1
        && cfg.padding == 0
        && cfg.stride == 1
        && cfg.dilation == 1)
}

fn conv1d_is_promoted_k3_l32(conv: &Conv1d, xs: &Tensor) -> Result<bool> {
    let (b, cin, len) = xs.dims3()?;
    let (cout, wcin, kernel) = conv.weight().dims3()?;
    let cfg = conv.config();
    Ok(matches!(xs.device(), Device::Cuda(_))
        && xs.dtype() == DType::F32
        && conv.weight().dtype() == DType::F32
        && xs.is_contiguous()
        && conv.weight().is_contiguous()
        && b == 1
        && cin == 512
        && wcin == 512
        && cout == 512
        && kernel == 3
        && len == 32
        && cfg.groups == 1
        && cfg.padding == 1
        && cfg.stride == 1
        && cfg.dilation == 1)
}

fn conv1d_forward_with_algo(
    conv: &Conv1d,
    xs: &Tensor,
    algo: CudnnFwdAlgo,
) -> Result<Tensor> {
    let cfg = conv.config();
    let y = xs.conv1d_with_algo(
        conv.weight(),
        cfg.padding,
        cfg.stride,
        cfg.dilation,
        cfg.groups,
        Some(algo),
    )?;
    match conv.bias() {
        None => Ok(y),
        Some(bias) => {
            let cout = bias.dims1()?;
            y.broadcast_add(&bias.reshape((1, cout, 1))?)
        }
    }
}

fn conv1d_forward_with_policy(
    conv: &Conv1d,
    xs: &Tensor,
    policy: MotionBricksConv1dPolicy,
) -> Result<Tensor> {
    // Preserve the original dispatch exactly. The opt-in capture observes the
    // PUBLIC module output (after bias); it cannot assert a backend kernel.
    let y = if policy == MotionBricksConv1dPolicy::Sm61Validated {
        if conv1d_is_promoted_k1(conv, xs)? {
            conv1d_k1_gemm(conv, xs)?
        } else if conv1d_is_promoted_k3_l32(conv, xs)? {
            conv1d_forward_with_algo(conv, xs, CudnnFwdAlgo::Direct)?
        } else {
            conv.forward(xs)?
        }
    } else {
        conv.forward(xs)?
    };
    let cfg = conv.config();
    crate::motionbricks_conv_parity::emit(
        xs, conv.weight(), &y, cfg.padding, cfg.stride, cfg.dilation, cfg.groups,
    )?;
    Ok(y)
}

#[derive(Clone, Debug)]
struct FcBlock {
    fc: Vec<Linear>,
    out: Linear,
}

impl FcBlock {
    fn load(
        size_in: usize,
        layer_width: usize,
        size_out: usize,
        num_layers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut fc = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let in_dim = if i == 0 { size_in } else { layer_width };
            fc.push(linear(
                in_dim,
                layer_width,
                vb.pp(format!("fc_layers.{i}")),
            )?);
        }
        let out = linear(layer_width, size_out, vb.pp("forward_projection"))?;
        Ok(Self { fc, out })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for layer in &self.fc {
            h = leaky_relu(&layer.forward(&h)?)?;
        }
        self.out.forward(&h)
    }
}

#[derive(Clone, Debug)]
struct SelfAttention {
    qkv: Linear,
    out: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn load(d_model: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let w = vb.get((3 * d_model, d_model), "in_proj_weight")?;
        let b = vb.get(3 * d_model, "in_proj_bias")?;
        let qkv = Linear::new(w, Some(b));
        let out = linear(d_model, d_model, vb.pp("out_proj"))?;
        Ok(Self {
            qkv,
            out,
            n_heads,
            head_dim: d_model / n_heads,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, d) = xs.dims3()?;
        // Match PyTorch self-attention's packed in-projection: one GEMM for QKV.
        let qkv = self.qkv.forward(xs)?;
        // Candle's matmul backend requires contiguous operands.  The head split
        // and K^T are view-only transposes, so materialize them before GEMM.
        let q = qkv
            .narrow(2, 0, d)?
            .contiguous()?
            .reshape((b, s, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = qkv
            .narrow(2, d, d)?
            .contiguous()?
            .reshape((b, s, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = qkv
            .narrow(2, 2 * d, d)?
            .contiguous()?
            .reshape((b, s, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let scale = Tensor::new(1.0f32 / (self.head_dim as f32).sqrt(), xs.device())?;
        let q = q.broadcast_mul(&scale)?.contiguous()?;
        let kt = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&kt)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?.contiguous()?;
        let ctx = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, d))?;
        self.out.forward(&ctx)
    }
}

#[derive(Clone, Debug)]
struct TransformerLayer {
    attn: SelfAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerLayer {
    fn load(d_model: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: SelfAttention::load(d_model, n_heads, vb.pp("self_attn"))?,
            linear1: linear(d_model, 2048, vb.pp("linear1"))?,
            linear2: linear(2048, d_model, vb.pp("linear2"))?,
            norm1: layer_norm(d_model, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(d_model, 1e-5, vb.pp("norm2"))?,
        })
    }

    fn attention_residual_norm(&self, xs: &Tensor) -> Result<Tensor> {
        let attn = self.attn.forward(xs)?;
        self.norm1.forward(&xs.add(&attn)?)
    }

    fn ffn_residual_norm(&self, xs: &Tensor) -> Result<Tensor> {
        let ff = self.linear2.forward(&self.linear1.forward(xs)?.relu()?)?;
        self.norm2.forward(&xs.add(&ff)?)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // PyTorch TransformerEncoderLayer defaults in eval mode:
        // post-norm, ReLU, dropout disabled by eval().
        let attn = self.attn.forward(xs)?;
        let x = self.norm1.forward(&xs.add(&attn)?)?;
        let ff = self.linear2.forward(&self.linear1.forward(&x)?.relu()?)?;
        self.norm2.forward(&x.add(&ff)?)
    }
}

#[derive(Clone, Debug)]
struct TransformerEncoder {
    layers: Vec<TransformerLayer>,
}

impl TransformerEncoder {
    fn load(d_model: usize, n_heads: usize, n_layers: usize, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(TransformerLayer::load(
                d_model,
                n_heads,
                vb.pp(format!("layers.{i}")),
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        Ok(h)
    }

    fn layer_attention_residual_norm(&self, layer: usize, xs: &Tensor) -> Result<Tensor> {
        self.layers[layer].attention_residual_norm(xs)
    }

    fn layer_ffn_residual_norm(&self, layer: usize, xs: &Tensor) -> Result<Tensor> {
        self.layers[layer].ffn_residual_norm(xs)
    }
}

#[derive(Clone, Debug)]
struct ResConv1dBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl ResConv1dBlock {
    fn load(width: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let conv1 = conv1d(
            width,
            width,
            3,
            Conv1dConfig {
                padding: dilation,
                stride: 1,
                dilation,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("conv1"),
        )?;
        let conv2 = conv1d(
            width,
            width,
            1,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("conv2"),
        )?;
        Ok(Self { conv1, conv2 })
    }

    fn forward_with_policy(
        &self,
        xs: &Tensor,
        policy: MotionBricksConv1dPolicy,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let h = conv1d_forward_with_policy(&self.conv1, &xs.relu()?, policy)?;
        let h = conv1d_forward_with_policy(&self.conv2, &h.relu()?, policy)?;
        h.add(&residual)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_with_policy(xs, MotionBricksConv1dPolicy::Baseline)
    }
}

#[derive(Clone, Debug)]
struct Resnet1d {
    blocks: Vec<ResConv1dBlock>,
}

impl Resnet1d {
    fn load(width: usize, depth: usize, vb: VarBuilder) -> Result<Self> {
        // Official runtime config: dilation_growth_rate=3, reverse_dilation=True.
        let mut dilations = (0..depth)
            .map(|i| 3usize.pow(i as u32))
            .collect::<Vec<_>>();
        dilations.reverse();
        let mut blocks = Vec::with_capacity(depth);
        for (i, dilation) in dilations.into_iter().enumerate() {
            blocks.push(ResConv1dBlock::load(
                width,
                dilation,
                vb.pp(format!("model.{i}")),
            )?);
        }
        Ok(Self { blocks })
    }

    fn forward_with_policy(
        &self,
        xs: &Tensor,
        policy: MotionBricksConv1dPolicy,
    ) -> Result<Tensor> {
        let mut h = xs.clone();
        for block in &self.blocks {
            h = block.forward_with_policy(&h, policy)?;
        }
        Ok(h)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_with_policy(xs, MotionBricksConv1dPolicy::Baseline)
    }

    fn benchmark_collect_conv_cases(
        &self,
        xs: &Tensor,
        prefix: &str,
        out: &mut Vec<Conv1dFrontierCase>,
    ) -> Result<Tensor> {
        let mut h = xs.clone();
        for (i, block) in self.blocks.iter().enumerate() {
            let residual = h.clone();
            let conv1_input = h.relu()?;
            out.push(Conv1dFrontierCase {
                source: format!("{prefix}.block{i}.conv1"),
                input: conv1_input.clone(),
                conv: block.conv1.clone(),
            });
            let conv1_out = block.conv1.forward(&conv1_input)?;
            let conv2_input = conv1_out.relu()?;
            out.push(Conv1dFrontierCase {
                source: format!("{prefix}.block{i}.conv2"),
                input: conv2_input.clone(),
                conv: block.conv2.clone(),
            });
            h = block.conv2.forward(&conv2_input)?.add(&residual)?;
        }
        Ok(h)
    }
}

#[derive(Clone, Debug)]
struct DecoderStage {
    resnet: Resnet1d,
    conv: Conv1d,
}

#[derive(Clone, Debug)]
struct DoubleCondDecoder {
    input_conv: Conv1d,
    stages: Vec<DecoderStage>,
    post_conv: Conv1d,
    output_conv: Conv1d,
    external_linears: Vec<Linear>,
    target_linears: Vec<Linear>,
    width: usize,
    down_t: usize,
}

impl DoubleCondDecoder {
    #[allow(clippy::too_many_arguments)]
    fn load(
        input_emb_width: usize,
        output_emb_width: usize,
        target_cond_dim: usize,
        external_cond_dim: usize,
        down_t: usize,
        width: usize,
        depth: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg3 = Conv1dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let input_conv = conv1d(
            output_emb_width,
            width,
            3,
            cfg3,
            vb.pp("model.0"),
        )?;
        let mut stages = Vec::with_capacity(down_t);
        for i in 0..down_t {
            let model_idx = i + 2;
            stages.push(DecoderStage {
                resnet: Resnet1d::load(width, depth, vb.pp(format!("model.{model_idx}.0")))?,
                conv: conv1d(
                    width,
                    width,
                    3,
                    cfg3,
                    vb.pp(format!("model.{model_idx}.2")),
                )?,
            });
        }
        let post_idx = 2 + down_t;
        let post_conv = conv1d(width, width, 3, cfg3, vb.pp(format!("model.{post_idx}")))?;
        let output_conv = conv1d(
            width,
            input_emb_width,
            3,
            cfg3,
            vb.pp(format!("model.{}", post_idx + 2)),
        )?;

        let mut external_linears = Vec::with_capacity(down_t);
        let mut target_linears = Vec::with_capacity(down_t);
        for i in 0..down_t {
            let frames_per_position = 1usize << (down_t - i);
            external_linears.push(linear(
                width + frames_per_position * external_cond_dim,
                width,
                vb.pp(format!("external_cond_blocks.{}", i * 2)),
            )?);
            target_linears.push(linear(
                target_cond_dim,
                width / frames_per_position,
                vb.pp(format!("target_cond_blocks.{}", i * 2)),
            )?);
        }

        Ok(Self {
            input_conv,
            stages,
            post_conv,
            output_conv,
            external_linears,
            target_linears,
            width,
            down_t,
        })
    }

    fn forward_with_policy(
        &self,
        xs: &Tensor,
        external_cond: &Tensor,
        target_cond: &Tensor,
        has_target_cond: &Tensor,
        policy: MotionBricksConv1dPolicy,
    ) -> Result<Tensor> {
        let (b, _, _) = xs.dims3()?;
        let mut h = conv1d_forward_with_policy(&self.input_conv, xs, policy)?.relu()?;
        for i in 0..self.down_t {
            let frames_per_position = 1usize << (self.down_t - i);
            let num_positions = h.dim(2)?;
            let timesteps = num_positions * frames_per_position;

            let h_target = self.target_linears[i].forward(target_cond)?.relu()?;
            let h_frame = h
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, timesteps, self.width / frames_per_position))?;
            let mask = has_target_cond
                .narrow(1, 0, timesteps)?
                .reshape((b, timesteps, 1))?;
            h = blend(&mask, &h_target.narrow(1, 0, timesteps)?, &h_frame)?
                .reshape((b, num_positions, self.width))?
                .transpose(1, 2)?;

            let external_dim = external_cond.dim(2)?;
            let ext = external_cond
                .narrow(1, 0, timesteps)?
                .reshape((b, num_positions, frames_per_position * external_dim))?;
            let ht = h.transpose(1, 2)?;
            let merged = Tensor::cat(&[&ht, &ext], D::Minus1)?;
            h = self.external_linears[i]
                .forward(&merged)?
                .relu()?
                .transpose(1, 2)?;

            h = self.stages[i].resnet.forward_with_policy(&h, policy)?;
            h = upsample_nearest_2x(&h)?;
            h = conv1d_forward_with_policy(&self.stages[i].conv, &h, policy)?;
        }
        h = conv1d_forward_with_policy(&self.post_conv, &h, policy)?.relu()?;
        conv1d_forward_with_policy(&self.output_conv, &h, policy)
    }

    fn forward(
        &self,
        xs: &Tensor,
        external_cond: &Tensor,
        target_cond: &Tensor,
        has_target_cond: &Tensor,
    ) -> Result<Tensor> {
        self.forward_with_policy(
            xs,
            external_cond,
            target_cond,
            has_target_cond,
            MotionBricksConv1dPolicy::Baseline,
        )
    }

    fn benchmark_collect_conv_cases(
        &self,
        xs: &Tensor,
        external_cond: &Tensor,
        target_cond: &Tensor,
        has_target_cond: &Tensor,
        prefix: &str,
    ) -> Result<Vec<Conv1dFrontierCase>> {
        let (b, _, _) = xs.dims3()?;
        let mut out = Vec::new();
        out.push(Conv1dFrontierCase {
            source: format!("{prefix}.input_conv"),
            input: xs.clone(),
            conv: self.input_conv.clone(),
        });
        let mut h = self.input_conv.forward(xs)?.relu()?;
        for i in 0..self.down_t {
            let frames_per_position = 1usize << (self.down_t - i);
            let num_positions = h.dim(2)?;
            let timesteps = num_positions * frames_per_position;
            let h_target = self.target_linears[i].forward(target_cond)?.relu()?;
            let h_frame = h.transpose(1, 2)?.contiguous()?.reshape((
                b, timesteps, self.width / frames_per_position
            ))?;
            let mask = has_target_cond.narrow(1, 0, timesteps)?.reshape((b, timesteps, 1))?;
            h = blend(&mask, &h_target.narrow(1, 0, timesteps)?, &h_frame)?
                .reshape((b, num_positions, self.width))?
                .transpose(1, 2)?;
            let external_dim = external_cond.dim(2)?;
            let ext = external_cond.narrow(1, 0, timesteps)?.reshape((
                b, num_positions, frames_per_position * external_dim
            ))?;
            let ht = h.transpose(1, 2)?;
            let merged = Tensor::cat(&[&ht, &ext], D::Minus1)?;
            h = self.external_linears[i].forward(&merged)?.relu()?.transpose(1, 2)?;
            h = self.stages[i].resnet.benchmark_collect_conv_cases(
                &h, &format!("{prefix}.stage{i}.resnet"), &mut out
            )?;
            h = upsample_nearest_2x(&h)?;
            out.push(Conv1dFrontierCase {
                source: format!("{prefix}.stage{i}.upsample_conv"),
                input: h.clone(),
                conv: self.stages[i].conv.clone(),
            });
            h = self.stages[i].conv.forward(&h)?;
        }
        out.push(Conv1dFrontierCase {
            source: format!("{prefix}.post_conv"),
            input: h.clone(),
            conv: self.post_conv.clone(),
        });
        h = self.post_conv.forward(&h)?.relu()?;
        out.push(Conv1dFrontierCase {
            source: format!("{prefix}.output_conv"),
            input: h.clone(),
            conv: self.output_conv.clone(),
        });
        let _ = self.output_conv.forward(&h)?;
        Ok(out)
    }
}

#[derive(Clone, Debug)]
pub struct RootRuntimeInput {
    pub global_root_values: Tensor, // [B,8,5]
    pub local_root_values: Tensor,  // [B,8,4]
    pub poses: Tensor,              // [B,8,304]
}

#[derive(Clone, Debug)]
pub struct RootRuntimeOutput {
    pub num_token_logits: Tensor,        // [B,12]
    pub pred_global_root_values: Tensor, // [B,64,5]
}

#[derive(Clone, Debug)]
pub struct RootBenchmarkState {
    first_input: Tensor,
    second_input: Tensor,
    dense_frame: Tensor,
    dense_global: Tensor,
    has_target: Tensor,
}

#[derive(Clone, Debug)]
pub struct MotionBricksRootBackbone {
    proj_local_pose: Linear,
    proj_local_root: Linear,
    proj_global_root: Linear,
    proj_start: FcBlock,
    proj_end: FcBlock,
    input_position_emb: Tensor,
    proj_num_tokens: Embedding,
    shared_transformer: TransformerEncoder,
    num_token_out: Linear,
    middle_token_emb: Embedding,
    position_emb: Tensor,
    root_token_transformer: TransformerEncoder,
    conv_no_frame_emb: Tensor,
    conv_output: DoubleCondDecoder,
}

impl MotionBricksRootBackbone {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj_local_pose: linear(MB_POSE_DIM, 256, vb.pp("_proj_local_pose"))?,
            proj_local_root: linear(MB_LOCAL_ROOT_DIM, 64, vb.pp("_proj_local_root_value"))?,
            proj_global_root: linear(MB_GLOBAL_ROOT_DIM, 64, vb.pp("_proj_global_root_value"))?,
            proj_start: FcBlock::load(384, 512, 512, 2, vb.pp("_proj_start_input"))?,
            proj_end: FcBlock::load(384, 512, 512, 2, vb.pp("_proj_end_input"))?,
            input_position_emb: vb.pp("_input_position_emb").get((8, 512), "weight")?,
            proj_num_tokens: embedding(13, 512, vb.pp("_proj_input_num_tokens"))?,
            shared_transformer: TransformerEncoder::load(
                512,
                16,
                3,
                vb.pp("_shared_transformer_model"),
            )?,
            num_token_out: linear(512, 12, vb.pp("_proj_num_token_output_logit"))?,
            middle_token_emb: embedding(12, 512, vb.pp("_middle_token_emb"))?,
            position_emb: vb.pp("_position_emb").get((16, 1, 512), "embed")?,
            root_token_transformer: TransformerEncoder::load(
                512,
                16,
                3,
                vb.pp("_root_token_transformer_model"),
            )?,
            conv_no_frame_emb: vb.get(512, "_conv_no_frame_emb")?,
            conv_output: DoubleCondDecoder::load(
                5,
                512,
                5,
                512,
                2,
                512,
                4,
                vb.pp("_conv_output"),
            )?,
        })
    }

    pub fn benchmark_prepare_fixed_16(&self, input: &RootRuntimeInput) -> Result<RootBenchmarkState> {
        let (b, frames, _) = input.poses.dims3()?;
        if frames != 8 {
            candle_core::bail!("root runtime reference expects exactly 8 constraint frames")
        }
        let pose = self.proj_local_pose.forward(&input.poses)?;
        let local = self.proj_local_root.forward(&input.local_root_values)?;
        let global = self.proj_global_root.forward(&input.global_root_values)?;
        let merged = Tensor::cat(&[&pose, &local, &global], D::Minus1)?;
        let start = self.proj_start.forward(&merged.narrow(1, 0, 4)?)?;
        let end = self.proj_end.forward(&merged.narrow(1, 4, 4)?)?;
        let frame_emb = Tensor::cat(&[&start, &end], 1)?;
        let positioned = frame_emb.broadcast_add(&self.input_position_emb.unsqueeze(0)?)?;

        let num_idx = Tensor::new(&[10i64; 1], input.poses.device())?;
        let time_emb = self
            .proj_num_tokens
            .forward(&num_idx)?
            .reshape((1, 1, 512))?
            .broadcast_as((b, 1, 512))?;
        let first_input = Tensor::cat(&[&time_emb, &positioned], 1)?;

        let middle = self
            .middle_token_emb
            .forward(&num_idx)?
            .reshape((1, 1, 512))?
            .broadcast_as((b, 1, 512))?;
        let pos = self
            .position_emb
            .reshape((1, MB_NUM_TOKENS, 512))?
            .broadcast_as((b, MB_NUM_TOKENS, 512))?;
        let second_input = Tensor::cat(&[&middle, &positioned, &pos], 1)?;

        let mid_len = MB_FRAMES - 8;
        let no_frame = self
            .conv_no_frame_emb
            .reshape((1, 1, 512))?
            .broadcast_as((b, mid_len, 512))?;
        let frame_start = frame_emb.narrow(1, 0, 4)?;
        let frame_end = frame_emb.narrow(1, 4, 4)?;
        let dense_frame = Tensor::cat(&[&frame_start, &no_frame, &frame_end], 1)?;
        let zeros = Tensor::zeros(
            (b, mid_len, 5),
            input.global_root_values.dtype(),
            input.poses.device(),
        )?;
        let global_start = input.global_root_values.narrow(1, 0, 4)?;
        let global_end = input.global_root_values.narrow(1, 4, 4)?;
        let dense_global = Tensor::cat(&[&global_start, &zeros, &global_end], 1)?;
        let mut mask = vec![0f32; b * MB_FRAMES];
        for bi in 0..b {
            for t in 0..4 {
                mask[bi * MB_FRAMES + t] = 1.0;
                mask[bi * MB_FRAMES + (MB_FRAMES - 4 + t)] = 1.0;
            }
        }
        let has_target = Tensor::from_vec(mask, (b, MB_FRAMES), input.poses.device())?;

        Ok(RootBenchmarkState {
            first_input,
            second_input,
            dense_frame,
            dense_global,
            has_target,
        })
    }

    pub fn benchmark_shared_transformer(&self, state: &RootBenchmarkState) -> Result<Tensor> {
        self.shared_transformer.forward(&state.first_input)
    }

    pub fn benchmark_num_token_head(&self, shared: &Tensor) -> Result<Tensor> {
        self.num_token_out.forward(&shared.i((.., 0, ..))?)
    }

    pub fn benchmark_root_token_transformer(&self, state: &RootBenchmarkState) -> Result<Tensor> {
        let second = self.root_token_transformer.forward(&state.second_input)?;
        second.narrow(1, 9, MB_NUM_TOKENS)
    }

    pub fn benchmark_conv_decoder_with_policy(
        &self,
        token_hidden: &Tensor,
        state: &RootBenchmarkState,
        policy: MotionBricksConv1dPolicy,
    ) -> Result<Tensor> {
        self.conv_output
            .forward_with_policy(
                &token_hidden.transpose(1, 2)?,
                &state.dense_frame,
                &state.dense_global,
                &state.has_target,
                policy,
            )?
            .transpose(1, 2)
    }

    pub fn benchmark_conv_decoder(
        &self,
        token_hidden: &Tensor,
        state: &RootBenchmarkState,
    ) -> Result<Tensor> {
        self.benchmark_conv_decoder_with_policy(
            token_hidden,
            state,
            MotionBricksConv1dPolicy::Baseline,
        )
    }

    pub fn benchmark_conv_frontier_cases(
        &self,
        input: &RootRuntimeInput,
    ) -> Result<Vec<Conv1dFrontierCase>> {
        let state = self.benchmark_prepare_fixed_16(input)?;
        let token_hidden = self.benchmark_root_token_transformer(&state)?;
        self.conv_output.benchmark_collect_conv_cases(
            &token_hidden.transpose(1, 2)?,
            &state.dense_frame,
            &state.dense_global,
            &state.has_target,
            "root",
        )
    }

    pub fn benchmark_shared_layer0_attention(
        &self,
        state: &RootBenchmarkState,
    ) -> Result<Tensor> {
        self.shared_transformer
            .layer_attention_residual_norm(0, &state.first_input)
    }

    pub fn benchmark_shared_layer0_ffn(&self, attn_norm: &Tensor) -> Result<Tensor> {
        self.shared_transformer
            .layer_ffn_residual_norm(0, attn_norm)
    }

    pub fn forward_fixed_16_with_policy(
        &self,
        input: &RootRuntimeInput,
        policy: MotionBricksConv1dPolicy,
    ) -> Result<RootRuntimeOutput> {
        let (b, frames, _) = input.poses.dims3()?;
        if frames != 8 {
            candle_core::bail!("root runtime reference expects exactly 8 constraint frames")
        }
        let pose = self.proj_local_pose.forward(&input.poses)?;
        let local = self.proj_local_root.forward(&input.local_root_values)?;
        let global = self.proj_global_root.forward(&input.global_root_values)?;
        let merged = Tensor::cat(&[&pose, &local, &global], D::Minus1)?;
        let start = self.proj_start.forward(&merged.narrow(1, 0, 4)?)?;
        let end = self.proj_end.forward(&merged.narrow(1, 4, 4)?)?;
        let frame_emb = Tensor::cat(&[&start, &end], 1)?;
        let positioned = frame_emb.broadcast_add(&self.input_position_emb.unsqueeze(0)?)?;

        let num_idx = Tensor::new(&[10i64; 1], input.poses.device())?;
        let time_emb = self
            .proj_num_tokens
            .forward(&num_idx)?
            .reshape((1, 1, 512))?
            .broadcast_as((b, 1, 512))?;
        let first_input = Tensor::cat(&[&time_emb, &positioned], 1)?;
        let first_out = self.shared_transformer.forward(&first_input)?;
        let num_token_logits = self.num_token_out.forward(&first_out.i((.., 0, ..))?)?;

        let middle = self
            .middle_token_emb
            .forward(&num_idx)?
            .reshape((1, 1, 512))?
            .broadcast_as((b, 1, 512))?;
        let pos = self
            .position_emb
            .reshape((1, MB_NUM_TOKENS, 512))?
            .broadcast_as((b, MB_NUM_TOKENS, 512))?;
        let second_input = Tensor::cat(&[&middle, &positioned, &pos], 1)?;
        let second = self.root_token_transformer.forward(&second_input)?;
        let token_hidden = second.narrow(1, 9, MB_NUM_TOKENS)?;

        let mid_len = MB_FRAMES - 8;
        let no_frame = self
            .conv_no_frame_emb
            .reshape((1, 1, 512))?
            .broadcast_as((b, mid_len, 512))?;
        let frame_start = frame_emb.narrow(1, 0, 4)?;
        let frame_end = frame_emb.narrow(1, 4, 4)?;
        let dense_frame = Tensor::cat(&[&frame_start, &no_frame, &frame_end], 1)?;
        let zeros = Tensor::zeros(
            (b, mid_len, 5),
            input.global_root_values.dtype(),
            input.poses.device(),
        )?;
        let global_start = input.global_root_values.narrow(1, 0, 4)?;
        let global_end = input.global_root_values.narrow(1, 4, 4)?;
        let dense_global = Tensor::cat(&[&global_start, &zeros, &global_end], 1)?;
        let mut mask = vec![0f32; b * MB_FRAMES];
        for bi in 0..b {
            for t in 0..4 {
                mask[bi * MB_FRAMES + t] = 1.0;
                mask[bi * MB_FRAMES + (MB_FRAMES - 4 + t)] = 1.0;
            }
        }
        let has_target = Tensor::from_vec(mask, (b, MB_FRAMES), input.poses.device())?;
        let pred = self
            .conv_output
            .forward_with_policy(
                &token_hidden.transpose(1, 2)?,
                &dense_frame,
                &dense_global,
                &has_target,
                policy,
            )?
            .transpose(1, 2)?;
        Ok(RootRuntimeOutput {
            num_token_logits,
            pred_global_root_values: pred,
        })
    }

    pub fn forward_fixed_16(&self, input: &RootRuntimeInput) -> Result<RootRuntimeOutput> {
        self.forward_fixed_16_with_policy(input, MotionBricksConv1dPolicy::Baseline)
    }
}

#[derive(Clone, Debug)]
pub struct PoseRuntimeInput {
    pub pose_tokens: Tensor, // [B,16,8], i64
    pub root_values: Tensor, // [B,64,4]
    pub pose_cond: Tensor,   // [B,64,304]
    pub has_pose_cond: Tensor, // [B,64] f32 0/1
}

#[derive(Clone, Debug)]
pub struct PoseBenchmarkState {
    transformer_input: Tensor,
}

#[derive(Clone, Debug)]
pub struct MotionBricksPoseBackbone {
    pose_token_emb: Embedding,
    proj_pose_token: FcBlock,
    proj_root: Linear,
    proj_pose_cond: Linear,
    num_valid_emb: Embedding,
    proj_input: Linear,
    position_emb: Tensor,
    transformer: TransformerEncoder,
    logits: Linear,
}

impl MotionBricksPoseBackbone {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            pose_token_emb: embedding(88, 32, vb.pp("_pose_token_emb"))?,
            proj_pose_token: FcBlock::load(256, 640, 640, 2, vb.pp("_proj_pose_token_emb"))?,
            proj_root: linear(16, 256, vb.pp("_proj_local_root_values"))?,
            proj_pose_cond: linear(MB_POSE_DIM, 160, vb.pp("_proj_local_pose"))?,
            num_valid_emb: embedding(11, 128, vb.pp("_proj_num_valid_positions"))?,
            proj_input: linear(1024, 1024, vb.pp("_proj_input.0"))?,
            position_emb: vb.pp("_position_emb").get((16, 1, 1024), "embed")?,
            transformer: TransformerEncoder::load(1024, 16, 16, vb.pp("_transformer_model"))?,
            logits: linear(1024, 80, vb.pp("_proj_pose_output_logit"))?,
        })
    }

    pub fn benchmark_prepare_fixed_16(&self, input: &PoseRuntimeInput) -> Result<PoseBenchmarkState> {
        let (b, positions, heads) = input.pose_tokens.dims3()?;
        if positions != 16 || heads != 8 {
            candle_core::bail!("pose runtime reference expects pose_tokens [B,16,8]")
        }
        let root = self
            .proj_root
            .forward(&input.root_values.reshape((b, 16, 16))?)?;
        let cond = self.proj_pose_cond.forward(&input.pose_cond)?;

        let offsets = Tensor::new(&[0i64, 11, 22, 33, 44, 55, 66, 77], input.pose_tokens.device())?
            .reshape((1, 1, 8))?;
        let ids = input.pose_tokens.broadcast_add(&offsets)?;
        let token = self
            .pose_token_emb
            .forward(&ids)?
            .reshape((b, 16, 256))?;
        let token = self
            .proj_pose_token
            .forward(&token)?
            .reshape((b, 64, 160))?;
        let mask = input.has_pose_cond.reshape((b, 64, 1))?;
        let pose = blend(&mask, &cond, &token)?.reshape((b, 16, 640))?;

        let num_idx = Tensor::new(&[10i64; 1], input.pose_tokens.device())?;
        let num = self
            .num_valid_emb
            .forward(&num_idx)?
            .reshape((1, 1, 128))?
            .broadcast_as((b, 16, 128))?;
        let merged = Tensor::cat(&[&pose, &root, &num], D::Minus1)?;
        let pos = self
            .position_emb
            .reshape((1, 16, 1024))?
            .broadcast_as((b, 16, 1024))?;
        let transformer_input = self.proj_input.forward(&merged)?.relu()?.add(&pos)?;
        Ok(PoseBenchmarkState { transformer_input })
    }

    pub fn benchmark_transformer(&self, state: &PoseBenchmarkState) -> Result<Tensor> {
        self.transformer.forward(&state.transformer_input)
    }

    pub fn benchmark_logit_head(&self, hidden: &Tensor) -> Result<Tensor> {
        let b = hidden.dim(0)?;
        self.logits.forward(hidden)?.reshape((b, 16, 8, 10))
    }

    pub fn benchmark_layer0_attention(&self, state: &PoseBenchmarkState) -> Result<Tensor> {
        self.transformer
            .layer_attention_residual_norm(0, &state.transformer_input)
    }

    pub fn benchmark_layer0_ffn(&self, attn_norm: &Tensor) -> Result<Tensor> {
        self.transformer.layer_ffn_residual_norm(0, attn_norm)
    }

    pub fn forward_fixed_16(&self, input: &PoseRuntimeInput) -> Result<Tensor> {
        let (b, positions, heads) = input.pose_tokens.dims3()?;
        if positions != 16 || heads != 8 {
            candle_core::bail!("pose runtime reference expects pose_tokens [B,16,8]")
        }
        let root = self
            .proj_root
            .forward(&input.root_values.reshape((b, 16, 16))?)?;
        let cond = self.proj_pose_cond.forward(&input.pose_cond)?;

        let offsets = Tensor::new(&[0i64, 11, 22, 33, 44, 55, 66, 77], input.pose_tokens.device())?
            .reshape((1, 1, 8))?;
        let ids = input.pose_tokens.broadcast_add(&offsets)?;
        let token = self
            .pose_token_emb
            .forward(&ids)?
            .reshape((b, 16, 256))?;
        let token = self
            .proj_pose_token
            .forward(&token)?
            .reshape((b, 64, 160))?;
        let mask = input.has_pose_cond.reshape((b, 64, 1))?;
        let pose = blend(&mask, &cond, &token)?.reshape((b, 16, 640))?;

        let num_idx = Tensor::new(&[10i64; 1], input.pose_tokens.device())?;
        let num = self
            .num_valid_emb
            .forward(&num_idx)?
            .reshape((1, 1, 128))?
            .broadcast_as((b, 16, 128))?;
        let merged = Tensor::cat(&[&pose, &root, &num], D::Minus1)?;
        let pos = self
            .position_emb
            .reshape((1, 16, 1024))?
            .broadcast_as((b, 16, 1024))?;
        let h = self.proj_input.forward(&merged)?.relu()?.add(&pos)?;
        let h = self.transformer.forward(&h)?;
        self.logits.forward(&h)?.reshape((b, 16, 8, 10))
    }
}

#[derive(Clone, Debug)]
pub struct VqvaeRuntimeInput {
    pub pose_tokens: Tensor,     // [B,16,8], i64 0..9
    pub target_cond: Tensor,     // [B,64,304]
    pub has_target_cond: Tensor, // [B,64] f32
    pub external_cond: Tensor,   // [B,64,2]
}

#[derive(Clone, Debug)]
pub struct VqvaeBenchmarkState {
    quantized: Tensor,
}

#[derive(Clone, Debug)]
pub struct MotionBricksPoseDecoder {
    codebook: Tensor, // [8,10,32]
    decoder: DoubleCondDecoder,
}

impl MotionBricksPoseDecoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let codebook = vb.get((8, 10, 32), "codebook")?;
        let decoder = DoubleCondDecoder::load(
            MB_POSE_DECODER_DIM,
            256,
            MB_POSE_DIM,
            2,
            2,
            512,
            4,
            vb.pp("decoder"),
        )?;
        Ok(Self { codebook, decoder })
    }

    pub fn benchmark_codebook_lookup(&self, input: &VqvaeRuntimeInput) -> Result<VqvaeBenchmarkState> {
        let (b, tokens, heads) = input.pose_tokens.dims3()?;
        if heads != 8 {
            candle_core::bail!("vqvae decoder expects 8 pose heads")
        }
        let mut chunks = Vec::with_capacity(8);
        for h in 0..8 {
            let idx = input.pose_tokens.i((.., .., h))?.flatten_all()?;
            let cb = self.codebook.i(h)?;
            chunks.push(cb.index_select(&idx, 0)?.reshape((b, tokens, 32))?);
        }
        let refs = chunks.iter().collect::<Vec<_>>();
        let quantized = Tensor::cat(&refs, D::Minus1)?.transpose(1, 2)?;
        Ok(VqvaeBenchmarkState { quantized })
    }

    pub fn benchmark_decoder_from_codebook_with_policy(
        &self,
        state: &VqvaeBenchmarkState,
        input: &VqvaeRuntimeInput,
        policy: MotionBricksConv1dPolicy,
    ) -> Result<Tensor> {
        self.decoder
            .forward_with_policy(
                &state.quantized,
                &input.external_cond,
                &input.target_cond,
                &input.has_target_cond,
                policy,
            )?
            .transpose(1, 2)
    }

    pub fn benchmark_decoder_from_codebook(
        &self,
        state: &VqvaeBenchmarkState,
        input: &VqvaeRuntimeInput,
    ) -> Result<Tensor> {
        self.benchmark_decoder_from_codebook_with_policy(
            state,
            input,
            MotionBricksConv1dPolicy::Baseline,
        )
    }

    pub fn benchmark_conv_frontier_cases(
        &self,
        input: &VqvaeRuntimeInput,
    ) -> Result<Vec<Conv1dFrontierCase>> {
        let state = self.benchmark_codebook_lookup(input)?;
        self.decoder.benchmark_collect_conv_cases(
            &state.quantized,
            &input.external_cond,
            &input.target_cond,
            &input.has_target_cond,
            "vq",
        )
    }

    pub fn forward_with_policy(
        &self,
        input: &VqvaeRuntimeInput,
        policy: MotionBricksConv1dPolicy,
    ) -> Result<Tensor> {
        let (b, tokens, heads) = input.pose_tokens.dims3()?;
        if heads != 8 {
            candle_core::bail!("vqvae decoder expects 8 pose heads")
        }
        let mut chunks = Vec::with_capacity(8);
        for h in 0..8 {
            let idx = input.pose_tokens.i((.., .., h))?.flatten_all()?;
            let cb = self.codebook.i(h)?;
            chunks.push(cb.index_select(&idx, 0)?.reshape((b, tokens, 32))?);
        }
        let refs = chunks.iter().collect::<Vec<_>>();
        let quant = Tensor::cat(&refs, D::Minus1)?.transpose(1, 2)?;
        self.decoder
            .forward_with_policy(
                &quant,
                &input.external_cond,
                &input.target_cond,
                &input.has_target_cond,
                policy,
            )?
            .transpose(1, 2)
    }

    pub fn forward(&self, input: &VqvaeRuntimeInput) -> Result<Tensor> {
        self.forward_with_policy(input, MotionBricksConv1dPolicy::Baseline)
    }
}

#[derive(Clone, Debug)]
pub struct MotionBricksRuntimeReference {
    pub root: MotionBricksRootBackbone,
    pub pose: MotionBricksPoseBackbone,
    pub decoder: MotionBricksPoseDecoder,
}

impl MotionBricksRuntimeReference {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            root: MotionBricksRootBackbone::load(vb.pp("root"))?,
            pose: MotionBricksPoseBackbone::load(vb.pp("pose"))?,
            decoder: MotionBricksPoseDecoder::load(vb.pp("vqvae"))?,
        })
    }
}
