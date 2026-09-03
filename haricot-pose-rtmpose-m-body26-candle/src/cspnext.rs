use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{ops, Conv2d, Conv2dConfig, Module, VarBuilder};

use crate::profiling::LayerProfiler;

fn fused_conv(
    vb: &VarBuilder,
    path: &str,
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Result<Conv2d> {
    let p = vb.pp(path);
    let weight = p.get((out_c, in_c / groups, k, k), "fused.weight")?;
    let bias = p.get(out_c, "fused.bias")?;
    Ok(Conv2d::new(
        weight,
        Some(bias),
        Conv2dConfig {
            padding,
            stride,
            dilation: 1,
            groups,
            cudnn_fwd_algo: None,
        },
    ))
}

fn plain_conv(
    vb: &VarBuilder,
    path: &str,
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    bias: bool,
) -> Result<Conv2d> {
    let p = vb.pp(path);
    let weight = p.get((out_c, in_c / groups, k, k), "weight")?;
    let bias = if bias { Some(p.get(out_c, "bias")?) } else { None };
    Ok(Conv2d::new(
        weight,
        bias,
        Conv2dConfig {
            padding,
            stride,
            dilation: 1,
            groups,
            cudnn_fwd_algo: None,
        },
    ))
}

#[derive(Debug)]
struct ConvSiLU {
    conv: Conv2d,
    path: String,
    category: &'static str,
}

impl ConvSiLU {
    fn load(
        vb: &VarBuilder,
        path: &str,
        in_c: usize,
        out_c: usize,
        k: usize,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Result<Self> {
        let category = if groups > 1 && groups == in_c && out_c == in_c {
            if k == 5 {
                "conv_depthwise_5x5"
            } else {
                "conv_depthwise"
            }
        } else if k == 1 {
            "conv_pointwise_1x1"
        } else {
            "conv_regular"
        };
        Ok(Self {
            conv: fused_conv(vb, path, in_c, out_c, k, stride, padding, groups)?,
            path: path.to_string(),
            category,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        ops::silu(&self.conv.forward(x)?)
    }

    fn forward_profiled(&self, x: &Tensor, profiler: &mut LayerProfiler) -> Result<Tensor> {
        let conv_name = format!("{}.conv", self.path);
        let y = profiler.measure_tensor(&conv_name, self.category, || self.conv.forward(x))?;
        let silu_name = format!("{}.silu", self.path);
        profiler.measure_tensor(&silu_name, "activation_silu", || ops::silu(&y))
    }
}

#[derive(Debug)]
struct DepthwiseSeparable {
    depthwise: ConvSiLU,
    pointwise: ConvSiLU,
}

impl DepthwiseSeparable {
    fn load(vb: &VarBuilder, path: &str, channels: usize, k: usize) -> Result<Self> {
        Ok(Self {
            depthwise: ConvSiLU::load(
                vb,
                &format!("{path}.depthwise_conv"),
                channels,
                channels,
                k,
                1,
                k / 2,
                channels,
            )?,
            pointwise: ConvSiLU::load(
                vb,
                &format!("{path}.pointwise_conv"),
                channels,
                channels,
                1,
                1,
                0,
                1,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.pointwise.forward(&self.depthwise.forward(x)?)
    }

    fn forward_profiled(&self, x: &Tensor, profiler: &mut LayerProfiler) -> Result<Tensor> {
        let x = self.depthwise.forward_profiled(x, profiler)?;
        self.pointwise.forward_profiled(&x, profiler)
    }
}

#[derive(Debug)]
struct CspNextBlock {
    conv1: ConvSiLU,
    conv2: DepthwiseSeparable,
    add_identity: bool,
    path: String,
}

impl CspNextBlock {
    fn load(vb: &VarBuilder, path: &str, channels: usize, add_identity: bool) -> Result<Self> {
        Ok(Self {
            conv1: ConvSiLU::load(
                vb,
                &format!("{path}.conv1"),
                channels,
                channels,
                3,
                1,
                1,
                1,
            )?,
            conv2: DepthwiseSeparable::load(vb, &format!("{path}.conv2"), channels, 5)?,
            add_identity,
            path: path.to_string(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv2.forward(&self.conv1.forward(x)?)?;
        if self.add_identity {
            x + &y
        } else {
            Ok(y)
        }
    }

    fn forward_profiled(&self, x: &Tensor, profiler: &mut LayerProfiler) -> Result<Tensor> {
        let y = self.conv1.forward_profiled(x, profiler)?;
        let y = self.conv2.forward_profiled(&y, profiler)?;
        if self.add_identity {
            let name = format!("{}.residual_add", self.path);
            profiler.measure_tensor(&name, "residual_add", || x + &y)
        } else {
            Ok(y)
        }
    }
}

#[derive(Debug)]
struct ChannelAttention {
    fc: Conv2d,
    path: String,
}

impl ChannelAttention {
    fn load(vb: &VarBuilder, path: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            fc: plain_conv(vb, &format!("{path}.fc"), channels, channels, 1, 1, 0, 1, true)?,
            path: path.to_string(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let pooled = x.mean_keepdim(2)?.mean_keepdim(3)?;
        // MMDetection CSPNeXt ChannelAttention uses Hardsigmoid, not sigmoid.
        // PyTorch Hardsigmoid(x) = clamp(x / 6 + 1/2, 0, 1).
        let gate = ((self.fc.forward(&pooled)? / 6.0)? + 0.5)?.clamp(0.0, 1.0)?;
        x.broadcast_mul(&gate)
    }

    fn forward_profiled(&self, x: &Tensor, profiler: &mut LayerProfiler) -> Result<Tensor> {
        let pool_name = format!("{}.global_avgpool", self.path);
        let pooled = profiler.measure_tensor(&pool_name, "channel_attention_pool", || {
            x.mean_keepdim(2)?.mean_keepdim(3)
        })?;
        let fc_name = format!("{}.fc", self.path);
        let fc = profiler.measure_tensor(&fc_name, "channel_attention_fc", || {
            self.fc.forward(&pooled)
        })?;
        let gate_name = format!("{}.hardsigmoid", self.path);
        let gate = profiler.measure_tensor(&gate_name, "channel_attention_gate", || {
            ((&fc / 6.0)? + 0.5)?.clamp(0.0, 1.0)
        })?;
        let mul_name = format!("{}.mul", self.path);
        profiler.measure_tensor(&mul_name, "channel_attention_mul", || x.broadcast_mul(&gate))
    }
}

#[derive(Debug)]
struct CspLayer {
    main: ConvSiLU,
    short: ConvSiLU,
    blocks: Vec<CspNextBlock>,
    attention: ChannelAttention,
    final_conv: ConvSiLU,
    path: String,
}

impl CspLayer {
    fn load(
        vb: &VarBuilder,
        path: &str,
        channels: usize,
        blocks: usize,
        add_identity: bool,
    ) -> Result<Self> {
        let mid = channels / 2;
        let mut inner = Vec::with_capacity(blocks);
        for i in 0..blocks {
            inner.push(CspNextBlock::load(
                vb,
                &format!("{path}.blocks.{i}"),
                mid,
                add_identity,
            )?);
        }
        Ok(Self {
            main: ConvSiLU::load(vb, &format!("{path}.main_conv"), channels, mid, 1, 1, 0, 1)?,
            short: ConvSiLU::load(vb, &format!("{path}.short_conv"), channels, mid, 1, 1, 0, 1)?,
            blocks: inner,
            attention: ChannelAttention::load(vb, &format!("{path}.attention"), mid * 2)?,
            final_conv: ConvSiLU::load(vb, &format!("{path}.final_conv"), mid * 2, channels, 1, 1, 0, 1)?,
            path: path.to_string(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let short = self.short.forward(x)?;
        let mut main = self.main.forward(x)?;
        for block in &self.blocks {
            main = block.forward(&main)?;
        }
        let joined = Tensor::cat(&[&main, &short], 1)?;
        let joined = self.attention.forward(&joined)?;
        self.final_conv.forward(&joined)
    }

    fn forward_profiled(&self, x: &Tensor, profiler: &mut LayerProfiler) -> Result<Tensor> {
        let short = self.short.forward_profiled(x, profiler)?;
        let mut main = self.main.forward_profiled(x, profiler)?;
        for block in &self.blocks {
            main = block.forward_profiled(&main, profiler)?;
        }
        let cat_name = format!("{}.concat", self.path);
        let joined = profiler.measure_tensor(&cat_name, "tensor_concat", || {
            Tensor::cat(&[&main, &short], 1)
        })?;
        let joined = self.attention.forward_profiled(&joined, profiler)?;
        self.final_conv.forward_profiled(&joined, profiler)
    }
}

fn pad_2d_min(x: &Tensor, p: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let dtype: DType = x.dtype();
    let dev: &Device = x.device();
    let top = (Tensor::zeros((b, c, p, w), dtype, dev)? - 1.0e4)?;
    let bottom = (Tensor::zeros((b, c, p, w), dtype, dev)? - 1.0e4)?;
    let y = Tensor::cat(&[&top, x, &bottom], 2)?;
    let left = (Tensor::zeros((b, c, h + 2 * p, p), dtype, dev)? - 1.0e4)?;
    let right = (Tensor::zeros((b, c, h + 2 * p, p), dtype, dev)? - 1.0e4)?;
    Tensor::cat(&[&left, &y, &right], 3)
}

#[derive(Debug)]
struct SppBottleneck {
    conv1: ConvSiLU,
    conv2: ConvSiLU,
    path: String,
}

impl SppBottleneck {
    fn load(vb: &VarBuilder, path: &str, channels: usize) -> Result<Self> {
        let mid = channels / 2;
        Ok(Self {
            conv1: ConvSiLU::load(vb, &format!("{path}.conv1"), channels, mid, 1, 1, 0, 1)?,
            conv2: ConvSiLU::load(vb, &format!("{path}.conv2"), mid * 4, channels, 1, 1, 0, 1)?,
            path: path.to_string(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?;
        let p5 = pad_2d_min(&x, 2)?.max_pool2d_with_stride(5, 1)?;
        let p9 = pad_2d_min(&x, 4)?.max_pool2d_with_stride(9, 1)?;
        let p13 = pad_2d_min(&x, 6)?.max_pool2d_with_stride(13, 1)?;
        let joined = Tensor::cat(&[&x, &p5, &p9, &p13], 1)?;
        self.conv2.forward(&joined)
    }

    fn forward_profiled(&self, x: &Tensor, profiler: &mut LayerProfiler) -> Result<Tensor> {
        let x = self.conv1.forward_profiled(x, profiler)?;
        let p5_name = format!("{}.pool5", self.path);
        let p5 = profiler.measure_tensor(&p5_name, "spp_pool", || {
            pad_2d_min(&x, 2)?.max_pool2d_with_stride(5, 1)
        })?;
        let p9_name = format!("{}.pool9", self.path);
        let p9 = profiler.measure_tensor(&p9_name, "spp_pool", || {
            pad_2d_min(&x, 4)?.max_pool2d_with_stride(9, 1)
        })?;
        let p13_name = format!("{}.pool13", self.path);
        let p13 = profiler.measure_tensor(&p13_name, "spp_pool", || {
            pad_2d_min(&x, 6)?.max_pool2d_with_stride(13, 1)
        })?;
        let cat_name = format!("{}.concat", self.path);
        let joined = profiler.measure_tensor(&cat_name, "tensor_concat", || {
            Tensor::cat(&[&x, &p5, &p9, &p13], 1)
        })?;
        self.conv2.forward_profiled(&joined, profiler)
    }
}

#[derive(Debug)]
struct Stage {
    downsample: ConvSiLU,
    spp: Option<SppBottleneck>,
    csp: CspLayer,
}

impl Stage {
    fn load(
        vb: &VarBuilder,
        stage: usize,
        in_c: usize,
        out_c: usize,
        blocks: usize,
        add_identity: bool,
        use_spp: bool,
    ) -> Result<Self> {
        let base = format!("backbone.stage{stage}");
        let downsample = ConvSiLU::load(vb, &format!("{base}.0"), in_c, out_c, 3, 2, 1, 1)?;
        let spp = if use_spp {
            Some(SppBottleneck::load(vb, &format!("{base}.1"), out_c)?)
        } else {
            None
        };
        let csp_index = if use_spp { 2 } else { 1 };
        let csp = CspLayer::load(
            vb,
            &format!("{base}.{csp_index}"),
            out_c,
            blocks,
            add_identity,
        )?;
        Ok(Self { downsample, spp, csp })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = self.downsample.forward(x)?;
        if let Some(spp) = &self.spp {
            y = spp.forward(&y)?;
        }
        self.csp.forward(&y)
    }

    fn forward_profiled(&self, x: &Tensor, profiler: &mut LayerProfiler) -> Result<Tensor> {
        let mut y = self.downsample.forward_profiled(x, profiler)?;
        if let Some(spp) = &self.spp {
            y = spp.forward_profiled(&y, profiler)?;
        }
        self.csp.forward_profiled(&y, profiler)
    }
}

/// CSPNeXt-P5 medium backbone used by the official RTMPose-m Body26 model.
/// Input: [B,3,256,192], output: [B,768,8,6].
#[derive(Debug)]
pub struct CspNextM {
    stem0: ConvSiLU,
    stem1: ConvSiLU,
    stem2: ConvSiLU,
    stages: [Stage; 4],
}

impl CspNextM {
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let stem0 = ConvSiLU::load(vb, "backbone.stem.0", 3, 24, 3, 2, 1, 1)?;
        let stem1 = ConvSiLU::load(vb, "backbone.stem.1", 24, 24, 3, 1, 1, 1)?;
        let stem2 = ConvSiLU::load(vb, "backbone.stem.2", 24, 48, 3, 1, 1, 1)?;
        let stages = [
            Stage::load(vb, 1, 48, 96, 2, true, false)?,
            Stage::load(vb, 2, 96, 192, 4, true, false)?,
            Stage::load(vb, 3, 192, 384, 4, true, false)?,
            Stage::load(vb, 4, 384, 768, 2, false, true)?,
        ];
        Ok(Self { stem0, stem1, stem2, stages })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = self.stem0.forward(x)?;
        y = self.stem1.forward(&y)?;
        y = self.stem2.forward(&y)?;
        for stage in &self.stages {
            y = stage.forward(&y)?;
        }
        Ok(y)
    }

    pub fn forward_profiled(&self, x: &Tensor, profiler: &mut LayerProfiler) -> Result<Tensor> {
        let mut y = self.stem0.forward_profiled(x, profiler)?;
        y = self.stem1.forward_profiled(&y, profiler)?;
        y = self.stem2.forward_profiled(&y, profiler)?;
        for stage in &self.stages {
            y = stage.forward_profiled(&y, profiler)?;
        }
        Ok(y)
    }
}
