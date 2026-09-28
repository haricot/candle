use candle_core::{DType, D, Result, Tensor};
use candle_nn::{ops, Conv2d, Conv2dConfig, Linear, Module, VarBuilder};

use crate::profiling::LayerProfiler;

fn linear_no_bias(vb: &VarBuilder, path: &str, in_dim: usize, out_dim: usize) -> Result<Linear> {
    let weight = vb.pp(path).get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(weight, None))
}

#[derive(Debug)]
struct ScaleNorm {
    g: Tensor,
    scale: f64,
    eps: f64,
}

impl ScaleNorm {
    fn load(vb: &VarBuilder, path: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            g: vb.pp(path).get(1, "g")?,
            scale: (dim as f64).powf(-0.5),
            eps: 1e-5,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // RTMCC ScaleNorm is sensitive to fp16 overflow because the L2 norm
        // reduces over the flattened feature dimension. Keep the reduction in
        // F32 even when the rest of the model runs in F16, then cast back.
        let input_dtype = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let g = self.g.to_dtype(DType::F32)?;
        let norm = xf.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let denom = (norm * self.scale)?.clamp(self.eps, f64::MAX)?;
        xf.broadcast_div(&denom)?
            .broadcast_mul(&g)?
            .to_dtype(input_dtype)
    }

    fn forward_profiled(
        &self,
        x: &Tensor,
        profiler: &mut LayerProfiler,
        name: &str,
    ) -> Result<Tensor> {
        profiler.measure_tensor(name, "rtmcc_scale_norm", || self.forward(x))
    }
}

#[derive(Debug)]
struct RtmccBlock {
    ln: ScaleNorm,
    uv: Linear,
    o: Linear,
    gamma: Tensor,
    beta: Tensor,
    res_scale: Tensor,
}

impl RtmccBlock {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let p = vb.pp("head.gau");
        Ok(Self {
            ln: ScaleNorm::load(vb, "head.gau.ln", 256)?,
            uv: linear_no_bias(vb, "head.gau.uv", 256, 1152)?,
            o: linear_no_bias(vb, "head.gau.o", 512, 256)?,
            gamma: p.get((2, 128), "gamma")?,
            beta: p.get((2, 128), "beta")?,
            res_scale: p.pp("res_scale").get(256, "scale")?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.ln.forward(input)?;
        let uv = ops::silu(&self.uv.forward(&x)?)?;
        let u = uv.narrow(2, 0, 512)?;
        let v = uv.narrow(2, 512, 512)?;
        let base = uv.narrow(2, 1024, 128)?.unsqueeze(2)?;

        let gamma = self.gamma.reshape((1, 1, 2, 128))?;
        let beta = self.beta.reshape((1, 1, 2, 128))?;
        let qk_base = base.broadcast_mul(&gamma)?.broadcast_add(&beta)?;
        // `narrow`/`squeeze` preserve the parent strides. Candle's matmul
        // requires dense operands on this path (CUDA and CPU), so materialize
        // the GAU Q/K/V views before the two GEMMs.
        let q = qk_base.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?;
        let k = qk_base.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?;
        let v = v.contiguous()?;

        let kt = k.transpose(1, 2)?.contiguous()?;
        let qk = (q.matmul(&kt)? / (128f64).sqrt())?;
        let kernel = qk.relu()?.sqr()?.contiguous()?;
        let attended = kernel.matmul(&v)?;
        let gated = u.broadcast_mul(&attended)?;
        let main = self.o.forward(&gated)?;

        let scale = self.res_scale.reshape((1, 1, 256))?;
        let residual = input.broadcast_mul(&scale)?;
        residual + main
    }

    fn forward_profiled(&self, input: &Tensor, profiler: &mut LayerProfiler) -> Result<Tensor> {
        let x = self
            .ln
            .forward_profiled(input, profiler, "head.gau.scale_norm")?;
        let uv_linear = profiler.measure_tensor("head.gau.uv", "rtmcc_linear", || {
            self.uv.forward(&x)
        })?;
        let uv = profiler.measure_tensor("head.gau.uv_silu", "activation_silu", || {
            ops::silu(&uv_linear)
        })?;
        let u = uv.narrow(2, 0, 512)?;
        let v_view = uv.narrow(2, 512, 512)?;
        let base = uv.narrow(2, 1024, 128)?.unsqueeze(2)?;

        let gamma = self.gamma.reshape((1, 1, 2, 128))?;
        let beta = self.beta.reshape((1, 1, 2, 128))?;
        let qk_base = profiler.measure_tensor(
            "head.gau.qk_affine",
            "rtmcc_elementwise",
            || base.broadcast_mul(&gamma)?.broadcast_add(&beta),
        )?;
        let q = profiler.measure_tensor("head.gau.q_contiguous", "tensor_contiguous", || {
            qk_base.narrow(2, 0, 1)?.squeeze(2)?.contiguous()
        })?;
        let k = profiler.measure_tensor("head.gau.k_contiguous", "tensor_contiguous", || {
            qk_base.narrow(2, 1, 1)?.squeeze(2)?.contiguous()
        })?;
        let v = profiler.measure_tensor("head.gau.v_contiguous", "tensor_contiguous", || {
            v_view.contiguous()
        })?;
        let kt = profiler.measure_tensor("head.gau.kt_contiguous", "tensor_contiguous", || {
            k.transpose(1, 2)?.contiguous()
        })?;
        let qk = profiler.measure_tensor("head.gau.qk_matmul", "rtmcc_matmul", || {
            (q.matmul(&kt)? / (128f64).sqrt())
        })?;
        let kernel = profiler.measure_tensor("head.gau.kernel", "rtmcc_elementwise", || {
            qk.relu()?.sqr()?.contiguous()
        })?;
        let attended = profiler.measure_tensor("head.gau.attention_v_matmul", "rtmcc_matmul", || {
            kernel.matmul(&v)
        })?;
        let gated = profiler.measure_tensor("head.gau.gated_mul", "rtmcc_elementwise", || {
            u.broadcast_mul(&attended)
        })?;
        let main = profiler.measure_tensor("head.gau.o", "rtmcc_linear", || {
            self.o.forward(&gated)
        })?;

        let scale = self.res_scale.reshape((1, 1, 256))?;
        let residual = profiler.measure_tensor("head.gau.residual_scale", "rtmcc_elementwise", || {
            input.broadcast_mul(&scale)
        })?;
        profiler.measure_tensor("head.gau.residual_add", "residual_add", || residual + main)
    }
}

/// RTMCCHead for the official 26-keypoint, 192x256 RTMPose-m profile.
#[derive(Debug)]
pub struct RtmccHead26 {
    final_layer: Conv2d,
    mlp_norm: ScaleNorm,
    mlp: Linear,
    gau: RtmccBlock,
    cls_x: Linear,
    cls_y: Linear,
}

impl RtmccHead26 {
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let p = vb.pp("head.final_layer");
        let weight = p.get((26, 768, 7, 7), "weight")?;
        let bias = p.get(26, "bias")?;
        let final_layer = Conv2d::new(
            weight,
            Some(bias),
            Conv2dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        );
        Ok(Self {
            final_layer,
            mlp_norm: ScaleNorm::load(vb, "head.mlp.0", 48)?,
            mlp: linear_no_bias(vb, "head.mlp.1", 48, 256)?,
            gau: RtmccBlock::load(vb)?,
            cls_x: linear_no_bias(vb, "head.cls_x", 256, 384)?,
            cls_y: linear_no_bias(vb, "head.cls_y", 256, 512)?,
        })
    }

    /// Returns `(simcc_x, simcc_y)` with shapes `[B,26,384]` and `[B,26,512]`.
    pub fn forward(&self, feature: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = self.final_layer.forward(feature)?;
        let (b, k, h, w) = x.dims4()?;
        if k != 26 || h * w != 48 {
            candle_core::bail!(
                "RTMCCHead26 expected [B,26,8,6] feature projection, got {:?}",
                x.dims()
            )
        }
        let x = x.reshape((b, k, h * w))?;
        let x = self.mlp_norm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = self.gau.forward(&x)?;
        Ok((self.cls_x.forward(&x)?, self.cls_y.forward(&x)?))
    }

    pub fn forward_profiled(
        &self,
        feature: &Tensor,
        profiler: &mut LayerProfiler,
    ) -> Result<(Tensor, Tensor)> {
        let x = profiler.measure_tensor("head.final_layer", "head_conv_7x7", || {
            self.final_layer.forward(feature)
        })?;
        let (b, k, h, w) = x.dims4()?;
        if k != 26 || h * w != 48 {
            candle_core::bail!(
                "RTMCCHead26 expected [B,26,8,6] feature projection, got {:?}",
                x.dims()
            )
        }
        let x = x.reshape((b, k, h * w))?;
        let x = self
            .mlp_norm
            .forward_profiled(&x, profiler, "head.mlp.scale_norm")?;
        let x = profiler.measure_tensor("head.mlp.linear", "rtmcc_linear", || {
            self.mlp.forward(&x)
        })?;
        let x = self.gau.forward_profiled(&x, profiler)?;
        let simcc_x = profiler.measure_tensor("head.cls_x", "rtmcc_linear", || {
            self.cls_x.forward(&x)
        })?;
        let simcc_y = profiler.measure_tensor("head.cls_y", "rtmcc_linear", || {
            self.cls_y.forward(&x)
        })?;
        Ok((simcc_x, simcc_y))
    }
}
