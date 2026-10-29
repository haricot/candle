//! Reusable building blocks for a continuous-latent FlowLM/Mimi audio path.
//! No application-specific voice policies or checkpoint assets.

use candle::{Error, Result, Tensor};
use candle_nn::{
    conv1d_no_bias, linear_no_bias, Conv1d, Conv1dConfig, ConvTranspose1d, Linear, Module,
    VarBuilder,
};

/// FlowLM input projection loaded from the input_linear checkpoint key.
pub struct FlowLmInputProjection {
    layer: Linear,
}
impl FlowLmInputProjection {
    pub fn load(latent_dim: usize, hidden_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            layer: linear_no_bias(latent_dim, hidden_dim, vb.pp("input_linear"))?,
        })
    }
    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        self.layer.forward(latent)
    }
}

/// Continuous-latent Mimi projection; no vector-quantizer codebook is needed.
pub struct MimiContinuousProjection {
    layer: Conv1d,
}
impl MimiContinuousProjection {
    pub fn load(latent_dim: usize, decoder_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            layer: conv1d_no_bias(
                latent_dim,
                decoder_dim,
                1,
                Conv1dConfig::default(),
                vb.pp("output_proj"),
            )?,
        })
    }
    pub fn forward(&self, latent_bct: &Tensor) -> Result<Tensor> {
        self.layer.forward(latent_bct)
    }
}

/// Device-resident overlap-add for unpadded ConvTranspose1d with dilation=1.
/// Each push commits T*stride samples; finish emits the remaining kernel-stride.
/// Supports stride < kernel <= 2*stride (including Mimi kernel32/stride16).
/// The provided layer must have the matching stride and kernel configuration.
pub struct StreamingConvTranspose1d {
    layer: ConvTranspose1d,
    stride: usize,
    overlap: usize,
    pending_unbiased: Option<Tensor>,
    output_shape: Option<(usize, usize)>,
    finished: bool,
}

impl StreamingConvTranspose1d {
    pub fn new(layer: ConvTranspose1d, stride: usize, kernel: usize) -> Result<Self> {
        if stride == 0 || kernel <= stride || kernel > stride.saturating_mul(2) {
            return Err(Error::Msg(format!(
                "stream requires 0 < stride < kernel <= 2*stride, got {stride} and {kernel}"
            )));
        }
        Ok(Self {
            layer,
            stride,
            overlap: kernel - stride,
            pending_unbiased: None,
            output_shape: None,
            finished: false,
        })
    }

    pub fn push(&mut self, chunk: &Tensor) -> Result<Tensor> {
        if self.finished {
            return Err(Error::Msg("stream already finalized".into()));
        }
        let (_, _, frames) = chunk.dims3()?;
        if frames == 0 {
            return Err(Error::Msg("empty ConvTranspose1d chunk".into()));
        }
        let raw = self.layer.forward(chunk)?;
        let (batch, channels, samples) = raw.dims3()?;
        let committed = frames
            .checked_mul(self.stride)
            .ok_or_else(|| Error::Msg("sample count overflow".into()))?;
        let expected = committed
            .checked_add(self.overlap)
            .ok_or_else(|| Error::Msg("sample count overflow".into()))?;
        if samples != expected {
            return Err(Error::Msg(format!(
                "ConvTranspose1d configuration mismatch: expected {expected}, got {samples}"
            )));
        }
        if let Some(shape) = self.output_shape {
            if shape != (batch, channels) {
                return Err(Error::Msg("stream batch/channel shape changed".into()));
            }
        }

        // Keep previous tail without bias. The current convolution already
        // includes the bias, so adding an unbiased carry counts it only once.
        let head = raw.narrow(2, 0, self.overlap)?;
        let head = match self.pending_unbiased.as_ref() {
            Some(previous) => (&head + previous)?,
            None => head,
        };
        let body = raw.narrow(2, self.overlap, committed - self.overlap)?;
        let output = Tensor::cat(&[&head, &body], 2)?;
        let tail = raw.narrow(2, committed, self.overlap)?;
        let tail = match self.layer.bias() {
            Some(bias) => tail.broadcast_sub(&bias.reshape((1, channels, 1))?)?,
            None => tail,
        };
        self.pending_unbiased = Some(tail);
        self.output_shape = Some((batch, channels));
        Ok(output)
    }

    pub fn finish(&mut self) -> Result<Tensor> {
        if self.finished {
            return Err(Error::Msg("stream already finalized".into()));
        }
        let pending = self
            .pending_unbiased
            .take()
            .ok_or_else(|| Error::Msg("cannot finalize a stream without input".into()))?;
        let (_, channels, _) = pending.dims3()?;
        let tail = match self.layer.bias() {
            Some(bias) => pending.broadcast_add(&bias.reshape((1, channels, 1))?)?,
            None => pending,
        };
        self.finished = true;
        Ok(tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use candle_nn::ConvTranspose1dConfig;

    fn make_layer(device: &Device, biased: bool) -> Result<ConvTranspose1d> {
        let weight = Tensor::ones((1, 1, 4), DType::F32, device)?;
        let bias = if biased {
            Some(Tensor::from_vec(vec![0.5_f32], 1, device)?)
        } else {
            None
        };
        let config = ConvTranspose1dConfig {
            stride: 2,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups: 1,
        };
        Ok(ConvTranspose1d::new(weight, bias, config))
    }

    #[test]
    fn variable_chunks_equal_offline_including_nonzero_bias() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1_f32, 2., 3., 4.], (1, 1, 4), &device)?;
        for biased in [false, true] {
            let offline = make_layer(&device, biased)?.forward(&x)?;
            let mut stream = StreamingConvTranspose1d::new(make_layer(&device, biased)?, 2, 4)?;
            let mut outputs = Vec::new();
            let mut offset = 0;
            for frames in [1, 2, 1] {
                outputs.push(stream.push(&x.narrow(2, offset, frames)?)?);
                offset += frames;
            }
            outputs.push(stream.finish()?);
            let pieces: Vec<&Tensor> = outputs.iter().collect();
            let online = Tensor::cat(&pieces, 2)?;
            assert_eq!(online.dims(), offline.dims());
            let delta = (&online - &offline)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert_eq!(delta, 0.0);
            assert!(stream.push(&x.narrow(2, 0, 1)?).is_err());
            assert!(stream.finish().is_err());
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_shapes_and_empty_streams() -> Result<()> {
        let device = Device::Cpu;
        assert!(StreamingConvTranspose1d::new(make_layer(&device, false)?, 0, 4).is_err());
        assert!(StreamingConvTranspose1d::new(make_layer(&device, false)?, 2, 5).is_err());
        let mut stream = StreamingConvTranspose1d::new(make_layer(&device, false)?, 2, 4)?;
        assert!(stream.finish().is_err());
        Ok(())
    }
}
