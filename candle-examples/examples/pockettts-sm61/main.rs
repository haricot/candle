//! P2-R5-A0: deterministic operator smoke gate for a PocketTTS-oriented
//! FlowLM/Mimi execution path. This is not a model inference or audio-parity test.
use candle::{DType, Device, Error, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, Module};

const LATENT_DIM: usize = 32;
const CODEC_DIM: usize = 512;
const UPSAMPLE_STRIDE: usize = 16;
const UPSAMPLE_KERNEL: usize = 32;

fn expect_shape(tensor: &Tensor, expected: &[usize], label: &str) -> Result<()> {
    if tensor.dims() == expected {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "{label}: expected {expected:?}, got {:?}", tensor.dims()
        )))
    }
}

fn expect_first(tensor: &Tensor, expected: f32, label: &str) -> Result<()> {
    let data = tensor.flatten_all()?.to_vec1::<f32>()?;
    if data.first() == Some(&expected) {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "{label}: expected first element {expected}, got {:?}", data.first()
        )))
    }
}

/// Exercises a FlowLM-shaped linear projection, Mimi's continuous-latent
/// pointwise projection, and grouped causal transposed-convolution overlap.
/// Uses only f32 and explicit deterministic tensors (no weights or RNG).
fn preflight(device: &Device) -> Result<f32> {
    let dtype = DType::F32;

    let flow_input = Tensor::ones((1, 1, LATENT_DIM), dtype, device)?;
    let flow_weight = Tensor::ones((CODEC_DIM, LATENT_DIM), dtype, device)?;
    let flow_output = Linear::new(flow_weight, None).forward(&flow_input)?;
    expect_shape(&flow_output, &[1, 1, CODEC_DIM], "FlowLM projection")?;
    expect_first(&flow_output, LATENT_DIM as f32, "FlowLM projection")?;

    let latent = Tensor::ones((1, LATENT_DIM, 1), dtype, device)?;
    let quant_weight = Tensor::ones((CODEC_DIM, LATENT_DIM, 1), dtype, device)?;
    let projected = Conv1d::new(quant_weight, None, Conv1dConfig::default())
        .forward(&latent)?;
    expect_shape(&projected, &[1, CODEC_DIM, 1], "Mimi pointwise projection")?;
    expect_first(&projected, LATENT_DIM as f32, "Mimi pointwise projection")?;

    let conv_weight = Tensor::ones((CODEC_DIM, 1, UPSAMPLE_KERNEL), dtype, device)?;
    let cfg = ConvTranspose1dConfig {
        stride: UPSAMPLE_STRIDE,
        padding: 0,
        output_padding: 0,
        dilation: 1,
        groups: CODEC_DIM,
    };
    let deconv = ConvTranspose1d::new(conv_weight, None, cfg);
    let input = Tensor::ones((1, CODEC_DIM, 2), dtype, device)?;
    let full = deconv.forward(&input)?;
    expect_shape(&full, &[1, CODEC_DIM, 48], "Mimi grouped transposed conv")?;

    // Streaming state remains device-resident across arbitrarily sized chunks.
    let mut stream = candle_transformers::models::pockettts::StreamingConvTranspose1d::new(
        deconv, UPSAMPLE_STRIDE, UPSAMPLE_KERNEL,
    )?;
    let first = stream.push(&input.narrow(2, 0, 1)?)?;
    let second = stream.push(&input.narrow(2, 1, 1)?)?;
    let tail = stream.finish()?;
    let streamed = Tensor::cat(&[&first, &second, &tail], 2)?;
    expect_shape(&streamed, &[1, CODEC_DIM, 48], "Mimi streamed overlap")?;

    let delta = (&full - &streamed)?.abs()?.max_all()?.to_scalar::<f32>()?;
    if delta != 0.0 {
        return Err(Error::Msg(format!(
            "Mimi streamed overlap: expected exact agreement, max abs delta={delta}"
        )));
    }

    let first_channel = full.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
    for (sample_index, expected) in [(0, 1.0), (15, 1.0), (16, 2.0),
                                      (31, 2.0), (32, 1.0), (47, 1.0)] {
        if first_channel[sample_index] != expected {
            return Err(Error::Msg(format!(
                "Mimi overlap sample {sample_index}: expected {expected}, got {}",
                first_channel[sample_index]
            )));
        }
    }
    Ok(delta)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let backend = match args.as_slice() {
        [] => "cpu",
        [flag, value] if flag == "--device" => value.as_str(),
        _ => return Err(Error::Msg("usage: pockettts-sm61 [--device cpu|cuda]".into())),
    };
    let device = match backend {
        "cpu" => Device::Cpu,
        "cuda" => Device::new_cuda(0)?,
        _ => return Err(Error::Msg(format!("unsupported device: {backend}"))),
    };
    let delta = preflight(&device)?;
    println!(
        "POCKETTTS_SM61_A1=PASS backend={backend} dtype=f32 flow_projection=PASS mimi_projection=PASS grouped_ct1d=PASS stream_overlap_max_abs={delta}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_cpu_operator_gate() -> Result<()> {
        assert_eq!(preflight(&Device::Cpu)?, 0.0);
        Ok(())
    }
}
