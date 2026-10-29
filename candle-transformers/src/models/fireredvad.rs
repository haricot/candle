//! Streaming FireRedVAD DFSMN inference.
//!
//! The model consumes CMVN-normalized filterbank features and returns speech
//! probabilities plus an explicit recurrent cache. Audio capture, filterbanks,
//! smoothing and endpoint policy belong to the caller.
//! Architecture: <https://github.com/FireRedTeam/FireRedVAD> (Apache-2.0).

use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};

pub mod audio;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub projection_dim: usize,
    pub num_layers: usize,
    pub memory_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_dim: 80,
            hidden_dim: 256,
            projection_dim: 128,
            num_layers: 8,
            memory_size: 20,
        }
    }
}

struct Block {
    fc1: Linear,
    fc2: Linear,
    filter: Tensor,
}

pub struct Model {
    blocks: Vec<Block>,
    dnn: Linear,
    out: Linear,
    config: Config,
    dtype: DType,
    device: Device,
}

impl Model {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        if config.input_dim == 0
            || config.hidden_dim == 0
            || config.projection_dim == 0
            || config.num_layers == 0
            || config.memory_size == 0
        {
            candle::bail!("FireRedVAD dimensions must be positive")
        }
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let b = vb.pp(format!("blocks.{i}"));
            blocks.push(Block {
                fc1: linear(
                    if i == 0 {
                        config.input_dim
                    } else {
                        config.projection_dim
                    },
                    config.hidden_dim,
                    b.pp("fc1"),
                )?,
                fc2: if i == 0 {
                    linear(config.hidden_dim, config.projection_dim, b.pp("fc2"))?
                } else {
                    linear_no_bias(config.hidden_dim, config.projection_dim, b.pp("fc2"))?
                },
                filter: b.get((config.projection_dim, 1, config.memory_size), "filter")?,
            });
        }
        Ok(Self {
            blocks,
            dnn: linear(config.projection_dim, config.hidden_dim, vb.pp("dnn"))?,
            out: linear(config.hidden_dim, 1, vb.pp("out"))?,
            config: config.clone(),
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    /// Cache for one stream: `[layers, 1, projection_dim, memory_size - 1]`.
    pub fn zero_cache(&self) -> Result<Tensor> {
        Tensor::zeros(
            (
                self.config.num_layers,
                1,
                self.config.projection_dim,
                self.config.memory_size - 1,
            ),
            self.dtype,
            &self.device,
        )
    }

    /// Input `[1, frames, input_dim]`; output probabilities `[1, frames, 1]`.
    pub fn forward(&self, features: &Tensor, caches: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t, d) = features.dims3()?;
        let history = self.config.memory_size - 1;
        if b != 1
            || t == 0
            || d != self.config.input_dim
            || caches.dims()
                != [
                    self.config.num_layers,
                    1,
                    self.config.projection_dim,
                    history,
                ]
        {
            candle::bail!("FireRedVAD input/cache shape mismatch")
        }
        let mut x = features.clone();
        let mut new_cache = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let h = block.fc1.forward(&x)?.relu()?;
            let p = block.fc2.forward(&h)?;
            let p = if i == 0 { p.relu()? } else { p };
            let p = p.transpose(1, 2)?.contiguous()?;
            let all = Tensor::cat(&[&caches.i(i)?, &p], 2)?;
            new_cache.push(all.narrow(2, t, history)?.contiguous()?);
            let memory = (&p + all.conv1d(&block.filter, 0, 1, 1, self.config.projection_dim)?)?
                .transpose(1, 2)?
                .contiguous()?;
            x = if i == 0 { memory } else { (&x + memory)? };
        }
        let probability =
            candle_nn::ops::sigmoid(&self.out.forward(&self.dnn.forward(&x)?.relu()?)?)?;
        Ok((probability, Tensor::stack(&new_cache, 0)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn streaming_cache_matches_whole_sequence() -> Result<()> {
        let device = Device::Cpu;
        let cfg = Config {
            input_dim: 2,
            hidden_dim: 2,
            projection_dim: 2,
            num_layers: 1,
            memory_size: 3,
        };
        let mut tensors = HashMap::new();
        for name in ["blocks.0.fc1", "blocks.0.fc2", "dnn"] {
            tensors.insert(
                format!("{name}.weight"),
                Tensor::new(&[[0.1f32, 0.2], [0.3, 0.1]], &device)?,
            );
            tensors.insert(
                format!("{name}.bias"),
                Tensor::zeros(2, DType::F32, &device)?,
            );
        }
        tensors.insert(
            "blocks.0.filter".into(),
            Tensor::new(&[[[0.1f32, 0.2, 0.3]], [[0.2, 0.1, 0.3]]], &device)?,
        );
        tensors.insert("out.weight".into(), Tensor::new(&[[0.3f32, 0.4]], &device)?);
        tensors.insert("out.bias".into(), Tensor::zeros(1, DType::F32, &device)?);
        let model = Model::new(&cfg, VarBuilder::from_tensors(tensors, DType::F32, &device))?;
        let features = Tensor::new(
            &[[
                [0.1f32, 0.2],
                [0.3, 0.1],
                [0.4, 0.2],
                [0.1, 0.5],
                [0.2, 0.3],
            ]],
            &device,
        )?;
        let (whole, whole_cache) = model.forward(&features, &model.zero_cache()?)?;
        let (first, cache) = model.forward(&features.narrow(1, 0, 3)?, &model.zero_cache()?)?;
        let (last, cache) = model.forward(&features.narrow(1, 3, 2)?, &cache)?;
        let joined = Tensor::cat(&[first, last], 1)?;
        let difference = (&joined - whole)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(difference < 1e-6);
        assert_eq!(
            cache.flatten_all()?.to_vec1::<f32>()?,
            whole_cache.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }
}
