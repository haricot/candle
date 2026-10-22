//! FireRedTeam Stream-VAD DFSMN, with the exact eight recurrent cache tensors.
use crate::{frontend::KaldiFbank, weights};
use anyhow::{ensure, Result};
use candle::{Device, IndexOp, Module, Tensor};
use candle_nn::{linear, linear_no_bias, Linear};
use serde::{Deserialize, Serialize};
use std::path::Path;

struct Block {
    fc1: Linear,
    fc2: Linear,
    filter: Tensor,
}

pub struct FireRedModel {
    blocks: Vec<Block>,
    dnn: Linear,
    out: Linear,
    device: Device,
}

impl FireRedModel {
    pub fn load(root: &Path, device: &Device) -> Result<(Self, Vec<f32>, Vec<f32>)> {
        #[derive(Deserialize)]
        struct Config {
            architecture: String,
            mean: Vec<f32>,
            invstd: Vec<f32>,
        }
        let config: Config = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        ensure!(
            config.architecture == "fireredvad-stream-8x256x128-k20"
                && config.mean.len() == 80
                && config.invstd.len() == 80,
            "Unsupported FireRed architecture"
        );
        ensure!(
            config.mean.iter().all(|x| x.is_finite())
                && config.invstd.iter().all(|x| x.is_finite() && *x > 0.0),
            "Invalid CMVN values"
        );
        let vb = weights::load(root, "fireredvad-stream", device)?;
        let mut blocks = Vec::new();
        for i in 0..8 {
            let b = vb.pp(format!("blocks.{i}"));
            blocks.push(Block {
                fc1: linear(if i == 0 { 80 } else { 128 }, 256, b.pp("fc1"))?,
                fc2: if i == 0 {
                    linear(256, 128, b.pp("fc2"))?
                } else {
                    linear_no_bias(256, 128, b.pp("fc2"))?
                },
                filter: b.get((128, 1, 20), "filter")?,
            });
        }
        Ok((
            Self {
                blocks,
                dnn: linear(128, 256, vb.pp("dnn"))?,
                out: linear(256, 1, vb.pp("out"))?,
                device: device.clone(),
            },
            config.mean,
            config.invstd,
        ))
    }

    pub fn zero_cache(&self) -> candle::Result<Tensor> {
        Tensor::zeros((8, 1, 128, 19), candle::DType::F32, &self.device)
    }

    pub fn forward(&self, features: &Tensor, caches: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t, d) = features.dims3()?;
        ensure!(
            b == 1 && t > 0 && d == 80 && caches.dims() == [8, 1, 128, 19],
            "FireRed input/cache shape mismatch"
        );
        let mut x = features.clone();
        let mut new_cache = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            let h = block.fc1.forward(&x)?.relu()?;
            let p = block.fc2.forward(&h)?;
            let p = if i == 0 { p.relu()? } else { p };
            let p = p.transpose(1, 2)?.contiguous()?;
            let all = Tensor::cat(&[&caches.i(i)?, &p], 2)?;
            new_cache.push(all.narrow(2, t, 19)?.contiguous()?);
            // The historical cache supplies all left context; no right padding
            // or lookahead is introduced by the native grouped convolution.
            let memory = (&p + all.conv1d(&block.filter, 0, 1, 1, 128)?)?
                .transpose(1, 2)?
                .contiguous()?;
            x = if i == 0 { memory } else { (&x + memory)? };
        }
        let probability =
            candle_nn::ops::sigmoid(&self.out.forward(&self.dnn.forward(&x)?.relu()?)?)?;
        Ok((probability, Tensor::stack(&new_cache, 0)?))
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct VadEvent {
    pub kind: String,
    pub sample: usize,
}

#[derive(Default)]
pub struct Postprocessor {
    window: std::collections::VecDeque<f32>,
    state: u8,
    speech: usize,
    silence: usize,
    frame: usize,
}

impl Postprocessor {
    pub fn feed(&mut self, probabilities: &[f32]) -> Result<Vec<VadEvent>> {
        let mut events = Vec::new();
        for &p in probabilities {
            ensure!(
                p.is_finite() && (0.0..=1.0).contains(&p),
                "Invalid VAD probability"
            );
            self.frame += 1;
            self.window.push_back(p);
            if self.window.len() > 5 {
                self.window.pop_front();
            }
            let speech = self.window.iter().map(|&p| p as f64).sum::<f64>()
                / self.window.len() as f64
                >= 0.4;
            match self.state {
                0 if speech => {
                    self.speech = 1;
                    self.state = 1;
                }
                1 if speech => {
                    self.speech += 1;
                    if self.speech >= 8 {
                        self.state = 2;
                        self.silence = 0;
                        events.push(VadEvent {
                            kind: "speech_start".into(),
                            sample: self.frame.saturating_sub(self.speech + 5) * 160,
                        });
                    }
                }
                1 => {
                    self.state = 0;
                    self.speech = 0;
                }
                2 => {
                    self.speech += 1;
                    if !speech {
                        self.silence = 1;
                        self.state = 3;
                    }
                }
                3 => {
                    self.speech += 1;
                    if speech {
                        self.silence = 0;
                        self.state = 2;
                    } else {
                        self.silence += 1;
                        if self.silence >= 20 {
                            events.push(VadEvent {
                                kind: "speech_end".into(),
                                sample: self.frame * 160,
                            });
                            self.state = 0;
                            self.speech = 0;
                        }
                    }
                }
                _ => {}
            }
            if (self.state == 2 || self.state == 3) && self.speech >= 2000 {
                events.push(VadEvent {
                    kind: "speech_end".into(),
                    sample: self.frame * 160,
                });
                self.state = 0;
                self.speech = 0;
            }
        }
        Ok(events)
    }
}

pub struct FireRedVad {
    pub model: FireRedModel,
    pub cache: Tensor,
    fbank: KaldiFbank,
    mean: Vec<f32>,
    invstd: Vec<f32>,
    post: Postprocessor,
}

impl FireRedVad {
    pub fn load(root: &Path) -> Result<Self> {
        let (model, mean, invstd) = FireRedModel::load(root, &Device::Cpu)?;
        let cache = model.zero_cache()?;
        Ok(Self {
            model,
            cache,
            mean,
            invstd,
            fbank: KaldiFbank::default(),
            post: Postprocessor::default(),
        })
    }
    pub fn reset(&mut self) -> Result<()> {
        self.cache = self.model.zero_cache()?;
        self.fbank.reset();
        self.post = Postprocessor::default();
        Ok(())
    }
    pub fn features(&mut self, pcm: &[i16]) -> Vec<f32> {
        let mut features = self.fbank.feed(pcm);
        for (i, x) in features.iter_mut().enumerate() {
            *x = (*x - self.mean[i % 80]) * self.invstd[i % 80];
        }
        features
    }
    pub fn feed(&mut self, pcm: &[i16]) -> Result<(Vec<VadEvent>, Vec<f32>)> {
        let features = self.features(pcm);
        if features.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let x = Tensor::from_vec(features.clone(), (1, features.len() / 80, 80), &Device::Cpu)?;
        let (probability, cache) = self.model.forward(&x, &self.cache)?;
        self.cache = cache;
        let probs = probability.flatten_all()?.to_vec1::<f32>()?;
        Ok((self.post.feed(&probs)?, probs))
    }
}
