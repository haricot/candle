//! Deterministic Slaney log-Mel and a bounded, chunk-independent PCM frontend.
use super::FeatureExtractorConfig;
use candle::{DType, Device, Result, Tensor};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::{collections::VecDeque, sync::Arc};

#[derive(Clone)]
pub struct LogMel {
    cfg: FeatureExtractorConfig,
    window: Vec<f32>,
    bank: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl std::fmt::Debug for LogMel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogMel").field("config", &self.cfg).finish()
    }
}

#[derive(Debug, Clone)]
pub struct Features {
    pub values: Tensor,
    pub valid_frames: usize,
}

/// Only the overlapping analysis window and preceding PCM sample are retained.
#[derive(Debug, Clone, Default)]
pub struct FrontendState {
    buffer: VecDeque<f32>,
    start: usize,
    samples: usize,
    next_frame: usize,
    previous: Option<f32>,
    finished: bool,
}

impl FrontendState {
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }
    pub fn sample_count(&self) -> usize {
        self.samples
    }
    pub fn frame_count(&self) -> usize {
        self.next_frame
    }
}

fn hz_to_mel(hz: f64) -> f64 {
    if hz < 1000.0 {
        hz / (200.0 / 3.0)
    } else {
        15.0 + (hz / 1000.0).ln() / (6.4f64.ln() / 27.0)
    }
}
fn mel_to_hz(mel: f64) -> f64 {
    if mel < 15.0 {
        mel * (200.0 / 3.0)
    } else {
        1000.0 * ((mel - 15.0) * (6.4f64.ln() / 27.0)).exp()
    }
}

impl LogMel {
    pub fn new(cfg: FeatureExtractorConfig) -> Result<Self> {
        if cfg.sampling_rate == 0
            || cfg.feature_size == 0
            || cfg.hop_length == 0
            || cfg.win_length < 2
            || cfg.n_fft < cfg.win_length
            || !cfg.n_fft.is_power_of_two()
            || !cfg.preemphasis.is_finite()
        {
            candle::bail!("invalid Nemotron log-Mel configuration")
        }
        let window = (0..cfg.win_length)
            .map(|i| {
                (0.5 - 0.5
                    * (2.0 * std::f64::consts::PI * i as f64 / (cfg.win_length - 1) as f64).cos())
                    as f32
            })
            .collect();
        let bins = cfg.n_fft / 2 + 1;
        let max_mel = hz_to_mel(cfg.sampling_rate as f64 / 2.0);
        let edges: Vec<_> = (0..cfg.feature_size + 2)
            .map(|i| mel_to_hz(max_mel * i as f64 / (cfg.feature_size + 1) as f64))
            .collect();
        let mut bank = vec![0f32; cfg.feature_size * bins];
        for c in 0..cfg.feature_size {
            for k in 0..bins {
                let hz = k as f64 * cfg.sampling_rate as f64 / cfg.n_fft as f64;
                let lower = (hz - edges[c]) / (edges[c + 1] - edges[c]);
                let upper = (edges[c + 2] - hz) / (edges[c + 2] - edges[c + 1]);
                // librosa stores the triangular weights in F32 before applying
                // its F64 Slaney normalization and rounding back to F32.
                let triangle = lower.min(upper).max(0.0) as f32;
                bank[c * bins + k] = (triangle as f64 * 2.0 / (edges[c + 2] - edges[c])) as f32;
            }
        }
        let fft = FftPlanner::new().plan_fft_forward(cfg.n_fft);
        Ok(Self {
            cfg,
            window,
            bank,
            fft,
        })
    }

    pub fn config(&self) -> &FeatureExtractorConfig {
        &self.cfg
    }

    fn frame(&self, state: &FrontendState, frame: usize) -> Vec<f32> {
        let n = self.cfg.n_fft;
        let left = (n - self.cfg.win_length) / 2;
        let center = frame * self.cfg.hop_length;
        let mut spectrum = vec![Complex32::new(0.0, 0.0); n];
        for i in 0..self.cfg.win_length {
            let index = center as i128 + (left + i) as i128 - (n / 2) as i128;
            if index >= state.start as i128 && index < state.samples as i128 {
                spectrum[left + i].re =
                    state.buffer[(index as usize) - state.start] * self.window[i];
            }
        }
        self.fft.process(&mut spectrum);
        let bins = n / 2 + 1;
        let powers: Vec<_> = spectrum[..bins]
            .iter()
            .map(|v| {
                let magnitude = (v.re * v.re + v.im * v.im).sqrt();
                magnitude * magnitude
            })
            .collect();
        (0..self.cfg.feature_size)
            .map(|c| {
                let mut value = 0f32;
                for k in 0..bins {
                    value += self.bank[c * bins + k] * powers[k];
                }
                (value + 2f32.powi(-24)).ln()
            })
            .collect()
    }

    fn emit(&self, state: &mut FrontendState, output: &mut Vec<f32>) {
        output.extend(self.frame(state, state.next_frame));
        state.next_frame += 1;
        let keep_from = (state.next_frame * self.cfg.hop_length).saturating_sub(self.cfg.n_fft / 2);
        while state.start < keep_from && !state.buffer.is_empty() {
            state.buffer.pop_front();
            state.start += 1;
        }
    }

    /// Return newly available valid frames, time-major and flattened. Preemphasis
    /// and STFT alignment are preserved across arbitrary PCM packet boundaries.
    pub fn push(&self, state: &mut FrontendState, pcm: &[f32], rate: usize) -> Result<Vec<f32>> {
        if state.finished || rate != self.cfg.sampling_rate || pcm.iter().any(|v| !v.is_finite()) {
            candle::bail!("invalid PCM, sample rate or finished Nemotron frontend")
        }
        let mut output = Vec::new();
        for &sample in pcm {
            let emphasized = match state.previous {
                None => sample,
                Some(prev) => sample - self.cfg.preemphasis as f32 * prev,
            };
            state.previous = Some(sample);
            state.buffer.push_back(emphasized);
            state.samples += 1;
            while state.next_frame * self.cfg.hop_length + self.cfg.n_fft / 2 <= state.samples {
                self.emit(state, &mut output);
            }
        }
        Ok(output)
    }

    pub fn finish(&self, state: &mut FrontendState) -> Result<Vec<f32>> {
        if state.finished {
            return Ok(Vec::new());
        }
        let valid = state.samples / self.cfg.hop_length;
        if valid == 0 {
            candle::bail!("utterance requires at least one audio hop")
        }
        let mut output = Vec::new();
        while state.next_frame < valid {
            self.emit(state, &mut output);
        }
        state.finished = true;
        state.buffer.clear();
        state.start = state.samples;
        Ok(output)
    }

    pub fn process_mono(
        &self,
        pcm: &[f32],
        rate: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Features> {
        let mut state = FrontendState::default();
        let mut values = self.push(&mut state, pcm, rate)?;
        values.extend(self.finish(&mut state)?);
        let valid_frames = state.next_frame;
        // The official centered STFT returns one additional masked frame.
        values.resize((valid_frames + 1) * self.cfg.feature_size, 0.0);
        let values = Tensor::from_vec(
            values,
            (1, valid_frames + 1, self.cfg.feature_size),
            &Device::Cpu,
        )?
        .to_dtype(dtype)?
        .to_device(device)?;
        Ok(Features {
            values,
            valid_frames,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packet_boundaries_do_not_change_mel_and_storage_is_bounded() -> Result<()> {
        let (_, processor) = super::super::config::official_config();
        let frontend = LogMel::new(processor.feature_extractor)?;
        let pcm: Vec<_> = (0..16000).map(|i| (i as f32 * 0.017).sin() * 0.2).collect();
        let full = frontend.process_mono(&pcm, 16000, DType::F32, &Device::Cpu)?;
        for size in [1, 79, 320, 4096] {
            let mut state = FrontendState::default();
            let mut output = Vec::new();
            for chunk in pcm.chunks(size) {
                output.extend(frontend.push(&mut state, chunk, 16000)?);
                assert!(state.buffered_samples() <= 512);
            }
            output.extend(frontend.finish(&mut state)?);
            assert_eq!(
                output,
                full.values
                    .narrow(1, 0, full.valid_frames)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            );
            assert!(frontend.finish(&mut state)?.is_empty());
            assert!(frontend.push(&mut state, &[0.], 16000).is_err());
        }
        Ok(())
    }
}
