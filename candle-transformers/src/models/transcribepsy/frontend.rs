//! QVAC/NeMo log-Mel frontend for a single mono utterance.
//!
//! The official checkpoint provides the analysis window and mel filterbank:
//! no regenerated mel bins, rounding aliases, or per-machine probes. This
//! implementation computes FFT on the host and returns a Candle tensor.
//! TP3-R1 checks CPU F32 parity on one pinned public French utterance;
//! other devices and preprocessing configurations need separate qualification.

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde_json::Value;
use std::f32::consts::PI;

fn default_normalize() -> String {
    "per_feature".into()
}
fn default_window() -> String {
    "hann".into()
}
fn default_mag_power() -> f64 {
    2.0
}
fn default_true() -> bool {
    true
}
fn default_pad_to() -> usize {
    16
}
fn default_preemph() -> Option<f64> {
    Some(0.97)
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub sample_rate: usize,
    pub window_size: f64,
    pub window_stride: f64,
    pub n_fft: usize,
    pub features: usize,
    #[serde(default = "default_normalize")]
    pub normalize: String,
    #[serde(default = "default_window")]
    pub window: String,
    #[serde(default)]
    pub dither: f64,
    #[serde(default = "default_preemph")]
    pub preemph: Option<f64>,
    #[serde(default = "default_mag_power")]
    pub mag_power: f64,
    #[serde(default = "default_true")]
    pub log: bool,
    #[serde(default)]
    pub log_zero_guard_type: Option<String>,
    #[serde(default)]
    pub log_zero_guard_value: Option<Value>,
    #[serde(default = "default_pad_to")]
    pub pad_to: usize,
    #[serde(default)]
    pub pad_value: f64,
    #[serde(default)]
    pub exact_pad: bool,
    #[serde(default)]
    pub frame_splicing: Option<usize>,
}

impl Config {
    pub fn win_length(&self) -> usize {
        (self.window_size * self.sample_rate as f64).round() as usize
    }
    pub fn hop_length(&self) -> usize {
        (self.window_stride * self.sample_rate as f64).round() as usize
    }
    pub fn validate(&self) -> Result<()> {
        let n = self.n_fft;
        let w = self.win_length();
        let hop = self.hop_length();
        if self.sample_rate == 0
            || n < 2
            || !n.is_power_of_two()
            || w < 2
            || w > n
            || hop == 0
            || hop > n
            || self.features == 0
            || !self.window_size.is_finite()
            || !self.window_stride.is_finite()
            || self.window_size <= 0.0
            || self.window_stride <= 0.0
            || !self.dither.is_finite()
            || self.dither < 0.0
            || !self.mag_power.is_finite()
            || self.mag_power <= 0.0
            || !self.pad_value.is_finite()
            || self.preemph.is_some_and(|v| !v.is_finite())
            || self.frame_splicing.unwrap_or(1) != 1
            || !matches!(
                self.normalize.as_str(),
                "NA" | "none" | "" | "per_feature" | "all_features"
            )
        {
            candle::bail!("unsupported or nondeterministic TranscribePsy preprocessing")
        }
        if self.exact_pad && (n - hop) % 2 != 0 {
            candle::bail!("exact_pad requires even n_fft-hop")
        }
        if !matches!(
            self.log_zero_guard_type.as_deref().unwrap_or("add"),
            "add" | "clamp"
        ) {
            candle::bail!("invalid TranscribePsy log guard type")
        }
        self.log_guard()?;
        Ok(())
    }
    fn log_guard(&self) -> Result<f32> {
        let v = match &self.log_zero_guard_value {
            None | Some(Value::Null) => 2f32.powi(-24),
            Some(Value::String(s)) if s == "eps" => f32::EPSILON,
            Some(Value::String(s)) if s == "tiny" => f32::MIN_POSITIVE,
            Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN) as f32,
            _ => candle::bail!("unsupported TranscribePsy log guard"),
        };
        if !v.is_finite() || v <= 0.0 {
            candle::bail!("invalid log guard")
        }
        Ok(v)
    }
    fn padding(&self) -> usize {
        if self.exact_pad {
            (self.n_fft - self.hop_length()) / 2
        } else {
            self.n_fft / 2
        }
    }
    pub fn valid_frames(&self, count: usize) -> Result<usize> {
        let pad = self.padding();
        let total = count
            .checked_add(2 * pad)
            .ok_or_else(|| candle::Error::Msg("audio frame count overflow".into()))?;
        if total < self.n_fft {
            candle::bail!("utterance too short for NeMo frontend")
        }
        // NeMo's length mask excludes the final raw STFT frame.
        let valid = (total - self.n_fft) / self.hop_length();
        if valid == 0 {
            candle::bail!("NeMo log-Mel requires nonzero valid frames")
        }
        Ok(valid)
    }
}

#[derive(Debug, Clone)]
pub struct Features {
    pub values: Tensor,
    pub valid_frames: usize,
}

#[derive(Debug, Clone)]
pub struct LogMel {
    cfg: Config,
    window: Vec<f32>,
    bank: Vec<f32>,
    device: Device,
}

impl LogMel {
    /// `vb` must point at the checkpoint's `preprocessor` prefix.
    pub fn load(cfg: Config, vb: VarBuilder<'_>) -> Result<Self> {
        cfg.validate()?;
        let n = cfg.n_fft / 2 + 1;
        let featurizer = vb.pp("featurizer");
        let window = featurizer
            .get((cfg.win_length(),), "window")?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        let bank = featurizer
            .get((1, cfg.features, n), "fb")?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if window.iter().chain(bank.iter()).any(|v| !v.is_finite()) {
            candle::bail!("nonfinite native QVAC frontend coefficients")
        }
        Ok(Self {
            cfg,
            window,
            bank,
            device: vb.device().clone(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Evaluation only: NeMo's training dither is intentionally inactive.
    /// No implicit resampling, VAD, microphone acquisition or file I/O.
    pub fn process_mono(&self, pcm: &[f32], sample_rate: usize) -> Result<Features> {
        let cfg = &self.cfg;
        if sample_rate != cfg.sample_rate || pcm.is_empty() || pcm.iter().any(|v| !v.is_finite()) {
            candle::bail!("invalid mono PCM or sample-rate mismatch")
        }
        let valid = cfg.valid_frames(pcm.len())?;
        let n = cfg.n_fft;
        let hop = cfg.hop_length();
        let win_len = cfg.win_length();
        let mut wave = pcm.to_vec();
        if let Some(alpha) = cfg.preemph {
            for i in (1..wave.len()).rev() {
                wave[i] -= alpha as f32 * wave[i - 1];
            }
        }
        let pad = cfg.padding();
        let mut padded = vec![0f32; pad];
        padded.extend_from_slice(&wave);
        padded.resize(padded.len() + pad, 0.0);
        let raw_frames = (padded.len() - n) / hop + 1;
        let bins = n / 2 + 1;
        let mut matrix = vec![0f32; cfg.features * raw_frames];
        let window_left = (n - win_len) / 2;
        let guard = cfg.log_guard()?;
        let mut frame = vec![0f32; n];
        for time in 0..raw_frames {
            frame.fill(0.0);
            for i in 0..win_len {
                frame[window_left + i] = padded[time * hop + window_left + i] * self.window[i];
            }
            let spectrum = real_fft(&frame)?;
            let mut powers = vec![0f32; bins];
            for k in 0..bins {
                let (real, imag) = spectrum[k];
                let magnitude = (real * real + imag * imag).sqrt();
                powers[k] = magnitude.powf(cfg.mag_power as f32);
            }
            for channel in 0..cfg.features {
                let mut value = 0f32;
                for bin in 0..bins {
                    value += self.bank[channel * bins + bin] * powers[bin];
                }
                matrix[channel * raw_frames + time] = if cfg.log {
                    match cfg.log_zero_guard_type.as_deref().unwrap_or("add") {
                        "add" => (value + guard).ln(),
                        "clamp" => value.max(guard).ln(),
                        _ => unreachable!("guard checked in config"),
                    }
                } else {
                    value
                };
            }
        }
        normalise(&mut matrix, cfg.features, raw_frames, valid, &cfg.normalize)?;
        let target = if cfg.pad_to == 0 {
            raw_frames
        } else {
            raw_frames.max(valid).div_ceil(cfg.pad_to) * cfg.pad_to
        };
        let mut output = vec![cfg.pad_value as f32; target * cfg.features];
        for t in 0..valid {
            for channel in 0..cfg.features {
                output[t * cfg.features + channel] = matrix[channel * raw_frames + t];
            }
        }
        if output.iter().any(|v| !v.is_finite()) {
            candle::bail!("nonfinite TranscribePsy log-Mel output")
        }
        let values = Tensor::from_vec(output, (1, target, cfg.features), &Device::Cpu)?
            .to_device(&self.device)?;
        Ok(Features {
            values,
            valid_frames: valid,
        })
    }
}

fn normalise(
    data: &mut [f32],
    channels: usize,
    frames: usize,
    valid: usize,
    mode: &str,
) -> Result<()> {
    match mode {
        "NA" | "none" | "" => return Ok(()),
        "per_feature" | "all_features" if valid < 2 => {
            candle::bail!("NeMo normalization requires >=2 frames")
        }
        _ => {}
    }
    if mode == "per_feature" {
        for c in 0..channels {
            let mean = (0..valid).map(|t| data[c * frames + t] as f64).sum::<f64>() / valid as f64;
            let std = ((0..valid)
                .map(|t| {
                    let delta = data[c * frames + t] as f64 - mean;
                    delta * delta
                })
                .sum::<f64>()
                / (valid - 1) as f64)
                .sqrt()
                + 1e-5;
            for t in 0..valid {
                data[c * frames + t] = ((data[c * frames + t] as f64 - mean) / std) as f32;
            }
        }
    } else if mode == "all_features" {
        let count = channels * valid;
        if count < 2 {
            candle::bail!("insufficient mel values")
        }
        let mut sum = 0f64;
        for c in 0..channels {
            for t in 0..valid {
                sum += data[c * frames + t] as f64;
            }
        }
        let mean = sum / count as f64;
        let mut var = 0f64;
        for c in 0..channels {
            for t in 0..valid {
                let delta = data[c * frames + t] as f64 - mean;
                var += delta * delta;
            }
        }
        let std = (var / (count - 1) as f64).sqrt() + 1e-5;
        for c in 0..channels {
            for t in 0..valid {
                data[c * frames + t] = ((data[c * frames + t] as f64 - mean) / std) as f32;
            }
        }
    }
    Ok(())
}

/// In-place radix-2 DIT FFT, F32 host baseline. A future DSP-optimized
/// frontend can replace this implementation without changing the API.
fn real_fft(input: &[f32]) -> Result<Vec<(f32, f32)>> {
    let n = input.len();
    if n == 0 || !n.is_power_of_two() {
        candle::bail!("FFT requires a power-of-two length")
    }
    let mut re = input.to_vec();
    let mut im = vec![0f32; n];
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut span = 2usize;
    while span <= n {
        let half = span / 2;
        for first in (0..n).step_by(span) {
            for k in 0..half {
                let angle = -2.0 * PI * k as f32 / span as f32;
                let wr = angle.cos();
                let wi = angle.sin();
                let tr = wr * re[first + k + half] - wi * im[first + k + half];
                let ti = wr * im[first + k + half] + wi * re[first + k + half];
                let ur = re[first + k];
                let ui = im[first + k];
                re[first + k] = ur + tr;
                im[first + k] = ui + ti;
                re[first + k + half] = ur - tr;
                im[first + k + half] = ui - ti;
            }
        }
        span *= 2;
    }
    Ok((0..n / 2 + 1).map(|k| (re[k], im[k])).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn config() -> Config {
        Config {
            sample_rate: 16,
            window_size: 0.25,
            window_stride: 0.25,
            n_fft: 8,
            features: 2,
            normalize: "NA".into(),
            window: "hann".into(),
            dither: 0.0,
            preemph: Some(0.5),
            mag_power: 2.0,
            log: true,
            log_zero_guard_type: Some("add".into()),
            log_zero_guard_value: Some(Value::from(1e-6)),
            pad_to: 4,
            pad_value: 0.0,
            exact_pad: false,
            frame_splicing: None,
        }
    }
    #[test]
    fn native_qvac_filterbank_and_valid_frame_mask() -> Result<()> {
        let cfg = config();
        let mut map = HashMap::new();
        map.insert(
            "preprocessor.featurizer.window".to_owned(),
            Tensor::ones((4,), DType::F32, &Device::Cpu)?,
        );
        map.insert(
            "preprocessor.featurizer.fb".to_owned(),
            Tensor::ones((1, 2, 5), DType::F32, &Device::Cpu)?,
        );
        let vb = VarBuilder::from_tensors(map.clone(), DType::F32, &Device::Cpu);
        let mel = LogMel::load(cfg, vb.pp("preprocessor"))?;
        let samples = vec![0.25f32; 16];
        let out = mel.process_mono(&samples, 16)?;
        assert_eq!(out.valid_frames, 4);
        assert_eq!(out.values.dims(), &[1, 8, 2]);
        let data = out.values.flatten_all()?.to_vec1::<f32>()?;
        assert!(data.iter().all(|v| v.is_finite()));
        assert!(data[4 * 2..].iter().all(|v| *v == 0.0));
        assert!(mel.process_mono(&samples, 48).is_err());
        map.remove("preprocessor.featurizer.fb");
        let invalid = LogMel::load(
            config(),
            VarBuilder::from_tensors(map, DType::F32, &Device::Cpu).pp("preprocessor"),
        );
        assert!(invalid.is_err(), "native QVAC filterbank is mandatory");
        Ok(())
    }

    #[test]
    fn fft_impulse_has_flat_one_sided_magnitude() -> Result<()> {
        let mut impulse = vec![0f32; 8];
        impulse[0] = 1.0;
        assert!(real_fft(&impulse)?
            .iter()
            .all(|&(a, b)| (a - 1.0).abs() < 1e-6 && b.abs() < 1e-6));
        Ok(())
    }

    #[test]
    fn eval_config_uses_nemo_defaults_and_rejects_invalid_dither() -> Result<()> {
        let defaults: Config = serde_json::from_value(serde_json::json!({
            "sample_rate":16000, "window_size":0.025, "window_stride":0.01,
            "n_fft":512, "features":128, "normalize":"NA"
        }))
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
        assert_eq!(defaults.preemph, Some(0.97));
        assert_eq!(defaults.pad_to, 16);
        let mut cfg = config();
        cfg.dither = -0.01;
        assert!(cfg.validate().is_err());
        cfg.dither = 1e-5; // ignored during eval, as in NeMo
        assert!(cfg.validate().is_ok());
        let _ = cfg.valid_frames(16)?;
        assert!(cfg.valid_frames(0).is_err());
        Ok(())
    }
}
