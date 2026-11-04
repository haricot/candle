//! Model-specific 16 kHz PCM16 feature extraction, without file or device I/O.
use candle::Result;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::{f64::consts::PI, sync::Arc};

fn slaney_hz(mel: f64) -> f64 {
    if mel < 15.0 {
        mel * 200.0 / 3.0
    } else {
        1000.0 * ((mel - 15.0) * 6.4f64.ln() / 27.0).exp()
    }
}

/// Reproduces WhisperFeatureExtractor(chunk_length=8, do_normalize=true),
/// including LEFT padding before normalization, reflect-centering, periodic Hann,
/// Slaney-normalized Mel, log10 floor and dynamic-range clipping.
pub struct WhisperFeatures {
    fft: Arc<dyn Fft<f64>>,
    window: Vec<f64>,
    banks: Vec<Vec<(usize, f64)>>,
}

impl Default for WhisperFeatures {
    fn default() -> Self {
        let upper = 15.0 + (8000.0f64 / 1000.0).ln() * 27.0 / 6.4f64.ln();
        let edges: Vec<_> = (0..82)
            .map(|i| slaney_hz(i as f64 * upper / 81.0))
            .collect();
        let banks = (0..80)
            .map(|b| {
                let (l, c, r) = (edges[b], edges[b + 1], edges[b + 2]);
                (0..201)
                    .filter_map(|k| {
                        let hz = k as f64 * 16000.0 / 400.0;
                        let w =
                            ((hz - l) / (c - l)).min((r - hz) / (r - c)).max(0.0) * 2.0 / (r - l);
                        (w > 0.0).then_some((k, w))
                    })
                    .collect()
            })
            .collect();
        Self {
            fft: FftPlanner::new().plan_fft_forward(400),
            window: (0..400)
                .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / 400.0).cos())
                .collect(),
            banks,
        }
    }
}

impl WhisperFeatures {
    pub fn compute(&self, pcm: &[i16]) -> Result<Vec<f32>> {
        if pcm.is_empty() {
            candle::bail!("Smart Turn requires a nonempty audio window")
        }
        let mut wave = vec![0f32; 128000];
        let used = pcm.len().min(128000);
        for (dst, &x) in wave[128000 - used..]
            .iter_mut()
            .zip(&pcm[pcm.len() - used..])
        {
            *dst = x as f32 / 32768.0;
        }
        // Accumulate accurately, then round as NumPy's float32 normalization does.
        let mean = (wave.iter().map(|&x| x as f64).sum::<f64>() / 128000.0) as f32;
        let variance = (wave
            .iter()
            .map(|&x| {
                let d = x - mean;
                (d * d) as f64
            })
            .sum::<f64>()
            / 128000.0) as f32;
        let scale = (variance + 1e-7f32).sqrt();
        for x in &mut wave {
            *x = (*x - mean) / scale;
        }
        let mut output = vec![0f32; 80 * 800];
        let mut fft = vec![Complex::new(0.0, 0.0); 400];
        let mut scratch = vec![Complex::new(0.0, 0.0); self.fft.get_inplace_scratch_len()];
        for t in 0..800 {
            for (i, dst) in fft.iter_mut().enumerate() {
                let p = t as isize * 160 + i as isize - 200;
                let p = if p < 0 {
                    -p
                } else if p >= 128000 {
                    255998 - p
                } else {
                    p
                } as usize;
                *dst = Complex::new(wave[p] as f64 * self.window[i], 0.0);
            }
            self.fft.process_with_scratch(&mut fft, &mut scratch);
            // transformers.audio_utils.spectrogram stores FFT as complex64
            // before calculating float64 powers. Preserve that rounding point.
            let power: Vec<_> = fft[..201]
                .iter()
                .map(|c| {
                    let re = c.re as f32 as f64;
                    let im = c.im as f32 as f64;
                    re * re + im * im
                })
                .collect();
            for (b, bank) in self.banks.iter().enumerate() {
                let energy: f64 = bank.iter().map(|&(i, w)| power[i] * w).sum();
                output[b * 800 + t] = energy.max(1e-10).log10() as f32;
            }
        }
        let max = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for x in &mut output {
            *x = (x.max(max - 8.0) + 4.0) / 4.0;
        }
        if !output.iter().all(|x| x.is_finite()) {
            candle::bail!("Nonfinite Smart Turn features")
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whisper_uses_last_eight_seconds() {
        let features = WhisperFeatures::default();
        let pcm: Vec<i16> = (0..144000).map(|i| (i % 997) as i16).collect();
        assert_eq!(
            features.compute(&pcm).unwrap(),
            features.compute(&pcm[16000..]).unwrap()
        );
        assert!(features.compute(&[]).is_err());
    }
}
