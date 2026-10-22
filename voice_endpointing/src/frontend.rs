//! The models use DIFFERENT frontends. Neither reuses the ASR's NeMo features.
use anyhow::{ensure, Result};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::{f64::consts::PI, path::Path, sync::Arc};

pub const SAMPLE_RATE: usize = 16_000;

pub fn read_wav(path: &Path) -> Result<Vec<i16>> {
    let mut wav = hound::WavReader::open(path)?;
    let s = wav.spec();
    ensure!(
        s.channels == 1
            && s.sample_rate == 16_000
            && s.bits_per_sample == 16
            && s.sample_format == hound::SampleFormat::Int,
        "Expected mono PCM16 WAV at 16000 Hz; convert explicitly before inference"
    );
    Ok(wav
        .samples::<i16>()
        .collect::<std::result::Result<_, _>>()?)
}

pub fn write_wav(path: &Path, pcm: &[i16]) -> Result<()> {
    let mut wav = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for &x in pcm {
        wav.write_sample(x)?;
    }
    wav.finalize()?;
    Ok(())
}

/// Kaldi's snip_edges=true, 25 ms / 10 ms, 80 HTK/Kaldi Mel bins,
/// Povey window, remove DC, preemphasis .97, unscaled PCM16 amplitudes.
/// The overlap is retained across arbitrary input chunk boundaries.
pub struct KaldiFbank {
    window: Vec<f32>,
    banks: Vec<Vec<(usize, f32)>>,
    fft: Arc<dyn Fft<f32>>,
    pending: Vec<f32>,
}

impl Default for KaldiFbank {
    fn default() -> Self {
        let window = (0..400)
            .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / 399.0).cos()).powf(0.85) as f32)
            .collect();
        let mel = |hz: f32| 1127.0f32 * (1.0 + hz / 700.0).ln();
        let lo = mel(20.0);
        let hi = mel(8000.0);
        let step = (hi - lo) / 81.0;
        let banks = (0..80)
            .map(|b| {
                let left = lo + b as f32 * step;
                let center = lo + (b + 1) as f32 * step;
                let right = lo + (b + 2) as f32 * step;
                (0..256)
                    .filter_map(|k| {
                        let m = mel(k as f32 * 16000.0 / 512.0);
                        if m > left && m < right {
                            Some((
                                k,
                                if m <= center {
                                    (m - left) / (center - left)
                                } else {
                                    (right - m) / (right - center)
                                },
                            ))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        Self {
            window,
            banks,
            fft: FftPlanner::new().plan_fft_forward(512),
            pending: Vec::new(),
        }
    }
}

impl KaldiFbank {
    pub fn reset(&mut self) {
        self.pending.clear();
    }

    pub fn feed(&mut self, pcm: &[i16]) -> Vec<f32> {
        self.pending.extend(pcm.iter().map(|&x| x as f32));
        let count = if self.pending.len() < 400 {
            0
        } else {
            1 + (self.pending.len() - 400) / 160
        };
        let mut features = Vec::with_capacity(count * 80);
        let mut fft = vec![Complex::new(0.0f32, 0.0); 512];
        let mut scratch = vec![Complex::new(0.0f32, 0.0); self.fft.get_inplace_scratch_len()];
        for t in 0..count {
            let frame = &self.pending[t * 160..t * 160 + 400];
            let mean = frame.iter().sum::<f32>() / 400.0;
            for i in 0..400 {
                let x = frame[i] - mean;
                let previous = if i == 0 { x } else { frame[i - 1] - mean };
                fft[i] = Complex::new((x - 0.97 * previous) * self.window[i], 0.0);
            }
            fft[400..].fill(Complex::new(0.0, 0.0));
            self.fft.process_with_scratch(&mut fft, &mut scratch);
            let power: Vec<f32> = fft[..257]
                .iter()
                .map(|c| c.re * c.re + c.im * c.im)
                .collect();
            for bank in &self.banks {
                let energy: f32 = bank.iter().map(|&(i, w)| power[i] * w).sum();
                features.push(energy.max(f32::EPSILON).ln());
            }
        }
        self.pending.drain(..count * 160);
        features
    }
}

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
        ensure!(
            !pcm.is_empty(),
            "Smart Turn requires a nonempty audio window"
        );
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
        ensure!(
            output.iter().all(|x| x.is_finite()),
            "Nonfinite Smart Turn features"
        );
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fbank_chunk_boundaries_preserve_every_frame() {
        let pcm: Vec<i16> = (0..4821).map(|i| ((i * 79) % 25000) as i16).collect();
        let expected = KaldiFbank::default().feed(&pcm);
        let mut fb = KaldiFbank::default();
        let actual: Vec<_> = pcm.chunks(137).flat_map(|x| fb.feed(x)).collect();
        assert_eq!(expected, actual);
        assert_eq!(actual.len(), 80 * (1 + (4821 - 400) / 160));
    }
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
