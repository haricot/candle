//! Model-specific 16 kHz PCM16 feature extraction, without file or device I/O.
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::{f64::consts::PI, sync::Arc};

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
}
