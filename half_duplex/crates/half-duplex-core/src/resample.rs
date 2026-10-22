//! Standalone mono PCM sample-rate normalization for HDB0 file-based tests.
//! Windowed-sinc low-pass interpolation avoids aliasing when downsampling.
//! This offline reference implementation is intentionally not a streaming VAD
//! resampler: HDB1 will define continuity, latency, and buffering semantics.
use crate::audio::{AudioError, PcmAudio};
use std::f64::consts::PI;

const TARGET_RATE: u32 = 16_000;
const RADIUS: i64 = 32;

/// Normalize a mono PCM buffer to 16 kHz without external dependencies.
///
/// Output sample indices are defined on the 16-kHz clock. Sample count is
/// rounded to the nearest output sample to preserve elapsed duration.
/// A Blackman-windowed sinc FIR suppresses frequencies above the new Nyquist
/// limit during downsampling. Near boundaries, taps are renormalized.
/// This is an offline, whole-buffer HDB0 tool, not production streaming DSP.
pub fn to_16khz(source: &PcmAudio) -> Result<PcmAudio, AudioError> {
    let src_rate = source.sample_rate();
    if !(8_000..=192_000).contains(&src_rate) {
        return Err(AudioError::InvalidSampleRate);
    }
    if src_rate == TARGET_RATE {
        return Ok(source.clone());
    }
    let target_len = (source.sample_count() as u128)
        .checked_mul(u128::from(TARGET_RATE))
        .and_then(|n| n.checked_add(u128::from(src_rate / 2)))
        .ok_or(AudioError::SampleOverflow)?
        / u128::from(src_rate);
    let target_len = usize::try_from(target_len).map_err(|_| AudioError::SampleOverflow)?;
    if target_len == 0 { return Err(AudioError::Empty); }

    // For downsampling, shrink the interpolation kernel bandwidth to avoid
    // aliasing. 0.95 provides a transition band below the target Nyquist.
    let cutoff = 0.95_f64 * (TARGET_RATE as f64 / src_rate as f64).min(1.0);
    let source_samples = source.samples();
    let source_len = i64::try_from(source_samples.len()).map_err(|_| AudioError::SampleOverflow)?;
    let mut output = Vec::with_capacity(target_len);
    for output_idx in 0..target_len {
        let input_pos = output_idx as f64 * src_rate as f64 / TARGET_RATE as f64;
        let input_center = input_pos.floor() as i64;
        let mut weighted = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for input_idx in (input_center - RADIUS + 1)..=(input_center + RADIUS) {
            if input_idx < 0 || input_idx >= source_len { continue; }
            let distance = input_pos - input_idx as f64;
            let relative = distance / RADIUS as f64;
            if relative.abs() >= 1.0 { continue; }
            let window = 0.42 + 0.50 * (PI * relative).cos()
                + 0.08 * (2.0 * PI * relative).cos();
            let phase = PI * cutoff * distance;
            let sinc = if phase.abs() < 1.0e-12 { 1.0 } else { phase.sin() / phase };
            let weight = cutoff * sinc * window;
            weighted += f64::from(source_samples[input_idx as usize]) * weight;
            weight_sum += weight;
        }
        if !weight_sum.is_finite() || weight_sum.abs() < 1.0e-12 {
            return Err(AudioError::NonFinite);
        }
        let sample = weighted / weight_sum;
        if !sample.is_finite() { return Err(AudioError::NonFinite); }
        output.push(sample as f32);
    }
    PcmAudio::new(TARGET_RATE, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_16k_keeps_samples_exact() {
        let input = PcmAudio::new(16_000, vec![0.2, -0.4, 0.7]).unwrap();
        assert_eq!(to_16khz(&input).unwrap(), input);
    }

    #[test]
    fn microphone_24k_preserves_duration_and_constant_signal() {
        let src = PcmAudio::new(24_000, vec![0.25; 2400]).unwrap();
        let dst = to_16khz(&src).unwrap();
        assert_eq!(dst.sample_rate(), 16_000);
        assert_eq!(dst.sample_count(), 1600);
        assert!((dst.duration_seconds() - src.duration_seconds()).abs() < 1.0 / 16000.0);
        assert!(dst.samples().iter().all(|s| (*s - 0.25).abs() < 1.0e-4));
    }

    #[test]
    fn preserve_micro_recording_length_exactly() {
        let input = PcmAudio::new(24_000, vec![0.0; 353_211]).unwrap();
        let output = to_16khz(&input).unwrap();
        assert_eq!(output.sample_count(), 235_474);
        assert!((output.duration_seconds() - input.duration_seconds()).abs() < 1.0 / 16_000.0);
    }

    fn sine(rate: u32, hz: f64, duration_ms: u32) -> PcmAudio {
        let n = (rate as usize * duration_ms as usize) / 1000;
        let samples: Vec<f32> = (0..n).map(|i|
            (2.0 * PI * hz * i as f64 / rate as f64).sin() as f32
        ).collect();
        PcmAudio::new(rate, samples).unwrap()
    }

    fn rms_core(samples: &[f32]) -> f64 {
        let core = &samples[200..samples.len()-200];
        (core.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>() / core.len() as f64).sqrt()
    }

    #[test]
    fn low_pass_rejects_10khz_alias_on_48k_to_16k() {
        let pass = to_16khz(&sine(48_000, 1_000.0, 150)).unwrap();
        let stop = to_16khz(&sine(48_000, 10_000.0, 150)).unwrap();
        assert!(rms_core(pass.samples()) > 0.60);
        assert!(rms_core(stop.samples()) < 0.08);
    }

    #[test]
    fn rejects_non_audio_sample_rates() {
        let src = PcmAudio::new(1, vec![0.0; 10]).unwrap();
        assert_eq!(to_16khz(&src), Err(AudioError::InvalidSampleRate));
    }
}
