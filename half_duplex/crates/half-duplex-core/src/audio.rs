use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioError {
    InvalidSampleRate,
    Empty,
    NonFinite,
    SampleOverflow,
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            Self::InvalidSampleRate => "sample rate must be positive",
            Self::Empty => "audio buffer is empty",
            Self::NonFinite => "non-finite PCM sample",
            Self::SampleOverflow => "PCM sample index overflow",
        })
    }
}
impl Error for AudioError {}

#[derive(Debug, Clone, PartialEq)]
pub struct PcmAudio {
    sample_rate: u32,
    samples: Vec<f32>,
}

impl PcmAudio {
    pub fn new(sample_rate: u32, samples: Vec<f32>) -> Result<Self, AudioError> {
        if sample_rate == 0 { return Err(AudioError::InvalidSampleRate); }
        if samples.is_empty() { return Err(AudioError::Empty); }
        if samples.iter().any(|v| !v.is_finite()) { return Err(AudioError::NonFinite); }
        Ok(Self { sample_rate, samples })
    }
    pub fn sample_rate(&self) -> u32 { self.sample_rate }
    pub fn samples(&self) -> &[f32] { &self.samples }
    pub fn sample_count(&self) -> usize { self.samples.len() }
    pub fn duration_seconds(&self) -> f64 { self.samples.len() as f64 / self.sample_rate as f64 }
    pub fn peak(&self) -> f32 { self.samples.iter().map(|v| v.abs()).fold(0.0_f32, f32::max) }
    pub fn rms(&self) -> f64 {
        (self.samples.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()
            / self.samples.len() as f64).sqrt()
    }
}

/// Mono PCM chunk with a monotonic absolute sample position (NOT wall clock time).
#[derive(Debug, Clone, PartialEq)]
pub struct AudioChunk {
    pub start_sample: u64,
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}
impl AudioChunk {
    pub fn new(start_sample: u64, sample_rate: u32, samples: Vec<f32>) -> Result<Self, AudioError> {
        PcmAudio::new(sample_rate, samples.clone())?;
        let n = u64::try_from(samples.len()).map_err(|_| AudioError::SampleOverflow)?;
        start_sample.checked_add(n).ok_or(AudioError::SampleOverflow)?;
        Ok(Self { start_sample, sample_rate, samples })
    }
    pub fn end_sample(&self) -> u64 { self.start_sample + self.samples.len() as u64 }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pcm_stats_and_validation() {
        let a = PcmAudio::new(16_000, vec![0.5, -0.5]).unwrap();
        assert_eq!(a.sample_count(), 2);
        assert_eq!(a.peak(), 0.5);
        assert!((a.rms() - 0.5).abs() < 1e-7);
        assert_eq!(PcmAudio::new(0, vec![0.0]), Err(AudioError::InvalidSampleRate));
        assert_eq!(PcmAudio::new(16_000, vec![]), Err(AudioError::Empty));
        assert_eq!(PcmAudio::new(16_000, vec![f32::NAN]), Err(AudioError::NonFinite));
    }
    #[test]
    fn overflow_is_rejected() {
        assert_eq!(AudioChunk::new(u64::MAX, 16_000, vec![0.0]).unwrap_err(), AudioError::SampleOverflow);
    }
}
