//! The models use DIFFERENT frontends. Neither reuses the ASR's NeMo features.
use anyhow::{ensure, Result};
pub use candle_transformers::models::fireredvad::audio::KaldiFbank;
use std::path::Path;

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

/// Compatibility adapter retaining the original SDK error type.
#[derive(Default)]
pub struct WhisperFeatures {
    inner: candle_transformers::models::smart_turn::audio::WhisperFeatures,
}
impl WhisperFeatures {
    pub fn compute(&self, pcm: &[i16]) -> Result<Vec<f32>> {
        Ok(self.inner.compute(pcm)?)
    }
}
