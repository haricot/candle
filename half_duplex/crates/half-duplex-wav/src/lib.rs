//! Deliberately small WAV reader for HDB0: RIFF/WAVE PCM16 or IEEE float32,
//! mono or stereo. A streaming decoder/resampler belongs to HDB1.
use half_duplex_core::PcmAudio;
use std::error::Error;
use std::fmt;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WavError {
    Truncated,
    NotRiffWave,
    UnsupportedFormat,
    InvalidHeader,
    NoData,
    InvalidSamples,
    TooLarge,
    Io(String),
}
impl fmt::Display for WavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{self:?}") }
}
impl Error for WavError {}

#[derive(Debug, Clone, Copy)]
struct Format { encoding: u16, channels: u16, sample_rate: u32, bits: u16, block_align: u16 }

fn read_u16(data: &[u8], at: usize) -> Result<u16, WavError> {
    let bytes = data.get(at..at+2).ok_or(WavError::Truncated)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}
fn read_u32(data: &[u8], at: usize) -> Result<u32, WavError> {
    let bytes = data.get(at..at+4).ok_or(WavError::Truncated)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Read up to 64 MiB of a WAV file. Reject unsupported codec settings clearly.
pub fn read_wav(path: impl AsRef<Path>) -> Result<PcmAudio, WavError> {
    let path = path.as_ref();
    let meta = std::fs::metadata(path).map_err(|e| WavError::Io(e.to_string()))?;
    if meta.len() > 64 * 1024 * 1024 { return Err(WavError::TooLarge); }
    let data = std::fs::read(path).map_err(|e| WavError::Io(e.to_string()))?;
    decode_wav(&data)
}

pub fn decode_wav(data: &[u8]) -> Result<PcmAudio, WavError> {
    if data.len() < 12 { return Err(WavError::Truncated); }
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" { return Err(WavError::NotRiffWave); }
    let riff_size = read_u32(data, 4)? as usize;
    let declared_end = riff_size.checked_add(8).ok_or(WavError::InvalidHeader)?;
    if declared_end > data.len() || declared_end < 12 { return Err(WavError::Truncated); }
    let mut position = 12usize;
    let mut fmt = None;
    let mut samples_bytes = None;
    while position + 8 <= declared_end {
        let id = &data[position..position+4];
        let size = read_u32(data, position + 4)? as usize;
        let start = position.checked_add(8).ok_or(WavError::InvalidHeader)?;
        let end = start.checked_add(size).ok_or(WavError::InvalidHeader)?;
        if end > declared_end { return Err(WavError::Truncated); }
        if id == b"fmt " {
            if size < 16 { return Err(WavError::InvalidHeader); }
            let spec = Format {
                encoding: read_u16(data, start)?,
                channels: read_u16(data, start + 2)?,
                sample_rate: read_u32(data, start + 4)?,
                block_align: read_u16(data, start + 12)?,
                bits: read_u16(data, start + 14)?,
            };
            fmt = Some(spec);
        } else if id == b"data" && samples_bytes.is_none() {
            samples_bytes = Some(&data[start..end]);
        }
        position = end.checked_add(size & 1).ok_or(WavError::InvalidHeader)?;
    }
    if position != declared_end { return Err(WavError::InvalidHeader); }
    let spec = fmt.ok_or(WavError::InvalidHeader)?;
    let bytes = samples_bytes.ok_or(WavError::NoData)?;
    if spec.channels == 0 || spec.channels > 2 || spec.sample_rate == 0 { return Err(WavError::UnsupportedFormat); }
    let stride = match (spec.encoding, spec.bits) {
        (1, 16) => 2usize,
        (3, 32) => 4usize,
        _ => return Err(WavError::UnsupportedFormat),
    };
    let frame_bytes = stride * spec.channels as usize;
    if spec.block_align as usize != frame_bytes || bytes.len() % frame_bytes != 0 {
        return Err(WavError::InvalidHeader);
    }
    if bytes.is_empty() { return Err(WavError::NoData); }
    let mut mono = Vec::with_capacity(bytes.len() / frame_bytes);
    for frame in bytes.chunks_exact(frame_bytes) {
        let mut value = 0_f32;
        for channel in 0..spec.channels as usize {
            let offset = channel * stride;
            let sample = if spec.encoding == 1 {
                i16::from_le_bytes([frame[offset], frame[offset+1]]) as f32 / 32768.0
            } else {
                f32::from_le_bytes(frame[offset..offset+4].try_into().map_err(|_| WavError::InvalidSamples)?)
            };
            if !sample.is_finite() { return Err(WavError::InvalidSamples); }
            value += sample / spec.channels as f32;
        }
        mono.push(value);
    }
    PcmAudio::new(spec.sample_rate, mono).map_err(|_| WavError::InvalidSamples)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn make_wav(enc: u16, bits: u16, channels: u16, samples: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        // RIFF chunks are word-aligned: an odd-sized data payload must have
        // a trailing pad byte, included in RIFF size but not in data size.
        let pad = samples.len() & 1;
        b.extend_from_slice(&(36u32 + samples.len() as u32 + pad as u32).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&enc.to_le_bytes());
        b.extend_from_slice(&channels.to_le_bytes());
        b.extend_from_slice(&16_000u32.to_le_bytes());
        let block = channels * (bits / 8);
        b.extend_from_slice(&(16_000u32 * block as u32).to_le_bytes());
        b.extend_from_slice(&block.to_le_bytes());
        b.extend_from_slice(&bits.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(samples.len() as u32).to_le_bytes());
        b.extend_from_slice(samples);
        if pad != 0 { b.push(0); }
        b
    }
    #[test]
    fn pcm16_mono_and_stereo() {
        let wav = make_wav(1, 16, 1, &[0, 0, 0, 64]);
        let audio = decode_wav(&wav).unwrap();
        assert_eq!(audio.sample_rate(), 16_000);
        assert_eq!(audio.sample_count(), 2);
        assert!((audio.samples()[1] - 0.5).abs() < 1e-6);
        let stereo = make_wav(1, 16, 2, &[0, 64, 0, 192]);
        assert_eq!(decode_wav(&stereo).unwrap().samples(), &[0.0]);
    }
    #[test]
    fn float32_and_rejection() {
        let wav = make_wav(3, 32, 1, &0.25_f32.to_le_bytes());
        assert!((decode_wav(&wav).unwrap().samples()[0] - 0.25).abs() < 1e-6);
        // Valid odd-length RIFF chunk: the 24-bit encoding is unsupported,
        // but the RIFF structure itself must be accepted as well-formed.
        let pcm24 = make_wav(1, 24, 1, &[0, 0, 0]);
        assert_eq!(pcm24.len(), 48); // 44 header + 3 data + 1 pad
        assert_eq!(decode_wav(&pcm24), Err(WavError::UnsupportedFormat));
        // Without the mandatory pad byte, the RIFF container is truncated.
        assert_eq!(decode_wav(&pcm24[..pcm24.len() - 1]), Err(WavError::Truncated));
        let mut bad = wav;
        bad[44..48].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(decode_wav(&bad), Err(WavError::InvalidSamples));
    }
    #[test]
    fn corrupt_header_is_rejected() {
        assert_eq!(decode_wav(b"not a wav"), Err(WavError::Truncated));
        assert_eq!(decode_wav(&[0; 44]), Err(WavError::NotRiffWave));
    }
}
