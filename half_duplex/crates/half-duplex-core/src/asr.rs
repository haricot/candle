use crate::audio::PcmAudio;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    pub text: String,
    pub language: String,
    pub backend_id: String,
}
impl Transcript {
    pub fn new(text: impl Into<String>, language: impl Into<String>, backend_id: impl Into<String>) -> Self {
        Self { text: text.into(), language: language.into(), backend_id: backend_id.into() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsrError { NotImplemented(String), InvalidAudio(String), Inference(String) }
impl fmt::Display for AsrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotImplemented(s) => write!(f, "ASR not implemented: {s}"),
            Self::InvalidAudio(s) => write!(f, "invalid audio: {s}"),
            Self::Inference(s) => write!(f, "ASR inference: {s}"),
        }
    }
}
impl Error for AsrError {}

/// Implement with TranscribePsy/Candle at HDB2; or other ASR engines later.
pub trait AsrBackend: Send {
    fn id(&self) -> &str;
    fn transcribe(&mut self, audio: &PcmAudio) -> Result<Transcript, AsrError>;
}

#[derive(Debug, Default)]
pub struct UnavailableAsr;
impl AsrBackend for UnavailableAsr {
    fn id(&self) -> &str { "unavailable-hdb0" }
    fn transcribe(&mut self, _audio: &PcmAudio) -> Result<Transcript, AsrError> {
        Err(AsrError::NotImplemented("TranscribePsy port scheduled for HDB2".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unavailable_backend_is_explicit() {
        let mut b = UnavailableAsr;
        let pcm = PcmAudio::new(16_000, vec![0.0]).unwrap();
        assert!(matches!(b.transcribe(&pcm), Err(AsrError::NotImplemented(_))));
    }
}
