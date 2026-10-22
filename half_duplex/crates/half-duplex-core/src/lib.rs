//! Model-independent half-duplex state and PCM primitives.
//! HDB0 is a software-only scaffold; it deliberately does NOT transcribe speech.

pub mod asr;
pub mod audio;
pub mod events;
pub mod ring_buffer;
pub mod resample;
pub mod turn_state;

pub use asr::{AsrBackend, AsrError, Transcript, UnavailableAsr};
pub use audio::{AudioChunk, AudioError, PcmAudio};
pub use events::{Event, Phase};
pub use ring_buffer::PcmRingBuffer;
pub use resample::to_16khz;
pub use turn_state::{TurnController, TurnError};
