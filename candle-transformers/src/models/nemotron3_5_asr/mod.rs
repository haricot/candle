//! NVIDIA Nemotron 3.5 ASR Streaming: PCM to tokens, batch size one.
//! Native log-Mel, cache-aware FastConformer, language conditioning and greedy RNN-T.
//! No Python, ONNX, download, WAV, microphone or controller dependency.

pub mod config;
pub mod encoder;
pub mod frontend;
pub mod model;
pub mod prompt;
pub mod rnnt;

pub use config::{ChunkPlan, Config, EncoderConfig, FeatureExtractorConfig, ProcessorConfig};
pub use model::{Model, StreamState};
pub use prompt::LanguageConditioner;
pub use rnnt::{Decision, DecodeTrace};

pub const MODEL_ID: &str = "nvidia/nemotron-3.5-asr-streaming-0.6b";
pub const MODEL_REVISION: &str = "ea30d66debe3740a08b573244286791d423d6b3e";
