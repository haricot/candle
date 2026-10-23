//! NVIDIA Nemotron 3.5 ASR Streaming model port, initial components.
//!
//! This module currently provides the official HF configuration contracts and
//! language-prompt projection. The audio frontend, cache-aware FastConformer
//! and RNN-T decoder are not implemented here yet. It cannot transcribe audio.
//! No Python, ONNX, download, WAV, microphone or controller dependency.

pub mod config;
pub mod prompt;

pub use config::{ChunkPlan, Config, EncoderConfig, FeatureExtractorConfig, ProcessorConfig};
pub use prompt::LanguageConditioner;

pub const MODEL_ID: &str = "nvidia/nemotron-3.5-asr-streaming-0.6b";
pub const MODEL_REVISION: &str = "ea30d66debe3740a08b573244286791d423d6b3e";
