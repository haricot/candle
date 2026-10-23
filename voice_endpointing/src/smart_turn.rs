//! Compatible Smart Turn package/PCM adapter around the Candle model.
use crate::{frontend::WhisperFeatures, weights};
use anyhow::{ensure, Result};
use candle::{Device, Tensor};
use candle_transformers::models::smart_turn;
use std::path::Path;

pub struct SmartTurn {
    model: smart_turn::Model,
    features: WhisperFeatures,
}

pub struct SmartOutput {
    pub probability: f32,
    pub encoder: Tensor,
    pub pooled: Tensor,
}

impl SmartTurn {
    pub fn load(root: &Path) -> Result<Self> {
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
        ensure!(
            cfg["architecture"] == "smart-turn-v3.2-whisper-tiny-400" && cfg["dtype"] == "float32",
            "Unsupported Smart Turn configuration"
        );
        let vb = weights::load(root, "smart-turn-v3.2-fp32", &Device::Cpu)?;
        Ok(Self {
            model: smart_turn::Model::new(&smart_turn::Config::default(), vb)?,
            features: WhisperFeatures::default(),
        })
    }

    pub fn features(&self, pcm: &[i16]) -> Result<Vec<f32>> {
        self.features.compute(pcm)
    }

    pub fn forward(&self, features: &Tensor) -> Result<SmartOutput> {
        ensure!(
            features.dims() == [1, 80, 800],
            "Smart Turn requires [1,80,800] features"
        );
        let output = self.model.forward_detailed(features)?;
        let probability = output.probability.flatten_all()?.to_vec1::<f32>()?[0];
        ensure!(
            probability.is_finite() && (0.0..=1.0).contains(&probability),
            "Invalid Smart Turn probability"
        );
        Ok(SmartOutput {
            probability,
            encoder: output.encoder,
            pooled: output.pooled,
        })
    }

    pub fn predict(&self, pcm: &[i16]) -> Result<f32> {
        let features = Tensor::from_vec(self.features(pcm)?, (1, 80, 800), &Device::Cpu)?;
        Ok(self.forward(&features)?.probability)
    }
}
