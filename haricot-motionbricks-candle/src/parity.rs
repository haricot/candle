use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::g1::{G1Assets, G1MotionFrame, G1_JOINTS};
use crate::math3d::{Mat3, Vec3};
use crate::motionbricks::{BridgeTimingMode, MotionBricksFeatureBridge, MotionBricksFeatures};

pub const MOTIONBRICKS_PINNED_REV: &str = "087f9ac01d46f6d8e4d0b73c01ae64799f292a38";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct G1ParityFrame {
    pub positions: Vec<[f32; 3]>,
    pub global_rotations: Vec<[[f32; 3]; 3]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct G1ParitySequence {
    pub schema: String,
    pub source: String,
    pub fps: f32,
    pub frames: Vec<G1ParityFrame>,
}

impl G1ParitySequence {
    pub fn new(source: impl Into<String>, fps: f32) -> Self {
        Self {
            schema: "haricot.motionbricks.g1-parity-input.v1".to_string(),
            source: source.into(),
            fps,
            frames: Vec::new(),
        }
    }

    pub fn push_g1(&mut self, frame: &G1MotionFrame) -> Result<()> {
        frame.validate()?;
        self.frames.push(G1ParityFrame {
            positions: frame.positions.iter().map(|p| p.to_array()).collect(),
            global_rotations: frame.global_rotations.iter().map(|r| r.m).collect(),
        });
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != "haricot.motionbricks.g1-parity-input.v1" {
            bail!("unsupported parity input schema {}", self.schema)
        }
        if !(self.fps > 0.0 && self.fps.is_finite()) {
            bail!("invalid parity fps {}", self.fps)
        }
        if self.frames.len() < 2 {
            bail!("official parity requires at least 2 frames")
        }
        for (t, frame) in self.frames.iter().enumerate() {
            if frame.positions.len() != G1_JOINTS || frame.global_rotations.len() != G1_JOINTS {
                bail!("frame {t}: expected {G1_JOINTS} G1 joints")
            }
            if frame.positions.iter().flatten().any(|v| !v.is_finite())
                || frame.global_rotations.iter().flatten().flatten().any(|v| !v.is_finite())
            {
                bail!("frame {t}: non-finite G1 parity data")
            }
        }
        Ok(())
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(self)? + "\n";
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading parity input {}", path.display()))?;
        let out: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing parity input {}", path.display()))?;
        out.validate()?;
        Ok(out)
    }

    pub fn to_g1_frames(&self) -> Result<Vec<G1MotionFrame>> {
        self.validate()?;
        let dt = 1.0 / self.fps;
        self.frames
            .iter()
            .map(|f| {
                let out = G1MotionFrame {
                    dt,
                    positions: f.positions.iter().copied().map(Vec3::from).collect(),
                    global_rotations: f
                        .global_rotations
                        .iter()
                        .copied()
                        .map(|m| Mat3 { m })
                        .collect(),
                    contacts: [0.0; 4],
                };
                out.validate()?;
                Ok(out)
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OfficialMotionBricksReference {
    pub schema: String,
    pub source: String,
    pub source_rev: String,
    pub fps: f32,
    pub frames: usize,
    pub dual: Vec<Vec<f32>>,
    pub global: Vec<Vec<f32>>,
    pub local: Vec<Vec<f32>>,
    pub normalized_dual: Vec<Vec<f32>>,
    pub normalized_global: Vec<Vec<f32>>,
    pub normalized_local: Vec<Vec<f32>>,
}

impl OfficialMotionBricksReference {
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading official reference {}", path.display()))?;
        let out: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing official reference {}", path.display()))?;
        out.validate()?;
        Ok(out)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != "haricot.motionbricks.official-feature-reference.v1" {
            bail!("unsupported official reference schema {}", self.schema)
        }
        if self.source_rev != MOTIONBRICKS_PINNED_REV {
            bail!(
                "official reference rev mismatch: got {}, expected {}",
                self.source_rev,
                MOTIONBRICKS_PINNED_REV
            )
        }
        if !(self.fps > 0.0 && self.fps.is_finite()) {
            bail!("invalid official reference fps")
        }
        let groups = [
            ("dual", &self.dual, 418usize),
            ("global", &self.global, 414usize),
            ("local", &self.local, 413usize),
            ("normalized_dual", &self.normalized_dual, 418usize),
            ("normalized_global", &self.normalized_global, 414usize),
            ("normalized_local", &self.normalized_local, 413usize),
        ];
        for (name, data, dim) in groups {
            if data.len() != self.frames {
                bail!("{name}: expected {} frames, got {}", self.frames, data.len())
            }
            for (t, row) in data.iter().enumerate() {
                if row.len() != dim {
                    bail!("{name}[{t}]: expected dim {dim}, got {}", row.len())
                }
                if row.iter().any(|v| !v.is_finite()) {
                    bail!("{name}[{t}]: non-finite values")
                }
            }
        }
        Ok(())
    }
}

pub fn encode_rust_sequence(
    assets: G1Assets,
    sequence: &G1ParitySequence,
) -> Result<Vec<MotionBricksFeatures>> {
    sequence.validate()?;
    let mut bridge = MotionBricksFeatureBridge::new(assets, BridgeTimingMode::ExactDelayed)?;
    let frames = sequence.to_g1_frames()?;
    let mut out = Vec::with_capacity(frames.len());
    for frame in &frames {
        if let Some(features) = bridge.push(frame)? {
            out.push(features);
        }
    }
    if let Some(features) = bridge.flush()? {
        out.push(features);
    }
    if out.len() != frames.len() {
        bail!("Rust bridge emitted {} frames for {} inputs", out.len(), frames.len())
    }
    Ok(out)
}
