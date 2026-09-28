use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::math3d::{Mat3, Vec3};
use crate::motion_frame::{HaricotMotionFrame, SMPL_JOINTS};

pub const G1_JOINTS: usize = 34;
pub const MOTIONBRICKS_DUAL_DIM: usize = 418;

pub const G1_NAMES: [&str; G1_JOINTS] = [
    "pelvis_skel",
    "left_hip_pitch_skel", "left_hip_roll_skel", "left_hip_yaw_skel",
    "left_knee_skel", "left_ankle_pitch_skel", "left_ankle_roll_skel", "left_toe_base",
    "right_hip_pitch_skel", "right_hip_roll_skel", "right_hip_yaw_skel",
    "right_knee_skel", "right_ankle_pitch_skel", "right_ankle_roll_skel", "right_toe_base",
    "waist_yaw_skel", "waist_roll_skel", "waist_pitch_skel",
    "left_shoulder_pitch_skel", "left_shoulder_roll_skel", "left_shoulder_yaw_skel",
    "left_elbow_skel", "left_wrist_roll_skel", "left_wrist_pitch_skel",
    "left_wrist_yaw_skel", "left_hand_roll_skel",
    "right_shoulder_pitch_skel", "right_shoulder_roll_skel", "right_shoulder_yaw_skel",
    "right_elbow_skel", "right_wrist_roll_skel", "right_wrist_pitch_skel",
    "right_wrist_yaw_skel", "right_hand_roll_skel",
];

pub const G1_PARENTS: [i32; G1_JOINTS] = [
    -1,
    0, 1, 2, 3, 4, 5, 6,
    0, 8, 9, 10, 11, 12, 13,
    0, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24,
    17, 26, 27, 28, 29, 30, 31, 32,
];

/// SMPL24 parent tree used by WHAM.
pub const SMPL_PARENTS: [i32; SMPL_JOINTS] = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    20, 21,
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct G1Assets {
    pub source: String,
    pub neutral_joints: Vec<[f32; 3]>,
    pub parents: Vec<i32>,
    /// Full DualRootGlobalJoints statistics, 418 dimensions. Optional for
    /// structural testing; required before feeding released MotionBricks weights.
    pub mean: Option<Vec<f32>>,
    pub std: Option<Vec<f32>>,
}

impl G1Assets {
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading G1 assets {}", path.display()))?;
        let out: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing G1 assets {}", path.display()))?;
        out.validate()?;
        Ok(out)
    }

    pub fn validate(&self) -> Result<()> {
        if self.neutral_joints.len() != G1_JOINTS {
            bail!("expected {G1_JOINTS} G1 neutral joints, got {}", self.neutral_joints.len())
        }
        if self.parents.len() != G1_JOINTS {
            bail!("expected {G1_JOINTS} G1 parents, got {}", self.parents.len())
        }
        for (idx, (&got, &expected)) in self.parents.iter().zip(G1_PARENTS.iter()).enumerate() {
            if got != expected {
                bail!("G1 parent mismatch at joint {idx}: got {got}, expected {expected}")
            }
        }
        match (&self.mean, &self.std) {
            (Some(mean), Some(std)) => {
                if mean.len() != MOTIONBRICKS_DUAL_DIM || std.len() != MOTIONBRICKS_DUAL_DIM {
                    bail!("MotionBricks stats must both have 418 elements")
                }
                if std.iter().any(|x| !x.is_finite() || *x < 0.0) {
                    bail!("MotionBricks std contains invalid values")
                }
            }
            (None, None) => {}
            _ => bail!("mean and std must either both be present or both absent"),
        }
        Ok(())
    }

    pub fn has_stats(&self) -> bool {
        self.mean.is_some() && self.std.is_some()
    }

    /// Geometry-only fallback for development. It has the correct G1Skeleton34
    /// topology but is NOT a substitute for the released MotionBricks joints.p.
    pub fn builtin_approx() -> Self {
        let j = vec![
            [0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0], [0.10, 0.0, 0.0], [0.10, 0.0, 0.0],
            [0.10, -0.40, 0.0], [0.10, -0.80, 0.0], [0.10, -0.80, 0.0], [0.10, -0.80, 0.15],
            [-0.10, 0.0, 0.0], [-0.10, 0.0, 0.0], [-0.10, 0.0, 0.0],
            [-0.10, -0.40, 0.0], [-0.10, -0.80, 0.0], [-0.10, -0.80, 0.0], [-0.10, -0.80, 0.15],
            [0.0, 0.15, 0.0], [0.0, 0.15, 0.0], [0.0, 0.35, 0.0],
            [0.20, 0.50, 0.0], [0.20, 0.50, 0.0], [0.20, 0.50, 0.0],
            [0.50, 0.50, 0.0], [0.75, 0.50, 0.0], [0.75, 0.50, 0.0],
            [0.75, 0.50, 0.0], [0.85, 0.50, 0.0],
            [-0.20, 0.50, 0.0], [-0.20, 0.50, 0.0], [-0.20, 0.50, 0.0],
            [-0.50, 0.50, 0.0], [-0.75, 0.50, 0.0], [-0.75, 0.50, 0.0],
            [-0.75, 0.50, 0.0], [-0.85, 0.50, 0.0],
        ];
        Self {
            source: "builtin-approx-geometry-only".to_string(),
            neutral_joints: j,
            parents: G1_PARENTS.to_vec(),
            mean: None,
            std: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct G1MotionFrame {
    pub dt: f32,
    pub positions: Vec<Vec3>,
    /// Global/world rotations, one per G1Skeleton34 joint.
    pub global_rotations: Vec<Mat3>,
    pub contacts: [f32; 4],
}

impl G1MotionFrame {
    pub fn validate(&self) -> Result<()> {
        if self.positions.len() != G1_JOINTS || self.global_rotations.len() != G1_JOINTS {
            bail!("invalid G1 frame dimensions")
        }
        if self.positions.iter().any(|p| !p.is_finite())
            || self.global_rotations.iter().any(|r| !r.is_finite())
        {
            bail!("non-finite G1 frame")
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct G1Retarget {
    assets: G1Assets,
}

impl G1Retarget {
    pub fn new(assets: G1Assets) -> Result<Self> {
        assets.validate()?;
        Ok(Self { assets })
    }

    pub fn assets(&self) -> &G1Assets {
        &self.assets
    }

    pub fn retarget(&self, human: &HaricotMotionFrame) -> Result<G1MotionFrame> {
        let human_global = human_global_rotations(human);
        let mut global_rotations = vec![Mat3::IDENTITY; G1_JOINTS];

        // Mapping is deliberately global-rotation based. Multiple serial G1 hinge
        // joints may share a target human global orientation; the released G1
        // neutral skeleton then supplies exact robot bone lengths during FK.
        const MAP: [usize; G1_JOINTS] = [
            0,
            1, 1, 1, 4, 7, 10, 10,
            2, 2, 2, 5, 8, 11, 11,
            3, 6, 9,
            13, 16, 16, 18, 20, 20, 20, 22,
            14, 17, 17, 19, 21, 21, 21, 23,
        ];
        for (g1, &human_idx) in MAP.iter().enumerate() {
            global_rotations[g1] = human_global[human_idx];
        }

        let neutral = self
            .assets
            .neutral_joints
            .iter()
            .copied()
            .map(Vec3::from)
            .collect::<Vec<_>>();
        let mut positions = vec![Vec3::ZERO; G1_JOINTS];
        // WHAM trajectory translation is relative and does not carry the robot's
        // absolute pelvis height. Anchor the neutral G1 contact joints to the
        // Haricot ground plane, then add WHAM's vertical root displacement.
        // MotionBricks expects an absolute global_root_y around the robot pelvis.
        const CONTACT_JOINTS: [usize; 4] = [6, 7, 13, 14];
        let min_contact_y = CONTACT_JOINTS
            .iter()
            .map(|&idx| neutral[idx].y)
            .fold(f32::INFINITY, f32::min);
        let neutral_pelvis_height = human.ground.origin.y - min_contact_y;
        positions[0] = Vec3::new(
            human.root_position.x,
            neutral_pelvis_height + human.root_position.y,
            human.root_position.z,
        );
        for joint in 1..G1_JOINTS {
            let parent = self.assets.parents[joint] as usize;
            let rest_offset = neutral[joint] - neutral[parent];
            positions[joint] = positions[parent] + global_rotations[parent].mul_vec(rest_offset);
        }

        let contacts = human.contacts.map(|p| if p >= 0.5 { 1.0 } else { 0.0 });
        let out = G1MotionFrame {
            dt: human.dt,
            positions,
            global_rotations,
            contacts,
        };
        out.validate()?;
        Ok(out)
    }

    pub fn max_bone_length_error(&self, frame: &G1MotionFrame) -> f32 {
        let neutral = self
            .assets
            .neutral_joints
            .iter()
            .copied()
            .map(Vec3::from)
            .collect::<Vec<_>>();
        let mut max_err = 0.0_f32;
        for joint in 1..G1_JOINTS {
            let parent = self.assets.parents[joint] as usize;
            let rest = (neutral[joint] - neutral[parent]).norm();
            if rest <= 1e-8 {
                continue;
            }
            let posed = (frame.positions[joint] - frame.positions[parent]).norm();
            max_err = max_err.max((rest - posed).abs());
        }
        max_err
    }
}

fn human_global_rotations(frame: &HaricotMotionFrame) -> [Mat3; SMPL_JOINTS] {
    let mut out = [Mat3::IDENTITY; SMPL_JOINTS];
    out[0] = frame.root_world_rotation;
    for joint in 1..SMPL_JOINTS {
        let parent = SMPL_PARENTS[joint] as usize;
        let local = Mat3::from_wham_rot6d(frame.body_rot6d[joint]);
        out[joint] = out[parent].mul_mat(local);
    }
    out
}
