use serde::{Deserialize, Serialize};

use crate::math3d::{Mat3, Vec3};

pub const HUMAN_JOINTS_3D: usize = 17;
pub const SMPL_JOINTS: usize = 24;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GroundFrame {
    pub origin: Vec3,
    pub normal: Vec3,
}

impl Default for GroundFrame {
    fn default() -> Self {
        Self {
            origin: Vec3::ZERO,
            normal: Vec3::Y,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BodyBasis {
    pub right: Vec3,
    pub up: Vec3,
    pub forward: Vec3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HaricotMotionFrame {
    pub dt: f32,
    /// Root-centered WHAM 17-joint estimate, converted to Haricot Y-up.
    pub joints_3d: [Vec3; HUMAN_JOINTS_3D],
    /// WHAM/SMPL local rotations, kept in the original PyTorch3D 6D convention.
    pub body_rot6d: [[f32; 6]; SMPL_JOINTS],
    /// Current world root orientation, already converted to Haricot Y-up.
    pub root_world_rotation: Mat3,
    /// Integrated world root translation. WHAM's predicted root velocity is a
    /// per-frame local displacement, not m/s.
    pub root_position: Vec3,
    /// Raw WHAM local displacement for this frame.
    pub root_delta_local: Vec3,
    /// World-space velocity in metres/second (delta / dt).
    pub root_velocity_world: Vec3,
    /// L heel/ankle, L toe, R heel/ankle, R toe probabilities.
    pub contacts: [f32; 4],
    pub confidence: f32,
    pub ground: GroundFrame,
    pub body_basis: BodyBasis,
}

