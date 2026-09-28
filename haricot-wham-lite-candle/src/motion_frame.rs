use anyhow::{bail, Result};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

use crate::math3d::{Mat3, Vec3};
use crate::WhamLiteStepOutput;

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

#[derive(Clone, Debug)]
pub struct DownloadedWhamStep {
    pub joints_3d: [[f32; 3]; HUMAN_JOINTS_3D],
    pub body_rot6d: [[f32; 6]; SMPL_JOINTS],
    pub root_rot6d: [f32; 6],
    pub root_velocity: [f32; 3],
    pub contact_logits: [f32; 4],
}

impl DownloadedWhamStep {
    /// Explicitly materialize the five WHAM tensors currently consumed by the
    /// CPU HaricotMotionFrame path. Keeping this separate from CPU assembly lets
    /// v0.3.4 measure the GPU->CPU boundary without changing numerical behavior.
    pub fn download(out: &WhamLiteStepOutput) -> Result<Self> {
        let joints = tensor_vec(out.joints_3d.reshape((HUMAN_JOINTS_3D * 3,))?)?;
        let pose = tensor_vec(out.body_rot6d.reshape((SMPL_JOINTS * 6,))?)?;
        let root = tensor_vec(out.root_rot6d.reshape((6,))?)?;
        let vel = tensor_vec(out.root_velocity.reshape((3,))?)?;
        let contact_logits = tensor_vec(out.contact_logits.reshape((4,))?)?;

        let mut joints_3d = [[0.0_f32; 3]; HUMAN_JOINTS_3D];
        for (idx, joint) in joints_3d.iter_mut().enumerate() {
            joint.copy_from_slice(&joints[idx * 3..idx * 3 + 3]);
        }
        let mut body_rot6d = [[0.0_f32; 6]; SMPL_JOINTS];
        for (idx, rot) in body_rot6d.iter_mut().enumerate() {
            rot.copy_from_slice(&pose[idx * 6..idx * 6 + 6]);
        }

        Ok(Self {
            joints_3d,
            body_rot6d,
            root_rot6d: root.try_into().map_err(|_| anyhow::anyhow!("root rot6d length mismatch"))?,
            root_velocity: vel.try_into().map_err(|_| anyhow::anyhow!("root velocity length mismatch"))?,
            contact_logits: contact_logits
                .try_into()
                .map_err(|_| anyhow::anyhow!("contact logits length mismatch"))?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct HaricotMotionFrameBuilder {
    dt: f32,
    prev_root_world_rotation: Mat3,
    root_position: Vec3,
}

impl HaricotMotionFrameBuilder {
    pub fn new(dt: f32, init_root_rot6d: [f32; 6]) -> Result<Self> {
        if !(dt > 0.0 && dt.is_finite()) {
            bail!("dt must be finite and > 0")
        }
        let ydown_to_yup = Mat3::rotation_x_pi();
        Ok(Self {
            dt,
            prev_root_world_rotation: ydown_to_yup.mul_mat(Mat3::from_wham_rot6d(init_root_rot6d)),
            root_position: Vec3::ZERO,
        })
    }

    pub fn from_wham_step(&mut self, out: &WhamLiteStepOutput) -> Result<HaricotMotionFrame> {
        let downloaded = DownloadedWhamStep::download(out)?;
        self.from_downloaded_wham_step(&downloaded)
    }

    pub fn from_downloaded_wham_step(
        &mut self,
        out: &DownloadedWhamStep,
    ) -> Result<HaricotMotionFrame> {
        let ydown_to_yup = Mat3::rotation_x_pi();
        let mut joints_3d = [Vec3::ZERO; HUMAN_JOINTS_3D];
        for (idx, j) in joints_3d.iter_mut().enumerate() {
            let p = Vec3::new(
                out.joints_3d[idx][0],
                out.joints_3d[idx][1],
                out.joints_3d[idx][2],
            );
            *j = ydown_to_yup.mul_vec(p);
        }

        let body_rot6d = out.body_rot6d;
        let root_delta_local = Vec3::new(
            out.root_velocity[0],
            out.root_velocity[1],
            out.root_velocity[2],
        );
        // Matches WHAM rollout_global_motion: root[:, :-1] @ root_v.
        let delta_world = self.prev_root_world_rotation.mul_vec(root_delta_local);
        self.root_position += delta_world;
        let root_velocity_world = delta_world / self.dt;

        let root_world_rotation =
            ydown_to_yup.mul_mat(Mat3::from_wham_rot6d(out.root_rot6d));
        self.prev_root_world_rotation = root_world_rotation;

        let contacts = [
            sigmoid(out.contact_logits[0]),
            sigmoid(out.contact_logits[1]),
            sigmoid(out.contact_logits[2]),
            sigmoid(out.contact_logits[3]),
        ];

        let right = root_world_rotation
            .mul_vec(Vec3::new(1.0, 0.0, 0.0))
            .normalized();
        let up = root_world_rotation
            .mul_vec(Vec3::new(0.0, 1.0, 0.0))
            .normalized();
        let forward = root_world_rotation
            .mul_vec(Vec3::new(0.0, 0.0, 1.0))
            .normalized();

        Ok(HaricotMotionFrame {
            dt: self.dt,
            joints_3d,
            body_rot6d,
            root_world_rotation,
            root_position: self.root_position,
            root_delta_local,
            root_velocity_world,
            contacts,
            confidence: 1.0,
            ground: GroundFrame::default(),
            body_basis: BodyBasis { right, up, forward },
        })
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tensor_vec(t: Tensor) -> Result<Vec<f32>> {
    Ok(t.to_vec1::<f32>()?)
}
