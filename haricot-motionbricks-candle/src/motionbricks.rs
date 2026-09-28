use anyhow::{bail, Result};

use crate::g1::{G1Assets, G1MotionFrame, G1_JOINTS, MOTIONBRICKS_DUAL_DIM};
use crate::math3d::{wrap_angle, Mat3, Vec3};

pub const MOTIONBRICKS_GLOBAL_DIM: usize = 414;
pub const MOTIONBRICKS_LOCAL_DIM: usize = 413;
pub const MOTIONBRICKS_BODY_DIM: usize = 409;

pub const IDX_GLOBAL_ROOT: std::ops::Range<usize> = 0..5;
pub const IDX_LOCAL_ROOT: std::ops::Range<usize> = 5..9;
pub const IDX_BODY: std::ops::Range<usize> = 9..418;
pub const IDX_RIC: std::ops::Range<usize> = 9..108;
pub const IDX_GLOBAL_ROT: std::ops::Range<usize> = 108..312;
pub const IDX_LOCAL_VEL: std::ops::Range<usize> = 312..414;
pub const IDX_CONTACTS: std::ops::Range<usize> = 414..418;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BridgeTimingMode {
    /// Matches MotionBricks' offline finite-difference convention exactly: the
    /// feature for t is emitted when t+1 arrives. `flush()` duplicates the last
    /// known velocity for the final frame, as MotionBricks does.
    ExactDelayed,
}

#[derive(Clone, Debug)]
pub struct MotionBricksFeatures {
    /// [global_root(5), local_root(4), body(409)]
    pub dual: Vec<f32>,
    /// [global_root(5), body(409)]
    pub global: Vec<f32>,
    /// [local_root(4), body(409)]
    pub local: Vec<f32>,
    pub normalized_dual: Option<Vec<f32>>,
    pub normalized_global: Option<Vec<f32>>,
    pub normalized_local: Option<Vec<f32>>,
}

impl MotionBricksFeatures {
    pub fn validate(&self) -> Result<()> {
        if self.dual.len() != MOTIONBRICKS_DUAL_DIM
            || self.global.len() != MOTIONBRICKS_GLOBAL_DIM
            || self.local.len() != MOTIONBRICKS_LOCAL_DIM
        {
            bail!(
                "MotionBricks feature dimension mismatch dual={} global={} local={}",
                self.dual.len(), self.global.len(), self.local.len()
            )
        }
        if self.dual.iter().any(|v| !v.is_finite())
            || self.global.iter().any(|v| !v.is_finite())
            || self.local.iter().any(|v| !v.is_finite())
        {
            bail!("non-finite MotionBricks features")
        }
        if let Some(x) = &self.normalized_dual {
            if x.len() != MOTIONBRICKS_DUAL_DIM || x.iter().any(|v| !v.is_finite()) {
                bail!("invalid normalized dual features")
            }
        }
        Ok(())
    }

    pub fn shared_body_matches(&self, atol: f32) -> bool {
        let global_body = &self.global[5..];
        let local_body = &self.local[4..];
        global_body
            .iter()
            .zip(local_body.iter())
            .all(|(&a, &b)| (a - b).abs() <= atol)
    }
}

#[derive(Clone, Debug)]
struct CanonicalG1Frame {
    dt: f32,
    positions: Vec<Vec3>,
    global_rotations: Vec<Mat3>,
    heading: f32,
}

#[derive(Clone, Debug)]
pub struct MotionBricksFeatureBridge {
    mode: BridgeTimingMode,
    assets: G1Assets,
    correction: Option<Mat3>,
    root_origin_xz: Option<Vec3>,
    previous: Option<CanonicalG1Frame>,
    last_velocity: Option<Vec<Vec3>>,
    last_rot_velocity: Option<f32>,
    emitted: usize,
}

impl MotionBricksFeatureBridge {
    pub fn new(assets: G1Assets, mode: BridgeTimingMode) -> Result<Self> {
        assets.validate()?;
        Ok(Self {
            mode,
            assets,
            correction: None,
            root_origin_xz: None,
            previous: None,
            last_velocity: None,
            last_rot_velocity: None,
            emitted: 0,
        })
    }

    pub fn mode(&self) -> BridgeTimingMode {
        self.mode
    }

    pub fn emitted_frames(&self) -> usize {
        self.emitted
    }

    pub fn push(&mut self, frame: &G1MotionFrame) -> Result<Option<MotionBricksFeatures>> {
        frame.validate()?;
        if !(frame.dt > 0.0 && frame.dt.is_finite()) {
            bail!("invalid G1 frame dt")
        }

        if self.correction.is_none() {
            let raw_heading = heading_from_hips(&frame.positions)?;
            self.correction = Some(Mat3::rotation_y(-raw_heading));
            let root = frame.positions[0];
            self.root_origin_xz = Some(Vec3::new(root.x, 0.0, root.z));
        }
        let current = self.canonicalize(frame)?;

        match self.mode {
            BridgeTimingMode::ExactDelayed => {
                let Some(previous) = self.previous.take() else {
                    self.previous = Some(current);
                    return Ok(None);
                };
                let dt = previous.dt;
                if (current.dt - dt).abs() > 1e-6 {
                    bail!("ExactDelayed requires constant dt: previous={dt} current={}", current.dt)
                }
                let velocity = current
                    .positions
                    .iter()
                    .zip(previous.positions.iter())
                    .map(|(&cur, &prev)| (cur - prev) / dt)
                    .collect::<Vec<_>>();
                let rot_velocity = wrap_angle(current.heading - previous.heading) / dt;
                let out = self.assemble(&previous, &velocity, rot_velocity)?;
                self.last_velocity = Some(velocity);
                self.last_rot_velocity = Some(rot_velocity);
                self.previous = Some(current);
                self.emitted += 1;
                Ok(Some(out))
            }
        }
    }

    /// Emit the final delayed frame. MotionBricks pads the final velocity by
    /// repeating the previous velocity; a one-frame stream falls back to zero.
    pub fn flush(&mut self) -> Result<Option<MotionBricksFeatures>> {
        let Some(previous) = self.previous.take() else {
            return Ok(None);
        };
        let velocity = self
            .last_velocity
            .clone()
            .unwrap_or_else(|| vec![Vec3::ZERO; G1_JOINTS]);
        let rot_velocity = self.last_rot_velocity.unwrap_or(0.0);
        let out = self.assemble(&previous, &velocity, rot_velocity)?;
        self.emitted += 1;
        Ok(Some(out))
    }

    fn canonicalize(&self, frame: &G1MotionFrame) -> Result<CanonicalG1Frame> {
        let correction = self.correction.ok_or_else(|| anyhow::anyhow!("bridge not initialized"))?;
        let origin = self.root_origin_xz.ok_or_else(|| anyhow::anyhow!("bridge not initialized"))?;
        let positions = frame
            .positions
            .iter()
            .map(|&p| correction.mul_vec(p - origin))
            .collect::<Vec<_>>();
        let global_rotations = frame
            .global_rotations
            .iter()
            .map(|&r| correction.mul_mat(r))
            .collect::<Vec<_>>();
        let heading = heading_from_hips(&positions)?;
        Ok(CanonicalG1Frame {
            dt: frame.dt,
            positions,
            global_rotations,
            heading,
        })
    }

    fn assemble(
        &self,
        frame: &CanonicalG1Frame,
        velocity: &[Vec3],
        local_root_rot_vel: f32,
    ) -> Result<MotionBricksFeatures> {
        if velocity.len() != G1_JOINTS {
            bail!("expected {G1_JOINTS} joint velocities")
        }
        let root = frame.positions[0];
        let mut dual = Vec::with_capacity(MOTIONBRICKS_DUAL_DIM);

        // Global root, 5 dims.
        dual.extend_from_slice(&[root.x, root.y, root.z, frame.heading.cos(), frame.heading.sin()]);

        // Local root, 4 dims. In DualRootGlobalJoints removing_heading=False,
        // so the x/z linear velocity stays in the canonical world frame.
        dual.extend_from_slice(&[
            local_root_rot_vel,
            velocity[0].x,
            velocity[0].z,
            root.y,
        ]);

        // ric_data: 33 non-root positions, subtract only projected root XZ;
        // keep world Y exactly as MotionBricks compute_position_features does.
        for p in frame.positions.iter().skip(1) {
            dual.extend_from_slice(&[p.x - root.x, p.y, p.z - root.z]);
        }

        // global_rot_data: 34 x MotionBricks cont6d (first two columns).
        for &rot in &frame.global_rotations {
            dual.extend_from_slice(&rot.to_motionbricks_cont6d());
        }

        // local_vel naming is historical here: in DualRootGlobalJoints this is
        // global/canonical-world velocity for all 34 joints.
        for &v in velocity {
            dual.extend_from_slice(&v.to_array());
        }

        // Match MotionBricks' default foot_detect_from_pos_and_vel semantics
        // exactly when no explicit contacts are supplied: left ankle-roll/toe,
        // then right ankle-roll/toe; speed < 0.15 m/s and height < 0.10 m.
        // We intentionally do not reuse WHAM's four contact logits here because
        // their training-label ordering is not part of WHAM's public runtime
        // contract. HaricotMotionFrame still retains those probabilities for
        // scoring/ground refinement.
        let contact_indices = [6usize, 7usize, 13usize, 14usize];
        let mut contacts = [0.0_f32; 4];
        for (slot, &joint) in contact_indices.iter().enumerate() {
            contacts[slot] = if velocity[joint].norm() < 0.15 && frame.positions[joint].y < 0.10 {
                1.0
            } else {
                0.0
            };
        }
        dual.extend_from_slice(&contacts);
        if dual.len() != MOTIONBRICKS_DUAL_DIM {
            bail!("internal MotionBricks dual layout bug: got {}", dual.len())
        }

        let mut global = Vec::with_capacity(MOTIONBRICKS_GLOBAL_DIM);
        global.extend_from_slice(&dual[IDX_GLOBAL_ROOT]);
        global.extend_from_slice(&dual[IDX_BODY]);

        let mut local = Vec::with_capacity(MOTIONBRICKS_LOCAL_DIM);
        local.extend_from_slice(&dual[IDX_LOCAL_ROOT]);
        local.extend_from_slice(&dual[IDX_BODY]);

        let (normalized_dual, normalized_global, normalized_local) =
            if let (Some(mean), Some(std)) = (&self.assets.mean, &self.assets.std) {
                let norm = dual
                    .iter()
                    .zip(mean.iter())
                    .zip(std.iter())
                    .map(|((&x, &m), &s)| (x - m) / (s * s + 1e-5).sqrt())
                    .collect::<Vec<_>>();
                let mut ng = Vec::with_capacity(MOTIONBRICKS_GLOBAL_DIM);
                ng.extend_from_slice(&norm[IDX_GLOBAL_ROOT]);
                ng.extend_from_slice(&norm[IDX_BODY]);
                let mut nl = Vec::with_capacity(MOTIONBRICKS_LOCAL_DIM);
                nl.extend_from_slice(&norm[IDX_LOCAL_ROOT]);
                nl.extend_from_slice(&norm[IDX_BODY]);
                (Some(norm), Some(ng), Some(nl))
            } else {
                (None, None, None)
            };

        let out = MotionBricksFeatures {
            dual,
            global,
            local,
            normalized_dual,
            normalized_global,
            normalized_local,
        };
        out.validate()?;
        Ok(out)
    }
}

/// Exact hips-position heading convention used by G1Skeleton34:
/// across = right_hip - left_hip; forward = up x across; angle = atan2(x, z).
pub fn heading_from_hips(positions: &[Vec3]) -> Result<f32> {
    if positions.len() != G1_JOINTS {
        bail!("expected {G1_JOINTS} positions for heading")
    }
    let right_hip = positions[8];
    let left_hip = positions[1];
    let across = Vec3::new(right_hip.x - left_hip.x, 0.0, right_hip.z - left_hip.z).normalized();
    if across.norm() <= 1e-8 {
        return Ok(0.0);
    }
    let forward = Vec3::Y.cross(across).normalized();
    Ok(forward.x.atan2(forward.z))
}
