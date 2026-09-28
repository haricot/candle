use candle_core::{D, IndexOp, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::config::WhamLiteConfig;
use crate::layers::{NeuralInitialization, Regressor, StackedLstmState};

const MAIN_JOINTS: [usize; 20] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
];

#[derive(Clone, Debug)]
struct MotionEncoder {
    embed: Linear,
    neural_init: NeuralInitialization,
    regressor: Regressor,
    cfg: WhamLiteConfig,
}

#[derive(Clone, Debug)]
struct MotionEncoderState {
    recurrent: StackedLstmState,
    prev_kp3d: Tensor,
}

impl MotionEncoder {
    fn load(cfg: &WhamLiteConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embed: linear(cfg.input_dim, cfg.embed_dim, vb.pp("embed_layer"))?,
            neural_init: NeuralInitialization::load(
                cfg.init_kp_dim(),
                cfg.embed_dim,
                cfg.n_layers,
                vb.pp("neural_init"),
            )?,
            regressor: Regressor::load(
                cfg.embed_dim,
                cfg.embed_dim,
                &[cfg.kp3d_dim()],
                cfg.kp3d_dim(),
                cfg.n_layers,
                vb.pp("regressor"),
            )?,
            cfg: cfg.clone(),
        })
    }

    fn start(&self, init_kp3d: &Tensor, init_input: &Tensor) -> Result<MotionEncoderState> {
        let init = Tensor::cat(&[init_kp3d, init_input], D::Minus1)?;
        Ok(MotionEncoderState {
            recurrent: self.neural_init.forward(&init)?,
            prev_kp3d: init_kp3d.clone(),
        })
    }

    fn step(
        &self,
        x: &Tensor,
        state: &mut MotionEncoderState,
    ) -> Result<(Tensor, Tensor)> {
        let embedded = self.embed.forward(x)?;
        let (preds, hidden, next_state) =
            self.regressor
                .step(&embedded, &[&state.prev_kp3d], &state.recurrent)?;
        let pred_kp3d = preds[0].clone();
        let context = Tensor::cat(&[&hidden, &pred_kp3d], D::Minus1)?;
        state.prev_kp3d = pred_kp3d.clone();
        state.recurrent = next_state;
        Ok((pred_kp3d, context))
    }
}

#[derive(Clone, Debug)]
struct TrajectoryDecoder {
    regressor: Regressor,
}

#[derive(Clone, Debug)]
struct TrajectoryDecoderState {
    recurrent: StackedLstmState,
    prev_root: Tensor,
}

impl TrajectoryDecoder {
    fn load(cfg: &WhamLiteConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            regressor: Regressor::load(
                cfg.context_dim,
                cfg.context_dim,
                &[3, 6],
                12,
                cfg.n_layers,
                vb.pp("regressor"),
            )?,
        })
    }

    fn start(&self, batch: usize, init_root_rot6d: &Tensor) -> Result<TrajectoryDecoderState> {
        Ok(TrajectoryDecoderState {
            recurrent: self.regressor.zero_state(batch)?,
            prev_root: init_root_rot6d.clone(),
        })
    }

    fn step(
        &self,
        context: &Tensor,
        cam_angvel: &Tensor,
        state: &mut TrajectoryDecoderState,
    ) -> Result<(Tensor, Tensor)> {
        let (preds, _, next_state) = self.regressor.step(
            context,
            &[&state.prev_root, cam_angvel],
            &state.recurrent,
        )?;
        let pred_vel = preds[0].clone();
        let pred_root = preds[1].clone();
        state.prev_root = pred_root.clone();
        state.recurrent = next_state;
        Ok((pred_root, pred_vel))
    }
}

#[derive(Clone, Debug)]
struct MotionDecoder {
    neural_init: NeuralInitialization,
    regressor: Regressor,
    cfg: WhamLiteConfig,
}

#[derive(Clone, Debug)]
struct MotionDecoderState {
    recurrent: StackedLstmState,
    prev_pose: Tensor,
}

impl MotionDecoder {
    fn load(cfg: &WhamLiteConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            neural_init: NeuralInitialization::load(
                cfg.main_pose_dim(),
                cfg.context_dim,
                cfg.n_layers,
                vb.pp("neural_init"),
            )?,
            regressor: Regressor::load(
                cfg.context_dim,
                cfg.context_dim,
                &[cfg.pose_dim(), 10, 3, cfg.contact_dim],
                cfg.pose_dim(),
                cfg.n_layers,
                vb.pp("regressor"),
            )?,
            cfg: cfg.clone(),
        })
    }

    fn main_pose_from_full(&self, full_pose: &Tensor) -> Result<Tensor> {
        let b = full_pose.dim(0)?;
        let pose = full_pose.reshape((b, self.cfg.pose_joints, 6))?;
        let mut joints = Vec::with_capacity(MAIN_JOINTS.len());
        for joint_idx in MAIN_JOINTS {
            joints.push(pose.i((.., joint_idx, ..))?.contiguous()?);
        }
        Tensor::stack(&joints, 1)?.reshape((b, self.cfg.main_pose_dim()))
    }

    fn start(&self, init_pose_rot6d: &Tensor) -> Result<MotionDecoderState> {
        let init_main = self.main_pose_from_full(init_pose_rot6d)?;
        Ok(MotionDecoderState {
            recurrent: self.neural_init.forward(&init_main)?,
            prev_pose: init_pose_rot6d.clone(),
        })
    }

    fn step(
        &self,
        context: &Tensor,
        state: &mut MotionDecoderState,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (preds, _, next_state) =
            self.regressor
                .step(context, &[&state.prev_pose], &state.recurrent)?;
        let pose = preds[0].clone();
        state.prev_pose = pose.clone();
        state.recurrent = next_state;
        Ok((pose, preds[1].clone(), preds[2].clone(), preds[3].clone()))
    }
}

#[derive(Clone, Debug)]
pub struct WhamLiteInput {
    /// [B,T,37] = normalized COCO17 xy + bbox center/scale.
    pub pose2d: Tensor,
    /// [B,51] root-centered initial 17x3 pose estimate.
    pub init_kp3d: Tensor,
    /// [B,144] initial SMPL-like 24x6D pose.
    pub init_pose_rot6d: Tensor,
    /// [B,6] initial root rotation in 6D representation.
    pub init_root_rot6d: Tensor,
    /// [B,T,6]. Set to zero in the no-SLAM / Haricot runtime path.
    pub cam_angvel: Tensor,
}

#[derive(Clone, Debug)]
pub struct WhamLiteOutput {
    /// [B,T,17,3]
    pub joints_3d: Tensor,
    /// [B,T,144] = 24 * rotation6d
    pub body_rot6d: Tensor,
    /// [B,T+1,6], including the supplied initial root as element 0.
    pub root_rot6d: Tensor,
    /// [B,T,3]
    pub root_velocity: Tensor,
    /// [B,T,4], raw WHAM contact head output/logits.
    pub contact_logits: Tensor,
    /// [B,T,10]
    pub shape: Tensor,
    /// [B,T,3]
    pub weak_camera: Tensor,
}

#[derive(Clone, Debug)]
pub struct WhamLiteStepOutput {
    /// [B,17,3]
    pub joints_3d: Tensor,
    /// [B,144]
    pub body_rot6d: Tensor,
    /// [B,6]
    pub root_rot6d: Tensor,
    /// [B,3]
    pub root_velocity: Tensor,
    /// [B,4]
    pub contact_logits: Tensor,
    /// [B,10]
    pub shape: Tensor,
    /// [B,3]
    pub weak_camera: Tensor,
}

#[derive(Clone, Debug)]
pub struct HaricotWhamStreamState {
    motion_encoder: MotionEncoderState,
    trajectory_decoder: TrajectoryDecoderState,
    motion_decoder: MotionDecoderState,
}

#[derive(Clone, Debug)]
pub struct HaricotWhamLite {
    motion_encoder: MotionEncoder,
    trajectory_decoder: TrajectoryDecoder,
    motion_decoder: MotionDecoder,
    cfg: WhamLiteConfig,
}

impl HaricotWhamLite {
    pub fn load(cfg: &WhamLiteConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            motion_encoder: MotionEncoder::load(cfg, vb.pp("motion_encoder"))?,
            trajectory_decoder: TrajectoryDecoder::load(cfg, vb.pp("trajectory_decoder"))?,
            motion_decoder: MotionDecoder::load(cfg, vb.pp("motion_decoder"))?,
            cfg: cfg.clone(),
        })
    }

    /// Create the persistent recurrent state for a new tracked person.
    ///
    /// `first_pose2d` is the first normalized 37-D frame and is used by WHAM's
    /// neural initializer exactly as in the sequence forward path. Call `step`
    /// with this same frame to obtain the first prediction.
    pub fn start_stream(
        &self,
        first_pose2d: &Tensor,
        init_kp3d: &Tensor,
        init_pose_rot6d: &Tensor,
        init_root_rot6d: &Tensor,
    ) -> Result<HaricotWhamStreamState> {
        let batch = first_pose2d.dim(0)?;
        Ok(HaricotWhamStreamState {
            motion_encoder: self.motion_encoder.start(init_kp3d, first_pose2d)?,
            trajectory_decoder: self
                .trajectory_decoder
                .start(batch, init_root_rot6d)?,
            motion_decoder: self.motion_decoder.start(init_pose_rot6d)?,
        })
    }

    /// Process exactly one normalized pose frame while preserving all recurrent state.
    pub fn step(
        &self,
        pose2d: &Tensor,
        cam_angvel: &Tensor,
        state: &mut HaricotWhamStreamState,
    ) -> Result<WhamLiteStepOutput> {
        let (joints_flat, context) = self.motion_encoder.step(pose2d, &mut state.motion_encoder)?;
        let (root_rot6d, root_velocity) = self.trajectory_decoder.step(
            &context,
            cam_angvel,
            &mut state.trajectory_decoder,
        )?;
        let (body_rot6d, shape, weak_camera, contact_logits) =
            self.motion_decoder.step(&context, &mut state.motion_decoder)?;
        let batch = pose2d.dim(0)?;
        let joints_3d = joints_flat.reshape((batch, self.cfg.n_joints, 3))?;

        Ok(WhamLiteStepOutput {
            joints_3d,
            body_rot6d,
            root_rot6d,
            root_velocity,
            contact_logits,
            shape,
            weak_camera,
        })
    }

    /// Reference sequence forward. Internally it uses the same recurrent stateful
    /// API that production streaming uses, but returns WHAM-compatible stacked tensors.
    pub fn forward(&self, input: &WhamLiteInput) -> Result<WhamLiteOutput> {
        let (batch, frames, input_dim) = input.pose2d.dims3()?;
        if frames == 0 {
            candle_core::bail!("WHAM-lite requires at least one frame")
        }
        if input_dim != self.cfg.input_dim {
            candle_core::bail!(
                "expected pose2d input_dim={}, got {}",
                self.cfg.input_dim,
                input_dim
            )
        }

        let first_pose = input.pose2d.i((.., 0, ..))?.contiguous()?;
        let mut state = self.start_stream(
            &first_pose,
            &input.init_kp3d,
            &input.init_pose_rot6d,
            &input.init_root_rot6d,
        )?;

        let mut joints = Vec::with_capacity(frames);
        let mut body = Vec::with_capacity(frames);
        let mut roots = Vec::with_capacity(frames + 1);
        let mut velocities = Vec::with_capacity(frames);
        let mut contacts = Vec::with_capacity(frames);
        let mut shapes = Vec::with_capacity(frames);
        let mut cameras = Vec::with_capacity(frames);
        roots.push(input.init_root_rot6d.clone());

        for t in 0..frames {
            let pose = input.pose2d.i((.., t, ..))?.contiguous()?;
            let cam = input.cam_angvel.i((.., t, ..))?.contiguous()?;
            let out = self.step(&pose, &cam, &mut state)?;
            joints.push(out.joints_3d);
            body.push(out.body_rot6d);
            roots.push(out.root_rot6d);
            velocities.push(out.root_velocity);
            contacts.push(out.contact_logits);
            shapes.push(out.shape);
            cameras.push(out.weak_camera);
        }

        Ok(WhamLiteOutput {
            joints_3d: Tensor::stack(&joints, 1)?.reshape((batch, frames, self.cfg.n_joints, 3))?,
            body_rot6d: Tensor::stack(&body, 1)?,
            root_rot6d: Tensor::stack(&roots, 1)?,
            root_velocity: Tensor::stack(&velocities, 1)?,
            contact_logits: Tensor::stack(&contacts, 1)?,
            shape: Tensor::stack(&shapes, 1)?,
            weak_camera: Tensor::stack(&cameras, 1)?,
        })
    }
}
