use anyhow::{bail, Context, Result};
use candle_core::{D, DType, Device, Tensor};

use crate::g1::G1Assets;
use crate::motionbricks::{MotionBricksFeatures, MOTIONBRICKS_GLOBAL_DIM, MOTIONBRICKS_LOCAL_DIM};
use crate::motionbricks_runtime::{
    MotionBricksConv1dPolicy, MotionBricksRuntimeReference, PoseRuntimeInput, RootRuntimeInput,
    RootRuntimeOutput, VqvaeRuntimeInput, MB_FRAMES, MB_NUM_TOKENS, MB_POSE_DIM, MB_POSE_HEADS,
};

pub const MOTIONBRICKS_PRODUCTION_SM61: u32 = 61;
pub const MOTIONBRICKS_PRODUCTION_CUDNN_SM61: &str = "9.10.2.21";
pub const MOTIONBRICKS_FPS: f32 = 30.0;
const STATS_EPS: f32 = 1e-5;

/// ASD V2: old 9.1.0.2 K1/K3 promotion NOT valid on cuDNN 9.10.2.21.
pub fn production_conv1d_policy(_: Option<u32>, _: Option<&str>) -> MotionBricksConv1dPolicy {
    MotionBricksConv1dPolicy::Baseline
}

#[derive(Clone, Debug)]
pub struct MotionBricksBridgeConditions {
    pub root: RootRuntimeInput,
    pub pose_cond: Tensor,     // [1,64,304]
    pub has_pose_cond: Tensor, // [1,64] f32 0/1
}

#[derive(Clone, Debug)]
pub struct MotionBricksPipelineOutput {
    pub root: RootRuntimeOutput,
    pub pose_logits: Tensor, // [1,16,8,10]
    pub pose_tokens: Tensor, // [1,16,8] i64
    pub recon_state: Tensor, // [1,64,413]
}

fn normalized_global<'a>(frame: &'a MotionBricksFeatures) -> Result<&'a [f32]> {
    let values = frame
        .normalized_global
        .as_deref()
        .context("MotionBricks official statistics are required: normalized_global missing")?;
    if values.len() != MOTIONBRICKS_GLOBAL_DIM {
        bail!(
            "normalized global feature dimension mismatch: got {}, expected {}",
            values.len(),
            MOTIONBRICKS_GLOBAL_DIM
        )
    }
    Ok(values)
}

fn normalized_local<'a>(frame: &'a MotionBricksFeatures) -> Result<&'a [f32]> {
    let values = frame
        .normalized_local
        .as_deref()
        .context("MotionBricks official statistics are required: normalized_local missing")?;
    if values.len() != MOTIONBRICKS_LOCAL_DIM {
        bail!(
            "normalized local feature dimension mismatch: got {}, expected {}",
            values.len(),
            MOTIONBRICKS_LOCAL_DIM
        )
    }
    Ok(values)
}

/// MotionBricks INTERNAL_POSE_FEATURE_MODE =
/// joint_positions_and_rotations_and_hip_height.
///
/// In the 413-dim local representation this is:
///   local_root.global_root_y (index 3)
///   + ric_data (99 dims)
///   + global_rot_data (204 dims)
/// = 304 dims.
fn normalized_pose304(frame: &MotionBricksFeatures) -> Result<Vec<f32>> {
    let local = normalized_local(frame)?;
    let mut out = Vec::with_capacity(MB_POSE_DIM);
    out.push(local[3]);
    // local[4..307] = ric99 + global_rot204.
    out.extend_from_slice(&local[4..307]);
    if out.len() != MB_POSE_DIM {
        bail!("internal pose feature dimension bug: got {}", out.len())
    }
    Ok(out)
}

pub fn bridge_features_to_conditions(
    features: &[MotionBricksFeatures],
    device: &Device,
) -> Result<MotionBricksBridgeConditions> {
    if features.len() != MB_FRAMES {
        bail!(
            "full pipeline fixed16 profile requires {MB_FRAMES} bridge frames, got {}",
            features.len()
        )
    }

    let constraint_frames = [0usize, 1, 2, 3, 60, 61, 62, 63];
    let mut root_global = Vec::with_capacity(8 * 5);
    let mut root_local = Vec::with_capacity(8 * 4);
    let mut root_poses = Vec::with_capacity(8 * MB_POSE_DIM);

    for &idx in &constraint_frames {
        let global = normalized_global(&features[idx])?;
        let local = normalized_local(&features[idx])?;
        root_global.extend_from_slice(&global[..5]);
        root_local.extend_from_slice(&local[..4]);
        root_poses.extend_from_slice(&normalized_pose304(&features[idx])?);
    }

    let root = RootRuntimeInput {
        global_root_values: Tensor::from_vec(root_global, (1, 8, 5), device)?,
        local_root_values: Tensor::from_vec(root_local, (1, 8, 4), device)?,
        poses: Tensor::from_vec(root_poses, (1, 8, MB_POSE_DIM), device)?,
    };

    let mut pose_cond = vec![0.0f32; MB_FRAMES * MB_POSE_DIM];
    let mut has_pose_cond = vec![0.0f32; MB_FRAMES];
    for &idx in &constraint_frames {
        let pose = normalized_pose304(&features[idx])?;
        pose_cond[idx * MB_POSE_DIM..(idx + 1) * MB_POSE_DIM].copy_from_slice(&pose);
        has_pose_cond[idx] = 1.0;
    }

    Ok(MotionBricksBridgeConditions {
        root,
        pose_cond: Tensor::from_vec(pose_cond, (1, MB_FRAMES, MB_POSE_DIM), device)?,
        has_pose_cond: Tensor::from_vec(has_pose_cond, (1, MB_FRAMES), device)?,
    })
}

fn pose_global_root_condition(pred_global_root_values: &Tensor) -> Result<Tensor> {
    let x = pred_global_root_values.narrow(2, 0, 1)?;
    let z = pred_global_root_values.narrow(2, 2, 1)?;
    let heading = pred_global_root_values.narrow(2, 3, 2)?;
    Ok(Tensor::cat(&[&x, &z, &heading], 2)?)
}

fn stats_scale(std: f32) -> f32 {
    (std * std + STATS_EPS).sqrt()
}

/// Build the VQ external root condition directly on the GPU.
///
/// The released pose VQ-VAE uses the local representation and
/// `root_without_hip_height_without_heading`, i.e. the two X/Z root velocity
/// dimensions. As DualRootGlobalJoints has `removing_heading=false`, these are
/// simply forward finite differences of global X/Z at 30 Hz. The final velocity
/// repeats the previous one, matching MotionBricks `compute_vel_xyz`.
fn decoder_external_root_xz(
    pred_global_root_values: &Tensor,
    assets: &G1Assets,
) -> Result<Tensor> {
    let mean = assets
        .mean
        .as_ref()
        .context("MotionBricks official mean statistics are required")?;
    let std = assets
        .std
        .as_ref()
        .context("MotionBricks official std statistics are required")?;
    if mean.len() != 418 || std.len() != 418 {
        bail!("MotionBricks dual statistics must contain 418 values")
    }

    // Global root indices in the dual representation: [x,y,z,cos,sin] = 0..5.
    let global_xz_norm = Tensor::cat(
        &[
            &pred_global_root_values.narrow(2, 0, 1)?,
            &pred_global_root_values.narrow(2, 2, 1)?,
        ],
        2,
    )?;
    let global_mean = Tensor::from_vec(vec![mean[0], mean[2]], (1, 1, 2), pred_global_root_values.device())?;
    let global_scale = Tensor::from_vec(
        vec![stats_scale(std[0]), stats_scale(std[2])],
        (1, 1, 2),
        pred_global_root_values.device(),
    )?;
    let global_xz = global_xz_norm
        .broadcast_mul(&global_scale)?
        .broadcast_add(&global_mean)?;

    let next = global_xz.narrow(1, 1, MB_FRAMES - 1)?;
    let prev = global_xz.narrow(1, 0, MB_FRAMES - 1)?;
    let velocity = next.sub(&prev)?.affine(MOTIONBRICKS_FPS as f64, 0.0)?;
    let last = velocity.narrow(1, MB_FRAMES - 2, 1)?;
    let velocity = Tensor::cat(&[&velocity, &last], 1)?;

    // Dual local-root indices are [rot_vel, vel_x, vel_z, y] at 5..9.
    let local_mean = Tensor::from_vec(vec![mean[6], mean[7]], (1, 1, 2), pred_global_root_values.device())?;
    let local_scale = Tensor::from_vec(
        vec![stats_scale(std[6]), stats_scale(std[7])],
        (1, 1, 2),
        pred_global_root_values.device(),
    )?;
    Ok(velocity
        .broadcast_sub(&local_mean)?
        .broadcast_div(&local_scale)?)
}

pub fn run_motionbricks_from_bridge(
    model: &MotionBricksRuntimeReference,
    conditions: &MotionBricksBridgeConditions,
    assets: &G1Assets,
    policy: MotionBricksConv1dPolicy,
) -> Result<MotionBricksPipelineOutput> {
    let root = model
        .root
        .forward_fixed_16_with_policy(&conditions.root, policy)?;

    // Official pose inference starts from all MASK tokens (ID=10), runs one
    // denoising iteration, then takes the highest-probability token in the
    // deterministic benchmark path.
    let pose_tokens = Tensor::from_vec(
        vec![10i64; MB_NUM_TOKENS * MB_POSE_HEADS],
        (1, MB_NUM_TOKENS, MB_POSE_HEADS),
        root.pred_global_root_values.device(),
    )?;
    let pose_input = PoseRuntimeInput {
        pose_tokens,
        root_values: pose_global_root_condition(&root.pred_global_root_values)?,
        pose_cond: conditions.pose_cond.clone(),
        has_pose_cond: conditions.has_pose_cond.clone(),
    };
    let pose_logits = model.pose.forward_fixed_16(&pose_input)?;
    let selected_tokens = pose_logits.argmax(D::Minus1)?.to_dtype(DType::I64)?;

    let vq_input = VqvaeRuntimeInput {
        pose_tokens: selected_tokens.clone(),
        target_cond: conditions.pose_cond.clone(),
        has_target_cond: conditions.has_pose_cond.clone(),
        external_cond: decoder_external_root_xz(&root.pred_global_root_values, assets)?,
    };
    let recon_state = model.decoder.forward_with_policy(&vq_input, policy)?;

    Ok(MotionBricksPipelineOutput {
        root,
        pose_logits,
        pose_tokens: selected_tokens,
        recon_state,
    })
}

pub fn validate_pipeline_output(out: &MotionBricksPipelineOutput) -> Result<()> {
    if out.root.num_token_logits.dims() != [1, 12]
        || out.root.pred_global_root_values.dims() != [1, MB_FRAMES, 5]
        || out.pose_logits.dims() != [1, MB_NUM_TOKENS, MB_POSE_HEADS, 10]
        || out.pose_tokens.dims() != [1, MB_NUM_TOKENS, MB_POSE_HEADS]
        || out.recon_state.dims() != [1, MB_FRAMES, 413]
    {
        bail!(
            "full pipeline output shape gate failed: root_logits={:?} root={:?} pose={:?} tokens={:?} recon={:?}",
            out.root.num_token_logits.dims(),
            out.root.pred_global_root_values.dims(),
            out.pose_logits.dims(),
            out.pose_tokens.dims(),
            out.recon_state.dims(),
        )
    }
    Ok(())
}
