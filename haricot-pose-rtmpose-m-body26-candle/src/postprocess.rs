use crate::halpe26::{NAMES, NUM_KEYPOINTS};
use crate::preprocess::TopdownTransform;
use candle_core::{DType, Result, Tensor};
use serde::Serialize;

pub const SIMCC_SPLIT_RATIO: f32 = 2.0;

#[derive(Clone, Debug, Serialize)]
pub struct PoseKeypoint {
    pub index: usize,
    pub name: &'static str,
    pub x: f32,
    pub y: f32,
    /// Raw SimCC peak score. The official codec uses normalize=false, so this
    /// is not a probability and is not constrained to [0, 1].
    pub score: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct SimccJointAudit {
    pub index: usize,
    pub name: &'static str,
    pub argmax_x: usize,
    pub argmax_y: usize,
    pub peak_x: f32,
    pub peak_y: f32,
    pub raw_score: f32,
    pub edge_distance_x_bins: usize,
    pub edge_distance_y_bins: usize,
    pub exact_edge_peak: bool,
    pub near_edge_2bin: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct SimccAudit {
    pub simcc_x_shape: Vec<usize>,
    pub simcc_y_shape: Vec<usize>,
    pub batch_size: usize,
    pub joints_per_pose: usize,
    pub exact_edge_peak_count: usize,
    pub exact_edge_peak_pct: f32,
    pub near_edge_2bin_count: usize,
    pub near_edge_2bin_pct: f32,
    pub raw_score_min: f32,
    pub raw_score_max: f32,
    pub raw_score_mean: f32,
    /// Detailed per-joint audit for the first pose in the batch. The fix2
    /// batch benchmark intentionally replicates one ROI, so repeating all
    /// rows would add no information.
    pub first_pose_joints: Vec<SimccJointAudit>,
}

fn max_index_value(xs: &[f32]) -> (usize, f32) {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_i = i;
            best_v = v;
        }
    }
    (best_i, best_v)
}

fn edge_distance(index: usize, len: usize) -> usize {
    index.min(len.saturating_sub(1).saturating_sub(index))
}

/// Audit raw SimCC distributions for shape, score range and border saturation.
pub fn audit_simcc(simcc_x: &Tensor, simcc_y: &Tensor) -> Result<SimccAudit> {
    let x = simcc_x.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let y = simcc_y.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    if x.len() != y.len() {
        candle_core::bail!("SimCC batch mismatch: x={} y={}", x.len(), y.len());
    }
    if x.is_empty() {
        candle_core::bail!("SimCC audit requires a non-empty batch");
    }

    let mut exact_edge_peak_count = 0usize;
    let mut near_edge_2bin_count = 0usize;
    let mut score_min = f32::INFINITY;
    let mut score_max = f32::NEG_INFINITY;
    let mut score_sum = 0.0f64;
    let mut score_count = 0usize;
    let mut first_pose_joints = Vec::with_capacity(NUM_KEYPOINTS);

    for b in 0..x.len() {
        if x[b].len() != NUM_KEYPOINTS || y[b].len() != NUM_KEYPOINTS {
            candle_core::bail!("expected {NUM_KEYPOINTS} Halpe26 joints");
        }
        for k in 0..NUM_KEYPOINTS {
            let (ix, vx) = max_index_value(&x[b][k]);
            let (iy, vy) = max_index_value(&y[b][k]);
            let dx = edge_distance(ix, x[b][k].len());
            let dy = edge_distance(iy, y[b][k].len());
            let exact = dx == 0 || dy == 0;
            let near = dx <= 2 || dy <= 2;
            let score = vx.min(vy);

            exact_edge_peak_count += if exact { 1 } else { 0 };
            near_edge_2bin_count += if near { 1 } else { 0 };
            score_min = score_min.min(score);
            score_max = score_max.max(score);
            score_sum += score as f64;
            score_count += 1;

            if b == 0 {
                first_pose_joints.push(SimccJointAudit {
                    index: k,
                    name: NAMES[k],
                    argmax_x: ix,
                    argmax_y: iy,
                    peak_x: vx,
                    peak_y: vy,
                    raw_score: score,
                    edge_distance_x_bins: dx,
                    edge_distance_y_bins: dy,
                    exact_edge_peak: exact,
                    near_edge_2bin: near,
                });
            }
        }
    }

    let total = score_count.max(1);
    Ok(SimccAudit {
        simcc_x_shape: simcc_x.dims().to_vec(),
        simcc_y_shape: simcc_y.dims().to_vec(),
        batch_size: x.len(),
        joints_per_pose: NUM_KEYPOINTS,
        exact_edge_peak_count,
        exact_edge_peak_pct: exact_edge_peak_count as f32 * 100.0 / total as f32,
        near_edge_2bin_count,
        near_edge_2bin_pct: near_edge_2bin_count as f32 * 100.0 / total as f32,
        raw_score_min: score_min,
        raw_score_max: score_max,
        raw_score_mean: (score_sum / total as f64) as f32,
        first_pose_joints,
    })
}

/// Decode RTMPose SimCC outputs and map them back through each top-down ROI transform.
pub fn decode_simcc(
    simcc_x: &Tensor,
    simcc_y: &Tensor,
    transforms: &[TopdownTransform],
) -> Result<Vec<Vec<PoseKeypoint>>> {
    let x = simcc_x.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let y = simcc_y.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    if x.len() != y.len() || x.len() != transforms.len() {
        candle_core::bail!(
            "batch/transform mismatch: x={} y={} transforms={}",
            x.len(),
            y.len(),
            transforms.len()
        );
    }
    let mut all = Vec::with_capacity(x.len());
    for b in 0..x.len() {
        if x[b].len() != NUM_KEYPOINTS || y[b].len() != NUM_KEYPOINTS {
            candle_core::bail!("expected {NUM_KEYPOINTS} Halpe26 joints");
        }
        let mut pose = Vec::with_capacity(NUM_KEYPOINTS);
        for k in 0..NUM_KEYPOINTS {
            let (ix, vx) = max_index_value(&x[b][k]);
            let (iy, vy) = max_index_value(&y[b][k]);
            let score = vx.min(vy);
            let (px, py) = if score <= 0.0 {
                (-1.0, -1.0)
            } else {
                let mx = ix as f32 / SIMCC_SPLIT_RATIO;
                let my = iy as f32 / SIMCC_SPLIT_RATIO;
                transforms[b].model_to_image(mx, my)
            };
            pose.push(PoseKeypoint {
                index: k,
                name: NAMES[k],
                x: px,
                y: py,
                score,
            });
        }
        all.push(pose);
    }
    Ok(all)
}
