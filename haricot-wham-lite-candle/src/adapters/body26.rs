use crate::types::{Pose17Frame, Pose26Frame};

/// Halpe-26 keeps the canonical COCO-17 joints at indices 0..=16.
/// Therefore the exact WHAM-reference adapter is just the first 17 joints.
/// The additional head/neck/mid-hip/toe/heel joints are intentionally preserved
/// for a later contact/ground-refinement stage, not consumed by the V0 network.
pub fn to_coco17(frame: &Pose26Frame) -> Pose17Frame {
    let keypoints = std::array::from_fn(|i| frame.keypoints[i]);
    Pose17Frame {
        keypoints,
        bbox: frame.bbox,
        image_width: frame.image_width,
        image_height: frame.image_height,
    }
}
