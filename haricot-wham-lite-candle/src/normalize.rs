use crate::types::Pose17Frame;

#[derive(Clone, Debug)]
pub struct NormalizedPose17 {
    /// WHAM Stage-1 input: 17 * (x,y) + normalized bbox center/scale = 37 values.
    pub values: [f32; 37],
    /// 1.0 means the keypoint should be treated as masked/missing.
    pub mask: [f32; 17],
}

pub fn normalize_wham17(frame: &Pose17Frame, confidence_threshold: f32) -> NormalizedPose17 {
    let width = frame.image_width.max(1) as f32;
    let height = frame.image_height.max(1) as f32;
    let max_res = width.max(height);

    let cx = 0.5 * (frame.bbox.x1 + frame.bbox.x2);
    let cy = 0.5 * (frame.bbox.y1 + frame.bbox.y2);
    let bbox_w = (frame.bbox.x2 - frame.bbox.x1).abs();
    let bbox_h = (frame.bbox.y2 - frame.bbox.y1).abs();

    // Matches WHAM inference's bbox expansion closely: derive the person square,
    // then use a 1.2 safety factor for the normalized pose patch.
    let box_size = bbox_w.max(bbox_h).max(1.0) * 1.2;

    let mut values = [0.0_f32; 37];
    let mut mask = [0.0_f32; 17];

    for (i, kp) in frame.keypoints.iter().enumerate() {
        values[2 * i] = 2.0 * (kp.x - cx) / box_size;
        values[2 * i + 1] = 2.0 * (kp.y - cy) / box_size;
        if kp.confidence < confidence_threshold {
            mask[i] = 1.0;
        }
    }

    values[34] = 2.0 * cx / max_res - width / max_res;
    values[35] = 2.0 * cy / max_res - height / max_res;
    values[36] = box_size / max_res;

    NormalizedPose17 { values, mask }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BBoxXyxy, Keypoint2D, Pose17Frame};

    #[test]
    fn centered_keypoint_maps_near_zero() {
        let kp = Keypoint2D {
            x: 320.0,
            y: 240.0,
            confidence: 1.0,
        };
        let frame = Pose17Frame {
            keypoints: [kp; 17],
            bbox: BBoxXyxy {
                x1: 220.0,
                y1: 140.0,
                x2: 420.0,
                y2: 340.0,
            },
            image_width: 640,
            image_height: 480,
        };
        let out = normalize_wham17(&frame, 0.3);
        assert!(out.values[0].abs() < 1e-6);
        assert!(out.values[1].abs() < 1e-6);
    }
}
