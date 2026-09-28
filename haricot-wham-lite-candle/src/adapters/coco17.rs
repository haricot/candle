use crate::types::Pose17Frame;

/// YOLO pose and WHAM's original ViTPose front-end both use COCO-17 ordering:
/// nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
pub fn identity(frame: Pose17Frame) -> Pose17Frame {
    frame
}
