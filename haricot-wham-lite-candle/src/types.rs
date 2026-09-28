use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Keypoint2D {
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BBoxXyxy {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pose17Frame {
    pub keypoints: [Keypoint2D; 17],
    pub bbox: BBoxXyxy,
    pub image_width: u32,
    pub image_height: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pose26Frame {
    pub keypoints: [Keypoint2D; 26],
    pub bbox: BBoxXyxy,
    pub image_width: u32,
    pub image_height: u32,
}
