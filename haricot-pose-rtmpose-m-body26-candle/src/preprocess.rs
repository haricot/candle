use candle_core::{DType, Device, Result, Tensor};
use image::{Rgb, RgbImage};
use serde::{Deserialize, Serialize};

pub const INPUT_W: usize = 192;
pub const INPUT_H: usize = 256;
pub const BBOX_PADDING: f32 = 1.25;
const ASPECT: f32 = INPUT_W as f32 / INPUT_H as f32;
const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const STD: [f32; 3] = [58.395, 57.12, 57.375];

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BboxXyxy {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BboxXyxy {
    pub fn full_image(w: u32, h: u32) -> Self {
        Self {
            x1: 0.0,
            y1: 0.0,
            x2: w as f32,
            y2: h as f32,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SourceBounds {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TopdownTransform {
    pub center_x: f32,
    pub center_y: f32,
    pub scale_w: f32,
    pub scale_h: f32,
}

impl TopdownTransform {
    pub fn from_bbox(bbox: BboxXyxy) -> Self {
        let center_x = (bbox.x1 + bbox.x2) * 0.5;
        let center_y = (bbox.y1 + bbox.y2) * 0.5;
        let mut scale_w = (bbox.x2 - bbox.x1).max(1.0) * BBOX_PADDING;
        let mut scale_h = (bbox.y2 - bbox.y1).max(1.0) * BBOX_PADDING;
        if scale_w > scale_h * ASPECT {
            scale_h = scale_w / ASPECT;
        } else {
            scale_w = scale_h * ASPECT;
        }
        Self {
            center_x,
            center_y,
            scale_w,
            scale_h,
        }
    }

    pub fn source_bounds(&self) -> SourceBounds {
        SourceBounds {
            x1: self.center_x - self.scale_w * 0.5,
            y1: self.center_y - self.scale_h * 0.5,
            x2: self.center_x + self.scale_w * 0.5,
            y2: self.center_y + self.scale_h * 0.5,
        }
    }

    pub fn model_to_image(&self, x: f32, y: f32) -> (f32, f32) {
        (
            x / INPUT_W as f32 * self.scale_w + self.center_x - self.scale_w * 0.5,
            y / INPUT_H as f32 * self.scale_h + self.center_y - self.scale_h * 0.5,
        )
    }

    fn output_to_source(&self, x: f32, y: f32) -> (f32, f32) {
        let src_x =
            (x - INPUT_W as f32 * 0.5) * self.scale_w / INPUT_W as f32 + self.center_x;
        let src_y =
            (y - INPUT_H as f32 * 0.5) * self.scale_h / INPUT_H as f32 + self.center_y;
        (src_x, src_y)
    }
}

fn pixel_or_zero(img: &RgbImage, x: i32, y: i32, channel: usize) -> f32 {
    if x < 0 || y < 0 || x >= img.width() as i32 || y >= img.height() as i32 {
        0.0
    } else {
        img.get_pixel(x as u32, y as u32).0[channel] as f32
    }
}

fn bilinear(img: &RgbImage, x: f32, y: f32, channel: usize) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let p00 = pixel_or_zero(img, x0, y0, channel);
    let p10 = pixel_or_zero(img, x1, y0, channel);
    let p01 = pixel_or_zero(img, x0, y1, channel);
    let p11 = pixel_or_zero(img, x1, y1, channel);
    let a = p00 * (1.0 - dx) + p10 * dx;
    let b = p01 * (1.0 - dx) + p11 * dx;
    a * (1.0 - dy) + b * dy
}

/// Render the exact zero-rotation top-down geometry as an RGB debug image.
/// This is diagnostic only: model preprocessing still samples in f32 directly,
/// so PNG quantization never enters the numerical inference path.
pub fn render_topdown_rgb(image: &RgbImage, bbox: BboxXyxy) -> (RgbImage, TopdownTransform) {
    let transform = TopdownTransform::from_bbox(bbox);
    let mut out = RgbImage::new(INPUT_W as u32, INPUT_H as u32);
    for oy in 0..INPUT_H {
        for ox in 0..INPUT_W {
            let (sx, sy) = transform.output_to_source(ox as f32, oy as f32);
            let mut rgb = [0u8; 3];
            for c in 0..3 {
                rgb[c] = bilinear(image, sx, sy, c).round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(ox as u32, oy as u32, Rgb(rgb));
        }
    }
    (out, transform)
}

/// MMPose-compatible zero-rotation top-down affine path.
/// The image crate already returns RGB, matching PoseDataPreprocessor after `bgr_to_rgb=true`.
pub fn prepare_topdown_rgb(
    image: &RgbImage,
    bbox: BboxXyxy,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, TopdownTransform)> {
    let transform = TopdownTransform::from_bbox(bbox);
    let mut chw = vec![0f32; 3 * INPUT_H * INPUT_W];
    for oy in 0..INPUT_H {
        for ox in 0..INPUT_W {
            let (sx, sy) = transform.output_to_source(ox as f32, oy as f32);
            for c in 0..3 {
                let value = bilinear(image, sx, sy, c);
                chw[c * INPUT_H * INPUT_W + oy * INPUT_W + ox] =
                    (value - MEAN[c]) / STD[c];
            }
        }
    }
    let tensor =
        Tensor::from_vec(chw, (1, 3, INPUT_H, INPUT_W), device)?.to_dtype(dtype)?;
    Ok((tensor, transform))
}
