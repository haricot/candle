//! Check the real ModelOpt reader on a CPU, without enabling CUDA kernels.
use candle_core::quantized::nvfp4_e7c_safetensors::ModelOptNvfp4Shards;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn fixture(scale_dtype: Dtype, scale: u8, global: f32) -> Fixture {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "candle-nvfp4-{}-{}.safetensors",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let packed = (0u8..32).collect::<Vec<_>>();
    let scales = [scale; 4];
    let global = global.to_le_bytes();
    let tensors = vec![
        (
            "op.weight",
            TensorView::new(Dtype::U8, vec![2, 16], &packed).unwrap(),
        ),
        (
            "op.weight_scale",
            TensorView::new(scale_dtype, vec![2, 2], &scales).unwrap(),
        ),
        (
            "op.weight_scale_2",
            TensorView::new(Dtype::F32, vec![], &global).unwrap(),
        ),
    ];
    serialize_to_file(tensors, None, &path).unwrap();
    Fixture(path)
}

#[test]
fn modelopt_loader_preserves_packed_bytes_and_scale_convention() {
    let file = fixture(Dtype::F8_E4M3, 0x38, 2.0);
    let (packed, scales, global) = ModelOptNvfp4Shards::single_file(&file.0)
        .load(2, 32, "op")
        .unwrap();
    assert_eq!(packed, (0u8..32).collect::<Vec<_>>());
    assert_eq!(scales, [0x38; 4]);
    assert_eq!(global, 2.0);
    assert!(ModelOptNvfp4Shards::single_file(&file.0)
        .load(2, 64, "op")
        .is_err());
    assert!(ModelOptNvfp4Shards::single_file(&file.0)
        .load(2, 32, "missing")
        .is_err());
}

#[test]
fn modelopt_loader_rejects_mxfp4_scale_dtype_and_nonfinite_scales() {
    for (dtype, scale, global) in [
        (Dtype::U8, 127, 1.0), // E8M0 is not a ModelOpt E4M3FN scale.
        (Dtype::F8_E4M3, 0xb8, 1.0),
        (Dtype::F8_E4M3, 0x7f, 1.0),
        (Dtype::F8_E4M3, 0x38, f32::INFINITY),
        (Dtype::F8_E4M3, 0x38, 0.0),
    ] {
        let file = fixture(dtype, scale, global);
        assert!(ModelOptNvfp4Shards::single_file(&file.0)
            .load(2, 32, "op")
            .is_err());
    }
}
