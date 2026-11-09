//! Identical public Candle F32/CPU contract on official upstream, fp8_candle, extended.
//! This test intentionally imports NO fork-only APIs and requires no CUDA.
use candle_core::{Device, Result, Tensor};

#[test]
fn fp8_r2_upstream_cpu_canary() -> Result<()> {
    let device = Device::Cpu;
    let a: Vec<f32> = (0..64)
        .map(|i| (((i * 17 + 3) % 101) as f32 - 50.0) / 37.0)
        .collect();
    let b: Vec<f32> = (0..80)
        .map(|i| (((i * 13 + 7) % 89) as f32 - 44.0) / 41.0)
        .collect();
    let x = Tensor::from_vec(a, (8, 8), &device)?;
    let w = Tensor::from_vec(b, (10, 8), &device)?;
    let y = x.matmul(&w.t()?)?.affine(0.75, 0.125)?;
    let rows = y.to_vec2::<f32>()?;
    let flattened: Vec<String> = rows.iter().flatten().map(|v| format!("{v:.8}")).collect();
    assert_eq!(flattened.len(), 80);
    println!("F8_R2_UPSTREAM_CANARY=[{}]", flattened.join(","));
    Ok(())
}
