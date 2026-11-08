#![cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
//! Generic model-independent native CUDA residency evidence and forward test.
//! These two synthetic Safetensors shards contain arbitrary projection labels.
use candle_core::quantized::fp8_sm61_dispatch::{Fp8Backend, Fp8WorkloadProfile};
use candle_core::quantized::fp8_sm61_loader::{
    load_cuda_native, sample_cuda_memory, Fp8DispatchStrategy, Fp8NativeDispatchConfig,
    Fp8NativeResidentModel,
};
use candle_core::{DType, Device, Result, Tensor};
use std::path::PathBuf;
fn shards() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    vec![
        dir.join("generic_shard_a.safetensors"),
        dir.join("generic_shard_b.safetensors"),
    ]
}
fn simt_config() -> Fp8NativeDispatchConfig {
    Fp8NativeDispatchConfig {
        safety_headroom_bytes: 1024 * 1024,
        workload_reserve_bytes: 1024 * 1024,
        ..Default::default()
    }
}
fn expanded_config() -> Fp8NativeDispatchConfig {
    Fp8NativeDispatchConfig {
        allow_auto_f16: true,
        strategy: Fp8DispatchStrategy::Uniform(Some(Fp8WorkloadProfile {
            simt_request_ms: 20.0,
            expanded_request_ms: 5.0,
            extra_initialization_ms: 10.0,
            planned_requests_while_resident: 50,
        })),
        expected_calls_per_projection: 50,
        ..simt_config()
    }
}
fn check(model: &Fp8NativeResidentModel, expect_f16: bool) {
    let e = &model.evidence;
    assert_eq!(model.weights.projections.len(), 3);
    assert!(e.single_representation_per_projection);
    assert_eq!(e.accounted_weight_bytes, e.planned_weight_bytes);
    assert!(e.accounted_weight_bytes > 0);
    assert_eq!(e.model_wide_f16_projections, if expect_f16 { 3 } else { 0 });
    assert_eq!(
        e.model_wide_simt_projections,
        if expect_f16 { 0 } else { 3 }
    );
    assert_eq!(e.before.total_bytes, e.after.total_bytes);
    assert!(e.before.total_bytes >= e.before.free_bytes);
    assert!(e.after.total_bytes >= e.after.free_bytes);
    assert!(e.after.free_bytes >= e.requested_reserve_bytes + e.requested_headroom_bytes);
    for (row, selection) in model
        .weights
        .projections
        .iter()
        .zip(&model.weights.dispatch.entries)
    {
        assert_eq!(row.linear.resident_weight_bytes(), selection.resident_bytes);
        assert!(matches!(selection.backend, Fp8Backend::ExpandedF16) == expect_f16);
    }
}
fn outputs(model: &Fp8NativeResidentModel, x: &Tensor) -> Result<Vec<Vec<f32>>> {
    let mut ys = Vec::new();
    for name in ["alpha.0", "beta.linear", "gamma.projection"] {
        let y = model.forward(name, x)?.to_dtype(DType::F32)?;
        ys.push(y.flatten_all()?.to_vec1::<f32>()?);
    }
    Ok(ys)
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa == 0.0 && bb == 0.0 {
        return 1.0;
    }
    dot / (aa * bb).sqrt()
}
#[test]
fn native_multi_shard_dispatch_and_cuda_residence() -> Result<()> {
    let d = Device::new_cuda(0)?;
    let cuda = match &d {
        Device::Cuda(dev) => dev,
        _ => unreachable!(),
    };
    let x = Tensor::from_vec(
        (0..8 * 128)
            .map(|i| ((i % 53) as f32 - 26.0) / 19.0)
            .collect::<Vec<_>>(),
        (8, 128),
        &d,
    )?
    .to_dtype(DType::F16)?;
    let initial = sample_cuda_memory(cuda)?;
    let simt = load_cuda_native(&shards(), cuda, &simt_config())?;
    check(&simt, false);
    let simt_y = outputs(&simt, &x)?;
    let simt_e = simt.evidence.clone();
    drop(simt);
    let after_simt_drop = sample_cuda_memory(cuda)?;
    let expanded = load_cuda_native(&shards(), cuda, &expanded_config())?;
    check(&expanded, true);
    let f16_y = outputs(&expanded, &x)?;
    let f16_e = expanded.evidence.clone();
    let mut min_cosine = 1.0f64;
    for (a, b) in simt_y.iter().zip(f16_y.iter()) {
        min_cosine = min_cosine.min(cosine(a, b));
    }
    assert!(min_cosine >= 0.9999, "native SIMT/FP16 cosine {min_cosine}");
    drop(expanded);
    let final_sample = sample_cuda_memory(cuda)?;
    assert_eq!(initial.total_bytes, final_sample.total_bytes);
    // Driver readings can include independent applications and allocator caches.
    // We deliberately do NOT assert observed delta == logical bytes.
    println!("F8_R4D_NATIVE_EVIDENCE_JSON={{\"status\":\"PASS\",\"kind\":\"cuda_driver_samples\",\"synthetic_model_independent\":true,\"projection_count\":3,\"single_representation\":true,\"simt_accounted_bytes\":{},\"f16_accounted_bytes\":{},\"simt_delta_device_bytes\":{},\"f16_delta_device_bytes\":{},\"simt_before_free_bytes\":{},\"simt_after_free_bytes\":{},\"f16_before_free_bytes\":{},\"f16_after_free_bytes\":{},\"post_drop_simt_free_bytes\":{},\"final_free_bytes\":{},\"min_cosine\":{:.10},\"memory_scope\":\"device_wide_boundary_samples_not_peak\"}}",
        simt_e.accounted_weight_bytes,f16_e.accounted_weight_bytes,
        simt_e.observed_driver_used_delta_bytes,f16_e.observed_driver_used_delta_bytes,
        simt_e.before.free_bytes,simt_e.after.free_bytes,
        f16_e.before.free_bytes,f16_e.after.free_bytes,
        after_simt_drop.free_bytes,final_sample.free_bytes,min_cosine);
    Ok(())
}
#[test]
fn native_preflight_rejects_oversubscribed_reserves() -> Result<()> {
    let d = Device::new_cuda(0)?;
    let cuda = match &d {
        Device::Cuda(dev) => dev,
        _ => unreachable!(),
    };
    let snap = sample_cuda_memory(cuda)?;
    let mut req = simt_config();
    req.workload_reserve_bytes = snap.free_bytes;
    assert!(load_cuda_native(&shards(), cuda, &req).is_err());
    Ok(())
}
