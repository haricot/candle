use candle_core::quantized::fp8_sm61_dispatch::{
    dispatch_uniform_residence, required_bytes, Fp8Backend, Fp8DispatchBudget, Fp8LinearSpec,
    Fp8WorkloadProfile,
};
use candle_core::quantized::fp8_sm61_loader::{
    inspect_safetensors, Fp8DispatchStrategy, Fp8LoadRequest,
};
use std::{collections::BTreeMap, path::PathBuf};

fn shards() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    vec![
        dir.join("generic_shard_a.safetensors"),
        dir.join("generic_shard_b.safetensors"),
    ]
}

fn budget(bytes: usize, allow: bool) -> Fp8DispatchBudget {
    Fp8DispatchBudget {
        free_device_bytes: bytes,
        workload_reserve_bytes: 4096,
        safety_headroom_bytes: 4096,
        min_steady_speedup: 1.03,
        allow_auto_f16: allow,
    }
}

#[test]
fn synthetic_shards_independent_of_architecture() {
    let rows = inspect_safetensors(&shards()).unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows.iter()
            .map(|r| r.scale_dtype.as_str())
            .collect::<Vec<_>>(),
        vec!["BF16", "F16", "F32"]
    );
    assert!(rows
        .iter()
        .all(|r| r.out_features == 16 && r.in_features == 128));
    assert!(rows.iter().all(|r| r.block_n == 128 && r.block_k == 128));
    assert_eq!(rows[0].identifier, "alpha.0");
    assert_eq!(rows[1].identifier, "beta.linear");
    assert_eq!(rows[2].identifier, "gamma.projection");
}

#[test]
fn synthetic_uniform_dispatch_respects_memory_and_residence() {
    let rows = inspect_safetensors(&shards()).unwrap();
    let specs: Vec<_> = rows
        .iter()
        .map(|r| Fp8LinearSpec {
            out_features: r.out_features,
            in_features: r.in_features,
            block_n: r.block_n,
            block_k: r.block_k,
            resident_calls: 50,
            profile: None,
        })
        .collect();
    let total: usize = specs.iter().map(|s| required_bytes(s).unwrap().0).sum();
    let expanded: usize = specs.iter().map(|s| required_bytes(s).unwrap().1).sum();
    let profile = Some(Fp8WorkloadProfile {
        simt_request_ms: 20.0,
        expanded_request_ms: 10.0,
        extra_initialization_ms: 100.0,
        planned_requests_while_resident: 50,
    });
    let simt = dispatch_uniform_residence(&specs, budget(total + 8192, false), profile).unwrap();
    assert_eq!(simt.expanded_f16_count, 0);
    assert_eq!(simt.selected_total_bytes, total);
    let low = dispatch_uniform_residence(&specs, budget(total + 8192, true), profile).unwrap();
    assert_eq!(low.expanded_f16_count, 0);
    let high = dispatch_uniform_residence(&specs, budget(expanded + 8192, true), profile).unwrap();
    assert_eq!(high.expanded_f16_count, 3);
    assert!(high
        .entries
        .iter()
        .all(|e| e.backend == Fp8Backend::ExpandedF16));
    let short = dispatch_uniform_residence(
        &specs,
        budget(expanded + 8192, true),
        Some(Fp8WorkloadProfile {
            planned_requests_while_resident: 1,
            ..profile.unwrap()
        }),
    )
    .unwrap();
    assert_eq!(short.expanded_f16_count, 0);
}

#[test]
fn request_defaults_without_profiles_to_simt() {
    let request = Fp8LoadRequest {
        budget: budget(64 << 20, true),
        strategy: Fp8DispatchStrategy::Uniform(None),
        expected_calls_per_projection: 100,
        geometry_profiles: BTreeMap::new(),
    };
    assert!(matches!(
        request.strategy,
        Fp8DispatchStrategy::Uniform(None)
    ));
}

#[test]
#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
fn synthetic_generic_cuda_dispatch_roundtrip() -> candle_core::Result<()> {
    use candle_core::quantized::fp8_linear_sm61::Fp8Sm61Mode;
    use candle_core::quantized::fp8_sm61_loader::load_cuda;
    use candle_core::{DType, Device, Tensor};
    let device = Device::new_cuda(0)?;
    let cuda = match &device {
        Device::Cuda(d) => d,
        _ => unreachable!(),
    };
    let paths = shards();
    let inputs = (0..8 * 128)
        .map(|i| ((i % 53) as f32 - 26.0) / 19.0)
        .collect::<Vec<_>>();
    let x = Tensor::from_vec(inputs, (8, 128), &device)?.to_dtype(DType::F16)?;
    let simt_req = Fp8LoadRequest {
        budget: budget(64 << 20, false),
        strategy: Fp8DispatchStrategy::Uniform(None),
        expected_calls_per_projection: 0,
        geometry_profiles: BTreeMap::new(),
    };
    let simt = load_cuda(&paths, cuda, &simt_req)?;
    assert_eq!(simt.projections.len(), 3);
    assert_eq!(simt.dispatch.expanded_f16_count, 0);
    assert!(simt
        .projections
        .iter()
        .all(|p| p.linear.mode() == Fp8Sm61Mode::SimtW8A16));
    assert_eq!(
        simt.realized_weight_bytes,
        simt.dispatch.selected_total_bytes
    );
    let mut reference = Vec::new();
    for proj in &simt.projections {
        reference.push((
            proj.identifier.clone(),
            proj.linear.forward(&x)?.to_vec2::<f32>()?,
        ));
    }
    drop(simt); // No concurrent FP8 + FP16 representations in this test.
    let expanded_req = Fp8LoadRequest {
        budget: budget(64 << 20, true),
        strategy: Fp8DispatchStrategy::Uniform(Some(Fp8WorkloadProfile {
            simt_request_ms: 20.0,
            expanded_request_ms: 10.0,
            extra_initialization_ms: 100.0,
            planned_requests_while_resident: 50,
        })),
        expected_calls_per_projection: 50,
        geometry_profiles: BTreeMap::new(),
    };
    let expanded = load_cuda(&paths, cuda, &expanded_req)?;
    assert_eq!(expanded.projections.len(), 3);
    assert_eq!(expanded.dispatch.expanded_f16_count, 3);
    assert_eq!(
        expanded.realized_weight_bytes,
        expanded.dispatch.selected_total_bytes
    );
    assert!(expanded
        .projections
        .iter()
        .all(|p| p.linear.mode() == Fp8Sm61Mode::ExpandedF16));
    for (name, ys) in reference {
        let zs = expanded.get(&name).unwrap().forward(&x)?.to_vec2::<f32>()?;
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (a, b) in ys.iter().flatten().zip(zs.iter().flatten()) {
            ab += f64::from(*a) * f64::from(*b);
            aa += f64::from(*a) * f64::from(*a);
            bb += f64::from(*b) * f64::from(*b);
        }
        let cos = ab / (aa * bb).sqrt();
        assert!(cos >= 0.9999, "synthetic linear cosine={cos} for {name}");
        eprintln!("F8_R4C_GENERIC_CUDA name={} cosine={:.9}", name, cos);
    }
    Ok(())
}

#[test]
fn synthetic_geometry_dispatch_ignores_names() -> candle_core::Result<()> {
    use candle_core::quantized::fp8_sm61_dispatch::{dispatch_residence, Fp8MeasuredProfile};
    let rows = inspect_safetensors(&shards())?;
    let profile = Fp8MeasuredProfile {
        simt_ms: 20.0,
        expanded_f16_ms: 8.0,
        extra_expand_ms: 10.0,
    };
    let mut req = Fp8LoadRequest {
        budget: budget(64 << 20, true),
        strategy: Fp8DispatchStrategy::ByGeometry,
        expected_calls_per_projection: 50,
        geometry_profiles: BTreeMap::new(),
    };
    req.geometry_profiles.insert((16, 128, 128, 128), profile);
    let specs: Vec<_> = rows
        .iter()
        .map(|r| {
            let key = (r.out_features, r.in_features, r.block_n, r.block_k);
            Fp8LinearSpec {
                out_features: r.out_features,
                in_features: r.in_features,
                block_n: r.block_n,
                block_k: r.block_k,
                resident_calls: req.expected_calls_per_projection,
                profile: req.geometry_profiles.get(&key).copied(),
            }
        })
        .collect();
    let selected = dispatch_residence(&specs, req.budget).unwrap();
    assert_eq!(selected.expanded_f16_count, 3);
    assert_eq!(selected.projection_count, 3);
    Ok(())
}
