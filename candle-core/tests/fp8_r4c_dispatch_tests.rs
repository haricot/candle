use candle_core::quantized::fp8_sm61_dispatch::{
    dispatch_residence, required_bytes, Fp8Backend, Fp8DispatchBudget, Fp8LinearSpec,
    Fp8MeasuredProfile,
};

#[test]
fn cpu_dispatch_is_architecture_agnostic() {
    let spec = Fp8LinearSpec {
        out_features: 1024,
        in_features: 1024,
        block_n: 128,
        block_k: 128,
        resident_calls: 40,
        profile: Some(Fp8MeasuredProfile {
            simt_ms: 8.0,
            expanded_f16_ms: 4.0,
            extra_expand_ms: 40.0,
        }),
    };
    let baseline = required_bytes(&spec).unwrap().0;
    let env = Fp8DispatchBudget {
        free_device_bytes: 10_000_000,
        workload_reserve_bytes: 2_000_000,
        safety_headroom_bytes: 2_000_000,
        min_steady_speedup: 1.03,
        allow_auto_f16: true,
    };
    let upgraded = dispatch_residence(&[spec], env).unwrap();
    assert_eq!(upgraded.entries[0].backend, Fp8Backend::ExpandedF16);
    assert!(upgraded.selected_total_bytes > baseline);
    let simt = dispatch_residence(
        &[Fp8LinearSpec {
            profile: None,
            ..spec
        }],
        env,
    )
    .unwrap();
    assert_eq!(simt.entries[0].backend, Fp8Backend::SimtW8A16);
}
