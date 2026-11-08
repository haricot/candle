use candle_core::quantized::fp8_sm61_dispatch::{
    dispatch_residence_for_activations, Fp8Activation, Fp8Backend, Fp8DispatchBudget,
    Fp8LinearSpec, Fp8MeasuredProfile,
};

fn spec() -> Fp8LinearSpec {
    Fp8LinearSpec {
        out_features: 5,
        in_features: 128,
        block_n: 128,
        block_k: 128,
        resident_calls: 100,
        profile: Some(Fp8MeasuredProfile {
            simt_ms: 8.,
            expanded_f16_ms: 1.,
            extra_expand_ms: 0.,
        }),
    }
}

fn budget() -> Fp8DispatchBudget {
    Fp8DispatchBudget {
        free_device_bytes: 1 << 20,
        workload_reserve_bytes: 0,
        safety_headroom_bytes: 0,
        min_steady_speedup: 1.03,
        allow_auto_f16: true,
    }
}

#[test]
fn bf16_eligibility_precedes_profiles_and_mixed_residency_admission() {
    let plan = dispatch_residence_for_activations(
        &[spec(), spec()],
        &[Fp8Activation::Bf16, Fp8Activation::F16],
        budget(),
    )
    .unwrap();
    assert_eq!(plan.entries[0].backend, Fp8Backend::SimtW8A16);
    assert_eq!(plan.entries[1].backend, Fp8Backend::ExpandedF16);
    assert_eq!(plan.expanded_f16_count, 1);
    assert_eq!(plan.selected_total_bytes, 5 * 128 + 4 + 5 * 128 * 2);
}

#[test]
fn contracts_and_model_wide_budget_are_checked() {
    assert!(dispatch_residence_for_activations(&[spec()], &[], budget()).is_err());
    assert!(dispatch_residence_for_activations(
        &[spec()],
        &[Fp8Activation::Bf16],
        Fp8DispatchBudget {
            free_device_bytes: 643,
            ..budget()
        },
    )
    .is_err());
}

#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
#[test]
#[ignore = "requires physical SM61; hosted CI compiles this contract only"]
fn physical_w8abf16_preserves_range_residency_and_prepared_dtype() -> candle_core::Result<()> {
    use candle_core::quantized::fp8_linear_sm61::Fp8Sm61Linear;
    use candle_core::{DType, Device, Tensor};
    let device = Device::new_cuda(0)?;
    let cuda = device.as_cuda_device()?;
    assert_eq!(
        cuda.cuda_stream()
            .context()
            .compute_capability()
            .map_err(candle_core::Error::msg)?,
        (6, 1)
    );
    let plan = dispatch_residence_for_activations(&[spec()], &[Fp8Activation::Bf16], budget())
        .map_err(candle_core::Error::msg)?;
    let linear = Fp8Sm61Linear::from_bytes_with_activation_dispatch(
        cuda,
        5,
        128,
        &vec![0x38; 5 * 128],
        &[1.],
        128,
        128,
        &plan.entries[0],
        Fp8Activation::Bf16,
    )?;
    assert_eq!(linear.resident_weight_bytes(), 5 * 128 + 4);
    let expanded = dispatch_residence_for_activations(&[spec()], &[Fp8Activation::F16], budget())
        .map_err(candle_core::Error::msg)?;
    assert!(Fp8Sm61Linear::from_bytes_with_activation_dispatch(
        cuda,
        5,
        128,
        &vec![0x38; 5 * 128],
        &[1.],
        128,
        128,
        &expanded.entries[0],
        Fp8Activation::Bf16,
    )
    .is_err());
    for rows in [1, 4, 5] {
        let x = Tensor::from_vec(vec![1e8f32; rows * 128], (rows, 128), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_device(&device)?;
        let rounded = x
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?[0][0];
        let y = linear.forward(&x)?;
        assert_eq!(y.dtype(), DType::F32);
        for value in y.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()? {
            let expected = rounded * 128.;
            assert!(value.is_finite() && (value - expected).abs() <= expected.abs() * 1e-5);
            assert!(value > 65504.);
        }
        assert!(linear
            .forward(&Tensor::zeros((rows, 128), DType::F16, &device)?)
            .is_err());
    }
    device.synchronize()?;
    println!("FP8_TYPED_W8ABF16_SM61=PASS");
    Ok(())
}
