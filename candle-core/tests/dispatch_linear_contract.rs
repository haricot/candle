use candle_core::dispatch::contract::*;
use candle_core::dispatch::providers::*;
use candle_core::dispatch::{
    Architecture, Bf16LinearBackend, DeviceCapabilities, LinearMemory, ResourceBudget,
};
use candle_core::{DType, Device, Result, Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};

fn budget() -> ResourceBudget {
    ResourceBudget {
        resident_bytes: 1 << 20,
        workspace_bytes: 1 << 20,
    }
}
fn cuda(major: i32, minor: i32) -> DeviceCapabilities {
    DeviceCapabilities {
        architecture: Architecture::Cuda { major, minor },
    }
}
fn request(weights: WeightFormat, activations: DType, output: DType) -> LinearRequest {
    LinearRequest {
        contract: LinearContract {
            weights,
            activations,
            accumulation: DType::F32,
            output,
            bias: BiasSemantics::None,
            approximation: Approximation::PreserveStoredValues,
        },
        in_features: 128,
        out_features: 5,
        max_rows: 5,
        input_layout: InputLayout::ContiguousZeroOffset,
    }
}
fn bf16() -> LinearRequest {
    request(WeightFormat::Bf16, DType::BF16, DType::BF16)
}
fn fp8(dtype: DType) -> LinearRequest {
    request(
        WeightFormat::Fp8E4m3Fn {
            block_n: 128,
            block_k: 128,
        },
        dtype,
        DType::F32,
    )
}
fn nvfp4() -> LinearRequest {
    request(WeightFormat::Nvfp4ModelOpt, DType::F16, DType::F32)
}
fn declared() -> [BuiltinLinearProvider; 5] {
    [
        BuiltinLinearProvider::Bf16 {
            backend: Bf16LinearBackend::CudaBf16,
            available: true,
        },
        BuiltinLinearProvider::Bf16 {
            backend: Bf16LinearBackend::CudaF32,
            available: true,
        },
        BuiltinLinearProvider::Bf16 {
            backend: Bf16LinearBackend::CpuF32,
            available: true,
        },
        BuiltinLinearProvider::Fp8Packed { available: true },
        BuiltinLinearProvider::Nvfp4W4A16 { available: true },
    ]
}
fn plan(
    r: LinearRequest,
    caps: DeviceCapabilities,
    b: ResourceBudget,
    p: Option<&dyn LinearPreference>,
) -> Result<LinearSelection> {
    let providers = declared();
    let refs: Vec<&dyn LinearProvider> =
        providers.iter().map(|x| x as &dyn LinearProvider).collect();
    select_linear(r, caps, b, &refs, p)
}
struct Require(ProviderId);
impl LinearPreference for Require {
    fn select(&self, _: &LinearRequest, eligible: &[ProviderAssessment]) -> Option<ProviderId> {
        assert!(eligible
            .iter()
            .all(|c| c.rejection.is_none() && c.memory.is_some()));
        Some(self.0)
    }
}

#[test]
fn common_selection_preserves_existing_bf16_architecture_rules() -> Result<()> {
    for (major, minor) in [(6, 0), (6, 1), (6, 2), (7, 0), (7, 5), (999, 0)] {
        let p = plan(bf16(), cuda(major, minor), budget(), None)?;
        assert_eq!(p.selected.provider, BF16_CUDA_F32);
        assert!(plan(
            bf16(),
            cuda(major, minor),
            budget(),
            Some(&Require(BF16_CUDA_NATIVE))
        )
        .is_err());
        assert!(plan(
            bf16(),
            cuda(major, minor),
            budget(),
            Some(&Require(BF16_CPU_F32))
        )
        .is_err());
    }
    assert_eq!(
        plan(bf16(), cuda(8, 0), budget(), None)?.selected.provider,
        BF16_CUDA_NATIVE
    );
    assert_eq!(
        plan(bf16(), cuda(8, 0), budget(), Some(&Require(BF16_CUDA_F32)))?
            .selected
            .provider,
        BF16_CUDA_F32
    );
    Ok(())
}

#[test]
fn packed_fp8_contract_retains_bf16_range_and_existing_residency() -> Result<()> {
    use candle_core::quantized::fp8_sm61_dispatch::*;
    for dtype in [DType::BF16, DType::F16] {
        let p = plan(fp8(dtype), cuda(6, 1), budget(), None)?;
        assert_eq!(p.selected.provider, FP8_CUDA_PACKED);
        let old = dispatch_residence_for_activations(
            &[Fp8LinearSpec {
                out_features: 5,
                in_features: 128,
                block_n: 128,
                block_k: 128,
                resident_calls: 0,
                profile: None,
            }],
            &[if dtype == DType::BF16 {
                Fp8Activation::Bf16
            } else {
                Fp8Activation::F16
            }],
            Fp8DispatchBudget {
                free_device_bytes: budget().resident_bytes,
                ..Default::default()
            },
        )
        .map_err(candle_core::Error::msg)?;
        assert_eq!(
            p.selected.memory.unwrap().resident_bytes,
            old.selected_total_bytes
        );
        assert_eq!(p.selected.memory.unwrap().workspace_bytes, 5 * 5 * 4);
    }
    for caps in [cuda(6, 0), cuda(6, 2), cuda(7, 0), cuda(8, 0), cuda(999, 0)] {
        assert!(plan(fp8(DType::BF16), caps, budget(), None).is_err());
    }
    Ok(())
}

#[test]
fn formats_and_numerical_contracts_cannot_be_reinterpreted() -> Result<()> {
    assert_eq!(
        plan(nvfp4(), cuda(6, 1), budget(), None)?
            .selected
            .memory
            .unwrap()
            .resident_bytes,
        5 * 128 / 2 + 5 * 128 / 16
    );
    assert!(plan(
        request(WeightFormat::Mxfp4Gguf, DType::F16, DType::F32),
        cuda(6, 1),
        budget(),
        None
    )
    .is_err());
    assert!(plan(
        request(WeightFormat::F16, DType::F16, DType::F32),
        cuda(6, 1),
        budget(),
        None
    )
    .is_err());
    for mut r in [bf16(), fp8(DType::BF16), nvfp4()] {
        r.contract.accumulation = DType::F16;
        assert!(plan(r, cuda(6, 1), budget(), None).is_err());
    }
    let mut r = nvfp4();
    r.contract.activations = DType::BF16;
    assert!(plan(r, cuda(6, 1), budget(), None).is_err());
    let mut r = fp8(DType::BF16);
    r.contract.output = DType::F16;
    assert!(plan(r, cuda(6, 1), budget(), None).is_err());
    let mut r = bf16();
    r.contract.bias = BiasSemantics::AfterAccumulation;
    assert!(plan(r, cuda(6, 1), budget(), None).is_err());
    Ok(())
}

#[test]
fn layout_geometry_and_budgets_precede_preference() -> Result<()> {
    for mut r in [fp8(DType::F16), nvfp4()] {
        let m = plan(r, cuda(6, 1), budget(), None)?
            .selected
            .memory
            .unwrap();
        assert!(plan(
            r,
            cuda(6, 1),
            ResourceBudget {
                resident_bytes: m.resident_bytes - 1,
                ..budget()
            },
            Some(&Require(FP8_CUDA_PACKED))
        )
        .is_err());
        assert!(plan(
            r,
            cuda(6, 1),
            ResourceBudget {
                workspace_bytes: m.workspace_bytes - 1,
                ..budget()
            },
            None
        )
        .is_err());
        r.input_layout = InputLayout::Strided;
        assert!(plan(r, cuda(6, 1), budget(), None).is_err());
        r.input_layout = InputLayout::ContiguousZeroOffset;
        r.max_rows = 65536;
        assert!(plan(r, cuda(6, 1), budget(), None).is_err());
    }
    let mut r = fp8(DType::BF16);
    r.in_features = 127;
    assert!(plan(r, cuda(6, 1), budget(), None).is_err());
    let mut r = nvfp4();
    r.in_features = 33;
    assert!(plan(r, cuda(6, 1), budget(), None).is_err());
    let mut r = bf16();
    r.max_rows = 0;
    assert!(plan(r, cuda(6, 1), budget(), None).is_err());
    r.max_rows = usize::MAX;
    assert!(plan(r, cuda(6, 1), budget(), None).is_err());
    Ok(())
}

/// A new F16 planning adapter demonstrates the common contract has no BF16-only enum gate.
/// This fixture is not an advertised executable CUDA provider.
struct FutureF16;
impl LinearProvider for FutureF16 {
    fn id(&self) -> ProviderId {
        ProviderId("test.f16.cuda.v1")
    }
    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Cuda
    }
    fn assess(&self, r: &LinearRequest, _: DeviceCapabilities) -> Result<ProviderAssessment> {
        assert_eq!(r.contract.weights, WeightFormat::F16);
        Ok(ProviderAssessment::eligible(
            self.id(),
            LinearMemory {
                source_bytes: 1280,
                resident_bytes: 1280,
                workspace_bytes: 100,
            },
        ))
    }
}
#[test]
fn adding_f16_planning_provider_does_not_change_common_selector() -> Result<()> {
    let r = request(WeightFormat::F16, DType::F16, DType::F32);
    let p = select_linear(r, cuda(7, 5), budget(), &[&FutureF16], None)?;
    assert_eq!(p.selected.provider, FutureF16.id());
    assert!(select_linear(
        r,
        cuda(7, 5),
        ResourceBudget {
            resident_bytes: 1279,
            ..budget()
        },
        &[&FutureF16],
        None
    )
    .is_err());
    assert!(select_linear(
        r,
        DeviceCapabilities {
            architecture: Architecture::Cpu
        },
        budget(),
        &[&FutureF16],
        None
    )
    .is_err());
    assert!(select_linear(r, cuda(7, 5), budget(), &[&FutureF16, &FutureF16], None).is_err());
    assert!(select_linear(
        r,
        cuda(7, 5),
        budget(),
        &[&FutureF16],
        Some(&Require(BF16_CUDA_NATIVE))
    )
    .is_err());
    Ok(())
}

#[test]
fn unavailable_provider_does_not_invoke_preference() -> Result<()> {
    struct Count(AtomicUsize);
    impl LinearPreference for Count {
        fn select(&self, _: &LinearRequest, _: &[ProviderAssessment]) -> Option<ProviderId> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Some(FP8_CUDA_PACKED)
        }
    }
    let count = Count(AtomicUsize::new(0));
    let absent = BuiltinLinearProvider::Fp8Packed { available: false };
    assert!(select_linear(
        fp8(DType::BF16),
        cuda(6, 1),
        budget(),
        &[&absent],
        Some(&count)
    )
    .is_err());
    assert_eq!(count.0.load(Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn unified_cpu_execution_enforces_contract_and_keeps_bf16_bias_rounding() -> Result<()> {
    let device = Device::Cpu;
    let mut r = bf16();
    r.in_features = 2;
    r.out_features = 1;
    r.max_rows = 2;
    r.input_layout = InputLayout::Strided;
    r.contract.bias = BiasSemantics::AfterBf16GemmRounding;
    let w = Tensor::from_vec(vec![1.00390625f32, 0.0], (1, 2), &device)?.to_dtype(DType::BF16)?;
    let b = Tensor::from_vec(vec![0.00390625f32], 1, &device)?.to_dtype(DType::BF16)?;
    let layer = PreparedLinear::prepare(
        r,
        &device,
        LinearWeights::Bf16 {
            weight: &w,
            bias: Some(&b),
        },
        budget(),
        None,
    )?;
    assert_eq!(layer.selection().selected.provider, BF16_CPU_F32);
    let ctx = candle_core::dispatch::DispatchContext::new(&device, budget(), None)?;
    let old = ctx.prepare_bf16_linear(&w, Some(&b), 2)?;
    let x = Tensor::from_vec(vec![1e8f32, 2e8, 0.0, 0.0], (2, 2), &device)?
        .to_dtype(DType::BF16)?
        .t()?;
    assert!(!x.is_contiguous());
    assert_eq!(
        layer.forward(&x)?.to_dtype(DType::F32)?.to_vec2::<f32>()?,
        old.forward(&x)?.to_dtype(DType::F32)?.to_vec2::<f32>()?
    );
    assert!(layer.forward(&x.to_dtype(DType::F16)?).is_err());
    assert!(layer
        .forward(&Tensor::ones((3, 2), DType::BF16, &device)?)
        .is_err());
    assert!(PreparedLinear::prepare(
        r,
        &device,
        LinearWeights::Bf16 {
            weight: &w,
            bias: None
        },
        budget(),
        None
    )
    .is_err());
    assert!(PreparedLinear::prepare(
        r,
        &device,
        LinearWeights::Fp8E4m3Fn {
            packed: &[0; 2],
            scales: &[1.]
        },
        budget(),
        None
    )
    .is_err());
    r.input_layout = InputLayout::ContiguousZeroOffset;
    let strict = PreparedLinear::prepare(
        r,
        &device,
        LinearWeights::Bf16 {
            weight: &w,
            bias: Some(&b),
        },
        budget(),
        None,
    )?;
    assert!(strict.forward(&x).is_err());
    let offset = Tensor::ones((3, 2), DType::BF16, &device)?.narrow(0, 1, 1)?;
    assert!(strict.forward(&offset).is_err());
    Ok(())
}

#[cfg(all(
    feature = "cuda",
    feature = "cuda-legacy-bf16",
    feature = "cuda-legacy-fp8",
    feature = "cuda-legacy-fp4"
))]
#[test]
#[ignore = "requires physical SM61; hosted CI compiles only"]
fn physical_common_facade_keeps_three_numeric_contracts() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let w = Tensor::ones((5, 128), DType::BF16, &device)?;
    let p8 = vec![0x38; 5 * 128];
    let p4 = vec![0x22; 5 * 64];
    let s4 = vec![0x38; 5 * 8];
    let lbf16 = PreparedLinear::prepare(
        bf16(),
        &device,
        LinearWeights::Bf16 {
            weight: &w,
            bias: None,
        },
        budget(),
        None,
    )?;
    let lfp8 = PreparedLinear::prepare(
        fp8(DType::BF16),
        &device,
        LinearWeights::Fp8E4m3Fn {
            packed: &p8,
            scales: &[1.],
        },
        budget(),
        None,
    )?;
    let lfp4 = PreparedLinear::prepare(
        nvfp4(),
        &device,
        LinearWeights::Nvfp4ModelOpt {
            packed: &p4,
            scales_e4m3: &s4,
            multiplier: 1.,
        },
        budget(),
        None,
    )?;
    assert_eq!(lbf16.selection().selected.provider, BF16_CUDA_F32);
    assert_eq!(lfp8.selection().selected.provider, FP8_CUDA_PACKED);
    assert_eq!(lfp4.selection().selected.provider, NVFP4_CUDA_W4A16);
    for rows in [1, 4, 5] {
        let x = Tensor::full(1e8f32, (rows, 128), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_device(&device)?;
        let a16 = Tensor::ones((rows, 128), DType::F16, &device)?;
        let expected = x
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?[0][0]
            * 128.;
        for y in [lbf16.forward(&x)?.to_dtype(DType::F32)?, lfp8.forward(&x)?] {
            for row in y.to_device(&Device::Cpu)?.to_vec2::<f32>()? {
                for value in row {
                    assert!(
                        value.is_finite()
                            && value > 65504.
                            && (value - expected).abs() <= expected.abs() * 1e-5
                    );
                }
            }
        }
        assert_eq!(
            lfp4.forward(&a16)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?,
            vec![vec![128.; 5]; rows]
        );
        assert!(lfp8.forward(&a16).is_err());
        assert!(lfp4.forward(&x).is_err());
        let offset = Tensor::ones((rows + 1, 128), DType::F16, &device)?.narrow(0, 1, rows)?;
        assert!(lfp4.forward(&offset).is_err());
    }
    let lfp8a16 = PreparedLinear::prepare(
        fp8(DType::F16),
        &device,
        LinearWeights::Fp8E4m3Fn {
            packed: &p8,
            scales: &[1.],
        },
        budget(),
        None,
    )?;
    let a16 = Tensor::ones((4, 128), DType::F16, &device)?;
    assert_eq!(
        lfp8a16
            .forward(&a16)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?,
        vec![vec![128.; 5]; 4]
    );
    device.synchronize()?;
    println!("CANDLE_COMMON_LINEAR_SM61=PASS");
    Ok(())
}

#[test]
fn one_kernel_geometry_rejection_does_not_block_another_adapter() -> Result<()> {
    struct OtherPacked;
    impl LinearProvider for OtherPacked {
        fn id(&self) -> ProviderId {
            ProviderId("test.fp8.other-geometry.v1")
        }
        fn device_kind(&self) -> DeviceKind {
            DeviceKind::Cuda
        }
        fn assess(&self, _: &LinearRequest, _: DeviceCapabilities) -> Result<ProviderAssessment> {
            Ok(ProviderAssessment::eligible(
                self.id(),
                LinearMemory {
                    source_bytes: 1,
                    resident_bytes: 1,
                    workspace_bytes: 1,
                },
            ))
        }
    }
    let mut r = fp8(DType::BF16);
    r.in_features = 127;
    let narrow = BuiltinLinearProvider::Fp8Packed { available: true };
    let selected = select_linear(r, cuda(6, 1), budget(), &[&narrow, &OtherPacked], None)?;
    assert!(matches!(
        selected.candidates[0].rejection,
        Some(ProviderRejection::Constraint(_))
    ));
    assert_eq!(selected.selected.provider, OtherPacked.id());
    Ok(())
}
