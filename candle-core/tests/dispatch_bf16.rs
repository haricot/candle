use candle_core::dispatch::*;
use candle_core::{DType, Device, Result, Tensor, Var};
use half::bf16;
use std::sync::Arc;

fn generous() -> ResourceBudget {
    ResourceBudget {
        resident_bytes: 1 << 20,
        workspace_bytes: 1 << 20,
    }
}

fn spec() -> LinearSpec {
    LinearSpec {
        in_features: 4,
        out_features: 2,
        bias: true,
        max_rows: 8,
    }
}

#[test]
fn capability_eligibility_precedes_policy_and_never_moves_cuda_to_cpu() -> Result<()> {
    let inventory = LinearInventory {
        cpu_f32: true,
        cuda_f32: true,
        cuda_bf16: true,
    };
    for (major, minor) in [(6, 0), (6, 1), (6, 2), (7, 0), (7, 2), (7, 5), (999, 0)] {
        let caps = DeviceCapabilities {
            architecture: Architecture::Cuda { major, minor },
        };
        let decision = select_bf16_linear(spec(), caps, inventory, generous(), None)?;
        assert_eq!(decision.backend, Bf16LinearBackend::CudaF32);
        assert!(select_bf16_linear(
            spec(),
            caps,
            inventory,
            generous(),
            Some(&RequireBackend(Bf16LinearBackend::CudaBf16))
        )
        .is_err());
        assert!(select_bf16_linear(
            spec(),
            caps,
            inventory,
            generous(),
            Some(&RequireBackend(Bf16LinearBackend::CpuF32))
        )
        .is_err());
        assert!(!caps.native_bf16_gemm());
    }
    for (major, minor) in [(8, 0), (8, 6), (8, 9), (9, 0)] {
        let caps = DeviceCapabilities {
            architecture: Architecture::Cuda { major, minor },
        };
        assert_eq!(
            select_bf16_linear(spec(), caps, inventory, generous(), None)?.backend,
            Bf16LinearBackend::CudaBf16
        );
        assert_eq!(
            select_bf16_linear(
                spec(),
                caps,
                inventory,
                generous(),
                Some(&RequireBackend(Bf16LinearBackend::CudaF32))
            )?
            .backend,
            Bf16LinearBackend::CudaF32
        );
    }
    Ok(())
}

#[test]
fn unavailable_providers_and_resource_limits_are_explained() -> Result<()> {
    let caps = DeviceCapabilities {
        architecture: Architecture::Cuda { major: 8, minor: 0 },
    };
    let inventory = LinearInventory {
        cpu_f32: true,
        cuda_f32: true,
        cuda_bf16: false,
    };
    let decision = select_bf16_linear(spec(), caps, inventory, generous(), None)?;
    assert_eq!(decision.backend, Bf16LinearBackend::CudaF32);
    assert_eq!(
        decision.candidates[0].rejection,
        Some(RejectionReason::ProviderUnavailable)
    );
    let memory = decision.memory;
    assert_eq!(memory.source_bytes, 20);
    assert_eq!(memory.resident_bytes, 40);
    assert_eq!(memory.workspace_bytes, 384);
    assert!(select_bf16_linear(
        spec(),
        caps,
        inventory,
        ResourceBudget {
            resident_bytes: 39,
            ..generous()
        },
        None
    )
    .is_err());
    assert!(select_bf16_linear(
        spec(),
        caps,
        inventory,
        ResourceBudget {
            workspace_bytes: 383,
            ..generous()
        },
        None
    )
    .is_err());
    assert_eq!(
        ResourceBudget::from_available(1000, 100, 100, 300)?,
        ResourceBudget {
            resident_bytes: 500,
            workspace_bytes: 300
        }
    );
    assert!(ResourceBudget::from_available(100, 101, 0, 0).is_err());
    assert!(LinearSpec {
        max_rows: 0,
        ..spec()
    }
    .validate()
    .is_err());
    assert!(LinearSpec {
        in_features: i32::MAX as usize + 1,
        ..spec()
    }
    .validate()
    .is_err());
    assert!(LinearSpec {
        in_features: i32::MAX as usize,
        out_features: i32::MAX as usize,
        max_rows: i32::MAX as usize,
        bias: true
    }
    .memory(Bf16LinearBackend::CpuF32)
    .is_err());
    Ok(())
}

fn bf(values: &[f32], shape: impl Into<candle_core::Shape>, device: &Device) -> Result<Tensor> {
    Tensor::from_slice(values, shape, &Device::Cpu)?
        .to_dtype(DType::BF16)?
        .to_device(device)
}

fn reference(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Vec<f32>> {
    let w = weight
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    let x = input
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let bias = bias
        .map(|b| {
            b.to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()
        })
        .transpose()?;
    let mut expected = Vec::new();
    for row in x.chunks(weight.dim(1)?) {
        for (n, weights) in w.iter().enumerate() {
            let dot = row
                .iter()
                .zip(weights)
                .map(|(&x, &w)| (x as f64) * (w as f64))
                .sum::<f64>() as f32;
            let rounded = bf16::from_f32(dot).to_f32();
            expected.push(bf16::from_f32(rounded + bias.as_ref().map_or(0., |b| b[n])).to_f32());
        }
    }
    Ok(expected)
}

fn layouts_and_range(device: &Device) -> Result<()> {
    let context = DispatchContext::new(device, generous(), None)?;
    let weight = bf(&[1., -2., 0.5, 0.25, -1., 0.25, 1., 2.], (2, 4), device)?;
    let bias = bf(&[9., 0.5, -1.], 3, device)?.narrow(0, 1, 2)?;
    let layer = context.prepare_bf16_linear(&weight, Some(&bias), 8)?;
    let base = bf(
        &[
            131072., -65536., 2., 4., 262144., 65536., -4., 8., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
            10., 11., 12., 13., 14., 15., 16.,
        ],
        (3, 2, 4),
        device,
    )?;
    let offset = base.narrow(0, 1, 1)?;
    assert!(offset.is_contiguous());
    assert!(offset.layout().start_offset() > 0);
    let prefix = base.narrow(0, 0, 1)?;
    let strided = base.transpose(0, 1)?;
    assert!(!strided.is_contiguous());
    let broadcast = base.narrow(0, 0, 1)?.broadcast_as((2, 2, 4))?;
    for x in [&base, &offset, &prefix, &strided, &broadcast] {
        let actual = layer.forward(x)?;
        device.synchronize()?;
        assert_eq!(actual.dtype(), DType::BF16);
        let mut shape = x.dims().to_vec();
        *shape.last_mut().unwrap() = 2;
        assert_eq!(actual.dims(), shape);
        assert_eq!(
            actual
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            reference(x, &weight, Some(&bias))?
        );
    }
    let weights_strided = weight.t()?.contiguous()?.t()?;
    let equivalent = context.prepare_bf16_linear(&weights_strided, Some(&bias), 8)?;
    let one = base
        .narrow(0, 0, 1)?
        .reshape((2, 4))?
        .narrow(0, 0, 1)?
        .squeeze(0)?;
    assert_eq!(equivalent.forward(&one)?.dims(), [2]);
    assert_eq!(
        equivalent
            .forward(&one)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?,
        reference(&one, &weight, Some(&bias))?
    );
    assert_eq!(
        layer
            .forward(&Tensor::zeros((0, 4), DType::BF16, device)?)?
            .dims(),
        [0, 2]
    );
    let tiny = bf(&[1e-20, -1e-20, 1e-10, -1e-10], (1, 4), device)?;
    let tiny_layer = context.prepare_bf16_linear(&weight, None, 1)?;
    assert_eq!(
        tiny_layer
            .forward(&tiny)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?[0],
        reference(&tiny, &weight, None)?
    );
    Ok(())
}

#[test]
fn prepared_cpu_linear_preserves_range_layouts_and_bias() -> Result<()> {
    layouts_and_range(&Device::Cpu)
}

#[test]
fn bias_is_added_after_bf16_gemm_rounding() -> Result<()> {
    let context = DispatchContext::new(&Device::Cpu, generous(), None)?;
    let weight = bf(&[1., 1. / 256.], (1, 2), &Device::Cpu)?;
    let bias = bf(&[-1.], 1, &Device::Cpu)?;
    let layer = context.prepare_bf16_linear(&weight, Some(&bias), 1)?;
    let input = bf(&[1., 1.], (1, 2), &Device::Cpu)?;
    assert_eq!(
        layer
            .forward(&input)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?,
        [[0.]]
    );
    Ok(())
}

#[test]
fn weight_snapshots_survive_source_mutation_and_share_admission_on_clone() -> Result<()> {
    let weight = Var::from_tensor(&bf(&[1., 2., 3., 4.], (2, 2), &Device::Cpu)?)?;
    let bias = Var::from_tensor(&bf(&[0.5, 1.], 2, &Device::Cpu)?)?;
    let context = DispatchContext::new(
        &Device::Cpu,
        ResourceBudget {
            resident_bytes: 24,
            ..generous()
        },
        None,
    )?;
    let other = DispatchContext::new(&Device::Cpu, generous(), None)?;
    assert_ne!(context.id(), other.id());
    let layer = context.prepare_bf16_linear(&weight, Some(&bias), 1)?;
    let clone = layer.clone();
    assert_eq!(layer.context_id(), clone.context_id());
    assert_eq!(context.reserved_resident_bytes(), 24);
    assert!(context
        .clone()
        .prepare_bf16_linear(&weight, Some(&bias), 1)
        .is_err());
    weight.set(&bf(&[0.; 4], (2, 2), &Device::Cpu)?)?;
    bias.set(&bf(&[0.; 2], 2, &Device::Cpu)?)?;
    let input = bf(&[1., 1.], (1, 2), &Device::Cpu)?;
    assert_eq!(
        layer
            .forward(&input)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?,
        [[3.5, 8.]]
    );
    drop(layer);
    assert_eq!(context.reserved_resident_bytes(), 24);
    drop(clone);
    assert_eq!(context.reserved_resident_bytes(), 0);
    assert!(context.prepare_bf16_linear(&weight, Some(&bias), 1).is_ok());
    Ok(())
}

#[test]
fn rejects_bad_shapes_dtypes_policy_and_row_limits() -> Result<()> {
    let context = DispatchContext::new(&Device::Cpu, generous(), None)?;
    let w = bf(&[1.; 8], (2, 4), &Device::Cpu)?;
    assert!(context
        .prepare_bf16_linear(&w.to_dtype(DType::F32)?, None, 1)
        .is_err());
    assert!(context
        .prepare_bf16_linear(&w, Some(&bf(&[1.], 1, &Device::Cpu)?), 1)
        .is_err());
    let layer = context.prepare_bf16_linear(&w, None, 1)?;
    assert!(layer.forward(&bf(&[1.; 8], (2, 4), &Device::Cpu)?).is_err());
    assert!(layer.forward(&bf(&[1.; 3], (1, 3), &Device::Cpu)?).is_err());
    assert!(layer
        .forward(&Tensor::new(1f32, &Device::Cpu)?.to_dtype(DType::BF16)?)
        .is_err());
    assert!(layer
        .forward(&Tensor::ones((1, 4), DType::F32, &Device::Cpu)?)
        .is_err());
    let forced = DispatchContext::new(
        &Device::Cpu,
        generous(),
        Some(Arc::new(RequireBackend(Bf16LinearBackend::CudaF32))),
    )?;
    assert!(forced.prepare_bf16_linear(&w, None, 1).is_err());
    assert_eq!(forced.reserved_resident_bytes(), 0);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a physical CUDA GPU; compile-only CI does not execute this test"]
fn cuda_physical_bf16_linear_contracts() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let first = DispatchContext::new(&device, generous(), None)?;
    // Copy every BF16 bit pattern through a strided GPU view, including NaNs,
    // signed zeros and subnormals. Storage support must not round via F16/F32.
    let bits = (0..=u16::MAX).map(bf16::from_bits).collect::<Vec<_>>();
    let source = Tensor::from_vec(bits.clone(), (256, 256), &Device::Cpu)?.to_device(&device)?;
    let copied = source
        .t()?
        .force_contiguous()?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<bf16>()?;
    for (i, value) in copied.iter().enumerate() {
        assert_eq!(value.to_bits(), bits[(i % 256) * 256 + i / 256].to_bits());
    }
    layouts_and_range(&device)?;
    let second = Device::new_cuda_with_stream(0)?;
    let weight = bf(&[1., 2., 3., 4.], (2, 2), &device)?;
    let linear = first.prepare_bf16_linear(&weight, None, 1)?;
    assert!(linear.forward(&bf(&[1., 1.], (1, 2), &second)?).is_err());
    println!(
        "DISPATCH_DEVICE={:?} BACKEND={:?}",
        first.capabilities(),
        linear.decision().backend
    );
    device.synchronize()?;
    Ok(())
}
