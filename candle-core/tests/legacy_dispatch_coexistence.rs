use candle_core::dispatch::{DispatchContext, ResourceBudget};
use candle_core::quantized::fp8_sm61_dispatch::{
    dispatch_residence_for_activations, Fp8Activation, Fp8Backend, Fp8DispatchBudget, Fp8LinearSpec,
};
use candle_core::{DType, Device, Result, Tensor};

fn spec() -> Fp8LinearSpec {
    Fp8LinearSpec {
        out_features: 5,
        in_features: 128,
        block_n: 128,
        block_k: 128,
        resident_calls: 1,
        profile: None,
    }
}

fn fp8_plan() -> Result<candle_core::quantized::fp8_sm61_dispatch::Fp8DispatchPlan> {
    dispatch_residence_for_activations(
        &[spec()],
        &[Fp8Activation::Bf16],
        Fp8DispatchBudget {
            free_device_bytes: 1 << 20,
            ..Default::default()
        },
    )
    .map_err(candle_core::Error::msg)
}

#[test]
fn assembled_formats_keep_distinct_contracts() -> Result<()> {
    let plan = fp8_plan()?;
    assert_eq!(plan.entries[0].backend, Fp8Backend::SimtW8A16);
    assert_eq!(plan.selected_total_bytes, 5 * 128 + 4);
    let ctx = DispatchContext::new(
        &Device::Cpu,
        ResourceBudget {
            resident_bytes: 1 << 20,
            workspace_bytes: 1 << 20,
        },
        None,
    )?;
    let w = Tensor::ones((5, 128), DType::BF16, &Device::Cpu)?;
    let layer = ctx.prepare_bf16_linear(&w, None, 1)?;
    let x = Tensor::ones((1, 128), DType::BF16, &Device::Cpu)?;
    assert_eq!(
        layer.forward(&x)?.to_dtype(DType::F32)?.to_vec2::<f32>()?,
        vec![vec![128.; 5]]
    );
    assert_eq!(candle_core::quantized::GgmlDType::Mxfp4.block_size(), 32);
    Ok(())
}

#[cfg(all(
    feature = "cuda",
    feature = "cuda-legacy-bf16",
    feature = "cuda-legacy-fp8",
    feature = "cuda-legacy-fp4"
))]
#[test]
#[ignore = "requires physical SM61; no GPU execution in hosted CI"]
fn physical_three_formats_share_one_cuda_device() -> Result<()> {
    use candle_core::quantized::fp4_e7c_runtime::{Fp4E7CLinear, Fp4Execution};
    use candle_core::quantized::fp8_linear_sm61::Fp8Sm61Linear;
    let device = Device::new_cuda(0)?;
    let cuda = device.as_cuda_device()?;
    assert_eq!(
        cuda.cuda_stream()
            .context()
            .compute_capability()
            .map_err(candle_core::Error::msg)?,
        (6, 1)
    );
    let ctx = DispatchContext::new(
        &device,
        ResourceBudget {
            resident_bytes: 1 << 20,
            workspace_bytes: 1 << 20,
        },
        None,
    )?;
    let w = Tensor::ones((5, 128), DType::BF16, &Device::Cpu)?.to_device(&device)?;
    let bf16 = ctx.prepare_bf16_linear(&w, None, 1)?;
    let plan = fp8_plan()?;
    let fp8 = Fp8Sm61Linear::from_bytes_with_activation_dispatch(
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
    let fp4 = Fp4E7CLinear::from_modelopt_bytes(
        cuda,
        5,
        128,
        &vec![0x22; 5 * 64],
        &vec![0x38; 5 * 8],
        1.,
    )?;
    let x = Tensor::from_vec(vec![1e8f32; 128], (1, 128), &Device::Cpu)?
        .to_dtype(DType::BF16)?
        .to_device(&device)?;
    let expected = x
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?[0][0]
        * 128.;
    for y in [bf16.forward(&x)?.to_dtype(DType::F32)?, fp8.forward(&x)?] {
        for v in y.to_device(&Device::Cpu)?.to_vec2::<f32>()?[0].iter() {
            assert!(v.is_finite() && (*v - expected).abs() <= expected.abs() * 1e-5 && *v > 65504.);
        }
    }
    let a16 = Tensor::ones((1, 128), DType::F16, &Device::Cpu)?.to_device(&device)?;
    assert_eq!(
        fp4.forward(&a16, Fp4Execution::Nvfp4W4A16Reference)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?,
        vec![vec![128.; 5]]
    );
    device.synchronize()?;
    println!("CANDLE_LEGACY_DISPATCH_CROSS_FORMAT_SM61=PASS");
    Ok(())
}
