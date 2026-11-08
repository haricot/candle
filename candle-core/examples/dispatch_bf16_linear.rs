use candle_core::dispatch::{DispatchContext, ResourceBudget};
use candle_core::{DType, Device, Result, Tensor};

fn main() -> Result<()> {
    let device = if std::env::args().any(|a| a == "--cuda") {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let context = DispatchContext::new(
        &device,
        ResourceBudget::from_available(1 << 20, 0, 0, 1 << 18)?,
        None,
    )?;
    let weight = Tensor::new(&[[1f32, 2.], [-1., 1.]], &Device::Cpu)?
        .to_dtype(DType::BF16)?
        .to_device(&device)?;
    let layer = context.prepare_bf16_linear(&weight, None, 4)?;
    let input = Tensor::new(&[[131072f32, 65536.]], &Device::Cpu)?
        .to_dtype(DType::BF16)?
        .to_device(&device)?;
    let output = layer.forward(&input)?;
    device.synchronize()?;
    println!(
        "capabilities={:?} decision={:?}",
        context.capabilities(),
        layer.decision()
    );
    println!(
        "output={:?}",
        output
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?
    );
    Ok(())
}
