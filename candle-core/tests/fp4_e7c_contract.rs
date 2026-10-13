//! E7C regression: MXFP4 baseline remains independent of ModelOpt NVFP4.
use candle_core::{Device, Result, Tensor};
use candle_core::quantized::{GgmlDType, QTensor};

#[test]
fn mxfp4_gguf_quantization_is_unmodified() -> Result<()> {
    let values = (0..128).map(|i| ((i as f32)*0.09).sin()*1.5).collect::<Vec<_>>();
    let t = Tensor::from_vec(values,(4,32),&Device::Cpu)?;
    let q = QTensor::quantize(&t,GgmlDType::Mxfp4)?;
    assert_eq!(q.dtype(), GgmlDType::Mxfp4);
    assert_eq!(q.data()?.len(), 4*17);
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp4"))]
#[test]
fn e7c_rejects_mxfp4_as_nvfp4() -> Result<()> {
    use candle_core::quantized::fp4_e7c_runtime::{Fp4E7CLinear, Fp4Execution, Fp4Format};
    let values = (0..128).map(|i| ((i as f32)*0.09).sin()*1.5).collect::<Vec<_>>();
    let t = Tensor::from_vec(values,(4,32),&Device::Cpu)?;
    let q = QTensor::quantize(&t,GgmlDType::Mxfp4)?;
    let fp4 = Fp4E7CLinear::from_mxfp4(q)?;
    assert_eq!(fp4.format(),Fp4Format::Mxfp4Gguf);
    let x = Tensor::zeros((1,32), candle_core::DType::F32, &Device::Cpu)?;
    assert!(fp4.forward(&x,Fp4Execution::Nvfp4W4A8Dp4a).is_err());
    assert!(fp4.forward(&x,Fp4Execution::Mxfp4Existing).is_ok());
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp4"))]
#[test]
#[ignore = "requires ARCH_E7C_FIXTURE and a physical CUDA sm_61 GPU"]
fn modelopt_packed_loader_and_dp4a_gpu_sm61() -> Result<()> {
    use candle_core::quantized::fp4_e7c_runtime::{Fp4E7CLinear, Fp4Execution, Fp4Format};
    use candle_core::quantized::nvfp4_e7c_safetensors::ModelOptNvfp4Shards;
    use candle_core::DType;
    use half::f16;
    let path = std::env::var("ARCH_E7C_FIXTURE")
        .map_err(|e| candle_core::Error::Msg(format!("ARCH_E7C_FIXTURE required: {e}")))?;
    let device = Device::new_cuda(0)?;
    let cuda = match &device { Device::Cuda(cuda) => cuda, _ => unreachable!() };
    let source = ModelOptNvfp4Shards::single_file(path);
    let op = Fp4E7CLinear::from_modelopt_shards(cuda,2,64,"model.layers.0.mlp.experts.0.w1",&source)?;
    assert_eq!(op.format(),Fp4Format::Nvfp4ModelOpt);
    let x = Tensor::from_vec(vec![f16::from_f32(0.125);2*64],(2,64),&device)?;
    let y_ref = op.forward(&x,Fp4Execution::Nvfp4W4A8Reference)?;
    let y_simt = op.forward(&x,Fp4Execution::Nvfp4W4A8Dp4a)?;
    assert_eq!(y_simt.dims(),&[2,2]);
    assert_eq!(y_simt.dtype(),DType::F32);
    let ref_values=y_ref.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let simt_values=y_simt.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let diff=ref_values.iter().zip(simt_values.iter())
        .map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    assert!(diff < 0.003f32, "E7C ModelOpt DP4A numeric mismatch: {diff}");
    assert!(op.forward(&x,Fp4Execution::Mxfp4Existing).is_err());
    Ok(())
}
