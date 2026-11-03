//! E7C regression: MXFP4 baseline remains independent of ModelOpt NVFP4.
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Result, Tensor};

#[test]
fn mxfp4_gguf_quantization_is_unmodified() -> Result<()> {
    let values = (0..128)
        .map(|i| ((i as f32) * 0.09).sin() * 1.5)
        .collect::<Vec<_>>();
    let t = Tensor::from_vec(values, (4, 32), &Device::Cpu)?;
    let q = QTensor::quantize(&t, GgmlDType::Mxfp4)?;
    assert_eq!(q.dtype(), GgmlDType::Mxfp4);
    assert_eq!(q.data()?.len(), 4 * 17);
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp4"))]
#[test]
fn e7c_rejects_mxfp4_as_nvfp4() -> Result<()> {
    use candle_core::quantized::fp4_e7c_runtime::{Fp4E7CLinear, Fp4Execution, Fp4Format};
    let values = (0..128)
        .map(|i| ((i as f32) * 0.09).sin() * 1.5)
        .collect::<Vec<_>>();
    let t = Tensor::from_vec(values, (4, 32), &Device::Cpu)?;
    let q = QTensor::quantize(&t, GgmlDType::Mxfp4)?;
    let fp4 = Fp4E7CLinear::from_mxfp4(q)?;
    assert_eq!(fp4.format(), Fp4Format::Mxfp4Gguf);
    let x = Tensor::zeros((1, 32), candle_core::DType::F32, &Device::Cpu)?;
    assert!(fp4.forward(&x, Fp4Execution::Nvfp4W4A8Dp4a).is_err());
    assert!(fp4.forward(&x, Fp4Execution::Mxfp4Existing).is_ok());
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp4"))]
#[test]
fn modelopt_native_modes_match_reference_with_partial_tiles() -> Result<()> {
    use candle_core::quantized::fp4_e7c_runtime::{Fp4E7CLinear, Fp4Execution};
    use candle_core::DType;
    use half::f16;
    let device = Device::new_cuda(0)?;
    let cuda = match &device {
        Device::Cuda(d) => d,
        _ => unreachable!(),
    };
    let (n, k) = (5usize, 64usize);
    let packed = (0..n * k / 2)
        .map(|i| (i * 37 + 11) as u8)
        .collect::<Vec<_>>();
    let scales = (0..n * k / 16)
        .map(|i| [0x30, 0x38, 0x40][i % 3])
        .collect::<Vec<_>>();
    let op = Fp4E7CLinear::from_modelopt_bytes(cuda, n, k, &packed, &scales, 0.5)?;
    const E2M1: [f32; 16] = [
        0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.,
    ];
    for m in [1usize, 3, 8, 9, 17] {
        let activations = (0..m * k)
            .map(|i| f16::from_f32((i as i32 % 19 - 9) as f32 / 8.0))
            .collect::<Vec<_>>();
        let x = Tensor::from_vec(activations.clone(), (m, k), &device)?;
        let scalar = op.forward(&x, Fp4Execution::Nvfp4W4A16Reference)?;
        let reference = op.forward(&x, Fp4Execution::Nvfp4W4A8Reference)?;
        let tiled = op.forward(&x, Fp4Execution::Nvfp4W4A8Dp4a)?;
        assert_eq!(tiled.dtype(), DType::F32);
        assert_eq!(tiled.dims(), &[m, n]);
        let a = scalar.flatten_all()?.to_vec1::<f32>()?;
        let b = reference.flatten_all()?.to_vec1::<f32>()?;
        let c = tiled.flatten_all()?.to_vec1::<f32>()?;
        for token in 0..m {
            for row in 0..n {
                let mut expected = 0f32;
                for col in 0..k {
                    let byte = packed[row * k / 2 + col / 2];
                    let code = if col % 2 == 0 { byte & 15 } else { byte >> 4 };
                    let scale = match scales[row * k / 16 + col / 16] {
                        0x30 => 0.5,
                        0x38 => 1.0,
                        0x40 => 2.0,
                        _ => unreachable!(),
                    };
                    expected +=
                        E2M1[code as usize] * scale * 0.5 * activations[token * k + col].to_f32();
                }
                let index = token * n + row;
                assert!(a[index].is_finite() && b[index].is_finite() && c[index].is_finite());
                assert!(
                    (a[index] - expected).abs() < 1e-4,
                    "W4A16 m={m} index={index}"
                );
                assert!(
                    (b[index] - c[index]).abs() < 0.005,
                    "W4A8 reference/DP4A m={m} index={index}"
                );
            }
        }
    }
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
    let cuda = match &device {
        Device::Cuda(cuda) => cuda,
        _ => unreachable!(),
    };
    let source = ModelOptNvfp4Shards::single_file(path);
    let op = Fp4E7CLinear::from_modelopt_shards(
        cuda,
        2,
        64,
        "model.layers.0.mlp.experts.0.w1",
        &source,
    )?;
    assert_eq!(op.format(), Fp4Format::Nvfp4ModelOpt);
    let x = Tensor::from_vec(vec![f16::from_f32(0.125); 2 * 64], (2, 64), &device)?;
    let y_ref = op.forward(&x, Fp4Execution::Nvfp4W4A8Reference)?;
    let y_simt = op.forward(&x, Fp4Execution::Nvfp4W4A8Dp4a)?;
    assert_eq!(y_simt.dims(), &[2, 2]);
    assert_eq!(y_simt.dtype(), DType::F32);
    let ref_values = y_ref
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let simt_values = y_simt
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let diff = ref_values
        .iter()
        .zip(simt_values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff < 0.003f32,
        "E7C ModelOpt DP4A numeric mismatch: {diff}"
    );
    assert!(op.forward(&x, Fp4Execution::Mxfp4Existing).is_err());
    Ok(())
}
