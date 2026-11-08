//! Model-independent packed FP8 contracts. No BF16 arithmetic or native FP8 is required.
#![cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
use candle_core::quantized::fp8_linear_sm61::{e4m3fn_decode, Fp8Sm61Linear, Fp8Sm61Mode};
use candle_core::{DType, Device, Result, Tensor};

#[test]
#[ignore = "requires a physical CUDA device; hosted CI compiles only"]
fn packed_fp8_f16_bf16_range_residency_layout_and_modes() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let cuda = device.as_cuda_device()?;
    let (major, minor) = cuda
        .cuda_stream()
        .context()
        .compute_capability()
        .map_err(candle_core::Error::msg)?;
    let (n, k, bn, bk) = (5usize, 256usize, 2usize, 128usize);
    let pattern = [0x38, 0xb8, 0x30, 0x40, 0xb0];
    let packed: Vec<u8> = (0..n * k)
        .map(|i| {
            if i / k == 0 {
                0x38
            } else {
                pattern[(i + i / k) % pattern.len()]
            }
        })
        .collect();
    let scales = [1f32, 0.5, 0.25, 2.0, 0.5, 4.0];
    let linear = Fp8Sm61Linear::from_bytes(
        cuda,
        n,
        k,
        &packed,
        &scales,
        bn,
        bk,
        Fp8Sm61Mode::SimtW8A16,
        128,
    )?;
    assert_eq!(linear.resident_weight_bytes(), n * k + scales.len() * 4);
    assert_eq!(linear.mode(), Fp8Sm61Mode::SimtW8A16);
    for dtype in [DType::F16, DType::BF16] {
        for rows in [1usize, 2, 3, 4, 5, 8, 9] {
            let magnitude = if dtype == DType::BF16 {
                1e8f32
            } else {
                16384f32
            };
            let values: Vec<f32> = (0..rows * k)
                .map(|i| match i % 11 {
                    0 => magnitude,
                    1 => -magnitude * 0.125,
                    2 => 1e-8,
                    _ => (i % 17) as f32 / 16.,
                })
                .collect();
            // Convert on CPU, upload raw BF16/F16 bits: this provider needs no GPU BF16 casts.
            let cpu = Tensor::from_vec(values, (rows, k), &Device::Cpu)?.to_dtype(dtype)?;
            let rounded = cpu.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let input = cpu.to_device(&device)?;
            let output = linear.forward(&input)?;
            assert_eq!(output.dtype(), DType::F32);
            assert_eq!(output.dims(), [rows, n]);
            let got = output.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            for token in 0..rows {
                for row in 0..n {
                    let (mut expected, mut absolute_sum) = (0f64, 0f64);
                    for col in 0..k {
                        let scale = scales[(row / bn) * k.div_ceil(bk) + col / bk];
                        let w = e4m3fn_decode(packed[row * k + col]) * scale;
                        let product = f64::from(w) * f64::from(rounded[token][col]);
                        expected += product;
                        absolute_sum += product.abs();
                    }
                    let actual = f64::from(got[token][row]);
                    assert!(actual.is_finite() && (actual-expected).abs() <= 2e-5*absolute_sum + 1e-5,
                        "dtype={dtype:?}, rows={rows}, token={token}, row={row}: actual={actual}, expected={expected}");
                }
                // Even F16 inputs can produce finite F32 sums beyond the F16 range.
                assert!(got[token][0] > 65504.);
            }
            assert_eq!(linear.resident_weight_bytes(), n * k + scales.len() * 4);
        }
    }
    let bf16 = Tensor::ones((1, k), DType::BF16, &Device::Cpu)?.to_device(&device)?;
    for mode in [Fp8Sm61Mode::ExpandedF16, Fp8Sm61Mode::Dp4aW8A8] {
        let other = Fp8Sm61Linear::from_bytes(cuda, n, k, &packed, &scales, bn, bk, mode, 128)?;
        let message = other.forward(&bf16).unwrap_err().to_string();
        assert!(
            message.contains("BF16 activations require packed SIMT"),
            "{message}"
        );
    }
    let offset = Tensor::ones((2, k), DType::BF16, &Device::Cpu)?
        .to_device(&device)?
        .narrow(0, 1, 1)?;
    assert!(linear.forward(&offset).is_err());
    let transpose = Tensor::ones((k, 2), DType::BF16, &Device::Cpu)?
        .to_device(&device)?
        .t()?;
    assert!(linear.forward(&transpose).is_err());
    // A zero-offset view can still alias a larger underlying allocation.
    let prefix = Tensor::ones((2, k), DType::BF16, &Device::Cpu)?
        .to_device(&device)?
        .narrow(0, 0, 1)?;
    assert!(linear.forward(&prefix).is_err());
    assert!(linear
        .forward(&Tensor::ones((1, k), DType::F32, &device)?)
        .is_err());
    assert!(linear
        .forward(&Tensor::ones((1, k), DType::BF16, &Device::Cpu)?)
        .is_err());
    device.synchronize()?;
    println!("FP8_W8ABF16_PHYSICAL=PASS CUDA_CC={major}{minor}");
    Ok(())
}
