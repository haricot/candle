//! Stable NVFP4 container contract and the opt-in software-CUDA API.
//! Run GPU tests on a real legacy card with --features cuda-legacy-fp4.

use candle_core::quantized::nvfp4::NvFp4Weights;
use candle_core::Result;

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn deterministic_weights(shape: impl Into<candle_core::Shape>) -> NvFp4Weights {
    let shape = shape.into();
    let count = shape.elem_count();
    let packed = (0..count / 2)
        .map(|i| ((i % 16) as u8) | (((i * 7 + 3) % 16) as u8) << 4)
        .collect::<Vec<_>>();
    let scales = (0..count / 16)
        .map(|i| [0x30u8, 0x38, 0x40][i % 3])
        .collect::<Vec<_>>();
    NvFp4Weights::new(shape, packed, scales, 0.125).unwrap()
}

fn reference_dense(
    packed: &[u8],
    scales: &[u8],
    global_scale: f32,
    n: usize,
    k: usize,
    input: &[f32],
    batch: usize,
) -> Vec<f32> {
    let blocks = k / 16;
    let mut out = vec![0f32; batch * n];
    for b in 0..batch {
        for row in 0..n {
            let mut sum = 0f32;
            for q in 0..k {
                let pos = row * k + q;
                let byte = packed[pos / 2];
                let code = if pos % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let exp = scales[row * blocks + q / 16];
                let block_scale = match exp {
                    0x30 => 0.5,
                    0x38 => 1.0,
                    0x40 => 2.0,
                    _ => panic!("unexpected fixture block scale"),
                };
                sum += E2M1[code as usize] * block_scale * global_scale * input[b * k + q];
            }
            out[b * n + row] = sum;
        }
    }
    out
}

#[test]
fn native_layout_and_accessors() -> Result<()> {
    let dense = deterministic_weights((7, 256));
    assert_eq!(dense.shape().dims(), &[7, 256]);
    assert_eq!(dense.packed_e2m1().len(), 7 * 256 / 2);
    assert_eq!(dense.scales_e4m3().len(), 7 * 256 / 16);
    assert_eq!(dense.global_scale(), 0.125);
    let moe = deterministic_weights((3, 7, 256));
    assert_eq!(moe.shape().dims(), &[3, 7, 256]);
    Ok(())
}

#[test]
fn reject_invalid_native_payloads_before_cuda() {
    let packed = vec![0u8; 32];
    let scales = vec![0x38; 4];
    assert!(NvFp4Weights::new((1, 64), packed.clone(), scales.clone(), 1.).is_ok());
    assert!(NvFp4Weights::new((1, 48), vec![0; 24], vec![0x38; 3], 1.).is_err());
    assert!(NvFp4Weights::new((0, 64), vec![], vec![], 1.).is_err());
    assert!(NvFp4Weights::new((1, 64), vec![0; 31], scales.clone(), 1.).is_err());
    assert!(NvFp4Weights::new((1, 64), packed.clone(), vec![0x38; 3], 1.).is_err());
    assert!(NvFp4Weights::new((1, 64), packed.clone(), scales.clone(), 0.).is_err());
    assert!(NvFp4Weights::new((1, 64), packed.clone(), scales.clone(), f32::NAN).is_err());
    assert!(NvFp4Weights::new((1, 64), packed.clone(), vec![0x7f; 4], 1.).is_err());
    assert!(NvFp4Weights::new((1, 64), packed, vec![0xb8; 4], 1.).is_err());
    assert!(NvFp4Weights::new((1, 64, 2, 64), vec![], vec![], 1.).is_err());
}

#[cfg(feature = "cuda-legacy-fp4")]
#[test]
fn dense_api_decode_prefill_and_strided_inputs() -> Result<()> {
    use candle_core::{CudaDevice, Device, Tensor};

    let cuda = CudaDevice::new(0)?;
    let device = Device::Cuda(cuda.clone());
    let (n, k) = (7usize, 256usize);
    let weights = deterministic_weights((n, k));
    let gpu = weights.upload(&cuda)?;

    for batch in [1usize, 4, 8, 17] {
        let input = (0..batch * k)
            .map(|i| ((i as f32) * 0.013).sin() * 0.7 + ((i as f32) * 0.003).cos() * 0.2)
            .collect::<Vec<_>>();
        let expected = reference_dense(
            weights.packed_e2m1(),
            weights.scales_e4m3(),
            weights.global_scale(),
            n,
            k,
            &input,
            batch,
        );
        let xs = Tensor::from_vec(input, (batch, k), &device)?;
        let got = gpu
            .forward(&xs)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(got.len(), expected.len());
        for (index, (&a, &b)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.15,
                "batch={batch}, elem={index}: {a} != {b}"
            );
        }
    }

    // The same weights also handle arbitrarily many leading dimensions,
    // and non-contiguous inputs without assuming zero storage offset.
    let batch = 2usize;
    let input = (0..batch * k)
        .map(|i| ((i as f32) * 0.01).cos())
        .collect::<Vec<_>>();
    let xs = Tensor::from_vec(input, (batch, k), &device)?
        .reshape((batch, 2, k / 2))?
        .transpose(0, 1)?; // non-contiguous shape [2, batch, k/2] (reject mismatched k)
    assert!(gpu.forward(&xs).is_err());
    // Non-contiguous view with a valid trailing k, including a nonzero offset.
    let input = Tensor::from_vec(
        (0..4 * k)
            .map(|i| ((i as f32) * 0.01).cos())
            .collect::<Vec<_>>(),
        (4, k),
        &device,
    )?;
    let slice = input.narrow(0, 1, 2)?;
    let expected_input = slice
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = reference_dense(
        weights.packed_e2m1(),
        weights.scales_e4m3(),
        weights.global_scale(),
        n,
        k,
        &expected_input,
        2,
    );
    let got = gpu
        .forward(&slice)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for (&a, &b) in got.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 0.15, "narrowed input: {a} != {b}");
    }
    Ok(())
}

#[cfg(feature = "cuda-legacy-fp4")]
#[test]
fn indexed_moe_api_validates_ids_and_matches_reference() -> Result<()> {
    use candle_core::{CudaDevice, Device, Tensor};

    let cuda = CudaDevice::new(0)?;
    let device = Device::Cuda(cuda.clone());
    let (experts, n, k, batch, topk) = (3usize, 5usize, 256usize, 2usize, 2usize);
    let weights = deterministic_weights((experts, n, k));
    let gpu = weights.upload(&cuda)?;
    let input = (0..batch * k)
        .map(|i| ((i as f32) * 0.007).cos() * 0.4)
        .collect::<Vec<_>>();
    let xs = Tensor::from_vec(input.clone(), (batch, k), &device)?;
    let ids = [0u32, 2, 1, 0];
    let idx = Tensor::from_slice(&ids, (batch, topk), &device)?;
    let got = gpu
        .indexed_moe(&xs, &idx)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for b in 0..batch {
        for t in 0..topk {
            let expert = ids[b * topk + t] as usize;
            let off = expert * n * k;
            let scale_off = expert * n * k / 16;
            let expected = reference_dense(
                &weights.packed_e2m1()[off / 2..(off + n * k) / 2],
                &weights.scales_e4m3()[scale_off..scale_off + n * k / 16],
                weights.global_scale(),
                n,
                k,
                &input[b * k..(b + 1) * k],
                1,
            );
            for row in 0..n {
                let actual = got[(b * topk + t) * n + row];
                assert!(
                    (actual - expected[row]).abs() < 0.15,
                    "b={b} t={t} row={row}: {actual} != {}",
                    expected[row]
                );
            }
        }
    }
    let bad_ids = Tensor::from_slice(&[0u32, 3, 1, 0], (batch, topk), &device)?;
    assert!(gpu.indexed_moe(&xs, &bad_ids).is_err());
    Ok(())
}
