//! MXFP4 (GGML_TYPE_MXFP4 = 39) tests.
//!
//! The golden bytes in this file were generated with llama.cpp's reference
//! implementation (`quantize_row_mxfp4_ref` in ggml-quants.c plus
//! `ggml_e8m0_to_fp32_half` from ggml-impl.h), compiled as a standalone C
//! program against a fixed LCG input. They pin the quantizer to bit-exact
//! compatibility with llama.cpp.
#![allow(clippy::excessive_precision)] // exact f32 literals emitted by the C golden generator

use candle_core::quantized::{gguf_file, GgmlDType, QStorage, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use std::borrow::Cow;
use std::io::Cursor;

const QK: usize = 32;

/// Two input blocks (LCG seed 42 and seed 7, see the module docs) with the
/// expected quantized bytes: 1 exponent byte + 16 nibble-packed code bytes.
struct Golden {
    input: [f32; QK],
    e: u8,
    qs: [u8; QK / 2],
}

fn goldens() -> [Golden; 2] {
    [
        Golden {
            input: [
                -0.835388184,
                -0.960388184,
                -1.0680542,
                -0.445983887,
                -1.15429688,
                -1.93328857,
                -1.16522217,
                -0.382568359,
                -0.77532959,
                -0.570251465,
                -1.63549805,
                -0.967651367,
                -0.876098633,
                -1.50268555,
                -0.160827637,
                -1.77832031,
                -0.683288574,
                -1.34307861,
                -1.62945557,
                -0.65802002,
                -0.162536621,
                -1.73706055,
                -1.17883301,
                -0.301757812,
                -1.18078613,
                -1.93463135,
                -0.867736816,
                -0.392028809,
                -1.70367432,
                -0.194030762,
                -1.03997803,
                -1.09802246,
            ],
            e: 0x7d,
            qs: [
                0xdd, 0xfe, 0xfe, 0xdc, 0x9e, 0xff, 0xee, 0xab, 0xed, 0xfc, 0xdf, 0xbe, 0xfe, 0xaf,
                0xe9, 0xef,
            ],
        },
        Golden {
            input: [
                0.194091797,
                -0.401489258,
                -0.336669922,
                0.383911133,
                0.985534668,
                -0.794616699,
                0.955200195,
                0.940307617,
                -0.512634277,
                -0.33392334,
                -0.315612793,
                0.548034668,
                0.80847168,
                0.579223633,
                -0.954284668,
                -0.98034668,
                -0.134216309,
                0.443725586,
                0.168823242,
                -0.544677734,
                -0.984619141,
                0.12713623,
                -0.690002441,
                0.810180664,
                0.102355957,
                -0.107421875,
                0.931091309,
                0.11529541,
                0.615844727,
                -0.0777587891,
                -0.147155762,
                -0.212036133,
            ],
            e: 0x7c,
            qs: [
                0xa3, 0x6d, 0x3d, 0xe5, 0xf7, 0x2f, 0xf7, 0x77, 0x2e, 0xad, 0x7d, 0x26, 0x67, 0x96,
                0xaf, 0xbf,
            ],
        },
    ]
}

/// The quantizer must produce byte-identical output to llama.cpp's
/// `quantize_row_mxfp4_ref` on the golden blocks.
#[test]
fn quantize_matches_llamacpp() -> Result<()> {
    for golden in goldens() {
        let xs = Tensor::from_slice(&golden.input, QK, &Device::Cpu)?;
        let qtensor = QTensor::quantize(&xs, GgmlDType::Mxfp4)?;
        let data = qtensor.data()?;
        assert_eq!(data.len(), 17, "MXFP4 block must be 17 bytes");
        assert_eq!(data[0], golden.e, "exponent byte mismatch");
        assert_eq!(&data[1..], &golden.qs, "nibble-packed codes mismatch");
    }
    Ok(())
}

/// Independent reference decoder: kvalues * 2^(e-128), per OCP MX spec.
/// Layout matches llama.cpp's dequantize_row_mxfp4: low nibbles decode
/// elements [0, 16), high nibbles decode elements [16, 32).
fn reference_decode(data: &[u8]) -> Vec<f32> {
    const KVALUES: [f32; 16] = [
        0., 1., 2., 3., 4., 6., 8., 12., 0., -1., -2., -3., -4., -6., -8., -12.,
    ];
    let e = data[0];
    let d = f32::powf(2.0, e as f32 - 128.0);
    let mut out = vec![0f32; QK];
    for j in 0..QK / 2 {
        out[j] = KVALUES[(data[1 + j] & 0x0F) as usize] * d;
        out[j + QK / 2] = KVALUES[(data[1 + j] >> 4) as usize] * d;
    }
    out
}

/// dequantize must match the independent reference decode.
#[test]
fn dequantize_matches_reference() -> Result<()> {
    for golden in goldens() {
        let xs = Tensor::from_slice(&golden.input, QK, &Device::Cpu)?;
        let qtensor = QTensor::quantize(&xs, GgmlDType::Mxfp4)?;
        let expected = reference_decode(&qtensor.data()?);
        let dequantized = qtensor.dequantize(&Device::Cpu)?;
        let got: Vec<f32> = dequantized.to_vec1()?;
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(g, e, "element {i}: {g} != {e}");
        }
    }
    Ok(())
}

/// Round trip through a GGUF file: write as MXFP4, read back, byte-identical.
#[test]
fn gguf_roundtrip() -> Result<()> {
    let golden = &goldens()[0];
    let xs = Tensor::from_slice(&golden.input, QK, &Device::Cpu)?;
    let qtensor = QTensor::quantize(&xs, GgmlDType::Mxfp4)?;

    let mut buf = Cursor::new(Vec::new());
    gguf_file::write(&mut buf, &[], &[("mxfp4_tensor", &qtensor)])?;

    let mut cursor = Cursor::new(buf.into_inner());
    let content = gguf_file::Content::read(&mut cursor)?;
    let tensor_info = content
        .tensor_infos
        .get("mxfp4_tensor")
        .expect("tensor must be in the file");
    assert_eq!(tensor_info.ggml_dtype, GgmlDType::Mxfp4);
    let loaded = content.tensor(&mut cursor, "mxfp4_tensor", &Device::Cpu)?;
    assert_eq!(loaded.data()?, qtensor.data()?);
    assert_eq!(loaded.shape(), qtensor.shape());
    Ok(())
}

/// Dtype plumbing: 17 bytes per 32 elements, and GGUF write/read maps id 39.
#[test]
fn dtype_plumbing() -> Result<()> {
    let dtype = GgmlDType::Mxfp4;
    assert_eq!(dtype.type_size(), 17);
    assert_eq!(dtype.block_size(), 32);
    Ok(())
}


#[cfg(feature = "cuda")]
#[test]
fn cuda_dequantize_matches_cpu() -> Result<()> {
    let cpu = Device::Cpu;
    let cuda = Device::new_cuda(0)?;

    // Use more than 8 blocks so the CUDA decoder crosses its 256-value
    // launch-group boundary rather than only validating a single block.
    let gs = goldens();
    let mut input = Vec::with_capacity(10 * QK);
    for i in 0..10 {
        input.extend_from_slice(&gs[i % gs.len()].input);
    }

    let src = Tensor::from_vec(input, (10 * QK,), &cpu)?;
    let q_cpu = QTensor::quantize(&src, GgmlDType::Mxfp4)?;
    let expected_f32 = q_cpu.dequantize(&cpu)?;

    let storage = QStorage::from_data(
        Cow::Owned(q_cpu.data()?.to_vec()),
        &cuda,
        GgmlDType::Mxfp4,
    )?;
    let q_cuda = QTensor::new(storage, (10 * QK,))?;

    let got_f32 = q_cuda.dequantize(&cuda)?.to_device(&cpu)?;
    assert_eq!(
        got_f32.to_vec1::<f32>()?,
        expected_f32.to_vec1::<f32>()?,
        "MXFP4 CUDA f32 dequantization differs from CPU reference"
    );

    let expected_f16_as_f32 = expected_f32
        .to_dtype(DType::F16)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let got_f16_as_f32 = q_cuda
        .dequantize_f16(&cuda)?
        .to_device(&cpu)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    assert_eq!(
        got_f16_as_f32,
        expected_f16_as_f32,
        "MXFP4 CUDA f16 dequantization differs from CPU reference"
    );

    Ok(())
}
