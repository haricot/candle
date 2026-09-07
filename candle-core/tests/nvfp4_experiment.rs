//! Experimental NVFP4 reference probes.
//!
//! This is intentionally not a Candle dtype yet. It models the software NVFP4
//! representation used by xInfer/ModelOpt:
//! - 16 E2M1 values per FP8 E4M3FN scale
//! - 2 packed E2M1 nibbles per byte
//! - one external/global F32 scale
//! - dequant = E2M1 * E4M3(block_scale) * global_scale
//!
//! The decoder is format-exact. The encoder below is an experimental
//! nearest-value calibrator for quality comparisons; it is not claimed to be
//! bit-identical to NVIDIA ModelOpt checkpoint quantization.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Result, Tensor};
use std::hint::black_box;
use std::time::Instant;

const NVFP4_BLOCK: usize = 16;
const E2M1_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[derive(Clone, Debug)]
struct Nvfp4Reference {
    packed: Vec<u8>,
    scales_e4m3: Vec<u8>,
    global_scale: f32,
    len: usize,
}

fn e4m3fn_to_f32(x: u8) -> f32 {
    let sign = (x >> 7) & 1;
    let exp = (x >> 3) & 0x0f;
    let mant = x & 0x07;

    let value = if exp == 0 {
        if mant == 0 {
            0.0
        } else {
            mant as f32 * 2f32.powi(-9)
        }
    } else if exp == 0x0f && mant == 0x07 {
        f32::NAN
    } else {
        let fraction = 1.0 + mant as f32 / 8.0;
        fraction * 2f32.powi(exp as i32 - 7)
    };

    if sign != 0 { -value } else { value }
}

fn f32_to_e4m3fn_nearest(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    if x == 0.0 {
        return if x.is_sign_negative() { 0x80 } else { 0x00 };
    }

    // Small reference encoder: exhaustive finite-code search gives deterministic
    // round-to-nearest behavior and keeps this experiment independent from CUDA.
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for raw in 0u16..=255 {
        let raw = raw as u8;
        if raw & 0x7f == 0x7f {
            continue;
        }
        let value = e4m3fn_to_f32(raw);
        if !value.is_finite() {
            continue;
        }
        let err = (value - x).abs();
        if err < best_err || (err == best_err && raw < best) {
            best = raw;
            best_err = err;
        }
    }
    best
}

fn nearest_e2m1(x: f32) -> u8 {
    let mut best = 0usize;
    let mut best_err = f32::INFINITY;
    for (i, &v) in E2M1_VALUES.iter().enumerate() {
        let err = (v - x).abs();
        if err < best_err {
            best = i;
            best_err = err;
        }
    }
    best as u8
}

fn nvfp4_quantize_experimental(xs: &[f32]) -> Nvfp4Reference {
    assert!(xs.len().is_multiple_of(NVFP4_BLOCK));

    let amax = xs.iter().fold(0f32, |m, &x| m.max(x.abs()));
    // NVFP4 has max |E2M1|=6 and max finite positive E4M3FN=448.
    // This calibration leaves the largest possible local scale at 448.
    let global_scale = if amax == 0.0 {
        1.0
    } else {
        amax / (6.0 * 448.0)
    };

    let mut packed = vec![0u8; xs.len() / 2];
    let mut scales_e4m3 = vec![0u8; xs.len() / NVFP4_BLOCK];

    for (block_idx, block) in xs.chunks_exact(NVFP4_BLOCK).enumerate() {
        let block_amax = block.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let desired_scale = if block_amax == 0.0 {
            0.0
        } else {
            (block_amax / (6.0 * global_scale)).min(448.0)
        };
        let scale_raw = f32_to_e4m3fn_nearest(desired_scale);
        scales_e4m3[block_idx] = scale_raw;
        let scale = e4m3fn_to_f32(scale_raw) * global_scale;

        for i in 0..NVFP4_BLOCK / 2 {
            let q0 = if scale == 0.0 {
                0
            } else {
                nearest_e2m1(block[2 * i] / scale)
            };
            let q1 = if scale == 0.0 {
                0
            } else {
                nearest_e2m1(block[2 * i + 1] / scale)
            };
            packed[block_idx * (NVFP4_BLOCK / 2) + i] = q0 | (q1 << 4);
        }
    }

    Nvfp4Reference {
        packed,
        scales_e4m3,
        global_scale,
        len: xs.len(),
    }
}

fn nvfp4_decode(q: &Nvfp4Reference) -> Vec<f32> {
    let mut out = vec![0f32; q.len];
    for block_idx in 0..q.scales_e4m3.len() {
        let scale = e4m3fn_to_f32(q.scales_e4m3[block_idx]) * q.global_scale;
        let packed = &q.packed[
            block_idx * (NVFP4_BLOCK / 2)..(block_idx + 1) * (NVFP4_BLOCK / 2)
        ];
        for (i, &byte) in packed.iter().enumerate() {
            out[block_idx * NVFP4_BLOCK + 2 * i] =
                E2M1_VALUES[(byte & 0x0f) as usize] * scale;
            out[block_idx * NVFP4_BLOCK + 2 * i + 1] =
                E2M1_VALUES[(byte >> 4) as usize] * scale;
        }
    }
    out
}

// Scalar reproduction of xInfer's legacy FP4 LUT result. xInfer uses
// __byte_perm to materialize the same signed doubled table:
// [0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12].
// Multiplying by 0.5 recovers E2M1.
fn xinfer_lut_decode_byte(byte: u8) -> (i8, i8) {
    const LUT: [i8; 16] = [
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
    ];
    (LUT[(byte & 0x0f) as usize], LUT[(byte >> 4) as usize])
}

fn error_metrics(reference: &[f32], observed: &[f32]) -> (f32, f32, f32, f64) {
    assert_eq!(reference.len(), observed.len());
    let mut max_abs = 0f32;
    let mut sum_abs = 0f32;
    let mut sum_sq = 0f64;
    let mut dot = 0f64;
    let mut nr = 0f64;
    let mut no = 0f64;
    for (&a, &b) in reference.iter().zip(observed.iter()) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
        sum_sq += (d as f64) * (d as f64);
        dot += a as f64 * b as f64;
        nr += (a as f64) * (a as f64);
        no += (b as f64) * (b as f64);
    }
    (
        max_abs,
        sum_abs / reference.len() as f32,
        (sum_sq / reference.len() as f64).sqrt() as f32,
        dot / (nr.sqrt() * no.sqrt()).max(f64::MIN_POSITIVE),
    )
}

#[test]
fn nvfp4_e4m3_reference_values() {
    let cases = [
        (0x00, 0.0f32),
        (0x01, 2f32.powi(-9)),
        (0x08, 2f32.powi(-6)),
        (0x38, 1.0),
        (0x40, 2.0),
        (0x48, 4.0),
        (0x70, 128.0),
        (0x7e, 448.0),
        (0xb8, -1.0),
    ];
    for (raw, expected) in cases {
        assert_eq!(e4m3fn_to_f32(raw), expected, "E4M3FN decode mismatch for {raw:#04x}");
    }
}

#[test]
fn nvfp4_xinfer_lut_matches_e2m1() {
    for lo in 0u8..16 {
        for hi in 0u8..16 {
            let byte = lo | (hi << 4);
            let (a, b) = xinfer_lut_decode_byte(byte);
            assert_eq!(a as f32 * 0.5, E2M1_VALUES[lo as usize]);
            assert_eq!(b as f32 * 0.5, E2M1_VALUES[hi as usize]);
        }
    }
}

#[test]
fn nvfp4_reference_roundtrip_is_finite() {
    let input = (0..4096)
        .map(|i| ((i as f32) * 0.017).sin() * 2.5 + ((i as f32) * 0.003).cos() * 0.4)
        .collect::<Vec<_>>();
    let q = nvfp4_quantize_experimental(&input);
    let decoded = nvfp4_decode(&q);
    assert_eq!(q.packed.len(), input.len() / 2);
    assert_eq!(q.scales_e4m3.len(), input.len() / NVFP4_BLOCK);
    assert!(q.global_scale.is_finite() && q.global_scale > 0.0);
    assert!(decoded.iter().all(|x| x.is_finite()));
}

#[test]
fn compare_mxfp4_nvfp4_quality() -> Result<()> {
    let cpu = Device::Cpu;
    let input = (0..8192)
        .map(|i| {
            let x = i as f32;
            (x * 0.011).sin() * 2.0 + (x * 0.0013).cos() * 0.35 + (x * 0.00017).sin() * 0.08
        })
        .collect::<Vec<_>>();

    let x = Tensor::from_vec(input.clone(), (input.len(),), &cpu)?;
    let mx = QTensor::quantize(&x, GgmlDType::Mxfp4)?;
    let mx_dec = mx.dequantize(&cpu)?.to_vec1::<f32>()?;

    let nv = nvfp4_quantize_experimental(&input);
    let nv_dec = nvfp4_decode(&nv);

    let mx_m = error_metrics(&input, &mx_dec);
    let nv_m = error_metrics(&input, &nv_dec);

    let mx_bits = mx.data()?.len() as f64 * 8.0 / input.len() as f64;
    let nv_bytes = nv.packed.len() + nv.scales_e4m3.len() + std::mem::size_of::<f32>();
    let nv_bits = nv_bytes as f64 * 8.0 / input.len() as f64;

    println!(
        "FP4_QUALITY format=MXFP4 bits_per_weight={mx_bits:.4} max_abs={:.6} mean_abs={:.6} rmse={:.6} cosine={:.8}",
        mx_m.0, mx_m.1, mx_m.2, mx_m.3
    );
    println!(
        "FP4_QUALITY format=NVFP4_EXPERIMENTAL bits_per_weight={nv_bits:.4} max_abs={:.6} mean_abs={:.6} rmse={:.6} cosine={:.8} global_scale={:.9}",
        nv_m.0, nv_m.1, nv_m.2, nv_m.3, nv.global_scale
    );

    assert!((mx_bits - 4.25).abs() < 1e-6);
    assert!(nv_bits > 4.49 && nv_bits < 4.51);
    assert!(nv_m.3.is_finite() && mx_m.3.is_finite());
    Ok(())
}

#[test]
#[ignore = "reference microbenchmark, run explicitly when comparing FP4 formats"]
fn compare_mxfp4_nvfp4_reference_decode_perf() -> Result<()> {
    let cpu = Device::Cpu;
    let input = (0..1_048_576)
        .map(|i| {
            let x = i as f32;
            (x * 0.0017).sin() * 1.7 + (x * 0.00031).cos() * 0.2
        })
        .collect::<Vec<_>>();

    let tensor = Tensor::from_vec(input.clone(), (input.len(),), &cpu)?;
    let mx = QTensor::quantize(&tensor, GgmlDType::Mxfp4)?;
    let nv = nvfp4_quantize_experimental(&input);

    let runs = 20usize;

    let start = Instant::now();
    for _ in 0..runs {
        black_box(mx.dequantize(&cpu)?.to_vec1::<f32>()?);
    }
    let mx_us = start.elapsed().as_secs_f64() * 1e6 / runs as f64;

    let start = Instant::now();
    for _ in 0..runs {
        black_box(nvfp4_decode(&nv));
    }
    let nv_us = start.elapsed().as_secs_f64() * 1e6 / runs as f64;

    println!("FP4_REFERENCE_PERF format=MXFP4 latency_us={mx_us:.3}");
    println!("FP4_REFERENCE_PERF format=NVFP4_EXPERIMENTAL latency_us={nv_us:.3}");
    Ok(())
}


#[test]
#[ignore = "large release-only quality gate matching the MXFP4 SM61 benchmark workload"]
fn compare_mxfp4_nvfp4_matmul_quality_same_workload() -> Result<()> {
    let cpu = Device::Cpu;
    let (n, k) = (4096usize, 4096usize);

    println!(
        "FP4_SAME_WORKLOAD_CONFIG profile={} n={n} k={k}",
        if cfg!(debug_assertions) { "debug" } else { "release" }
    );

    let weights = (0..n * k)
        .map(|i| ((i as f32) * 0.0013).sin() * 0.75 + ((i as f32) * 0.0007).cos() * 0.25)
        .collect::<Vec<_>>();
    let x = (0..k)
        .map(|i| ((i as f32) * 0.017).cos() * 0.5 + 0.1)
        .collect::<Vec<_>>();

    let w_cpu = Tensor::from_vec(weights.clone(), (n, k), &cpu)?;
    let x_cpu = Tensor::from_vec(x, (1, k), &cpu)?;
    let reference = x_cpu
        .matmul(&w_cpu.t()?)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mx = QTensor::quantize(&w_cpu, GgmlDType::Mxfp4)?;
    let mx_dec = mx.dequantize(&cpu)?;
    let mx_out = x_cpu
        .matmul(&mx_dec.t()?)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mx_m = error_metrics(&reference, &mx_out);
    let mx_bits = mx.data()?.len() as f64 * 8.0 / (n * k) as f64;

    let nv = nvfp4_quantize_experimental(&weights);
    let nv_dec = Tensor::from_vec(nvfp4_decode(&nv), (n, k), &cpu)?;
    let nv_out = x_cpu
        .matmul(&nv_dec.t()?)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let nv_m = error_metrics(&reference, &nv_out);
    let nv_bytes = nv.packed.len() + nv.scales_e4m3.len() + std::mem::size_of::<f32>();
    let nv_bits = nv_bytes as f64 * 8.0 / (n * k) as f64;

    println!(
        "FP4_SAME_WORKLOAD format=MXFP4 bits_per_weight={mx_bits:.4} max_abs={:.6} mean_abs={:.6} rmse={:.6} cosine={:.8}",
        mx_m.0, mx_m.1, mx_m.2, mx_m.3
    );
    println!(
        "FP4_SAME_WORKLOAD format=NVFP4_EXPERIMENTAL bits_per_weight={nv_bits:.4} max_abs={:.6} mean_abs={:.6} rmse={:.6} cosine={:.8} global_scale={:.9}",
        nv_m.0, nv_m.1, nv_m.2, nv_m.3, nv.global_scale
    );

    assert!((mx_bits - 4.25).abs() < 1e-6);
    assert!(nv_bits > 4.49 && nv_bits < 4.51);
    assert!(mx_m.3.is_finite() && nv_m.3.is_finite());

    Ok(())
}
