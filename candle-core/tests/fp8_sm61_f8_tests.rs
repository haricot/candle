#![cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
use candle_core::quantized::fp8_linear_sm61::{
    dp4a_launch_for_tokens, e4m3fn_decode, fp8_sm61_select_mode, Fp8Sm61Linear, Fp8Sm61Mode,
};
use candle_core::quantized::fp8_sm61_dispatch::{
    dispatch_residence, Fp8Backend, Fp8DispatchBudget, Fp8LinearSpec, Fp8MeasuredProfile,
};
use candle_core::{DType, Device, Result, Tensor};

#[test]
fn f8_sm61_e4m3fn_cpu_encoding_oracle() {
    assert_eq!(e4m3fn_decode(0x38), 1.0);
    assert_eq!(e4m3fn_decode(0xb8), -1.0);
    assert_eq!(e4m3fn_decode(0x01), 1.0 / 512.0);
    assert_eq!(e4m3fn_decode(0x7e), 448.0);
    assert!(e4m3fn_decode(0x7f).is_nan());
    assert_eq!(e4m3fn_decode(0x80).to_bits(), (-0f32).to_bits());
}

#[test]
fn f8e_r4c_dispatch_defaults_simt_cpu() -> Result<()> {
    for m in [1usize, 2, 3] {
        assert_eq!(dp4a_launch_for_tokens(m), ("fp8_sm61_dp4a_gemv", 1));
    }
    for m in [4usize, 8, 16, 64, 128] {
        assert_eq!(dp4a_launch_for_tokens(m), ("fp8_sm61_dp4a_prefill4", 4));
    }
    assert!(dp4a_launch_for_tokens(8).0 != "fp8_sm61_dp4a_prefill8");
    let fp8 = 1024 * 1024 + 1024 * 1024 / 128 * 4;
    let int8 = fp8;
    let f16 = 1024 * 1024 * 2;
    assert_eq!(
        fp8_sm61_select_mode(1, 1024, 1024, 128, fp8, int8, f16, 3 * 1024 * 1024)?,
        Fp8Sm61Mode::SimtW8A16
    );
    assert_eq!(
        fp8_sm61_select_mode(1, 1024, 1024, 64, fp8, int8, f16, 3 * 1024 * 1024)?,
        Fp8Sm61Mode::SimtW8A16
    );
    assert_eq!(
        fp8_sm61_select_mode(1, 2048, 2048, 128, fp8, int8, f16, 3 * 1024 * 1024)?,
        Fp8Sm61Mode::SimtW8A16
    );
    assert_eq!(
        fp8_sm61_select_mode(1, 4096, 1024, 128, fp8, int8, f16, 3 * 1024 * 1024)?,
        Fp8Sm61Mode::SimtW8A16
    );
    assert_eq!(
        fp8_sm61_select_mode(16, 1024, 1024, 128, fp8, int8, f16, 3 * 1024 * 1024)?,
        Fp8Sm61Mode::SimtW8A16
    );
    assert_eq!(
        fp8_sm61_select_mode(16, 1024, 1024, 128, fp8, int8, f16, 1500000)?,
        Fp8Sm61Mode::SimtW8A16
    );
    assert!(fp8_sm61_select_mode(0, 1024, 1024, 128, fp8, int8, f16, 3 * 1024 * 1024).is_err());
    Ok(())
}

#[test]
fn f8_sm61_linear_simt_dp4a_cuda() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let cuda = match &device {
        Device::Cuda(d) => d,
        _ => unreachable!(),
    };
    let (n, k, m) = (64usize, 256usize, 8usize);
    let patterns = [0x38u8, 0xb8, 0x30, 0xb0, 0x40, 0xc0, 0x28, 0xa8];
    let weights = (0..n * k)
        .map(|i| patterns[(i * 13 + i / 17) % patterns.len()])
        .collect::<Vec<_>>();
    let scales = vec![0.05f32; n * 2]; // per-row, per 128 K-block
    let input = (0..m * k)
        .map(|i| ((i % 97) as f32 - 46.) / 40.)
        .collect::<Vec<_>>();
    let x = Tensor::from_vec(input, (m, k), &device)?.to_dtype(DType::F16)?;
    let simt = Fp8Sm61Linear::from_bytes(
        cuda,
        n,
        k,
        &weights,
        &scales,
        1,
        128,
        Fp8Sm61Mode::SimtW8A16,
        128,
    )?;
    let dp4a = Fp8Sm61Linear::from_bytes(
        cuda,
        n,
        k,
        &weights,
        &scales,
        1,
        128,
        Fp8Sm61Mode::Dp4aW8A8,
        128,
    )?;
    let expanded = Fp8Sm61Linear::from_bytes(
        cuda,
        n,
        k,
        &weights,
        &scales,
        1,
        128,
        Fp8Sm61Mode::ExpandedF16,
        128,
    )?;
    let auto_decode = Fp8Sm61Linear::from_bytes_profiled(
        cuda,
        n,
        k,
        &weights,
        &scales,
        1,
        128,
        1,
        64 * 1024,
        128,
    )?;
    let auto_prefill = Fp8Sm61Linear::from_bytes_profiled(
        cuda,
        n,
        k,
        &weights,
        &scales,
        1,
        128,
        8,
        64 * 1024,
        128,
    )?;
    assert_eq!(simt.resident_weight_bytes(), n * k + scales.len() * 4);
    assert_eq!(dp4a.resident_weight_bytes(), n * k + n * (k / 128) * 4);
    assert_eq!(expanded.resident_weight_bytes(), n * k * 2);
    assert_eq!(auto_decode.mode(), Fp8Sm61Mode::SimtW8A16);
    assert_eq!(auto_prefill.mode(), Fp8Sm61Mode::SimtW8A16);
    let auto_prefill_g64 = Fp8Sm61Linear::from_bytes_profiled(
        cuda,
        n,
        k,
        &weights,
        &scales,
        1,
        128,
        8,
        64 * 1024,
        64,
    )?;
    assert_eq!(auto_prefill_g64.mode(), Fp8Sm61Mode::SimtW8A16);
    let spec = Fp8LinearSpec {
        out_features: n,
        in_features: k,
        block_n: 1,
        block_k: 128,
        resident_calls: 10,
        profile: Some(Fp8MeasuredProfile {
            simt_ms: 8.0,
            expanded_f16_ms: 3.0,
            extra_expand_ms: 5.0,
        }),
    };
    let env = Fp8DispatchBudget {
        free_device_bytes: 128 * 1024,
        workload_reserve_bytes: 32 * 1024,
        safety_headroom_bytes: 32 * 1024,
        min_steady_speedup: 1.03,
        allow_auto_f16: true,
    };
    let plan = dispatch_residence(&[spec], env).unwrap();
    assert_eq!(plan.entries[0].backend, Fp8Backend::ExpandedF16);
    let admitted = Fp8Sm61Linear::from_bytes_with_dispatch(
        cuda,
        n,
        k,
        &weights,
        &scales,
        1,
        128,
        &plan.entries[0],
    )?;
    assert_eq!(admitted.mode(), Fp8Sm61Mode::ExpandedF16);
    assert_eq!(
        admitted.resident_weight_bytes(),
        plan.entries[0].resident_bytes
    );
    let reference = simt.forward(&x)?.to_vec2::<f32>()?;
    let compare = dp4a.forward(&x)?.to_vec2::<f32>()?;
    let expanded_output = expanded.forward(&x)?.to_vec2::<f32>()?;
    assert_eq!(expanded_output.len(), m);
    assert!(expanded_output.iter().flatten().all(|v| v.is_finite()));
    let mut max_abs = 0f64;
    let mut ab = 0f64;
    let mut aa = 0f64;
    let mut bb = 0f64;
    for (xs, ys) in reference.iter().zip(compare.iter()) {
        for (&a, &b) in xs.iter().zip(ys.iter()) {
            assert!(a.is_finite() && b.is_finite());
            max_abs = max_abs.max((a - b).abs() as f64);
            aa += f64::from(a) * f64::from(a);
            ab += f64::from(a) * f64::from(b);
            bb += f64::from(b) * f64::from(b);
        }
    }
    let cosine = ab / (aa * bb).sqrt();
    eprintln!("F8_SM61_PARITY cosine={cosine:.9} max_abs={max_abs:.8} shape=[{m},{n},{k}]");
    assert!(
        cosine >= 0.9999,
        "F8E DP4A double requantization too inaccurate: {cosine}"
    );
    Ok(())
}

/// A real F8_E4M3 checkpoint is required for this test to claim qualification.
/// The default test run records NOT_TESTED; the one-pass driver refuses to
/// promote without a supplied checkpoint. This is layer parity, not logits.
#[test]
fn f8_sm61_real_safetensors_layer_optional() -> Result<()> {
    let (path, prefix) = match (
        std::env::var("HARICOT_F8_CHECKPOINT"),
        std::env::var("HARICOT_F8_PREFIX"),
    ) {
        (Ok(path), Ok(prefix)) => (path, prefix),
        _ => {
            eprintln!("F8D_REAL_LAYER=NOT_TESTED (supply --safetensors and --prefix)");
            return Ok(());
        }
    };
    let threshold: f64 = std::env::var("HARICOT_F8_MIN_COSINE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.9999);
    let device = Device::new_cuda(0)?;
    let cuda = match &device {
        Device::Cuda(d) => d,
        _ => unreachable!(),
    };
    let simt = Fp8Sm61Linear::from_safetensors(cuda, &path, &prefix, Fp8Sm61Mode::SimtW8A16, 128)?;
    let (n, k) = simt.shape();
    let mut runs = Vec::new();
    for &m in &[1usize, 8, 16] {
        // Mixed deterministic activation distributions: bounded periodic + outlier.
        let input = (0..m * k)
            .map(|i| {
                let v = ((i % 47) as f32 - 23.) / 34.;
                if i % 257 == 0 {
                    v * 3.0
                } else {
                    v
                }
            })
            .collect::<Vec<_>>();
        let x = Tensor::from_vec(input, (m, k), &device)?.to_dtype(DType::F16)?;
        let y = simt.forward(&x)?.to_vec2::<f32>()?;
        runs.push((m, x, y));
    }
    drop(simt);
    // DP4A with exclusive INT8 weights, no simultaneous FP8 GPU resident weights.
    let dp = Fp8Sm61Linear::from_safetensors(cuda, &path, &prefix, Fp8Sm61Mode::Dp4aW8A8, 128)?;
    let mut results = Vec::new();
    for (m, x, reference) in &runs {
        let candidate = dp.forward(x)?.to_vec2::<f32>()?;
        let (mut ab, mut aa, mut bb, mut squared, mut max_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
        let mut flips = 0usize;
        for (ys, zs) in reference.iter().zip(candidate.iter()) {
            let mut ai = 0usize;
            let mut bi = 0usize;
            for (j, (&a, &b)) in ys.iter().zip(zs.iter()).enumerate() {
                assert!(a.is_finite() && b.is_finite());
                ab += f64::from(a) * f64::from(b);
                aa += f64::from(a) * f64::from(a);
                bb += f64::from(b) * f64::from(b);
                let d = f64::from(a) - f64::from(b);
                squared += d * d;
                max_abs = max_abs.max(d.abs());
                if a > ys[ai] {
                    ai = j;
                }
                if b > zs[bi] {
                    bi = j;
                }
            }
            if ai != bi {
                flips += 1;
            }
        }
        let cosine = if aa == 0. && bb == 0. {
            1.0
        } else if aa > 0. && bb > 0. {
            ab / (aa * bb).sqrt()
        } else {
            0.0
        };
        let rmse = (squared / (*m * n) as f64).sqrt();
        eprintln!("F8_R1_REAL_ROW m={m} n={n} k={k} cosine={cosine:.9} rmse={rmse:.9} max_abs={max_abs:.9} argmax_flips={flips}");
        assert!(
            cosine >= threshold,
            "F8E-R1 real FP8 layer cosine {cosine} below {threshold}"
        );
        results.push((*m, cosine, rmse, max_abs, flips));
    }
    assert_eq!(results.len(), 3);
    drop(dp);
    // Independent FP16/cuBLAS expansion path detects loader/scale contract errors
    // that a comparison between two compressed kernels could otherwise miss.
    // Constructed only AFTER releasing the INT8 representation.
    let f16 = Fp8Sm61Linear::from_safetensors(cuda, &path, &prefix, Fp8Sm61Mode::ExpandedF16, 128)?;
    let mut f16_min_cos = 1.0f64;
    for (m, x, reference) in &runs {
        let candidate = f16.forward(x)?.to_vec2::<f32>()?;
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (ys, zs) in reference.iter().zip(candidate.iter()) {
            for (&a, &b) in ys.iter().zip(zs.iter()) {
                assert!(a.is_finite() && b.is_finite());
                ab += f64::from(a) * f64::from(b);
                aa += f64::from(a) * f64::from(a);
                bb += f64::from(b) * f64::from(b);
            }
        }
        let cosine = if aa == 0. && bb == 0. {
            1.0
        } else if aa > 0. && bb > 0. {
            ab / (aa * bb).sqrt()
        } else {
            0.0
        };
        f16_min_cos = f16_min_cos.min(cosine);
        eprintln!("F8_R1_REAL_F16_ROW m={m} n={n} k={k} cosine={cosine:.9}");
        assert!(
            cosine >= threshold,
            "F8E-R1 real FP8 loader differs from FP16 expansion: {cosine}"
        );
    }
    eprintln!("F8D_REAL_FP16_REFERENCE=PASS cases=3 min_cosine={f16_min_cos:.9}");
    eprintln!("F8D_REAL_LAYER=PASS_R1_CHECKPOINT prefix={prefix} shape=[{n},{k}] cases=3 min_cosine={:.9} argmax_flips={}",
              results.iter().map(|(_,c,_,_,_)|*c).fold(1.0f64,f64::min),
              results.iter().map(|(_,_,_,_,f)|*f).sum::<usize>());
    Ok(())
}

#[test]
fn expanded_f16_preserves_f32_output_range() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let cuda = match &device {
        Device::Cuda(d) => d,
        _ => unreachable!(),
    };
    let weights = vec![0x38; 2 * 128]; // exactly one in E4M3FN
    let scales = [1f32];
    let linear = Fp8Sm61Linear::from_bytes(
        cuda,
        2,
        128,
        &weights,
        &scales,
        2,
        128,
        Fp8Sm61Mode::ExpandedF16,
        128,
    )?;
    let x = Tensor::from_vec(vec![half::f16::MAX; 128], (1, 128), &device)?;
    let y = linear.forward(&x)?;
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(y.to_vec2::<f32>()?, vec![vec![65504.0 * 128.0; 2]]);
    let overflow = Fp8Sm61Linear::from_bytes(
        cuda,
        2,
        128,
        &vec![0x7e; 2 * 128],
        &[200.0],
        2,
        128,
        Fp8Sm61Mode::ExpandedF16,
        128,
    );
    assert!(overflow.is_err());
    Ok(())
}
