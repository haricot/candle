//! Runtime regression for SM61 SIMT FP16 grouped MoE.
#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::moe::moe_gemm;

const K: usize = 16;
const N: usize = 8;
const EXPERTS: usize = 2;

fn data(num_tokens: usize) -> (Vec<f32>, Vec<f32>) {
    // Binary-exact FP16 test values; reference uses FP32 arithmetic.
    let inputs = (0..num_tokens * K)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.125)
        .collect();
    let weights = (0..EXPERTS * N * K)
        .map(|i| ((i * 3 % 13) as f32 - 6.0) * 0.0625)
        .collect();
    (inputs, weights)
}

#[allow(clippy::too_many_arguments)]
fn check(
    input_rows: usize,
    sorted_token_ids: &[u32],
    expert_ids: &[u32],
    token_experts: &[usize],
    topk: usize,
    route_weights: Option<&[f32]>,
    is_prefill: bool,
) -> Result<()> {
    let gpu = Device::new_cuda(0)?;
    let (inputs, weights) = data(input_rows);
    let x = Tensor::from_vec(inputs.clone(), (input_rows, K), &gpu)?.to_dtype(DType::F16)?;
    let w = Tensor::from_vec(weights.clone(), (EXPERTS, N, K), &gpu)?.to_dtype(DType::F16)?;
    let sorted = Tensor::from_vec(sorted_token_ids.to_vec(), sorted_token_ids.len(), &gpu)?;
    let experts = Tensor::from_vec(expert_ids.to_vec(), expert_ids.len(), &gpu)?;
    let topk_weights = route_weights
        .map(|weights| Tensor::from_vec(weights.to_vec(), weights.len(), &gpu))
        .transpose()?;
    let result = moe_gemm(&x, &w, &topk_weights, &sorted, &experts, topk, is_prefill)?;
    gpu.synchronize()?;
    let output = result.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

    assert_eq!(output.len(), token_experts.len() * N);
    for (row, &expert) in token_experts.iter().enumerate() {
        let input_row = if route_weights.is_some() { row } else { row / topk };
        for n in 0..N {
            let reference: f32 = (0..K)
                .map(|k| inputs[input_row * K + k] * weights[(expert * N + n) * K + k])
                .sum();
            let scale = route_weights.map_or(1.0, |weights| weights[row]);
            let actual = output[row * N + n];
            let expected = reference * scale;
            assert!(
                (actual - expected).abs() <= 0.03,
                "SIMT MoE parity failed: prefill={is_prefill} row={row} n={n} \
                 expert={expert} actual={actual} expected={expected}"
            );
        }
    }
    Ok(())
}

#[test]
fn sm61_simt_prefill_topk2_fp16() -> Result<()> {
    // Two tokens, two routes each: expert 0 receives output rows 0 and 2,
    // expert 1 receives output rows 1 and 3.
    check(2, &[0, 2, 1, 3], &[0, 0, 1, 1], &[0, 1, 0, 1], 2, None, true)
}

#[test]
fn sm61_simt_decode_weighted_fp16() -> Result<()> {
    // Exercise the light prefix-sum path, nonidentity routing and FP32
    // per-output route weights. Experts are sorted in the routing arrays.
    check(
        3, &[1, 0, 2], &[0, 1, 1], &[1, 0, 1], 1,
        Some(&[0.5, 0.75, 1.25]), false,
    )
}
