//! Runtime regression for SM61 SIMT FP16 grouped MoE.
#![cfg(feature = "cuda")]

use candle::{DType, Device, Result, Tensor};
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

#[cfg(feature = "cuda-dkr-catalog")]
#[test]
#[ignore = "requires a selected verified native DKR catalogue"]
fn dkr_routed_gguf_moe_cuda() -> Result<()> {
    use candle::quantized::{GgmlDType,QTensor};
    use candle_nn::moe::moe_gemm_gguf;
    let gpu=Device::new_cuda(0)?;
    let dev=gpu.as_cuda_device()?;
    // This test only qualifies the native route. Static FFI has its own tests.
    assert!(dev.dkr_moe_enabled(),"set the native DKR catalogue for GGUF MoE qualification");
    let (e,n,k)=(2usize,5usize,256usize);
    let weights=(0..e*n*k).map(|i|if i<n*k{0.5f32}else{1.0f32}).collect::<Vec<_>>();
    let w=Tensor::from_vec(weights,(e,n,k),&gpu)?;
    let w=QTensor::quantize(&w,GgmlDType::Q8_0)?;
    let x=Tensor::from_vec(vec![0.25f32;2*k],(2,k),&gpu)?;
    let sorted=Tensor::from_vec(vec![0u32,2,1,3],4,&gpu)?;
    let experts=Tensor::from_vec(vec![0u32,0,1,1],4,&gpu)?;
    let result=moe_gemm_gguf(&x,&w,&None,&sorted,&experts,2,true,DType::F32)?;
    assert_eq!(result.dims(),&[4,n]);
    let values=result.flatten_all()?.to_vec1::<f32>()?;
    for (i,v) in values.iter().enumerate() {
        let expected=if (i/n)%2==0{32.}else{64.};
        assert!((v-expected).abs()<0.5,"GGUF MoE row {} got {v}",i/n);
    }
    // A selected native SIMT FP16 catalogue cannot dispatch BF16 via static FFI.
    assert!(dev.dkr_moe_admit(false,DType::BF16).is_err());
    Ok(())
}


#[cfg(feature = "cuda-dkr-embedded")]
#[test]
#[ignore = "SM61 headless archive only: six native GGUF MoE codecs"]
fn dkr_s52_gguf_moe_six_codecs() -> Result<()> {
    use candle::quantized::{GgmlDType, QTensor};
    use candle_nn::moe::moe_gemm_gguf;
    let gpu = Device::new_cuda(0)?;
    assert!(gpu.as_cuda_device()?.dkr_moe_enabled());
    let (experts, n, k) = (2usize,5usize,256usize);
    let w = (0..experts*n*k).map(|i| {
        let expert = i/(n*k);
        (if expert==0 {0.40f32} else {0.70f32}) +
            ((i * 7 % 13) as f32)*0.005
    }).collect::<Vec<_>>();
    let weights = Tensor::from_vec(w,(experts,n,k),&gpu)?;
    let x=Tensor::from_vec(vec![0.25f32;2*k],(2,k),&gpu)?;
    let sorted=Tensor::from_vec(vec![0u32,2,1,3],4,&gpu)?;
    let experts_id=Tensor::from_vec(vec![0u32,0,1,1],4,&gpu)?;
    for codec in [GgmlDType::Q8_0,GgmlDType::Q4K,GgmlDType::Q2K,
                  GgmlDType::Q3K,GgmlDType::Q5K,GgmlDType::Q6K] {
        let q=QTensor::quantize(&weights,codec)?;
        let dequant=q.dequantize(&gpu)?.flatten_all()?.to_vec1::<f32>()?;
        let out=moe_gemm_gguf(&x,&q,&None,&sorted,&experts_id,2,true,DType::F32)?;
        assert_eq!(out.dims(),&[4,n], "{codec:?}");
        let values=out.flatten_all()?.to_vec1::<f32>()?;
        for row in 0..4usize {
            let expert=row%2;
            for col in 0..n {
                let offset=(expert*n+col)*k;
                let expected=dequant[offset..offset+k].iter().sum::<f32>()*0.25;
                let got=values[row*n+col];
                assert!(got.is_finite() && (got-expected).abs()<0.75+expected.abs()*0.025,
                        "GGUF MoE {codec:?} row={row} col={col} got={got} expected={expected}");
            }
        }
        println!("DKR_R124_S52_GGUF_MOE codec={codec:?} rows=4 n={n} k={k}");
    }
    println!("DKR_R124_S52_GGUF_MOE=PASS codecs=6");
    Ok(())
}
