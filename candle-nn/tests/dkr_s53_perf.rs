//! R1.24 S5.3: synchronized end-to-end CUDA operation timings.
//! Tests are ignored by default. The Python headless harness chooses CUDA_CACHE_PATH.
#![cfg(feature = "cuda-dkr-embedded")]

use candle::{quantized::{GgmlDType, QMatMul, QTensor}, DType, Device, Module, Result, Tensor};
use candle_nn::moe::{moe_gemm, moe_gemm_gguf};
use std::{hint::black_box, time::Instant};

const WARMUPS: usize = 24;
const SAMPLES: usize = 80;

fn samples(
    device: &Device,
    phase: &str,
    round: usize,
    name: &str,
    tokens: usize,
    mut launch: impl FnMut() -> Result<Tensor>,
) -> Result<()> {
    device.synchronize()?;
    let start = Instant::now();
    let initial = launch()?;
    device.synchronize()?;
    let first_us = start.elapsed().as_micros();
    black_box(initial);
    for _ in 0..WARMUPS {
        let value = launch()?;
        device.synchronize()?;
        black_box(value);
    }
    let mut durations = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        device.synchronize()?;
        let start = Instant::now();
        let value = launch()?;
        device.synchronize()?;
        durations.push(start.elapsed().as_micros());
        black_box(value);
    }
    durations.sort_unstable();
    let p50 = durations[SAMPLES / 2];
    let p95 = durations[(SAMPLES * 95).div_ceil(100) - 1];
    println!("DKR_R124_S53_CASE phase={phase} round={round} name={name} first_us={first_us} p50_us={p50} p95_us={p95} min_us={} max_us={} tokens={tokens} samples={SAMPLES}",
             durations[0], durations[SAMPLES-1]);
    Ok(())
}

fn quantized(
    gpu: &Device,
    phase: &str,
    round: usize,
    name: &str,
    dtype: GgmlDType,
    batch: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    // Weight preparation and upload are intentionally outside the timed loop.
    let weight = (0..n*k).map(|i| 0.20 + ((i * 13 % 31) as f32)*0.0078125)
        .collect::<Vec<_>>();
    let w = Tensor::from_vec(weight, (n,k), gpu)?;
    let q = QTensor::quantize(&w,dtype)?;
    let mm = QMatMul::from_qtensor(q)?;
    let input = (0..batch*k).map(|i| 0.12 + ((i*7 % 23) as f32)*0.00390625)
        .collect::<Vec<_>>();
    let x = Tensor::from_vec(input,(batch,k),gpu)?.to_dtype(DType::F16)?;
    samples(gpu,phase,round,name,batch,|| {
        let result = mm.forward(&x)?;
        assert_eq!(result.dims(), &[batch,n]);
        Ok(result)
    })
}

fn sorted_routes(gpu: &Device, tokens: usize) -> Result<(Tensor,Tensor)> {
    // For topk=2, each token expands to two rows, one for each expert.
    let ids = (0..tokens).map(|i| (2*i) as u32)
        .chain((0..tokens).map(|i| (2*i+1) as u32))
        .collect::<Vec<_>>();
    let experts = (0..tokens).map(|_| 0u32)
        .chain((0..tokens).map(|_| 1u32))
        .collect::<Vec<_>>();
    Ok((Tensor::from_vec(ids,2*tokens,gpu)?,
        Tensor::from_vec(experts,2*tokens,gpu)?))
}

fn gguf_moe(gpu: &Device, phase: &str, round: usize) -> Result<()> {
    let (tokens, experts, n, k) = (8usize,2usize,128usize,1024usize);
    let weights = (0..experts*n*k)
        .map(|i| 0.3f32 + ((i*11 % 29) as f32)*0.0078125)
        .collect::<Vec<_>>();
    let w = Tensor::from_vec(weights,(experts,n,k),gpu)?;
    let q = QTensor::quantize(&w,GgmlDType::Q4K)?;
    let x=Tensor::from_vec(vec![0.25f32;tokens*k],(tokens,k),gpu)?;
    let (sorted,expert_ids)=sorted_routes(gpu,tokens)?;
    samples(gpu,phase,round,"moe_gguf_q4k_t8",tokens,|| {
        let y=moe_gemm_gguf(&x,&q,&None,&sorted,&expert_ids,2,true,DType::F32)?;
        assert_eq!(y.dims(),&[tokens*2,n]);
        Ok(y)
    })
}

fn simt_moe(gpu: &Device, phase: &str, round: usize, decode: bool) -> Result<()> {
    let (tokens,experts,n,k,topk) = if decode {(3usize,2usize,128usize,128usize,1usize)}
                                   else {(8usize,2usize,128usize,128usize,2usize)};
    let weights=(0..experts*n*k)
        .map(|i| ((i*7 % 31) as f32-15.)*0.015625).collect::<Vec<_>>();
    let w=Tensor::from_vec(weights,(experts,n,k),gpu)?.to_dtype(DType::F16)?;
    let x=Tensor::from_vec(vec![0.25f32;tokens*k],(tokens,k),gpu)?.to_dtype(DType::F16)?;
    let (sorted,expert_ids) = if decode {
        (Tensor::from_vec(vec![0u32,1,2],3,gpu)?,
         Tensor::from_vec(vec![0u32,1,1],3,gpu)?)
    } else {
        sorted_routes(gpu,tokens)?
    };
    let name=if decode {"moe_simt_f16_decode_t3"} else {"moe_simt_f16_prefill_t8"};
    samples(gpu,phase,round,name,tokens,|| {
        let y=moe_gemm(&x,&w,&None,&sorted,&expert_ids,topk,!decode)?;
        assert_eq!(y.dims(),&[tokens*topk,n]);
        Ok(y)
    })
}

#[test]
#[ignore = "S5.3 benchmark; use the sealed headless archive runner"]
fn dkr_s53_bench() -> Result<()> {
    let phase=std::env::var("DKR_R124_S53_PHASE").unwrap_or_default();
    assert!(phase=="cold" || phase=="warm","set DKR_R124_S53_PHASE=cold|warm");
    let round=std::env::var("DKR_R124_S53_ROUND")
        .expect("set DKR_R124_S53_ROUND")
        .parse::<usize>().expect("integer round");
    let created=Instant::now();
    let gpu=Device::new_cuda(0)?;
    let init_us=created.elapsed().as_micros();
    assert!(gpu.as_cuda_device()?.dkr_moe_enabled());
    println!("DKR_R124_S53_DEVICE phase={phase} round={round} init_us={init_us}");
    quantized(&gpu,&phase,round,"mmvq_q4k_b1",GgmlDType::Q4K,1,512,1024)?;
    quantized(&gpu,&phase,round,"mmvq_q8_0_b8",GgmlDType::Q8_0,8,512,1024)?;
    quantized(&gpu,&phase,round,"mmq_q4k_b16",GgmlDType::Q4K,16,512,1024)?;
    quantized(&gpu,&phase,round,"mmq_q8_0_b32",GgmlDType::Q8_0,32,512,1024)?;
    gguf_moe(&gpu,&phase,round)?;
    simt_moe(&gpu,&phase,round,true)?;
    simt_moe(&gpu,&phase,round,false)?;
    gpu.synchronize()?;
    println!("DKR_R124_S53_TEST=PASS phase={phase} round={round} cases=7 samples={SAMPLES}");
    Ok(())
}
