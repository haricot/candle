//! S5.5d: identical-input numeric evidence for separate embedded PTX and
//! Cudaforge CUBIN binaries. Each run serializes *every* output element.
#![cfg(feature = "cuda-dkr-embedded")]

use candle::quantized::{fp4_e7c_runtime::{Fp4E7CLinear, Fp4Execution}, fp8_linear_sm61::{Fp8Sm61Linear, Fp8Sm61Mode}, GgmlDType, QMatMul, QStorage, QTensor};
use candle::{DType, Device, Module, Tensor};
use candle_nn::moe::{moe_gemm, moe_gemm_gguf};
use serde_json::{json, Map, Value};
use std::{borrow::Cow, path::PathBuf};
use sha2::{Digest, Sha256};

const CODECS: [GgmlDType; 10] = [
    GgmlDType::Q4_0, GgmlDType::Q4_1, GgmlDType::Q5_0,
    GgmlDType::Q5_1, GgmlDType::Q8_0, GgmlDType::Q2K,
    GgmlDType::Q3K, GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K,
];
const MOE_CODECS: [GgmlDType; 6] = [
    GgmlDType::Q8_0, GgmlDType::Q4K, GgmlDType::Q2K,
    GgmlDType::Q3K, GgmlDType::Q5K, GgmlDType::Q6K,
];

fn capture(records: &mut Map<String, Value>, name: String, t: &Tensor) -> anyhow::Result<()> {
    let shape=t.dims().to_vec();
    let dtype=format!("{:?}",t.dtype());
    let values=t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    anyhow::ensure!(!values.is_empty(), "{name}: empty output");
    anyhow::ensure!(values.iter().all(|v| v.is_finite()), "{name}: nonfinite output");
    anyhow::ensure!(!records.contains_key(&name), "{name}: duplicate case");
    records.insert(name,json!({"shape":shape,"dtype":dtype,"values":values}));
    Ok(())
}

#[test]
#[ignore = "S5.5d PTX/CUBIN headless pair; DKR_S55D_OUTPUT must be set"]
fn dkr_s55d_capture_outputs() -> anyhow::Result<()> {
    let destination=PathBuf::from(std::env::var("DKR_S55D_OUTPUT")?);
    anyhow::ensure!(destination.is_absolute() && !destination.exists(),
                    "numeric output must be a fresh absolute path");
    let mode=std::env::var("DKR_S55D_MODE")?;
    anyhow::ensure!(mode=="ptx" || mode=="cubin","unknown image mode");
    let gpu=Device::new_cuda(0)?;
    anyhow::ensure!(gpu.as_cuda_device()?.dkr_moe_enabled(),"embedded DKR required");
    let mut records=Map::new();
    let mut inputs=Map::new();
    let cpu=Device::Cpu;
    let mut codec_count=0usize;
    let (n,k)=(65usize,256usize);
    let weights=(0..n*k).map(|i|0.2f32+((i*13%19) as f32)*0.025)
        .collect::<Vec<_>>();
    let w_cpu=Tensor::from_vec(weights,(n,k),&cpu)?;
    for codec in CODECS {
        // Bit-identical quantized weights in both builds: quantize only on
        // CPU and upload the exact same bytes to the selected CUDA backend.
        let q_cpu=QTensor::quantize(&w_cpu,codec)?;
        let bytes=q_cpu.data()?.to_vec();
        inputs.insert(format!("gguf/{codec:?}"),
            json!(format!("{:x}",Sha256::digest(&bytes))));
        let q=QTensor::new(QStorage::from_data(Cow::Owned(bytes),&gpu,codec)?,(n,k))?;
        let op=QMatMul::from_qtensor(q)?;
        for dtype in [DType::F16,DType::BF16,DType::F32] {
            for batch in [1usize,8,9,17] {
                let input=(0..batch*k).map(|i|
                    0.15f32+((i*7%23) as f32)*0.01).collect::<Vec<_>>();
                let x=Tensor::from_vec(input,(batch,k),&gpu)?.to_dtype(dtype)?;
                let y=op.forward(&x)?;
                gpu.synchronize()?;
                capture(&mut records,format!("gguf/{codec:?}/{dtype:?}/b{batch}"),&y)?;
                codec_count+=1;
            }
        }
    }
    anyhow::ensure!(codec_count==120,"missing 120 GGUF cases");

    // BF16 edge magnitudes, signed zeros and rounding boundaries.
    let values=[
        -0.0f32, 0., 0.00006103515625, -0.0001220703125,
        0.9990234375, 1.00390625, -1.0078125, 3.1415927,
        -8.25, 15.875, 127.5, -256., 384., 0.03125,
        -0.0625, 0.5,
    ];
    let x=Tensor::from_vec(values.to_vec(),(2,8),&gpu)?;
    capture(&mut records,"bf16/roundtrip".into(),&x.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?;
    let b=x.to_dtype(DType::BF16)?;
    let rhs=Tensor::from_vec(
        (0..64).map(|i|((i*7%29) as f32-14.)/16.).collect(),
        (8,8),&gpu)?.to_dtype(DType::BF16)?;
    capture(&mut records,"bf16/matmul".into(),&b.matmul(&rhs)?)?;
    capture(&mut records,"bf16/sum".into(),&b.sum(0)?)?;
    capture(&mut records,"bf16/unary".into(),&b.abs()?.affine(0.5,-0.125)?)?;

    // Explicit FP8 E4M3FN small, extreme and negative zero codes
    // (never 0x7f NaN); identical serialized bytes and FP16 inputs.
    let cuda=gpu.as_cuda_device()?;
    let (n8,k8)=(64usize,256usize);
    let patterns=[0x38u8,0xb8,0x30,0xb0,0x40,0xc0,0x28,0xa8,0x01,0x81,0x7e,0xfe,0x80];
    let bytes=(0..n8*k8).map(|i|patterns[(i*13+i/17)%patterns.len()])
        .collect::<Vec<_>>();
    let scales=vec![0.05f32;n8*2];
    for (label,kind) in [
        ("simt",Fp8Sm61Mode::SimtW8A16),
        ("dp4a",Fp8Sm61Mode::Dp4aW8A8),
    ] {
        let layer=Fp8Sm61Linear::from_bytes(
            cuda,n8,k8,&bytes,&scales,1,128,kind,128)?;
        for m in [1usize,3,8,9,17] {
            let activations=(0..m*k8).map(|i|{
                let value=((i%97) as f32-46.)/40.;
                if i%127==0 { value*2.5 } else {value}
            }).collect::<Vec<_>>();
            let x=Tensor::from_vec(activations,(m,k8),&gpu)?.to_dtype(DType::F16)?;
            let y=layer.forward(&x)?;
            gpu.synchronize()?;
            capture(&mut records,format!("fp8/{label}/b{m}"),&y)?;
        }
    }

    // Direct ModelOpt NVFP4 reference/DP4A with partial tiles,
    // E2M1 zeros, signs and extreme scale encodings.
    let (n4,k4)=(5usize,64usize);
    let packed=(0..n4*k4/2).map(|i|(i*37+11) as u8).collect::<Vec<_>>();
    let scales=(0..n4*k4/16).map(|i|[0x30u8,0x38,0x40][i%3])
        .collect::<Vec<_>>();
    let fp4=Fp4E7CLinear::from_modelopt_bytes(cuda,n4,k4,&packed,&scales,0.5)?;
    for m in [1usize,3,8,9,17] {
        // The test is not conditional on half::f16 visibility in candle-nn:
        // convert directly through Candle's FP16 kernel.
        let activations=(0..m*k4).map(|i|(i as i32%19-9) as f32/8.)
            .collect::<Vec<_>>();
        let x=Tensor::from_vec(activations,(m,k4),&gpu)?.to_dtype(DType::F16)?;
        for (route,kind) in [
            ("scalar",Fp4Execution::Nvfp4W4A16Reference),
            ("reference",Fp4Execution::Nvfp4W4A8Reference),
            ("dp4a",Fp4Execution::Nvfp4W4A8Dp4a),
        ] {
            let y=fp4.forward(&x,kind)?;
            gpu.synchronize()?;
            capture(&mut records,format!("fp4/{route}/b{m}"),&y)?;
        }
    }

    // Capture the GGUF MoE outputs, not merely their CPU parity verdict.
    let (tokens,experts,neurons,hidden)=(2usize,2usize,5usize,256usize);
    let weights=(0..experts*neurons*hidden).map(|i|
        if i<neurons*hidden {0.5f32}else{1.0}).collect::<Vec<_>>();
    let w_cpu=Tensor::from_vec(weights,(experts,neurons,hidden),&cpu)?;
    let x=Tensor::from_vec(vec![0.25f32;tokens*hidden],(tokens,hidden),&gpu)?;
    let sorted=Tensor::from_vec(vec![0u32,2,1,3],4,&gpu)?;
    let experts_id=Tensor::from_vec(vec![0u32,0,1,1],4,&gpu)?;
    for codec in MOE_CODECS {
        let q_cpu=QTensor::quantize(&w_cpu,codec)?;
        let bytes=q_cpu.data()?.to_vec();
        inputs.insert(format!("moe/gguf/{codec:?}"),
            json!(format!("{:x}",Sha256::digest(&bytes))));
        let q=QTensor::new(QStorage::from_data(Cow::Owned(bytes),&gpu,codec)?,
                           (experts,neurons,hidden))?;
        let y=moe_gemm_gguf(&x,&q,&None,&sorted,&experts_id,2,true,DType::F32)?;
        gpu.synchronize()?;
        capture(&mut records,format!("moe/gguf/{codec:?}"),&y)?;
    }

    // SIMT MoE decode (topk=1) and prefill (topk=2).
    let (e,n,k)=(2usize,128usize,128usize);
    let wf=(0..e*n*k).map(|i|((i*7%31) as f32-15.)*0.015625)
        .collect::<Vec<_>>();
    let w=Tensor::from_vec(wf,(e,n,k),&gpu)?.to_dtype(DType::F16)?;
    for (phase,tokens,topk) in [("decode",3usize,1usize),("prefill",8,2)] {
        let x=Tensor::from_vec(vec![0.25f32;tokens*k],(tokens,k),&gpu)?
            .to_dtype(DType::F16)?;
        let (sorted,ids)=if phase=="decode" {
            (Tensor::from_vec(vec![0u32,1,2],3,&gpu)?,
             Tensor::from_vec(vec![0u32,1,1],3,&gpu)?)
        }else{
            let sorted=(0..tokens).map(|i|(2*i) as u32)
                .chain((0..tokens).map(|i|(2*i+1) as u32))
                .collect::<Vec<_>>();
            let expert=(0..tokens).map(|_|0u32)
                .chain((0..tokens).map(|_|1u32)).collect::<Vec<_>>();
            (Tensor::from_vec(sorted,2*tokens,&gpu)?,
             Tensor::from_vec(expert,2*tokens,&gpu)?)
        };
        let y=moe_gemm(&x,&w,&None,&sorted,&ids,topk,phase=="prefill")?;
        gpu.synchronize()?;
        capture(&mut records,format!("moe/simt/{phase}"),&y)?;
    }

    // The exact same fixture is executed by two separately built binaries.
    // Python verifies the mode against archive feature provenance and loaded
    // CUDA images, so this environment label alone is not trusted evidence.
    let report=json!({
        "schema":"CANDLE-DKR-R124-S55D-NUMERIC-V1",
        "mode":mode,
        "target_cc":61,
        "fixture":"S55D_FIXED_INPUTS_V1",
        "gguf_cases":codec_count,
        "quantized_weight_sha256":inputs,
        "cases":records,
    });
    let count=report["cases"].as_object().map_or(0,|v|v.len());
    anyhow::ensure!(count==157, "expected 157 numeric cases, got {count}");
    std::fs::write(&destination,serde_json::to_vec(&report)?)?;
    println!("DKR_R124_S55D_CAPTURE=PASS mode={mode} cases={count} quantized={codec_count} output={}",destination.display());
    Ok(())
}
