//! S5.5d: offline, identical-GGUF, greedy Qwen3 cold-start probe.
#![allow(clippy::needless_range_loop)]
use anyhow::{anyhow, bail, Context, Result};
use candle::{quantized::gguf_file, DType, Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights;
use serde_json::json;
use std::{
    fs::File, io::{BufWriter, Write}, path::PathBuf,
    time::Instant,
};

fn options() -> Result<(PathBuf,PathBuf,PathBuf,Vec<u32>,usize)> {
    let mut args=std::env::args().skip(1);
    let mut model=None;let mut json_path=None;let mut logits_path=None;
    let mut prompt=None;let mut decode=None;
    while let Some(flag)=args.next(){
        let val=args.next().ok_or_else(||anyhow!("missing value for {flag}"))?;
        match flag.as_str() {
            "--model" => model=Some(PathBuf::from(val)),
            "--result" => json_path=Some(PathBuf::from(val)),
            "--logits" => logits_path=Some(PathBuf::from(val)),
            "--tokens" => {
                let ids=val.split(',').map(str::parse::<u32>)
                    .collect::<std::result::Result<Vec<_>,_>>()?;
                prompt=Some(ids);
            },
            "--decode" => decode=Some(val.parse::<usize>()?),
            _=>bail!("unsupported argument: {flag}"),
        }
    }
    let model=model.ok_or_else(||anyhow!("--model required"))?;
    let result=json_path.ok_or_else(||anyhow!("--result required"))?;
    let logits=logits_path.ok_or_else(||anyhow!("--logits required"))?;
    let tokens=prompt.ok_or_else(||anyhow!("--tokens required"))?;
    let n=decode.ok_or_else(||anyhow!("--decode required"))?;
    anyhow::ensure!(model.is_file() && model.extension().is_some_and(|s|s=="gguf"),
                    "model must be an existing local GGUF file");
    anyhow::ensure!(result.is_absolute() && logits.is_absolute()
       && !result.exists() && !logits.exists() && result!=logits,
       "output files must be distinct, absolute and not exist");
    anyhow::ensure!((1..=64).contains(&tokens.len()) && n>=2 && n<=32,
                    "token input or decode count outside qualifying range");
    anyhow::ensure!(tokens.iter().all(|&token|token<2_000_000),
                    "token ID not a valid bounded integer");
    Ok((model,result,logits,tokens,n))
}

fn capture(
    logits:&Tensor, target:&mut BufWriter<File>, vocabulary:&mut Option<usize>
)->Result<u32>{
    let output=logits.squeeze(0)?.to_dtype(DType::F32)?
        .flatten_all()?.to_vec1::<f32>()?;
    anyhow::ensure!(!output.is_empty() && output.iter().all(|v|v.is_finite()),
                    "nonfinite/empty Qwen3 logits");
    if let Some(previous)=vocabulary {
        anyhow::ensure!(*previous==output.len(),"vocab dimension changed");
    }else{
        *vocabulary=Some(output.len());
    }
    let mut best=0usize;
    for (i,value) in output.iter().enumerate().skip(1) {
        if *value>output[best] {best=i;}
    }
    for &value in &output {
        target.write_all(&value.to_le_bytes())?;
    }
    Ok(u32::try_from(best)?)
}

fn main() -> Result<()> {
    #[cfg(any(
        all(feature="ptx",feature="cubin"),
        all(feature="ptx",feature="promoted"),
        all(feature="cubin",feature="promoted")
    ))]
    compile_error!("PTX, CUBIN and promoted probe features are mutually exclusive");
    #[cfg(not(any(feature="ptx",feature="cubin",feature="promoted")))]
    compile_error!("model probe requires ptx, cubin or promoted");
    let mode=if cfg!(feature="promoted"){"promoted"}
        else if cfg!(feature="cubin"){"cubin"} else {"ptx"};
    if std::env::args().len()==2 && std::env::args().nth(1).as_deref()==Some("--self-test") {
        println!("DKR_R124_S55D_MODEL_PROBE=PASS mode={mode} no_cuda_build=1");
        return Ok(());
    }
    let (model_path,json_path,logits_path,tokens,decode)=options()?;
    let start=Instant::now();
    let t=Instant::now();
    let device=Device::new_cuda(0)?;
    device.synchronize()?;
    let device_init_ms=t.elapsed().as_secs_f64()*1000.;
    anyhow::ensure!(device.as_cuda_device()?.dkr_moe_enabled(),"native DKR unavailable");

    let t=Instant::now();
    let mut source=File::open(&model_path)
        .with_context(||format!("model file: {}",model_path.display()))?;
    let gguf=gguf_file::Content::read(&mut source)?;
    let tensor_count=gguf.tensor_infos.len();
    let header_ms=t.elapsed().as_secs_f64()*1000.;

    let t=Instant::now();
    let mut model=ModelWeights::from_gguf(gguf,&mut source,&device)?;
    device.synchronize()?;
    let model_build_ms=t.elapsed().as_secs_f64()*1000.;

    let mut writer=BufWriter::new(File::create(&logits_path)?);
    let mut vocabulary=None;
    let input=Tensor::new(tokens.as_slice(),&device)?.unsqueeze(0)?;
    device.synchronize()?;
    let t=Instant::now();
    let first=model.forward(&input,0)?;
    device.synchronize()?;
    let prefill_ms=t.elapsed().as_secs_f64()*1000.;
    let mut next=capture(&first,&mut writer,&mut vocabulary)?;
    let mut generated=vec![next];
    let mut per_token_ms=Vec::with_capacity(decode);
    for index in 0..decode {
        let input=Tensor::new(&[next],&device)?.unsqueeze(0)?;
        device.synchronize()?;
        let t=Instant::now();
        let out=model.forward(&input,tokens.len()+index)?;
        device.synchronize()?;
        per_token_ms.push(t.elapsed().as_secs_f64()*1000.);
        next=capture(&out,&mut writer,&mut vocabulary)?;
        generated.push(next);
    }
    writer.flush()?;
    let total_ms=start.elapsed().as_secs_f64()*1000.;
    let vocab=vocabulary.ok_or_else(||anyhow!("no generated output"))?;
    anyhow::ensure!(tokens.iter().all(|&x|(x as usize)<vocab),
                    "prompt token out of declared model vocabulary");
    let file_bytes=std::fs::metadata(&logits_path)?.len();
    anyhow::ensure!(file_bytes==(decode as u64+1)*vocab as u64*4,
                    "incomplete logits dump");
    let report=json!({
        "schema":"CANDLE-DKR-R124-S55D-MODEL-V1",
        "format":"qwen3-gguf","backend":mode,
        "target_cc":61,
        "model":model_path.display().to_string(),
        "tensor_count":tensor_count,
        "prompt_token_ids":tokens,
        "decode_steps":decode,
        "generated_token_ids":generated,
        "vocabulary":vocab,
        "logits_binary":logits_path.display().to_string(),
        "logits_bytes":file_bytes,
        "device_init_ms":device_init_ms,
        "gguf_header_ms":header_ms,
        "model_build_ms":model_build_ms,
        "prefill_first_ms":prefill_ms,
        "decode_per_token_ms":per_token_ms,
        "total_process_work_ms":total_ms,
        "timing_semantics":"GPU synchronized; model file OS page cache not evicted; logfile writing included only in total",
    });
    std::fs::write(&json_path,serde_json::to_vec_pretty(&report)?)?;
    println!("DKR_R124_S55D_MODEL=PASS mode={mode} vocab={vocab} prompt={} decode={} build_ms={model_build_ms:.3} prefill_ms={prefill_ms:.3} output={}",
        tokens.len(),decode,json_path.display());
    Ok(())
}
