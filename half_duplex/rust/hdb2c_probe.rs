//! HDB2-C real Candle numerical trace. Standalone additional Cargo bin.
//! Does not touch the existing parakeet-candle executable or its model weights.
//! Same get_logmel() and FastConformer encoder used by ParakeetTdt.generate().
use anyhow::{anyhow, bail, Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use parakeet_candle::parakeet::{from_config_value, get_logmel, load_audio, ParakeetModel};
use serde_json::json;
use std::{fs, io::Write, path::{Path, PathBuf}};

#[derive(Parser)]
struct Args {
    #[arg(long)] model_dir: PathBuf,
    #[arg(long)] input: PathBuf,
    #[arg(long)] output: PathBuf,
    #[arg(long, default_value_t=false)] cpu: bool,
}

fn dump(name: &str, t: &Tensor, out: &Path) -> Result<serde_json::Value> {
    let cpu = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let shape = cpu.dims().to_vec();
    let vector = cpu.flatten_all()?.to_vec1::<f32>()?;
    let path=out.join(format!("{name}.f32le"));
    let mut f=fs::File::create(&path)?;
    for value in vector { f.write_all(&value.to_le_bytes())?; }
    Ok(json!({"filename":path.file_name().unwrap().to_string_lossy(),"shape":shape,
        "dtype":"f32le","layout":"batch,time,features"}))
}
fn main() -> Result<()> {
    let a=Args::parse();
    fs::create_dir_all(&a.output)?;
    let dev=if a.cpu { Device::Cpu } else { Device::new_cuda(0).context("CUDA unavailable: no silent CPU fallback")? };
    let path=a.model_dir.join("model.safetensors");
    if !path.is_file() { bail!("Missing Candle safetensors: {}",path.display()); }
    let config_path=a.model_dir.join("config.json");
    let config:serde_json::Value=serde_json::from_reader(fs::File::open(&config_path)?)?;
    // SAFETY: The model weights are not modified during the process lifetime.
    let vb=unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32,&dev)? };
    let mut model=from_config_value(config, vb)?;
    let pre=model.preprocessor_config().clone();
    let audio=load_audio(&a.input,pre.sample_rate)?;
    let mel=get_logmel(&audio,&pre,&dev)?;
    let mel_meta=dump("candle_mel",&mel,&a.output)?;
    let (features,lengths)=match &mut model {
        ParakeetModel::Tdt(m) => m.encoder.forward(&mel,None)?,
        _ => bail!("Not a TDT checkpoint; refuse encoder parity probe"),
    };
    dev.synchronize()?;
    let feat_meta=dump("candle_encoder",&features,&a.output)?;
    let lengths=lengths.to_device(&Device::Cpu)?.to_vec1::<i64>()?;
    if lengths.len()!=1 { return Err(anyhow!("Expected batch 1 encoder lengths")); }
    let meta=json!({"stage":"HDB2C_CANDLE_REAL_TENSOR_PROBE","status":"REAL_CANDLE_MEL_AND_ENCODER",
        "device":format!("{dev:?}"),"source_pcm_frames":audio.len(),
        "mel":mel_meta,"encoder":feat_meta,"encoder_valid_frames":lengths[0],
        "note":"Raw probes, not evidence of NeMo/Candle parity on their own"});
    fs::write(a.output.join("candle_probe.json"),serde_json::to_vec_pretty(&meta)?)?;
    eprintln!("HDB2C_CANDLE_PROBE=REAL_TENSORS_SAVED");
    Ok(())
}
