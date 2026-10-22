//! R7: independent PCM -> native frontend -> R6 encoder -> actual TDT decode.
//! Shared-input operator replay is an explicitly separate execution mode.
use anyhow::{bail, ensure, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use parakeet_candle::parakeet::{from_config_value,get_logmel,load_audio,ParakeetModel,DecodingConfig};
use serde_json::{json,Value};
use std::{fs,io::Write,path::{Path,PathBuf}};

#[derive(Parser)]
struct Args {
    #[arg(long)] model_dir: PathBuf,
    #[arg(long)] input: PathBuf,
    #[arg(long)] output: PathBuf,
    #[arg(long)] shared_mel: Option<PathBuf>,
    #[arg(long)] mel_valid: Option<usize>,
    #[arg(long)] tdt_cases: Option<PathBuf>,
}

fn dump(dir:&Path,name:&str,t:&Tensor)->Result<()> {
    fs::create_dir_all(dir)?;
    let t=t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let mut file=fs::File::create(dir.join(format!("{name}.f32le")))?;
    for v in t.flatten_all()?.to_vec1::<f32>()? {file.write_all(&v.to_le_bytes())?;}
    fs::write(dir.join(format!("{name}.json")),serde_json::to_vec_pretty(&json!({"shape":t.dims(),"dtype":"f32le"}))?)?;
    Ok(())
}
fn read(dir:&Path,name:&str)->Result<Tensor> {
    let meta:Value=serde_json::from_slice(&fs::read(dir.join(format!("{name}.json")))?)?;
    let shape:Vec<usize>=meta["shape"].as_array().unwrap().iter().map(|x|x.as_u64().unwrap() as usize).collect();
    let bytes=fs::read(dir.join(format!("{name}.f32le")))?;
    ensure!(bytes.len()==shape.iter().product::<usize>()*4,"R7 input tensor length mismatch");
    let values:Vec<f32>=bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
    ensure!(values.iter().all(|v|v.is_finite()),"R7 nonfinite shared input");
    Ok(Tensor::from_vec(values,shape.as_slice(),&Device::Cpu)?)
}

fn main()->Result<()> {
    let a=Args::parse(); fs::create_dir_all(&a.output)?;
    let dev=Device::Cpu;
    let cfg:Value=serde_json::from_slice(&fs::read(a.model_dir.join("config.json"))?)?;
    let weights=a.model_dir.join("model.safetensors");
    // SAFETY: immutable checkpoint throughout this process.
    let vb=unsafe{VarBuilder::from_mmaped_safetensors(&[weights],DType::F32,&dev)?};
    let mut model=from_config_value(cfg,vb)?;
    let ParakeetModel::Tdt(m)=&mut model else {bail!("R7 requires TDT");};
    if let Some(cases)=a.tdt_cases {
        ensure!(a.shared_mel.is_none() && a.mel_valid.is_none(),"R7 mixed execution scopes");
        let manifest:Value=serde_json::from_slice(&fs::read(cases.join("manifest.json"))?)?;
        for row in manifest["predictor_cases"].as_array().unwrap() {
            let id=row["id"].as_str().unwrap();
            let folder=cases.join("predictor").join(id);
            let token=row["token"].as_u64().map(|t|Tensor::from_vec(vec![t as i64],(1,1),&dev)).transpose()?;
            let state=if row["has_state"].as_bool().unwrap(){Some((read(&folder,"input_h")?,read(&folder,"input_c")?))}else{None};
            if let Some((ref h,ref c))=state {dump(&a.output.join("predictor").join(id),"input_h",h)?;dump(&a.output.join("predictor").join(id),"input_c",c)?;}
            let (g,(h,c))=m.decoder.forward(token.as_ref(),state)?;
            let out=a.output.join("predictor").join(id);
            for (name,t) in [("g",&g),("h",&h),("c",&c)] {dump(&out,name,t)?;}
        }
        for row in manifest["decisions"].as_array().unwrap() {
            let id=row["id"].as_str().unwrap();
            let folder=cases.join("joint").join(id);
            let f=read(&folder,"enc")?; let g=read(&folder,"g")?;
            let logits=m.joint.forward(&f,&g)?;
            let out=a.output.join("joint").join(id);
            dump(&out,"enc",&f)?;dump(&out,"g",&g)?;dump(&out,"logits",&logits)?;
        }
        fs::write(a.output.join("scope.json"),b"{\"scope\":\"COMMON_INPUT_TDT_OPERATORS_ONLY\"}")?;
        return Ok(());
    }
    let audio=load_audio(&a.input,m.preprocessor_config.sample_rate)?;
    dump(&a.output,"pcm",&Tensor::from_vec(audio.clone(),(1,audio.len(),1),&dev)?)?;
    let (mel,valid,scope)=if let Some(folder)=a.shared_mel {
        (read(&folder,"mel")?,a.mel_valid.ok_or_else(||anyhow::anyhow!("missing shared valid length"))?,"SHARED_NEMO_MEL_ENCODER_TDT")
    }else{
        ensure!(a.mel_valid.is_none(),"R7 valid override without shared Mel");
        (get_logmel(&audio,&m.preprocessor_config,&dev)?,
         parakeet_candle::parakeet::audio::hdb2c_valid_frames(audio.len(),&m.preprocessor_config)?,"NATIVE_PCM_FRONTEND_ENCODER_TDT")
    };
    dump(&a.output,"mel",&mel)?;
    let input_lengths=Tensor::from_vec(vec![valid as i64],(1,),&dev)?;
    let (encoded,lengths)=m.encoder.forward(&mel,Some(&input_lengths))?;
    dump(&a.output,"encoder",&encoded)?;
    let (tokens,_)=m.decode(&encoded,Some(&lengths),None,None,&DecodingConfig::default())?;
    let tokens=&tokens[0];
    let ids:Vec<usize>=tokens.iter().map(|t|t.id).collect();
    let frames:Vec<usize>=tokens.iter().map(|t|(t.start/m.time_ratio()).round() as usize).collect();
    let durations:Vec<usize>=tokens.iter().map(|t|(t.duration/m.time_ratio()).round() as usize).collect();
    let text=parakeet_candle::parakeet::tokenizer::decode(&ids,&m.vocabulary).trim().to_string();
    let result=json!({"scope":scope,"device":"CPU","ids":ids,"frames":frames,"durations":durations,
        "text":text,"encoder_valid_frames":lengths.to_vec1::<i64>()?[0],"mel_valid_frames":valid,
        "durations_config":m.durations,"max_symbols":m.max_symbols,"blank":m.vocabulary.len()});
    fs::write(a.output.join("result.json"),serde_json::to_vec_pretty(&result)?)?;
    println!("HDB2C3_R7_NATIVE={scope} text={text}");
    Ok(())
}
