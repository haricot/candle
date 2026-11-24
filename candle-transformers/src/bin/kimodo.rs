//! Native Kimodo F32 checkpoint runner. No Python, no text encoder dependency.
use anyhow::{bail, ensure, Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::kimodo::{Config, Ddim, Guidance, Kimodo, RootStats, MOTION_DIM};
use clap::Parser;
use hf_hub::{HFClientSync, HFError};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use serde::Deserialize;
use std::{collections::HashMap, path::{Path, PathBuf}, time::Instant};

const MODEL: &str = "nvidia/Kimodo-SOMA-RP-v1.2";

#[derive(Parser, Debug)]
#[command(about = "Kimodo SOMA native F32 generation (normalized motion features)")]
struct Args {
    #[arg(long, default_value = MODEL)]
    model: String,
    #[arg(long, default_value = "main")]
    revision: String,
    /// Use an already downloaded model directory instead of hf-hub.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long)]
    cpu: bool,
    #[arg(long, default_value_t = 24)]
    frames: usize,
    #[arg(long, default_value_t = 3)]
    steps: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Optional F32 LLM2Vec features, shape [1,N,4096]. No text encoder is loaded.
    #[arg(long)]
    embedding: Option<PathBuf>,
    /// Optional normalized observation [1,T,369], with matching binary mask.
    #[arg(long, requires = "mask")]
    observed: Option<PathBuf>,
    #[arg(long, requires = "observed")]
    mask: Option<PathBuf>,
    #[arg(long, default_value_t = 0.0)]
    heading: f32,
    #[arg(long, default_value_t = 2.0)]
    text_cfg: f64,
    #[arg(long, default_value_t = 2.0)]
    constraint_cfg: f64,
    #[arg(long, default_value = "kimodo-output")]
    out: PathBuf,
    /// Load and validate the checkpoint without sampling.
    #[arg(long)]
    load_only: bool,
}

#[derive(Deserialize)]
struct ModelYaml {
    #[serde(rename = "_target_")]
    target: String,
    num_base_steps: usize,
    cfg_type: String,
    denoiser: DenoiserYaml,
}
#[derive(Deserialize)]
struct DenoiserYaml {
    #[serde(rename = "_target_")]
    target: String,
    ckpt_path: String,
    motion_mask_mode: String,
    motion_rep: RepYaml,
    llm_shape: Vec<usize>,
    use_text_mask: bool,
    latent_dim: usize,
    ff_size: usize,
    num_layers: usize,
    num_heads: usize,
    activation: String,
    dropout: f32,
    pe_dropout: f32,
    #[serde(default)]
    norm_first: bool,
    num_text_tokens_override: Option<usize>,
    #[serde(default)]
    input_first_heading_angle: bool,
}
#[derive(Deserialize)]
struct RepYaml {
    #[serde(rename = "_target_")]
    target: String,
    fps: f32,
    stats_path: String,
    skeleton: SkeletonYaml,
}
#[derive(Deserialize)]
struct SkeletonYaml {
    #[serde(rename = "_target_")]
    target: String,
}

fn parse_config(text: &str) -> Result<Config> {
    let m: ModelYaml = serde_yaml::from_str(text)?;
    let d = m.denoiser;
    ensure!(m.target == "kimodo.model.Kimodo"
        && d.target == "kimodo.model.twostage_denoiser.TwostageDenoiser"
        && d.motion_rep.target == "kimodo.motion_rep.KimodoMotionRep"
        && d.motion_rep.skeleton.target == "kimodo.skeleton.SOMASkeleton30",
        "unsupported architecture; expected SOMA30 KimodoMotionRep");
    ensure!(d.ckpt_path.ends_with("/model.safetensors")
        && d.motion_rep.stats_path.trim_end_matches('/').ends_with("/stats/motion"),
        "unsupported weight/stats paths");
    ensure!(d.motion_mask_mode == "concat" && m.cfg_type == "separated"
        && d.llm_shape.len() == 2 && d.llm_shape[1] == 4096
        && d.motion_rep.fps == 30.0 && d.dropout == 0.0 && d.pe_dropout == 0.0,
        "unsupported Kimodo checkpoint configuration");
    let tokens = d.num_text_tokens_override.unwrap_or(d.llm_shape[0]);
    ensure!(tokens > 0 && tokens <= 4096, "invalid text token count");
    Ok(Config {
        latent_dim: d.latent_dim, ff_size: d.ff_size, num_layers: d.num_layers,
        num_heads: d.num_heads, text_dim: d.llm_shape[1], text_tokens: tokens,
        use_text_mask: d.use_text_mask, input_first_heading_angle: d.input_first_heading_angle,
        norm_first: d.norm_first, activation: d.activation, num_base_steps: m.num_base_steps,
        fps: d.motion_rep.fps,
    })
}

const FILES: [&str; 8] = [
    "config.yaml", "model.safetensors",
    "stats/motion/global_root/mean.npy", "stats/motion/global_root/std.npy",
    "stats/motion/local_root/mean.npy", "stats/motion/local_root/std.npy",
    "stats/motion/body/mean.npy", "stats/motion/body/std.npy",
];

fn resolve_files(a: &Args) -> Result<(Vec<PathBuf>, String)> {
    if let Some(dir) = &a.model_dir {
        let files: Vec<_> = FILES.iter().map(|f| dir.join(f)).collect();
        for p in &files { ensure!(p.is_file(), "missing {}", p.display()); }
        return Ok((files, "local-unverified".into()));
    }
    let (owner, name) = a.model.split_once('/').context("expected owner/model")?;
    ensure!(!owner.is_empty() && !name.is_empty() && !name.contains('/'), "invalid model id");
    let api = HFClientSync::new()?;
    let repo = api.model(owner, name);
    let get = |file: &str, rev: &str| -> Result<PathBuf> {
        let request = repo.download_file().filename(file).maybe_revision(Some(rev.to_owned()));
        let path = match request.clone().local_files_only(true).send() {
            Err(HFError::LocalEntryNotFound { .. }) => {
                eprintln!("KIMODO_HF_DOWNLOAD file={file} revision={rev}");
                request.send()?
            }
            cached => cached?,
        };
        eprintln!("KIMODO_HF_FILE file={file} path={}", path.display());
        Ok(path)
    };
    let config = get(FILES[0], &a.revision)?;
    // hf-hub returns a snapshot path; pin all remaining downloads to this commit.
    let revision = config.parent().and_then(Path::file_name).and_then(|s| s.to_str())
        .context("cannot resolve config snapshot revision")?.to_owned();
    ensure!(revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()),
        "expected hf-hub snapshot commit, got {revision}; use --model-dir if necessary");
    if a.revision.len() == 40 && a.revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        ensure!(a.revision.eq_ignore_ascii_case(&revision), "snapshot revision mismatch");
    }
    let mut files = vec![config];
    for file in &FILES[1..] { files.push(get(file, &revision)?); }
    Ok((files, revision))
}

fn stat(path: &Path, n: usize, std: bool) -> Result<Vec<f32>> {
    let t = Tensor::read_npy(path)?;
    ensure!(matches!(t.dtype(), DType::F32 | DType::F64) && t.elem_count() == n,
        "unexpected stats dtype/shape: {}", path.display());
    let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    ensure!(v.iter().all(|x| x.is_finite() && (!std || *x >= 0.)),
        "invalid statistics: {}", path.display());
    Ok(v)
}

fn f32_npy(path: &Path, dev: &Device) -> Result<Tensor> {
    let t = Tensor::read_npy(path)?;
    ensure!(t.dtype() == DType::F32, "expected F32 array: {}", path.display());
    ensure!(t.flatten_all()?.to_vec1::<f32>()?.iter().all(|v| v.is_finite()), "nonfinite input");
    Ok(t.to_device(dev)?)
}

fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(a.model == MODEL, "this first native runner supports only {MODEL}");
    ensure!((2..=300).contains(&a.frames) && a.steps >= 2, "invalid frames/steps");
    ensure!(a.heading.is_finite() && a.text_cfg.is_finite() && a.constraint_cfg.is_finite(),
        "nonfinite generation arguments");
    if !a.load_only { ensure!(!a.out.exists(), "output exists: {}", a.out.display()); }
    let (files, revision) = resolve_files(&a)?;
    let cfg = parse_config(&std::fs::read_to_string(&files[0])?)?;
    ensure!(a.steps <= cfg.num_base_steps, "steps exceed training schedule");
    let stats = RootStats {
        global_mean: stat(&files[2], 5, false)?.try_into().unwrap(),
        global_std: stat(&files[3], 5, true)?.try_into().unwrap(),
        local_mean: stat(&files[4], 4, false)?.try_into().unwrap(),
        local_std: stat(&files[5], 4, true)?.try_into().unwrap(),
    };
    let mut mean = stats.global_mean.to_vec();
    mean.extend(stat(&files[6], MOTION_DIM-5, false)?);
    let mut std = stats.global_std.to_vec();
    std.extend(stat(&files[7], MOTION_DIM-5, true)?);
    println!("KIMODO_NATIVE_CONFIG=PASS model={} revision={} latent={} layers={} heads={} text_dim={} text_tokens={} motion_dim={}",
        a.model, revision, cfg.latent_dim, cfg.num_layers, cfg.num_heads, cfg.text_dim, cfg.text_tokens, MOTION_DIM);
    let device = if a.cpu { Device::Cpu } else { Device::new_cuda(0)? };
    let start = Instant::now();
    // Load official F32 tensors directly. Strip only the exact upstream prefix,
    // not arbitrary key aliases. VarBuilder performs device upload and shape validation.
    let all = candle::safetensors::load(&files[1], &Device::Cpu)?;
    let mut weights = HashMap::new();
    let mut params = 0usize;
    for (name, tensor) in all {
        let key = name.strip_prefix("denoiser.backbone.").unwrap_or(&name);
        ensure!(key.starts_with("root_model.") || key.starts_with("body_model."),
            "unexpected checkpoint tensor: {name}");
        ensure!(tensor.dtype() == DType::F32, "expected F32 checkpoint tensor: {name}");
        params += tensor.elem_count();
        ensure!(weights.insert(key.to_owned(), tensor).is_none(), "duplicate tensor: {key}");
    }
    ensure!((280_000_000..=285_000_000).contains(&params), "unexpected parameter count: {params}");
    let model = Kimodo::load(cfg.clone(), stats, VarBuilder::from_tensors(weights, DType::F32, &device))?;
    device.synchronize()?;
    println!("KIMODO_NATIVE_LOAD=PASS params={params} device={device:?} ms={}",start.elapsed().as_millis());
    if a.load_only { return Ok(()); }
    let shape = (1, a.frames, MOTION_DIM);
    let mut rng = StdRng::seed_from_u64(a.seed);
    let values: Vec<f32> = (0..a.frames*MOTION_DIM).map(|_| StandardNormal.sample(&mut rng)).collect();
    let mut x = Tensor::from_vec(values, shape, &device)?;
    let (text, ntext) = if let Some(p) = &a.embedding {
        let t = f32_npy(p, &device)?;
        let (b,n,d) = t.dims3()?;
        ensure!(b == 1 && n > 0 && n <= cfg.text_tokens && d == cfg.text_dim, "invalid embedding shape");
        (t,n)
    } else {
        (Tensor::zeros((1,1,cfg.text_dim), DType::F32, &device)?,0)
    };
    let (observed,mask) = if let (Some(o),Some(m)) = (&a.observed,&a.mask) {
        let o = f32_npy(o,&device)?;
        let m = f32_npy(m,&device)?;
        ensure!(o.dims() == [1,a.frames,MOTION_DIM] && m.dims() == o.dims(), "invalid constraint shape");
        ensure!(m.flatten_all()?.to_vec1::<f32>()?.iter().all(|v| *v == 0. || *v == 1.), "nonbinary mask");
        (o,m)
    } else {
        (Tensor::zeros(shape,DType::F32,&device)?,Tensor::zeros(shape,DType::F32,&device)?)
    };
    let conditioned = a.embedding.is_some() || a.mask.is_some();
    let guidance = if conditioned { Guidance::Separated{text:a.text_cfg,constraint:a.constraint_cfg} } else { Guidance::None };
    let schedule = Ddim::new(cfg.num_base_steps,a.steps)?;
    let begin = Instant::now();
    for i in (0..a.steps).rev() {
        let now=Instant::now();
        let clean=model.guided(&x,&text,&[ntext],&[a.heading],&observed,&mask,schedule.timesteps[i],&guidance)?;
        x=schedule.step(&x,&clean,i)?;
        device.synchronize()?;
        let v=x.flatten_all()?.to_vec1::<f32>()?;
        ensure!(v.iter().all(|v|v.is_finite()), "nonfinite motion at DDIM step {i}");
        println!("KIMODO_NATIVE_STEP index={i} train_t={} ms={}",schedule.timesteps[i],now.elapsed().as_millis());
    }
    let sample_ms=begin.elapsed().as_millis();
    let normalized=x.to_device(&Device::Cpu)?;
    let mu=Tensor::from_vec(mean,(1,1,MOTION_DIM),&Device::Cpu)?;
    let scale=Tensor::from_vec(std.into_iter().map(|s|(s*s+1e-5).sqrt()).collect::<Vec<_>>(),(1,1,MOTION_DIM),&Device::Cpu)?;
    let physical=normalized.broadcast_mul(&scale)?.broadcast_add(&mu)?;
    let vals=physical.flatten_all()?.to_vec1::<f32>()?;
    ensure!(vals.iter().all(|v|v.is_finite()), "nonfinite decoded features");
    // This is the trained feature representation, NOT a reconstructed SOMA77 BVH.
    let root=physical.narrow(2,5,3)?.broadcast_add(
        &Tensor::cat(&[&physical.narrow(2,0,1)?,&Tensor::zeros((1,a.frames,1),DType::F32,&Device::Cpu)?,&physical.narrow(2,2,1)?],2)?
    )?;
    if let Some(parent)=a.out.parent().filter(|p| !p.as_os_str().is_empty()) { std::fs::create_dir_all(parent)?; }
    std::fs::create_dir(&a.out)?;
    normalized.write_npy(a.out.join("motion_normalized.npy"))?;
    physical.write_npy(a.out.join("motion_features.npy"))?;
    root.write_npy(a.out.join("root_positions.npy"))?;
    let report=serde_json::json!({"schema":"haricot.kimodo.native.v1","model":a.model,"revision":revision,
        "frames":a.frames,"fps":cfg.fps,"steps":a.steps,"seed":a.seed,"parameters":params,
        "device":format!("{device:?}"),"dtype":"F32","sample_ms":sample_ms,
        "conditioned":conditioned,"embedding_supplied":a.embedding.is_some(),"constraints_supplied":a.mask.is_some(),
        "output":"SOMA30 motion features, not BVH","qualification":"functional only, no PyTorch parity"});
    std::fs::write(a.out.join("manifest.json"),serde_json::to_vec_pretty(&report)?)?;
    println!("KIMODO_NATIVE_SAMPLE=PASS frames={} steps={} ms={sample_ms} out={}",a.frames,a.steps,a.out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kimodo_cli_config() -> Result<()> {
        let c=parse_config(r#"
_target_: kimodo.model.Kimodo
num_base_steps: 1000
cfg_type: separated
denoiser:
  _target_: kimodo.model.twostage_denoiser.TwostageDenoiser
  ckpt_path: ${oc.select:checkpoint_dir}/model.safetensors
  motion_mask_mode: concat
  motion_rep:
    _target_: kimodo.motion_rep.KimodoMotionRep
    fps: 30
    stats_path: ${oc.select:checkpoint_dir}/stats/motion/
    skeleton:
      _target_: kimodo.skeleton.SOMASkeleton30
  llm_shape: [1,4096]
  use_text_mask: false
  latent_dim: 1024
  ff_size: 2048
  num_layers: 16
  num_heads: 8
  activation: gelu
  dropout: 0.0
  pe_dropout: 0.0
  norm_first: false
  num_text_tokens_override: 50
  input_first_heading_angle: true
"#)?;
        assert_eq!(c.text_tokens,50);
        assert_eq!(c.latent_dim,1024);
        Ok(())
    }
}
