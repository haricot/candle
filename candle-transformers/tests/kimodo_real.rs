//! Real-checkpoint parity with an explicitly captured NVIDIA CPU fixture.
use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::kimodo::{Config, Ddim, Guidance, Kimodo, RootStats, MOTION_DIM};
use serde::Deserialize;
use std::{path::PathBuf, time::Instant};

#[derive(Deserialize)]
struct Manifest {
    schema: String,
    model: String,
    model_revision: String,
    upstream: String,
    config: Config,
    stats: RootStats,
    steps: Vec<usize>,
    heading: Vec<f32>,
    text_lengths: Vec<usize>,
    frames: usize,
    parameters: usize,
    embedding_kind: String,
}

fn compare(name: &str, got: &Tensor, want: &Tensor, atol: f64, rtol: f64) -> Result<()> {
    if got.dims() != want.dims() {
        candle::bail!("{name}: shape mismatch {:?} != {:?}", got.dims(), want.dims());
    }
    let a = got.flatten_all()?.to_vec1::<f32>()?;
    let b = want.flatten_all()?.to_vec1::<f32>()?;
    let mut max_abs = 0f64;
    let mut sum_sq = 0f64;
    let mut violations = 0;
    for (&x, &y) in a.iter().zip(&b) {
        let e = (f64::from(x) - f64::from(y)).abs();
        if !x.is_finite() || !y.is_finite() || e > atol + rtol * f64::from(y).abs() {
            violations += 1;
        }
        max_abs = max_abs.max(e);
        sum_sq += e * e;
    }
    println!("KIMODO_B2_TENSOR name={name} elements={} max_abs={max_abs:.8e} rmse={:.8e} violations={violations} atol={atol} rtol={rtol}",
             a.len(), (sum_sq / a.len() as f64).sqrt());
    if violations != 0 {
        candle::bail!("{name}: {violations} tolerance violations");
    }
    Ok(())
}

fn parity(device: &Device) -> Result<()> {
    let dir = PathBuf::from(std::env::var("KIMODO_FIXTURE").map_err(candle::Error::wrap)?);
    let m: Manifest = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).map_err(candle::Error::wrap)?)
        .map_err(candle::Error::wrap)?;
    if m.schema != "haricot.kimodo.b2.v1" || m.model != "nvidia/Kimodo-SOMA-RP-v1.2"
        || m.upstream != "e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954"
        || m.model_revision.len() != 40 || m.parameters < 100_000_000
        || m.embedding_kind != "synthetic_nonzero" || m.steps.len() < 2 {
        candle::bail!("unexpected fixture identity");
    }
    let start = Instant::now();
    let weights = candle::safetensors::load(dir.join("weights.safetensors"), device)?;
    let count: usize = weights.values().map(|t| t.elem_count()).sum();
    if count != m.parameters || weights.values().any(|t| t.dtype() != DType::F32) {
        candle::bail!("weight count/dtype mismatch");
    }
    let schedule = Ddim::new(m.config.num_base_steps, m.steps.len())?;
    if schedule.timesteps != m.steps {
        candle::bail!("DDIM timestep mismatch: {:?} != {:?}", schedule.timesteps, m.steps);
    }
    let model = Kimodo::load(m.config.clone(), m.stats.clone(), VarBuilder::from_tensors(weights, DType::F32, device))?;
    let f = candle::safetensors::load(dir.join("oracle.safetensors"), device)?;
    let get = |key: &str| -> Result<&Tensor> {
        f.get(key).ok_or_else(|| candle::Error::msg(format!("missing tensor {key}")))
    };
    let noise = get("noise")?;
    let text = get("text")?;
    let observed = get("observed")?;
    let mask = get("mask")?;
    if noise.dims() != [1, m.frames, MOTION_DIM] {
        candle::bail!("unexpected noise shape");
    }
    println!("KIMODO_B2_LOAD params={count} device={device:?} ms={}", start.elapsed().as_millis());
    let direct = model.forward(noise, text, &m.text_lengths, &m.heading, observed, mask, *m.steps.last().unwrap())?;
    let want = get("direct")?;
    compare("root", &direct.narrow(2, 0, 5)?, &want.narrow(2, 0, 5)?, 0.003, 0.003)?;
    compare("body", &direct.narrow(2, 5, MOTION_DIM-5)?, &want.narrow(2, 5, MOTION_DIM-5)?, 0.003, 0.003)?;
    let local = m.stats.global_to_local(&want.narrow(2, 0, 5)?, m.config.fps)?;
    compare("root_bridge", &local, get("local_root")?, 0.001, 0.001)?;
    let guidance = Guidance::Separated { text: 2.0, constraint: 2.0 };
    for i in (0..m.steps.len()).rev() {
        let x = get(&format!("input_{i}"))?;
        let pred = model.guided(x, text, &m.text_lengths, &m.heading, observed, mask, m.steps[i], &guidance)?;
        compare(&format!("cfg_{i}"), &pred, get(&format!("clean_{i}"))?, 0.005, 0.005)?;
        let next = schedule.step(x, get(&format!("clean_{i}"))?, i)?;
        compare(&format!("ddim_{i}"), &next, get(&format!("next_{i}"))?, 0.002, 0.002)?;
    }
    let now = Instant::now();
    let final_motion = model.sample(noise, text, &m.text_lengths, &m.heading, observed, mask, m.steps.len(), guidance)?;
    compare("end_to_end", &final_motion, get("final")?, 0.015, 0.01)?;
    println!("KIMODO_B2_PARITY=PASS device={device:?} frames={} steps={} sample_ms={}",
             m.frames, m.steps.len(), now.elapsed().as_millis());
    Ok(())
}
#[test]
#[ignore = "requires local NVIDIA oracle and checkpoint"]
fn kimodo_real_cpu() -> Result<()> { parity(&Device::Cpu) }

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires local NVIDIA oracle, checkpoint and CUDA GPU"]
fn kimodo_real_cuda() -> Result<()> { parity(&Device::new_cuda(0)?) }
