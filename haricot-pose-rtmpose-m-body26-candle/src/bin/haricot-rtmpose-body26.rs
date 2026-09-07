use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use haricot_pose_rtmpose_m_body26_candle::{
    audit_simcc, decode_simcc, measure_sync_baseline, prepare_topdown_rgb, render_topdown_rgb,
    summarize_layer_audit, BboxXyxy, LayerProfiler, RtmPoseBody26, RtmPoseOutput,
};
use serde::Serialize;
use serde_json::json;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "haricot-rtmpose-body26")]
struct Args {
    #[arg(long)]
    weights: PathBuf,
    #[arg(long)]
    image: PathBuf,
    /// xyxy bounding box, e.g. 120,20,510,470. Defaults to the full image.
    #[arg(long)]
    bbox: Option<String>,
    #[arg(long)]
    cpu: bool,
    #[arg(long)]
    f16: bool,
    #[arg(long, default_value_t = 0.0)]
    score_threshold: f32,
    /// Extra synchronized warm-up iterations after the separately reported cold run.
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    /// Number of synchronized timed samples. Zero preserves the legacy one-cold-run behavior.
    #[arg(long, default_value_t = 0)]
    runs: usize,
    /// Replicate the same top-down ROI into a batch of 1..3 for pure landmarker scaling audits.
    #[arg(long, default_value_t = 1)]
    batch: usize,
    /// Save the exact 192x256 affine RGB crop before mean/std normalization.
    #[arg(long)]
    dump_input: Option<PathBuf>,
    /// Also write the complete JSON audit report to this file.
    #[arg(long)]
    json_out: Option<PathBuf>,
    /// Run the intrusive leaf-by-leaf execution audit with device synchronization around each op.
    #[arg(long)]
    layer_audit: bool,
    /// Number of profiled forwards used to compute per-layer medians.
    #[arg(long, default_value_t = 3)]
    layer_audit_runs: usize,
    /// Empty device synchronizations used to quantify profiling observer overhead.
    #[arg(long, default_value_t = 100)]
    sync_baseline_samples: usize,
    /// Capture best-effort nvidia-smi telemetry before and after the layer audit.
    #[arg(long)]
    gpu_telemetry: bool,
}

#[derive(Debug, Serialize)]
struct TimingStats {
    samples_ms: Vec<f64>,
    min_ms: f64,
    max_ms: f64,
    mean_ms: f64,
    median_ms: f64,
    p90_ms: f64,
    p95_ms: f64,
    per_pose_median_ms: f64,
    batch_fps_median: f64,
    poses_per_second_median: f64,
}

fn parse_bbox(s: &str) -> Result<BboxXyxy> {
    let vals = s
        .split(',')
        .map(str::trim)
        .map(str::parse::<f32>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    anyhow::ensure!(vals.len() == 4, "--bbox must be x1,y1,x2,y2");
    anyhow::ensure!(vals[2] > vals[0], "--bbox requires x2 > x1");
    anyhow::ensure!(vals[3] > vals[1], "--bbox requires y2 > y1");
    Ok(BboxXyxy {
        x1: vals[0],
        y1: vals[1],
        x2: vals[2],
        y2: vals[3],
    })
}

fn force_output_sync(output: &RtmPoseOutput) -> Result<()> {
    let _ = output
        .simcc_x
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let _ = output
        .simcc_y
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(())
}

fn run_synced(model: &RtmPoseBody26, input: &Tensor) -> Result<(RtmPoseOutput, f64)> {
    let start = Instant::now();
    let output = model.forward(input)?;
    force_output_sync(&output)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((output, elapsed_ms))
}

fn run_layer_profile(
    model: &RtmPoseBody26,
    input: &Tensor,
) -> Result<(RtmPoseOutput, Vec<haricot_pose_rtmpose_m_body26_candle::LayerEventSample>, f64)> {
    let mut profiler = LayerProfiler::new(input.device());
    profiler.synchronize()?;
    let start = Instant::now();
    let output = model.forward_profiled(input, &mut profiler)?;
    profiler.synchronize()?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((output, profiler.into_events(), elapsed_ms))
}

fn nvidia_smi_snapshot() -> Option<serde_json::Value> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,pstate,temperature.gpu,clocks.gr,clocks.mem,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let rows = stdout
        .lines()
        .filter_map(|line| {
            let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
            if fields.len() < 7 {
                return None;
            }
            Some(json!({
                "index":fields[0],
                "name":fields[1],
                "pstate":fields[2],
                "temperature_c":fields[3],
                "graphics_clock_mhz":fields[4],
                "memory_clock_mhz":fields[5],
                "power_w":fields[6]
            }))
        })
        .collect::<Vec<_>>();
    Some(json!({"rows":rows,"raw":stdout.trim()}))
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn timing_stats(samples: Vec<f64>, batch: usize) -> Option<TimingStats> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let min_ms = sorted[0];
    let max_ms = sorted[sorted.len() - 1];
    let mean_ms = samples.iter().sum::<f64>() / samples.len() as f64;
    let median_ms = percentile_sorted(&sorted, 0.5);
    let p90_ms = percentile_sorted(&sorted, 0.90);
    let p95_ms = percentile_sorted(&sorted, 0.95);
    let batch_fps_median = if median_ms > 0.0 {
        1000.0 / median_ms
    } else {
        f64::INFINITY
    };
    let poses_per_second_median = batch_fps_median * batch as f64;
    Some(TimingStats {
        samples_ms: samples,
        min_ms,
        max_ms,
        mean_ms,
        median_ms,
        p90_ms,
        p95_ms,
        per_pose_median_ms: median_ms / batch as f64,
        batch_fps_median,
        poses_per_second_median,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!((1..=3).contains(&args.batch), "--batch must be 1, 2, or 3");
    anyhow::ensure!(args.layer_audit_runs >= 1, "--layer-audit-runs must be >= 1");
    anyhow::ensure!(args.sync_baseline_samples >= 1, "--sync-baseline-samples must be >= 1");

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).context("CUDA device 0")?
    };
    let dtype = if args.f16 { DType::F16 } else { DType::F32 };

    let image = image::open(&args.image)
        .with_context(|| format!("open {:?}", args.image))?
        .to_rgb8();
    let bbox = match args.bbox.as_deref() {
        Some(s) => parse_bbox(s)?,
        None => BboxXyxy::full_image(image.width(), image.height()),
    };

    if let Some(path) = &args.dump_input {
        let (debug_crop, _) = render_topdown_rgb(&image, bbox);
        debug_crop
            .save(path)
            .with_context(|| format!("save affine crop {:?}", path))?;
    }

    let (single_input, transform) = prepare_topdown_rgb(&image, bbox, dtype, &device)?;
    let input = if args.batch == 1 {
        single_input
    } else {
        let inputs: Vec<&Tensor> = (0..args.batch).map(|_| &single_input).collect();
        Tensor::cat(&inputs, 0)?
    };
    let transforms = vec![transform; args.batch];

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&args.weights], dtype, &device)? };
    let model = RtmPoseBody26::load(&vb)?;

    let (mut output, cold_ms) = run_synced(&model, &input)?;

    let mut warmup_ms = Vec::with_capacity(args.warmup);
    for _ in 0..args.warmup {
        let (next, ms) = run_synced(&model, &input)?;
        output = next;
        warmup_ms.push(ms);
    }

    let mut measured_ms = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let (next, ms) = run_synced(&model, &input)?;
        output = next;
        measured_ms.push(ms);
    }
    let stats = timing_stats(measured_ms, args.batch);
    let inference_ms = stats.as_ref().map_or(cold_ms, |s| s.median_ms);

    let telemetry_before = if args.layer_audit && args.gpu_telemetry {
        nvidia_smi_snapshot()
    } else {
        None
    };
    let layer_audit = if args.layer_audit {
        let sync_baseline = measure_sync_baseline(&device, args.sync_baseline_samples)?;
        let mut audit_runs = Vec::with_capacity(args.layer_audit_runs);
        let mut audit_pass_ms = Vec::with_capacity(args.layer_audit_runs);
        for _ in 0..args.layer_audit_runs {
            let (next, events, pass_ms) = run_layer_profile(&model, &input)?;
            output = next;
            audit_runs.push(events);
            audit_pass_ms.push(pass_ms);
        }
        Some(summarize_layer_audit(audit_runs, audit_pass_ms, sync_baseline)?)
    } else {
        None
    };
    let telemetry_after = if args.layer_audit && args.gpu_telemetry {
        nvidia_smi_snapshot()
    } else {
        None
    };
    let observer_effect_ratio = layer_audit.as_ref().map(|audit| {
        if inference_ms > 0.0 {
            audit.profiled_pass.median_ms / inference_ms
        } else {
            f64::NAN
        }
    });

    let poses = decode_simcc(&output.simcc_x, &output.simcc_y, &transforms)?;
    let simcc_audit = audit_simcc(&output.simcc_x, &output.simcc_y)?;

    let points: Vec<_> = poses[0]
        .iter()
        .filter(|p| p.score >= args.score_threshold)
        .map(|p| {
            json!({
                "id":p.index,
                "name":p.name,
                "x":p.x,
                "y":p.y,
                "score":p.score
            })
        })
        .collect();

    let report = json!({
        "schema":"haricot.pose.body26.audit.v3",
        "backend":"candle-native",
        "model":"rtmpose-m-halpe26-256x192",
        "device":format!("{:?}", device),
        "dtype":format!("{:?}", dtype),
        "batch_size":args.batch,
        "batch_mode":if args.batch == 1 { "single_roi" } else { "replicated_same_roi_benchmark" },
        "inference_ms":inference_ms,
        "timing":{
            "scope":"model_forward_plus_synchronized_simcc_readback",
            "cold_ms":cold_ms,
            "warmup_count":args.warmup,
            "warmup_ms":warmup_ms,
            "measured_runs":args.runs,
            "stats":stats
        },
        "layer_audit":layer_audit,
        "layer_audit_observer_effect_ratio_vs_normal_median":observer_effect_ratio,
        "gpu_telemetry":{
            "enabled":args.gpu_telemetry,
            "before":telemetry_before,
            "after":telemetry_after
        },
        "bbox":bbox,
        "transform":transform,
        "crop_bounds_image":transform.source_bounds(),
        "dump_input":args.dump_input.as_ref().map(|p| p.display().to_string()),
        "json_out":args.json_out.as_ref().map(|p| p.display().to_string()),
        "score_semantics":"raw_simcc_peak_normalize_false_not_probability",
        "simcc_audit":simcc_audit,
        "keypoints":points
    });
    let pretty = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &args.json_out {
        std::fs::write(path, &pretty)
            .with_context(|| format!("write JSON audit {:?}", path))?;
    }
    println!("{pretty}");
    Ok(())
}
