use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use candle_core::{safetensors, DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use haricot_wham_lite_candle::{HaricotWhamLite, WhamLiteConfig, WhamLiteInput, WhamLiteOutput};

fn arg_value(name: &str) -> Option<String> {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == name {
            return args.next();
        }
    }
    None
}

fn parse_or<T>(name: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match arg_value(name) {
        Some(v) => v
            .parse::<T>()
            .map_err(|e| anyhow::anyhow!("invalid {name}={v}: {e}")),
        None => Ok(default),
    }
}

fn has_flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

fn main() -> Result<()> {
    let weights = arg_value("--weights").context("missing --weights <stage1.safetensors>")?;
    let device_ordinal: usize = parse_or("--device", 0usize)?;
    let device = if has_flag("--cpu") {
        Device::Cpu
    } else {
        Device::new_cuda(device_ordinal)
            .context("CUDA device unavailable; pass --cpu for CPU execution")?
    };

    let cfg = WhamLiteConfig::default();
    let weight_paths = [PathBuf::from(&weights)];
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_paths, DType::F32, &device)? };
    let model = HaricotWhamLite::load(&cfg, vb)?;

    let mut ran_mode = false;

    if let Some(reference) = arg_value("--numerical-gate") {
        ran_mode = true;
        let atol: f64 = parse_or("--atol", 5e-4_f64)?;
        let rmse_tol: f64 = parse_or("--rmse-tol", 1e-4_f64)?;
        run_numerical_gate(&model, &cfg, &device, Path::new(&reference), atol, rmse_tol)?;
    }

    if has_flag("--benchmark") {
        ran_mode = true;
        let warmup: usize = parse_or("--warmup", 20usize)?;
        let iterations: usize = parse_or("--iterations", 200usize)?;
        let frame_list = arg_value("--bench-frames").unwrap_or_else(|| "1,8,16,32".to_string());
        let frames = parse_frame_list(&frame_list)?;
        run_batch_benchmark(&model, &cfg, &device, &frames, warmup, iterations)?;
    }

    if has_flag("--stream-benchmark") {
        ran_mode = true;
        let warmup: usize = parse_or("--warmup", 20usize)?;
        let iterations: usize = parse_or("--iterations", 200usize)?;
        run_stream_benchmark(&model, &cfg, &device, warmup, iterations)?;
    }

    if !ran_mode {
        let frames: usize = parse_or("--frames", 32usize)?;
        run_smoke(&model, &cfg, &device, frames)?;
    }

    Ok(())
}

fn run_smoke(
    model: &HaricotWhamLite,
    cfg: &WhamLiteConfig,
    device: &Device,
    frames: usize,
) -> Result<()> {
    let input = deterministic_input(cfg, device, frames)?;
    let output = model.forward(&input)?;
    device.synchronize()?;

    println!("=== HARICOT WHAM LITE CANDLE V0.1 ===");
    println!("device={device:?}");
    println!("frames={frames}");
    println!("joints_3d={:?}", output.joints_3d.dims());
    println!("body_rot6d={:?}", output.body_rot6d.dims());
    println!("root_rot6d={:?}", output.root_rot6d.dims());
    println!("root_velocity={:?}", output.root_velocity.dims());
    println!("contact_logits={:?}", output.contact_logits.dims());

    if output.body_rot6d.elem_count() == 0 {
        bail!("empty model output")
    }
    Ok(())
}

fn run_batch_benchmark(
    model: &HaricotWhamLite,
    cfg: &WhamLiteConfig,
    device: &Device,
    frame_list: &[usize],
    warmup: usize,
    iterations: usize,
) -> Result<()> {
    println!("=== HARICOT WHAM LITE CUDA BENCH V0.1 ===");
    println!("device={device:?}");
    println!("mode=batch_recompute");
    println!("warmup={warmup}");
    println!("iterations={iterations}");

    for &frames in frame_list {
        let input = deterministic_input(cfg, device, frames)?;
        for _ in 0..warmup {
            let out = model.forward(&input)?;
            std::hint::black_box(out);
        }
        device.synchronize()?;

        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            device.synchronize()?;
            let start = Instant::now();
            let out = model.forward(&input)?;
            std::hint::black_box(out);
            device.synchronize()?;
            samples.push(start.elapsed());
        }
        print_stats(frames, &samples);
    }
    Ok(())
}

fn run_stream_benchmark(
    model: &HaricotWhamLite,
    cfg: &WhamLiteConfig,
    device: &Device,
    warmup: usize,
    iterations: usize,
) -> Result<()> {
    let input = deterministic_input(cfg, device, 1)?;
    let pose = input.pose2d.i((.., 0, ..))?.contiguous()?;
    let cam = input.cam_angvel.i((.., 0, ..))?.contiguous()?;
    let mut state = model.start_stream(
        &pose,
        &input.init_kp3d,
        &input.init_pose_rot6d,
        &input.init_root_rot6d,
    )?;

    for _ in 0..warmup {
        let out = model.step(&pose, &cam, &mut state)?;
        std::hint::black_box(out);
    }
    device.synchronize()?;

    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        device.synchronize()?;
        let start = Instant::now();
        let out = model.step(&pose, &cam, &mut state)?;
        std::hint::black_box(out);
        device.synchronize()?;
        samples.push(start.elapsed());
    }

    println!("=== HARICOT WHAM LITE STATEFUL STREAM BENCH V0.1 ===");
    println!("device={device:?}");
    println!("warmup={warmup}");
    println!("iterations={iterations}");
    println!("state_persistent=true");
    print_stats(1, &samples);
    Ok(())
}

fn run_numerical_gate(
    model: &HaricotWhamLite,
    cfg: &WhamLiteConfig,
    device: &Device,
    reference_path: &Path,
    atol: f64,
    rmse_tol: f64,
) -> Result<()> {
    let tensors = safetensors::load(reference_path, device)?;
    let input = WhamLiteInput {
        pose2d: get_ref(&tensors, "input.pose2d")?,
        init_kp3d: get_ref(&tensors, "input.init_kp3d")?,
        init_pose_rot6d: get_ref(&tensors, "input.init_pose_rot6d")?,
        init_root_rot6d: get_ref(&tensors, "input.init_root_rot6d")?,
        cam_angvel: get_ref(&tensors, "input.cam_angvel")?,
    };
    let frames = input.pose2d.dim(1)?;
    let output = model.forward(&input)?;
    device.synchronize()?;

    println!("=== HARICOT WHAM LITE NUMERICAL GATE V0.1 ===");
    println!("device={device:?}");
    println!("frames={frames}");
    println!("atol={atol:.3e}");
    println!("rmse_tol={rmse_tol:.3e}");

    let checks = [
        ("joints_3d", &output.joints_3d, "output.joints_3d"),
        ("body_rot6d", &output.body_rot6d, "output.body_rot6d"),
        ("root_rot6d", &output.root_rot6d, "output.root_rot6d"),
        ("root_velocity", &output.root_velocity, "output.root_velocity"),
        ("contact_logits", &output.contact_logits, "output.contact_logits"),
        ("shape", &output.shape, "output.shape"),
        ("weak_camera", &output.weak_camera, "output.weak_camera"),
    ];

    let mut numerical_pass = true;
    for (name, actual, reference_name) in checks {
        let expected = tensors
            .get(reference_name)
            .with_context(|| format!("missing {reference_name} in reference safetensors"))?;
        let metric = compare_tensors(actual, expected)?;
        let tensor_pass = metric.max_abs <= atol && metric.rmse <= rmse_tol;
        println!(
            "tensor={name} max_abs={:.9e} mean_abs={:.9e} rmse={:.9e} pass={}",
            metric.max_abs, metric.mean_abs, metric.rmse, tensor_pass
        );
        numerical_pass &= tensor_pass;
    }
    println!("PYTORCH_CANDLE_NUMERICAL_GATE={}", if numerical_pass { "PASS" } else { "FAIL" });

    // Stateful lifecycle gate: feed the same reference input one frame at a time and
    // ensure stacking the persistent-step outputs reproduces the sequence forward.
    let stream_output = forward_via_persistent_steps(model, cfg, &input)?;
    device.synchronize()?;
    let stream_checks = [
        ("stream.joints_3d", &stream_output.joints_3d, &output.joints_3d),
        ("stream.body_rot6d", &stream_output.body_rot6d, &output.body_rot6d),
        ("stream.root_rot6d", &stream_output.root_rot6d, &output.root_rot6d),
        (
            "stream.root_velocity",
            &stream_output.root_velocity,
            &output.root_velocity,
        ),
        (
            "stream.contact_logits",
            &stream_output.contact_logits,
            &output.contact_logits,
        ),
    ];
    let mut stream_pass = true;
    for (name, actual, expected) in stream_checks {
        let metric = compare_tensors(actual, expected)?;
        let tensor_pass = metric.max_abs <= 1e-7 && metric.rmse <= 1e-8;
        println!(
            "tensor={name} max_abs={:.9e} rmse={:.9e} pass={}",
            metric.max_abs, metric.rmse, tensor_pass
        );
        stream_pass &= tensor_pass;
    }
    println!("STATEFUL_STREAM_GATE={}", if stream_pass { "PASS" } else { "FAIL" });

    let passed = numerical_pass && stream_pass;
    println!("V0_1_GATE={}", if passed { "PASS" } else { "FAIL" });
    if !passed {
        bail!("v0.1 numerical/stateful gate failed")
    }
    Ok(())
}

fn forward_via_persistent_steps(
    model: &HaricotWhamLite,
    cfg: &WhamLiteConfig,
    input: &WhamLiteInput,
) -> Result<WhamLiteOutput> {
    let (batch, frames, _) = input.pose2d.dims3()?;
    let first = input.pose2d.i((.., 0, ..))?.contiguous()?;
    let mut state = model.start_stream(
        &first,
        &input.init_kp3d,
        &input.init_pose_rot6d,
        &input.init_root_rot6d,
    )?;

    let mut joints = Vec::with_capacity(frames);
    let mut body = Vec::with_capacity(frames);
    let mut roots = vec![input.init_root_rot6d.clone()];
    let mut velocities = Vec::with_capacity(frames);
    let mut contacts = Vec::with_capacity(frames);
    let mut shapes = Vec::with_capacity(frames);
    let mut cameras = Vec::with_capacity(frames);

    for t in 0..frames {
        let pose = input.pose2d.i((.., t, ..))?.contiguous()?;
        let cam = input.cam_angvel.i((.., t, ..))?.contiguous()?;
        let out = model.step(&pose, &cam, &mut state)?;
        joints.push(out.joints_3d);
        body.push(out.body_rot6d);
        roots.push(out.root_rot6d);
        velocities.push(out.root_velocity);
        contacts.push(out.contact_logits);
        shapes.push(out.shape);
        cameras.push(out.weak_camera);
    }

    Ok(WhamLiteOutput {
        joints_3d: Tensor::stack(&joints, 1)?.reshape((batch, frames, cfg.n_joints, 3))?,
        body_rot6d: Tensor::stack(&body, 1)?,
        root_rot6d: Tensor::stack(&roots, 1)?,
        root_velocity: Tensor::stack(&velocities, 1)?,
        contact_logits: Tensor::stack(&contacts, 1)?,
        shape: Tensor::stack(&shapes, 1)?,
        weak_camera: Tensor::stack(&cameras, 1)?,
    })
}

fn deterministic_input(
    cfg: &WhamLiteConfig,
    device: &Device,
    frames: usize,
) -> Result<WhamLiteInput> {
    if frames == 0 {
        bail!("frames must be >= 1")
    }

    let mut pose2d = Vec::with_capacity(frames * cfg.input_dim);
    for t in 0..frames {
        for i in 0..cfg.input_dim {
            let x = ((t * cfg.input_dim + i) as f32 * 0.017).sin() * 0.25;
            pose2d.push(x);
        }
    }

    let mut init_kp3d = Vec::with_capacity(cfg.kp3d_dim());
    for i in 0..cfg.kp3d_dim() {
        init_kp3d.push((i as f32 * 0.031).cos() * 0.05);
    }

    let mut cam = Vec::with_capacity(frames * 6);
    for t in 0..frames {
        for i in 0..6 {
            cam.push(((t * 6 + i) as f32 * 0.013).sin() * 0.002);
        }
    }

    Ok(WhamLiteInput {
        pose2d: Tensor::from_vec(pose2d, (1, frames, cfg.input_dim), device)?,
        init_kp3d: Tensor::from_vec(init_kp3d, (1, cfg.kp3d_dim()), device)?,
        init_pose_rot6d: neutral_pose6d(cfg.pose_joints, device)?,
        init_root_rot6d: neutral_pose6d(1, device)?.reshape((1, 6))?,
        cam_angvel: Tensor::from_vec(cam, (1, frames, 6), device)?,
    })
}

fn neutral_pose6d(joints: usize, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(joints * 6);
    for _ in 0..joints {
        values.extend_from_slice(&[1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }
    Ok(Tensor::from_vec(values, (1, joints * 6), device)?)
}

fn get_ref(tensors: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .get(name)
        .cloned()
        .with_context(|| format!("missing {name} in reference safetensors"))
}

#[derive(Clone, Copy, Debug)]
struct ErrorMetric {
    max_abs: f64,
    mean_abs: f64,
    rmse: f64,
}

fn compare_tensors(actual: &Tensor, expected: &Tensor) -> Result<ErrorMetric> {
    if actual.dims() != expected.dims() {
        bail!(
            "shape mismatch actual={:?} expected={:?}",
            actual.dims(),
            expected.dims()
        )
    }
    let n = actual.elem_count();
    let actual = actual.reshape((n,))?.to_vec1::<f32>()?;
    let expected = expected.reshape((n,))?.to_vec1::<f32>()?;

    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for (&a, &e) in actual.iter().zip(expected.iter()) {
        let d = (a as f64 - e as f64).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
        sum_sq += d * d;
    }
    let denom = n.max(1) as f64;
    Ok(ErrorMetric {
        max_abs,
        mean_abs: sum_abs / denom,
        rmse: (sum_sq / denom).sqrt(),
    })
}

fn parse_frame_list(s: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for item in s.split(',') {
        let value: usize = item
            .trim()
            .parse()
            .with_context(|| format!("invalid frame count in --bench-frames: {item}"))?;
        if value == 0 {
            bail!("frame counts must be >= 1")
        }
        out.push(value);
    }
    if out.is_empty() {
        bail!("--bench-frames cannot be empty")
    }
    Ok(out)
}

fn print_stats(frames: usize, samples: &[Duration]) {
    let mut us: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1e6).collect();
    us.sort_by(|a, b| a.total_cmp(b));
    let min = us.first().copied().unwrap_or(0.0);
    let median = percentile(&us, 0.50);
    let p95 = percentile(&us, 0.95);
    let mean = if us.is_empty() {
        0.0
    } else {
        us.iter().sum::<f64>() / us.len() as f64
    };
    println!(
        "frames={frames} min_us={min:.3} median_us={median:.3} mean_us={mean:.3} p95_us={p95:.3}"
    );
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
