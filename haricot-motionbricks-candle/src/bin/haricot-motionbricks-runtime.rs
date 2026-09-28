use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use candle_core::{conv::CudnnFwdAlgo, safetensors, DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};
use haricot_motionbricks_candle::{
    production_conv1d_policy, Conv1dFrontierCase, MotionBricksConv1dPolicy,
    MotionBricksRuntimeReference, PoseRuntimeInput, RootRuntimeInput, VqvaeRuntimeInput,
    MOTIONBRICKS_PRODUCTION_CUDNN_SM61, MOTIONBRICKS_PRODUCTION_SM61,
};

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
    if has_flag("--help") || has_flag("-h") {
        println!("haricot-motionbricks-runtime — v0.3.3 Production Conv1d Policy (r2 A/B/A retained)");
        println!("usage:");
        println!("  --weights <motionbricks-runtime-v1.safetensors>");
        println!("  [--reference <official-runtime-reference.safetensors>]");
        println!("  [--numerical-gate] [--benchmark] [--operator-breakdown] [--conv-frontier] [--conv-r1-sweep] [--conv-r2-aba]");
        println!("  [--asd-v2-audit] (CUDA baseline + Torch numerical gate, NO route promotion)");
        println!("  [--frontier-warmup 10] [--frontier-iterations 60] [--sweep-warmup 10] [--sweep-iterations 60]");
        println!("  [--aba-warmup 10] [--aba-iterations 60] [--warmup 10] [--iterations 100] [--cpu] [--device 0]");
        println!("  Historical --candle-asd-candidate/--asd-policy/--conv-r2-aba are DISABLED");
        println!("profile=fixed16-no-text-all-root-constraints");
        return Ok(());
    }

    let weights = arg_value("--weights").context("missing --weights <runtime.safetensors>")?;
    let device_ordinal: usize = parse_or("--device", 0usize)?;
    let device = if has_flag("--cpu") {
        Device::Cpu
    } else {
        Device::new_cuda(device_ordinal)
            .context("CUDA device unavailable; pass --cpu for CPU execution")?
    };
    let dispatch_sm = match arg_value("--dispatch-sm") {
        Some(v) => Some(
            v.parse::<u32>()
                .map_err(|e| anyhow::anyhow!("invalid --dispatch-sm={v}: {e}"))?,
        ),
        None => None,
    };
    let dispatch_cudnn = arg_value("--dispatch-cudnn");
    // ASD V2 has no promoted Conv1D route for MotionBricks on CUDA12.9.2/cuDNN9.10.
    // The legacy Candle ASD v0.1 and 9.1.0.2 policies are disabled fail-closed.
    if arg_value("--candle-asd-candidate").is_some()
        || arg_value("--asd-policy").is_some()
        || dispatch_sm.is_some() || dispatch_cudnn.is_some()
        || has_flag("--conv-r2-aba") {
        bail!("ASD V2 rejects the obsolete Conv1D dispatch and cuDNN 9.1.0.2 proof; requalify exact signatures on 9.10.2.21");
    }
    let asd_v2_audit = has_flag("--asd-v2-audit");
    if asd_v2_audit {
        if !has_flag("--numerical-gate") || has_flag("--benchmark")
            || has_flag("--operator-breakdown") || has_flag("--conv-frontier")
            || has_flag("--conv-r1-sweep") || has_flag("--cpu") {
            bail!("--asd-v2-audit requires CUDA --numerical-gate only (not benchmarking)");
        }
        if parse_or("--atol", 5e-4_f64)? != 5e-4_f64
            || parse_or("--rmse-tol", 1e-4_f64)? != 1e-4_f64 {
            bail!("ASD V2 baseline numerical tolerances fixed at 5e-4 / 1e-4");
        }
        if arg_value("--reference").is_none() { bail!("ASD V2 requires Torch --reference"); }
        println!("ASD_V2_CONV1D_DISPATCH=BASELINE_NO_PRODUCTION_RULE new_cudnn_qualification=REQUIRED");
    }
    let policy = production_conv1d_policy(None, None);
    let paths = [PathBuf::from(&weights)];
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::F32, &device)? };
    let model = MotionBricksRuntimeReference::load(vb)?;
    if asd_v2_audit {
        let reference = arg_value("--reference").context("ASD V2 requires Torch oracle")?;
        numerical_gate(&model, &device, Path::new(&reference), 5e-4, 1e-4,
                       MotionBricksConv1dPolicy::Baseline, None, None)?;
        println!("ASD_V2_BASELINE_NUMERICAL_GATE=PASS optimized_conv1d=false performance=NOT_MEASURED promotion=NONE");
        return Ok(());
    }

    let reference_path = arg_value("--reference");
    let mut ran = false;
    if has_flag("--numerical-gate") {
        ran = true;
        let reference = reference_path
            .as_deref()
            .context("--numerical-gate requires --reference <safetensors>")?;
        let atol: f64 = parse_or("--atol", 5e-4_f64)?;
        let rmse_tol: f64 = parse_or("--rmse-tol", 1e-4_f64)?;
        numerical_gate(
            &model,
            &device,
            Path::new(reference),
            atol,
            rmse_tol,
            policy,
            dispatch_sm,
            dispatch_cudnn.as_deref(),
        )?;
    }
    if has_flag("--benchmark") {
        ran = true;
        let reference = reference_path
            .as_deref()
            .context("--benchmark requires --reference <safetensors>")?;
        let warmup: usize = parse_or("--warmup", 10usize)?;
        let iterations: usize = parse_or("--iterations", 100usize)?;
        benchmark(
            &model,
            &device,
            Path::new(reference),
            warmup,
            iterations,
            policy,
        )?;
    }
    if has_flag("--operator-breakdown") {
        ran = true;
        let reference = reference_path
            .as_deref()
            .context("--operator-breakdown requires --reference <safetensors>")?;
        let warmup: usize = parse_or("--warmup", 10usize)?;
        let iterations: usize = parse_or("--iterations", 100usize)?;
        operator_breakdown(&model, &device, Path::new(reference), warmup, iterations)?;
    }
    if has_flag("--conv-frontier") {
        ran = true;
        let reference = reference_path
            .as_deref()
            .context("--conv-frontier requires --reference <safetensors>")?;
        let warmup: usize = parse_or("--frontier-warmup", 10usize)?;
        let iterations: usize = parse_or("--frontier-iterations", 60usize)?;
        conv_frontier(&model, &device, Path::new(reference), warmup, iterations)?;
    }
    if has_flag("--conv-r1-sweep") {
        ran = true;
        let reference = reference_path
            .as_deref()
            .context("--conv-r1-sweep requires --reference <safetensors>")?;
        let warmup: usize = parse_or("--sweep-warmup", 10usize)?;
        let iterations: usize = parse_or("--sweep-iterations", 60usize)?;
        conv_r1_sweep(&model, &device, Path::new(reference), warmup, iterations)?;
    }
    if has_flag("--conv-r2-aba") {
        bail!("ASD V2: old 9.1.0.2 A/B/A cannot certify 9.10.2.21");
    }
    if !ran {
        let reference = reference_path
            .as_deref()
            .context("smoke mode requires --reference <safetensors>")?;
        smoke(
            &model,
            &device,
            Path::new(reference),
            policy,
            dispatch_sm,
            dispatch_cudnn.as_deref(),
        )?;
    }
    Ok(())
}

struct Inputs {
    root: RootRuntimeInput,
    pose: PoseRuntimeInput,
    vqvae: VqvaeRuntimeInput,
}

fn get(map: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    map.get(name)
        .cloned()
        .with_context(|| format!("missing tensor {name}"))
}

fn load_inputs(map: &HashMap<String, Tensor>) -> Result<Inputs> {
    Ok(Inputs {
        root: RootRuntimeInput {
            global_root_values: get(map, "input.root.global_root_values")?,
            local_root_values: get(map, "input.root.local_root_values")?,
            poses: get(map, "input.root.poses")?,
        },
        pose: PoseRuntimeInput {
            pose_tokens: get(map, "input.pose.pose_tokens")?,
            root_values: get(map, "input.pose.root_values")?,
            pose_cond: get(map, "input.pose.pose_cond")?,
            has_pose_cond: get(map, "input.pose.has_pose_cond")?,
        },
        vqvae: VqvaeRuntimeInput {
            pose_tokens: get(map, "input.vqvae.pose_tokens")?,
            target_cond: get(map, "input.vqvae.target_cond")?,
            has_target_cond: get(map, "input.vqvae.has_target_cond")?,
            external_cond: get(map, "input.vqvae.external_cond")?,
        },
    })
}

fn smoke(
    model: &MotionBricksRuntimeReference,
    device: &Device,
    path: &Path,
    policy: MotionBricksConv1dPolicy,
    dispatch_sm: Option<u32>,
    dispatch_cudnn: Option<&str>,
) -> Result<()> {
    let map = safetensors::load(path, device)?;
    let input = load_inputs(&map)?;
    let root = model.root.forward_fixed_16_with_policy(&input.root, policy)?;
    let pose = model.pose.forward_fixed_16(&input.pose)?;
    let recon = model.decoder.forward_with_policy(&input.vqvae, policy)?;
    device.synchronize()?;

    println!("=== HARICOT MOTIONBRICKS RUNTIME V0.3 ===");
    println!("device={device:?}");
    println!("profile=fixed16-no-text-all-root-constraints");
    println!("conv1d_policy={}", policy.as_str());
    println!("dispatch_sm={}", dispatch_sm.map_or_else(|| "none".to_string(), |v| v.to_string()));
    println!("dispatch_cudnn={}", dispatch_cudnn.unwrap_or("none"));
    println!("conv_transpose_count=0");
    println!("attention_layout_policy=transpose_contiguous_matmul");
    println!("root.num_token_logits={:?}", root.num_token_logits.dims());
    println!(
        "root.pred_global_root_values={:?}",
        root.pred_global_root_values.dims()
    );
    println!("pose.pose_logits={:?}", pose.dims());
    println!("vqvae.recon_state={:?}", recon.dims());

    let pass = root.num_token_logits.dims() == [1, 12]
        && root.pred_global_root_values.dims() == [1, 64, 5]
        && pose.dims() == [1, 16, 8, 10]
        && recon.dims() == [1, 64, 413];
    println!("MOTIONBRICKS_RUNTIME_ARCH_GATE={}", if pass { "PASS" } else { "FAIL" });
    if !pass {
        bail!("MotionBricks runtime architecture gate failed")
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Stats {
    max_abs: f64,
    mean_abs: f64,
    rmse: f64,
}

fn tensor_stats(got: &Tensor, reference: &Tensor) -> Result<Stats> {
    if got.dims() != reference.dims() {
        bail!("shape mismatch {:?} != {:?}", got.dims(), reference.dims())
    }
    let n = got.elem_count();
    let a = got.reshape((n,))?.to_vec1::<f32>()?;
    let b = reference.reshape((n,))?.to_vec1::<f32>()?;
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x as f64 - y as f64).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
        sum_sq += d * d;
    }
    Ok(Stats {
        max_abs,
        mean_abs: sum_abs / n as f64,
        rmse: (sum_sq / n as f64).sqrt(),
    })
}

fn report(name: &str, got: &Tensor, reference: &Tensor, atol: f64, rmse_tol: f64) -> Result<bool> {
    let s = tensor_stats(got, reference)?;
    let pass = s.max_abs <= atol && s.rmse <= rmse_tol;
    println!(
        "tensor={name} max_abs={:.9e} mean_abs={:.9e} rmse={:.9e} atol={atol:.9e} rmse_tol={rmse_tol:.9e} pass={pass}",
        s.max_abs, s.mean_abs, s.rmse
    );
    Ok(pass)
}

fn numerical_gate(
    model: &MotionBricksRuntimeReference,
    device: &Device,
    path: &Path,
    atol: f64,
    rmse_tol: f64,
    policy: MotionBricksConv1dPolicy,
    dispatch_sm: Option<u32>,
    dispatch_cudnn: Option<&str>,
) -> Result<()> {
    let map = safetensors::load(path, device)?;
    let input = load_inputs(&map)?;
    let root = model.root.forward_fixed_16_with_policy(&input.root, policy)?;
    let pose = model.pose.forward_fixed_16(&input.pose)?;
    let recon = model.decoder.forward_with_policy(&input.vqvae, policy)?;
    device.synchronize()?;

    println!("=== HARICOT MOTIONBRICKS RUNTIME NUMERICAL GATE V0.3.2-R2 ===");
    println!("device={device:?}");
    println!("profile=fixed16-no-text-all-root-constraints");
    println!("conv1d_policy={}", policy.as_str());
    println!("dispatch_sm={}", dispatch_sm.map_or_else(|| "none".to_string(), |v| v.to_string()));
    println!("dispatch_cudnn={}", dispatch_cudnn.unwrap_or("none"));
    println!("atol={atol:.3e}");
    println!("rmse_tol={rmse_tol:.3e}");

    let mut root_pass = true;
    root_pass &= report(
        "root.num_token_logits",
        &root.num_token_logits,
        &get(&map, "output.root.num_token_logits")?,
        atol,
        rmse_tol,
    )?;
    root_pass &= report(
        "root.pred_global_root_values",
        &root.pred_global_root_values,
        &get(&map, "output.root.pred_global_root_values")?,
        atol,
        rmse_tol,
    )?;
    let pose_pass = report(
        "pose.pose_logits",
        &pose,
        &get(&map, "output.pose.pose_logits")?,
        atol,
        rmse_tol,
    )?;
    let vqvae_pass = report(
        "vqvae.recon_state",
        &recon,
        &get(&map, "output.vqvae.recon_state")?,
        atol,
        rmse_tol,
    )?;

    println!("MOTIONBRICKS_ROOT_NUMERICAL_GATE={}", if root_pass { "PASS" } else { "FAIL" });
    println!("MOTIONBRICKS_POSE_NUMERICAL_GATE={}", if pose_pass { "PASS" } else { "FAIL" });
    println!("MOTIONBRICKS_VQVAE_DECODER_GATE={}", if vqvae_pass { "PASS" } else { "FAIL" });
    let pass = root_pass && pose_pass && vqvae_pass;
    println!("V0_3_GATE={}", if pass { "PASS" } else { "FAIL" });
    println!(
        "V0_3_R2_TORCH_ORACLE_GATE={}",
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        bail!("MotionBricks runtime numerical parity failed")
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct BenchStats {
    min_us: f64,
    median_us: f64,
    mean_us: f64,
    p95_us: f64,
}

fn bench_one<F>(
    name: &str,
    device: &Device,
    warmup: usize,
    iterations: usize,
    mut f: F,
) -> Result<BenchStats>
where
    F: FnMut() -> Result<()>,
{
    if iterations == 0 {
        bail!("--iterations must be >= 1")
    }
    for _ in 0..warmup {
        f()?;
    }
    device.synchronize()?;
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        device.synchronize()?;
        let start = Instant::now();
        f()?;
        device.synchronize()?;
        samples.push(start.elapsed());
    }
    let stats = calc_stats(&samples);
    println!(
        "component={name} min_us={:.3} median_us={:.3} mean_us={:.3} p95_us={:.3}",
        stats.min_us, stats.median_us, stats.mean_us, stats.p95_us
    );
    Ok(stats)
}

fn benchmark(
    model: &MotionBricksRuntimeReference,
    device: &Device,
    path: &Path,
    warmup: usize,
    iterations: usize,
    policy: MotionBricksConv1dPolicy,
) -> Result<()> {
    let map = safetensors::load(path, device)?;
    let input = load_inputs(&map)?;
    println!("=== HARICOT MOTIONBRICKS CUDA RUNTIME BENCH V0.3.2-R2 ===");
    println!("device={device:?}");
    println!("profile=fixed16-no-text-all-root-constraints");
    println!("conv1d_policy={}", policy.as_str());
    println!("warmup={warmup}");
    println!("iterations={iterations}");

    let root = bench_one("root_backbone", device, warmup, iterations, || {
        let y = model.root.forward_fixed_16_with_policy(&input.root, policy)?;
        std::hint::black_box(y);
        Ok(())
    })?;
    let pose = bench_one("pose_backbone", device, warmup, iterations, || {
        let y = model.pose.forward_fixed_16(&input.pose)?;
        std::hint::black_box(y);
        Ok(())
    })?;
    let vq = bench_one("vqvae_pose_decoder", device, warmup, iterations, || {
        let y = model.decoder.forward_with_policy(&input.vqvae, policy)?;
        std::hint::black_box(y);
        Ok(())
    })?;
    let full = bench_one("runtime_components_sum", device, warmup, iterations, || {
        let r = model.root.forward_fixed_16_with_policy(&input.root, policy)?;
        let p = model.pose.forward_fixed_16(&input.pose)?;
        let v = model.decoder.forward_with_policy(&input.vqvae, policy)?;
        std::hint::black_box((r, p, v));
        Ok(())
    })?;

    println!("=== CUDA RUNTIME COMPONENT SHARES ===");
    println!("shares_are_median_ratios_not_additive=true");
    print_share("root_backbone", root.median_us, full.median_us);
    print_share("pose_backbone", pose.median_us, full.median_us);
    print_share("vqvae_pose_decoder", vq.median_us, full.median_us);
    println!("MOTIONBRICKS_V0_3_1_RUNTIME_BENCH=PASS");
    Ok(())
}

fn operator_breakdown(
    model: &MotionBricksRuntimeReference,
    device: &Device,
    path: &Path,
    warmup: usize,
    iterations: usize,
) -> Result<()> {
    let map = safetensors::load(path, device)?;
    let input = load_inputs(&map)?;

    // Precompute fixed inputs for each isolated operator so the measured closure
    // contains the named operator rather than its upstream dependencies.
    let root_state = model.root.benchmark_prepare_fixed_16(&input.root)?;
    let root_shared = model.root.benchmark_shared_transformer(&root_state)?;
    let root_token_hidden = model.root.benchmark_root_token_transformer(&root_state)?;
    let root_attn0 = model.root.benchmark_shared_layer0_attention(&root_state)?;

    let pose_state = model.pose.benchmark_prepare_fixed_16(&input.pose)?;
    let pose_hidden = model.pose.benchmark_transformer(&pose_state)?;
    let pose_attn0 = model.pose.benchmark_layer0_attention(&pose_state)?;

    let vq_state = model.decoder.benchmark_codebook_lookup(&input.vqvae)?;
    device.synchronize()?;

    println!("=== HARICOT MOTIONBRICKS CUDA OPERATOR BREAKDOWN V0.3.1 ===");
    println!("device={device:?}");
    println!("profile=fixed16-no-text-all-root-constraints");
    println!("measurement=host_wall_clock_with_cuda_synchronize");
    println!("conv1d_policy=official_standard_groups1");
    println!("conv_transpose_count=0");
    println!("warmup={warmup}");
    println!("iterations={iterations}");

    let runtime = bench_one("reference.runtime_full", device, warmup, iterations, || {
        let r = model.root.forward_fixed_16(&input.root)?;
        let p = model.pose.forward_fixed_16(&input.pose)?;
        let v = model.decoder.forward(&input.vqvae)?;
        std::hint::black_box((r, p, v));
        Ok(())
    })?;

    // This captures the fixed per-sample observer cost of the post-operation
    // synchronize + host timer. Subtract only for diagnostic reporting.
    let observer = bench_one("observer.sync_only", device, warmup, iterations, || Ok(()))?;

    let mut ops: Vec<(&str, BenchStats)> = Vec::new();
    ops.push((
        "root.input_projection_and_fixed16_prep",
        bench_one("root.input_projection_and_fixed16_prep", device, warmup, iterations, || {
            let y = model.root.benchmark_prepare_fixed_16(&input.root)?;
            std::hint::black_box(y);
            Ok(())
        })?,
    ));
    ops.push((
        "root.shared_transformer_3l",
        bench_one("root.shared_transformer_3l", device, warmup, iterations, || {
            let y = model.root.benchmark_shared_transformer(&root_state)?;
            std::hint::black_box(y);
            Ok(())
        })?,
    ));
    ops.push((
        "root.shared_layer0.attention_norm",
        bench_one("root.shared_layer0.attention_norm", device, warmup, iterations, || {
            let y = model.root.benchmark_shared_layer0_attention(&root_state)?;
            std::hint::black_box(y);
            Ok(())
        })?,
    ));
    ops.push((
        "root.shared_layer0.ffn_norm",
        bench_one("root.shared_layer0.ffn_norm", device, warmup, iterations, || {
            let y = model.root.benchmark_shared_layer0_ffn(&root_attn0)?;
            std::hint::black_box(y);
            Ok(())
        })?,
    ));
    ops.push((
        "root.num_token_head",
        bench_one("root.num_token_head", device, warmup, iterations, || {
            let y = model.root.benchmark_num_token_head(&root_shared)?;
            std::hint::black_box(y);
            Ok(())
        })?,
    ));
    ops.push((
        "root.root_token_transformer_3l",
        bench_one("root.root_token_transformer_3l", device, warmup, iterations, || {
            let y = model.root.benchmark_root_token_transformer(&root_state)?;
            std::hint::black_box(y);
            Ok(())
        })?,
    ));
    ops.push((
        "root.conv_decoder",
        bench_one("root.conv_decoder", device, warmup, iterations, || {
            let y = model.root.benchmark_conv_decoder(&root_token_hidden, &root_state)?;
            std::hint::black_box(y);
            Ok(())
        })?,
    ));

    ops.push((
        "pose.input_embedding_projection",
        bench_one("pose.input_embedding_projection", device, warmup, iterations, || {
            let y = model.pose.benchmark_prepare_fixed_16(&input.pose)?;
            std::hint::black_box(y);
            Ok(())
        })?,
    ));
    ops.push((
        "pose.transformer_16l",
        bench_one("pose.transformer_16l", device, warmup, iterations, || {
            let y = model.pose.benchmark_transformer(&pose_state)?;
            std::hint::black_box(y);
            Ok(())
        })?,
    ));
    ops.push((
        "pose.layer0.attention_norm",
        bench_one("pose.layer0.attention_norm", device, warmup, iterations, || {
            let y = model.pose.benchmark_layer0_attention(&pose_state)?;
            std::hint::black_box(y);
            Ok(())
        })?,
    ));
    ops.push((
        "pose.layer0.ffn_norm",
        bench_one("pose.layer0.ffn_norm", device, warmup, iterations, || {
            let y = model.pose.benchmark_layer0_ffn(&pose_attn0)?;
            std::hint::black_box(y);
            Ok(())
        })?,
    ));
    ops.push((
        "pose.logit_head",
        bench_one("pose.logit_head", device, warmup, iterations, || {
            let y = model.pose.benchmark_logit_head(&pose_hidden)?;
            std::hint::black_box(y);
            Ok(())
        })?,
    ));

    ops.push((
        "vq.codebook_lookup",
        bench_one("vq.codebook_lookup", device, warmup, iterations, || {
            let y = model.decoder.benchmark_codebook_lookup(&input.vqvae)?;
            std::hint::black_box(y);
            Ok(())
        })?,
    ));
    ops.push((
        "vq.decoder_conv_resnet_upsample",
        bench_one("vq.decoder_conv_resnet_upsample", device, warmup, iterations, || {
            let y = model
                .decoder
                .benchmark_decoder_from_codebook(&vq_state, &input.vqvae)?;
            std::hint::black_box(y);
            Ok(())
        })?,
    ));

    println!("=== CUDA OPERATOR BREAKDOWN SUMMARY ===");
    println!("observer_sync_median_us={:.3}", observer.median_us);
    println!("runtime_reference_median_us={:.3}", runtime.median_us);
    println!("observer_subtraction=diagnostic_only");
    println!("shares_are_diagnostic_not_additive=true");
    for (name, stats) in &ops {
        let net = (stats.median_us - observer.median_us).max(0.0);
        let pct = if runtime.median_us > 0.0 {
            100.0 * net / runtime.median_us
        } else {
            0.0
        };
        println!(
            "operator={name} raw_median_us={:.3} net_median_us={net:.3} runtime_share_pct={pct:.2}",
            stats.median_us
        );
    }

    let pose_transformer = ops
        .iter()
        .find(|(name, _)| *name == "pose.transformer_16l")
        .map(|(_, s)| s.median_us)
        .unwrap_or(0.0);
    let pose_net = (pose_transformer - observer.median_us).max(0.0);
    println!(
        "derived=pose.transformer_per_layer_mean net_us={:.3}",
        pose_net / 16.0
    );

    let net_of = |target: &str| -> f64 {
        ops.iter()
            .find(|(name, _)| *name == target)
            .map(|(_, s)| (s.median_us - observer.median_us).max(0.0))
            .unwrap_or(0.0)
    };
    let major_transformers = net_of("root.shared_transformer_3l")
        + net_of("root.root_token_transformer_3l")
        + net_of("pose.transformer_16l");
    let major_conv_decoders =
        net_of("root.conv_decoder") + net_of("vq.decoder_conv_resnet_upsample");
    let transformer_pct = if runtime.median_us > 0.0 {
        100.0 * major_transformers / runtime.median_us
    } else {
        0.0
    };
    let conv_pct = if runtime.median_us > 0.0 {
        100.0 * major_conv_decoders / runtime.median_us
    } else {
        0.0
    };
    let ratio = if major_conv_decoders > 0.0 {
        major_transformers / major_conv_decoders
    } else {
        f64::INFINITY
    };
    println!(
        "derived=major_transformers net_us={major_transformers:.3} runtime_share_pct={transformer_pct:.2}"
    );
    println!(
        "derived=major_conv_decoders net_us={major_conv_decoders:.3} runtime_share_pct={conv_pct:.2}"
    );
    println!("derived=transformer_to_conv_decoder_ratio value={ratio:.3}");
    println!("MOTIONBRICKS_V0_3_1_OPERATOR_BREAKDOWN=PASS");
    Ok(())
}


#[derive(Clone)]
struct CanonicalConvCase {
    case: Conv1dFrontierCase,
    multiplicity: usize,
    sources: Vec<String>,
}

#[derive(Clone, Copy, Default)]
struct FrontierFamilyTotals {
    instances: usize,
    baseline_proxy_us: f64,
    best_proxy_us: f64,
}

fn canonicalize_conv_cases(cases: Vec<Conv1dFrontierCase>) -> Result<Vec<CanonicalConvCase>> {
    let mut out: Vec<CanonicalConvCase> = Vec::new();
    let mut by_signature: HashMap<String, usize> = HashMap::new();
    for case in cases {
        let sig = case.signature()?;
        if let Some(&idx) = by_signature.get(&sig) {
            out[idx].multiplicity += 1;
            out[idx].sources.push(case.source);
        } else {
            let idx = out.len();
            by_signature.insert(sig, idx);
            out.push(CanonicalConvCase {
                sources: vec![case.source.clone()],
                case,
                multiplicity: 1,
            });
        }
    }
    out.sort_by_key(|c| c.case.signature().unwrap_or_default());
    Ok(out)
}

fn conv1d_k1_gemm(case: &Conv1dFrontierCase) -> Result<Tensor> {
    let (b, cin, len) = case.input.dims3()?;
    let (cout, wcin, kernel) = case.conv.weight().dims3()?;
    if kernel != 1 || cin != wcin {
        bail!("K1 GEMM candidate requires weight [cout,cin,1]")
    }
    let x = case
        .input
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b * len, cin))?;
    let wt = case
        .conv
        .weight()
        .reshape((cout, cin))?
        .transpose(0, 1)?
        .contiguous()?;
    let mut y = x.matmul(&wt)?;
    if let Some(bias) = case.conv.bias() {
        y = y.broadcast_add(&bias.reshape((1, cout))?)?;
    }
    Ok(y
        .reshape((b, len, cout))?
        .transpose(1, 2)?
        .contiguous()?)
}

fn frontier_family(kernel: usize, dilation: usize) -> String {
    if kernel == 1 {
        "k1".to_string()
    } else {
        format!("k{kernel}_d{dilation}")
    }
}

fn conv_frontier(
    model: &MotionBricksRuntimeReference,
    device: &Device,
    path: &Path,
    warmup: usize,
    iterations: usize,
) -> Result<()> {
    let map = safetensors::load(path, device)?;
    let input = load_inputs(&map)?;
    let mut all = model.root.benchmark_conv_frontier_cases(&input.root)?;
    all.extend(model.decoder.benchmark_conv_frontier_cases(&input.vqvae)?);
    device.synchronize()?;

    let observed_instances = all.len();
    let cases = canonicalize_conv_cases(all)?;
    println!("=== HARICOT MOTIONBRICKS STANDARD CONV1D DECODER FRONTIER V0.3.2 ===");
    println!("device={device:?}");
    println!("profile=fixed16-no-text-all-root-constraints");
    println!("scope=root+vq_decoders_only");
    println!("current_path=candle_nn::Conv1d");
    println!("build_expected=features:cudnn");
    println!("groups=1");
    println!("conv_transpose_count=0");
    println!("candidate_k1=explicit_gemm");
    println!("candidate_k3=deferred_until_frontier_proves_roi");
    println!("observed_conv_instances={observed_instances}");
    println!("canonical_shapes={}", cases.len());
    println!("warmup={warmup}");
    println!("iterations={iterations}");

    if observed_instances != 42 {
        bail!("expected exactly 42 decoder Conv1d instances, got {observed_instances}")
    }

    let observer = bench_one("frontier.observer.sync_only", device, warmup, iterations, || Ok(()))?;
    let mut families: HashMap<String, FrontierFamilyTotals> = HashMap::new();
    let mut k1_any_win = false;
    let mut k1_all_parity = true;

    for canonical in &cases {
        let sig = canonical.case.signature()?;
        let (b, cin, len) = canonical.case.input.dims3()?;
        let (cout, _, kernel) = canonical.case.conv.weight().dims3()?;
        let cfg = *canonical.case.conv.config();
        let family = frontier_family(kernel, cfg.dilation);
        println!(
            "shape={sig} family={family} multiplicity={} example_source={} b={b} cin={cin} cout={cout} len={len} kernel={kernel} dilation={} padding={} stride={}",
            canonical.multiplicity,
            canonical.sources[0],
            cfg.dilation,
            cfg.padding,
            cfg.stride
        );

        let current_name = format!("frontier.{sig}.current");
        let current = bench_one(&current_name, device, warmup, iterations, || {
            let y = canonical.case.conv.forward(&canonical.case.input)?;
            std::hint::black_box(y);
            Ok(())
        })?;
        let current_net = (current.median_us - observer.median_us).max(0.0);
        let mut best_net = current_net;

        if kernel == 1 {
            let current_out = canonical.case.conv.forward(&canonical.case.input)?;
            let gemm_out = conv1d_k1_gemm(&canonical.case)?;
            device.synchronize()?;
            let parity = tensor_stats(&gemm_out, &current_out)?;
            let parity_pass = parity.max_abs <= 2e-4 && parity.rmse <= 5e-5;
            k1_all_parity &= parity_pass;
            println!(
                "candidate={sig}.k1_gemm max_abs={:.9e} rmse={:.9e} parity_pass={parity_pass}",
                parity.max_abs, parity.rmse
            );
            let gemm_name = format!("frontier.{sig}.k1_gemm");
            let gemm = bench_one(&gemm_name, device, warmup, iterations, || {
                let y = conv1d_k1_gemm(&canonical.case)?;
                std::hint::black_box(y);
                Ok(())
            })?;
            let gemm_net = (gemm.median_us - observer.median_us).max(0.0);
            let speedup = if gemm_net > 0.0 { current_net / gemm_net } else { f64::INFINITY };
            if parity_pass && speedup >= 1.10 {
                k1_any_win = true;
                best_net = gemm_net.min(current_net);
            }
            println!(
                "frontier={sig} current_net_us={current_net:.3} k1_gemm_net_us={gemm_net:.3} speedup={speedup:.3} selected={}",
                if parity_pass && gemm_net < current_net { "k1_gemm" } else { "current" }
            );
            if parity_pass {
                best_net = best_net.min(gemm_net);
            }
        }

        let totals = families.entry(family).or_default();
        totals.instances += canonical.multiplicity;
        totals.baseline_proxy_us += current_net * canonical.multiplicity as f64;
        totals.best_proxy_us += best_net * canonical.multiplicity as f64;
    }

    println!("=== STANDARD CONV1D FRONTIER SUMMARY ===");
    println!("weighted_proxy_note=standalone_medians_times_instance_count_not_additive_runtime");
    println!("k1_candidate_gate=parity_and_at_least_1.10x_for_promotion_interest");
    let mut names = families.keys().cloned().collect::<Vec<_>>();
    names.sort();
    let mut k1_best = 0.0;
    let mut k1_base = 0.0;
    let mut k3_base = 0.0;
    for name in names {
        let t = families[&name];
        let speedup = if t.best_proxy_us > 0.0 {
            t.baseline_proxy_us / t.best_proxy_us
        } else {
            1.0
        };
        println!(
            "family={name} instances={} baseline_proxy_us={:.3} best_proxy_us={:.3} proxy_speedup={speedup:.3}",
            t.instances, t.baseline_proxy_us, t.best_proxy_us
        );
        if name == "k1" {
            k1_base += t.baseline_proxy_us;
            k1_best += t.best_proxy_us;
        } else if name.starts_with("k3_") {
            k3_base += t.baseline_proxy_us;
        }
    }
    let k1_saving = (k1_base - k1_best).max(0.0);
    let k1_speedup = if k1_best > 0.0 { k1_base / k1_best } else { 1.0 };
    println!("derived=k1_baseline_proxy_us value={k1_base:.3}");
    println!("derived=k1_best_proxy_us value={k1_best:.3}");
    println!("derived=k1_proxy_saving_us value={k1_saving:.3}");
    println!("derived=k1_proxy_speedup value={k1_speedup:.3}");
    println!("derived=k3_baseline_proxy_us value={k3_base:.3}");
    println!("k1_all_parity={k1_all_parity}");
    println!("k1_any_shape_ge_1_10x={k1_any_win}");

    let next = if !k1_all_parity {
        "fix_k1_gemm_parity"
    } else if k3_base > k1_best * 1.25 {
        "raw_cuda_k3_frontier"
    } else if k1_any_win {
        "k1_gemm_dispatch_ab"
    } else {
        "decoder_nonconv_and_k3_profile"
    };
    println!("decision_next={next}");
    println!("stop_rule=do_not_add_raw_cuda_k3_unless_decision_next_is_raw_cuda_k3_frontier");
    println!("MOTIONBRICKS_V0_3_2_STANDARD_CONV1D_FRONTIER=PASS");
    Ok(())
}


fn cudnn_algo_name(algo: CudnnFwdAlgo) -> &'static str {
    match algo {
        CudnnFwdAlgo::ImplicitGemm => "implicit_gemm",
        CudnnFwdAlgo::ImplicitPrecompGemm => "implicit_precomp_gemm",
        CudnnFwdAlgo::Gemm => "gemm",
        CudnnFwdAlgo::Direct => "direct",
        CudnnFwdAlgo::Fft => "fft",
        CudnnFwdAlgo::FftTiling => "fft_tiling",
        CudnnFwdAlgo::Winograd => "winograd",
        CudnnFwdAlgo::WinogradNonFused => "winograd_non_fused",
        CudnnFwdAlgo::Count => "count",
    }
}

fn conv_with_cudnn_algo(case: &Conv1dFrontierCase, algo: CudnnFwdAlgo) -> Conv1d {
    let base = *case.conv.config();
    Conv1d::new(
        case.conv.weight().clone(),
        case.conv.bias().cloned(),
        Conv1dConfig {
            cudnn_fwd_algo: Some(algo),
            ..base
        },
    )
}

fn one_line_error(err: &anyhow::Error) -> String {
    format!("{err:#}").replace('\n', " | ")
}

fn conv_r1_sweep(
    model: &MotionBricksRuntimeReference,
    device: &Device,
    path: &Path,
    warmup: usize,
    iterations: usize,
) -> Result<()> {
    let map = safetensors::load(path, device)?;
    let input = load_inputs(&map)?;
    let mut all = model.root.benchmark_conv_frontier_cases(&input.root)?;
    all.extend(model.decoder.benchmark_conv_frontier_cases(&input.vqvae)?);
    let cases = canonicalize_conv_cases(all)?;
    device.synchronize()?;

    println!("=== HARICOT MOTIONBRICKS V0.3.2-R1 K1 GEMM + K3-L32 CUDNN SWEEP ===");
    println!("device={device:?}");
    println!("profile=fixed16-no-text-all-root-constraints");
    println!("k1_r1_evidence=explicit_gemm_winner_not_default_r2_policy");
    println!("k1_evidence_sm61_speedup=3.267x");
    println!("k3_target=b1_cin512_cout512_k3_l32_d1_p1_s1");
    println!("decision_policy=hardware_relative_not_absolute_microseconds");
    println!("portable_key=device_or_sm+cudnn_version+dtype+shape");
    println!("warmup={warmup}");
    println!("iterations={iterations}");

    let observer = bench_one("r1.observer.sync_only", device, warmup, iterations, || Ok(()))?;

    let find_case = |len: usize| -> Result<&CanonicalConvCase> {
        cases.iter().find(|c| {
            let Ok((_, cin, l)) = c.case.input.dims3() else { return false; };
            let Ok((cout, _, k)) = c.case.conv.weight().dims3() else { return false; };
            let cfg = c.case.conv.config();
            cin == 512 && cout == 512 && k == 3 && l == len
                && cfg.dilation == 1 && cfg.padding == 1 && cfg.stride == 1
        }).with_context(|| format!("missing canonical K3 d1 512x512 L={len} case"))
    };

    let c16 = find_case(16)?;
    let c32 = find_case(32)?;
    let c64 = find_case(64)?;

    let bench_current = |tag: &str, c: &CanonicalConvCase| -> Result<BenchStats> {
        bench_one(tag, device, warmup, iterations, || {
            let y = c.case.conv.forward(&c.case.input)?;
            std::hint::black_box(y);
            Ok(())
        })
    };

    let b16 = bench_current("r1.peer_l16.current", c16)?;
    let b32 = bench_current("r1.target_l32.current", c32)?;
    let b64 = bench_current("r1.peer_l64.current", c64)?;
    let n16 = (b16.median_us - observer.median_us).max(0.0);
    let n32 = (b32.median_us - observer.median_us).max(0.0);
    let n64 = (b64.median_us - observer.median_us).max(0.0);

    // FLOP count is linear in sequence length for a fixed Conv1d shape.  Use
    // both neighboring lengths to form a conservative device-local expectation
    // for L32 rather than hard-coding a Pascal-specific microsecond threshold.
    let peer_expected_l32 = (n16 * 2.0).max(n64 * 0.5);
    let cliff_ratio = if peer_expected_l32 > 0.0 { n32 / peer_expected_l32 } else { f64::INFINITY };
    println!("derived=peer_expected_l32_net_us value={peer_expected_l32:.3}");
    println!("derived=current_l32_cliff_ratio value={cliff_ratio:.3}");

    let baseline_out = c32.case.conv.forward(&c32.case.input)?;
    device.synchronize()?;
    let algos = [
        CudnnFwdAlgo::ImplicitGemm,
        CudnnFwdAlgo::ImplicitPrecompGemm,
        CudnnFwdAlgo::Gemm,
        CudnnFwdAlgo::Direct,
        CudnnFwdAlgo::Fft,
        CudnnFwdAlgo::FftTiling,
        CudnnFwdAlgo::Winograd,
        CudnnFwdAlgo::WinogradNonFused,
    ];

    let mut best_name = "current".to_string();
    let mut best_net = n32;
    let mut supported = 0usize;
    let mut parity_supported = 0usize;
    for algo in algos {
        let name = cudnn_algo_name(algo);
        let tuned = conv_with_cudnn_algo(&c32.case, algo);
        let once = match tuned.forward(&c32.case.input) {
            Ok(y) => y,
            Err(e) => {
                let e = anyhow::Error::new(e);
                println!("candidate={name} status=unsupported error={}", one_line_error(&e));
                continue;
            }
        };
        supported += 1;
        device.synchronize()?;
        let parity = tensor_stats(&once, &baseline_out)?;
        let parity_pass = parity.max_abs <= 2e-4 && parity.rmse <= 5e-5;
        if parity_pass { parity_supported += 1; }
        println!(
            "candidate={name} status=supported max_abs={:.9e} rmse={:.9e} parity_pass={parity_pass}",
            parity.max_abs, parity.rmse
        );
        if !parity_pass { continue; }
        let tag = format!("r1.target_l32.cudnn_{name}");
        let stats = match bench_one(&tag, device, warmup, iterations, || {
            let y = tuned.forward(&c32.case.input)?;
            std::hint::black_box(y);
            Ok(())
        }) {
            Ok(v) => v,
            Err(e) => {
                println!("candidate={name} benchmark_status=failed error={}", one_line_error(&e));
                continue;
            }
        };
        let net = (stats.median_us - observer.median_us).max(0.0);
        let speedup = if net > 0.0 { n32 / net } else { f64::INFINITY };
        let to_peer = if peer_expected_l32 > 0.0 { net / peer_expected_l32 } else { f64::INFINITY };
        println!("frontier={name} net_us={net:.3} speedup_vs_current={speedup:.3} ratio_to_peer_expected={to_peer:.3}");
        if net < best_net {
            best_net = net;
            best_name = name.to_string();
        }
    }

    let best_speedup = if best_net > 0.0 { n32 / best_net } else { f64::INFINITY };
    let best_to_peer = if peer_expected_l32 > 0.0 { best_net / peer_expected_l32 } else { f64::INFINITY };
    println!("=== K3-L32 CUDNN SWEEP SUMMARY ===");
    println!("supported_algorithms={supported}");
    println!("parity_supported_algorithms={parity_supported}");
    println!("current_net_us={n32:.3}");
    println!("best_candidate={best_name}");
    println!("best_net_us={best_net:.3}");
    println!("best_speedup_vs_current={best_speedup:.3}");
    println!("best_ratio_to_peer_expected={best_to_peer:.3}");
    println!("sm61_500us_reference=diagnostic_only_not_a_portable_gate");

    let decision = if cliff_ratio < 1.50 {
        "no_k3_specialization_needed"
    } else if best_name != "current" && best_to_peer <= 1.35 && best_speedup >= 1.25 {
        "cudnn_k3_l32_dispatch_ab"
    } else if best_name != "current" && best_speedup >= 1.50 {
        "cudnn_k3_l32_dispatch_ab_then_reassess"
    } else {
        "raw_cuda_k3_l32_only"
    };
    println!("decision_next={decision}");
    println!("raw_cuda_scope_if_needed=only_b1_c512_c512_k3_l32_d1_p1_s1");
    println!("dispatch_portability=retune_per_gpu_arch_and_cudnn_version");
    println!("MOTIONBRICKS_V0_3_2_R1_CUDNN_SWEEP=PASS");
    Ok(())
}


#[derive(Clone, Copy, Debug)]
struct AbaResult {
    baseline_midpoint_us: f64,
    candidate_us: f64,
    speedup: f64,
    control_drift_pct: f64,
    candidate_beats_both: bool,
}

fn bench_aba<FBase, FCandidate>(
    name: &str,
    device: &Device,
    warmup: usize,
    iterations: usize,
    mut baseline: FBase,
    mut candidate: FCandidate,
) -> Result<AbaResult>
where
    FBase: FnMut() -> Result<()>,
    FCandidate: FnMut() -> Result<()>,
{
    let a1 = bench_one(
        &format!("aba.{name}.A1_baseline"),
        device,
        warmup,
        iterations,
        || baseline(),
    )?;
    let b = bench_one(
        &format!("aba.{name}.B_candidate"),
        device,
        warmup,
        iterations,
        || candidate(),
    )?;
    let a2 = bench_one(
        &format!("aba.{name}.A2_baseline"),
        device,
        warmup,
        iterations,
        || baseline(),
    )?;

    let midpoint = (a1.median_us + a2.median_us) * 0.5;
    let speedup = if b.median_us > 0.0 {
        midpoint / b.median_us
    } else {
        f64::INFINITY
    };
    let control_drift_pct = if midpoint > 0.0 {
        100.0 * (a2.median_us - a1.median_us).abs() / midpoint
    } else {
        0.0
    };
    let candidate_beats_both = b.median_us < a1.median_us && b.median_us < a2.median_us;
    println!(
        "aba_summary={name} A1_median_us={:.3} B_median_us={:.3} A2_median_us={:.3} baseline_midpoint_us={midpoint:.3} speedup={speedup:.3} control_drift_pct={control_drift_pct:.2} candidate_beats_both={candidate_beats_both}",
        a1.median_us, b.median_us, a2.median_us
    );
    Ok(AbaResult {
        baseline_midpoint_us: midpoint,
        candidate_us: b.median_us,
        speedup,
        control_drift_pct,
        candidate_beats_both,
    })
}

fn pair_parity(name: &str, baseline: &Tensor, candidate: &Tensor) -> Result<bool> {
    let s = tensor_stats(candidate, baseline)?;
    let pass = s.max_abs <= 5e-4 && s.rmse <= 1e-4;
    println!(
        "dispatch_parity={name} max_abs={:.9e} mean_abs={:.9e} rmse={:.9e} atol=5.000000000e-4 rmse_tol=1.000000000e-4 pass={pass}",
        s.max_abs, s.mean_abs, s.rmse
    );
    Ok(pass)
}

fn conv_r2_aba(
    model: &MotionBricksRuntimeReference,
    device: &Device,
    path: &Path,
    warmup: usize,
    iterations: usize,
    candidate: MotionBricksConv1dPolicy,
    asd_selected: bool,
) -> Result<()> {
    if candidate != MotionBricksConv1dPolicy::Sm61Validated {
        bail!("A/B/A candidate must be the explicitly selected complete bundle");
    }
    let map = safetensors::load(path, device)?;
    let input = load_inputs(&map)?;
    let baseline = MotionBricksConv1dPolicy::Baseline;

    println!("=== HARICOT MOTIONBRICKS V0.3.2-R2 SM61 CONV1D DISPATCH A/B/A ===");
    println!("device={device:?}");
    println!("profile=fixed16-no-text-all-root-constraints");
    println!("dispatch_scope=motionbricks_local_only");
    println!("candle_core_modified=false");
    println!("grouped_conv_transpose_branch_dependency=false");
    println!("validated_dispatch_sm=61");
    println!("validated_cudnn_key={MOTIONBRICKS_PRODUCTION_CUDNN_SM61}");
    println!("dtype=f32");
    println!("baseline_policy={}", baseline.as_str());
    println!("candidate_policy={}", candidate.as_str());
    println!("aba_candidate_source={}", if asd_selected { "flow_asd_atomic_bundle" } else { "legacy_motionbricks_local" });
    println!("candidate_k1=b1_c512_c512_k1_l16_or_l32_d1_p0_s1:explicit_gemm");
    println!("candidate_k3=b1_c512_c512_k3_l32_d1_p1_s1:cudnn_direct");
    println!("performance_gate=candidate_median_must_beat_both_A_controls");
    println!("control_drift=diagnostic_only");
    println!("warmup={warmup}");
    println!("iterations={iterations}");

    let root_state = model.root.benchmark_prepare_fixed_16(&input.root)?;
    let root_hidden = model.root.benchmark_root_token_transformer(&root_state)?;
    let vq_state = model.decoder.benchmark_codebook_lookup(&input.vqvae)?;

    let mut dispatch_cases = model.root.benchmark_conv_frontier_cases(&input.root)?;
    dispatch_cases.extend(model.decoder.benchmark_conv_frontier_cases(&input.vqvae)?);
    let mut k1_hits = 0usize;
    let mut k3_l32_hits = 0usize;
    for case in &dispatch_cases {
        let (b, cin, len) = case.input.dims3()?;
        let (cout, wcin, kernel) = case.conv.weight().dims3()?;
        let cfg = case.conv.config();
        if b == 1
            && cin == 512
            && wcin == 512
            && cout == 512
            && kernel == 1
            && (len == 16 || len == 32)
            && cfg.groups == 1
            && cfg.padding == 0
            && cfg.stride == 1
            && cfg.dilation == 1
        {
            k1_hits += 1;
        }
        if b == 1
            && cin == 512
            && wcin == 512
            && cout == 512
            && kernel == 3
            && len == 32
            && cfg.groups == 1
            && cfg.padding == 1
            && cfg.stride == 1
            && cfg.dilation == 1
        {
            k3_l32_hits += 1;
        }
    }
    println!("dispatch_inventory_conv_instances={}", dispatch_cases.len());
    println!("dispatch_inventory_k1_hits={k1_hits}");
    println!("dispatch_inventory_k3_l32_direct_hits={k3_l32_hits}");
    if dispatch_cases.len() != 42 || k1_hits != 16 || k3_l32_hits != 4 {
        bail!(
            "unexpected decoder dispatch inventory: total={} k1={} k3_l32={} (expected 42/16/4)",
            dispatch_cases.len(),
            k1_hits,
            k3_l32_hits
        )
    }

    let root_dec_a = model
        .root
        .benchmark_conv_decoder_with_policy(&root_hidden, &root_state, baseline)?;
    let root_dec_b = model
        .root
        .benchmark_conv_decoder_with_policy(&root_hidden, &root_state, candidate)?;
    let vq_dec_a = model
        .decoder
        .benchmark_decoder_from_codebook_with_policy(&vq_state, &input.vqvae, baseline)?;
    let vq_dec_b = model
        .decoder
        .benchmark_decoder_from_codebook_with_policy(&vq_state, &input.vqvae, candidate)?;
    let root_a = model.root.forward_fixed_16_with_policy(&input.root, baseline)?;
    let root_b = model.root.forward_fixed_16_with_policy(&input.root, candidate)?;
    let vq_a = model.decoder.forward_with_policy(&input.vqvae, baseline)?;
    let vq_b = model.decoder.forward_with_policy(&input.vqvae, candidate)?;
    device.synchronize()?;

    println!("=== DISPATCH NUMERICAL PARITY ===");
    let mut parity_pass = true;
    parity_pass &= pair_parity("root.decoder_only", &root_dec_a, &root_dec_b)?;
    parity_pass &= pair_parity("vq.decoder_only", &vq_dec_a, &vq_dec_b)?;
    parity_pass &= pair_parity(
        "root.full.pred_global_root_values",
        &root_a.pred_global_root_values,
        &root_b.pred_global_root_values,
    )?;
    parity_pass &= pair_parity("vq.full.recon_state", &vq_a, &vq_b)?;
    parity_pass &= report(
        "candidate.root.num_token_logits_vs_torch",
        &root_b.num_token_logits,
        &get(&map, "output.root.num_token_logits")?,
        5e-4,
        1e-4,
    )?;
    parity_pass &= report(
        "candidate.root.pred_global_root_values_vs_torch",
        &root_b.pred_global_root_values,
        &get(&map, "output.root.pred_global_root_values")?,
        5e-4,
        1e-4,
    )?;
    parity_pass &= report(
        "candidate.vqvae.recon_state_vs_torch",
        &vq_b,
        &get(&map, "output.vqvae.recon_state")?,
        5e-4,
        1e-4,
    )?;
    println!(
        "MOTIONBRICKS_V0_3_2_R2_DISPATCH_NUMERICAL_GATE={}",
        if parity_pass { "PASS" } else { "FAIL" }
    );
    if !parity_pass {
        bail!("v0.3.2-r2 dispatch numerical parity failed before A/B/A")
    }

    println!("=== DECODER A/B/A ===");
    let root_decoder = bench_aba(
        "root.decoder_only",
        device,
        warmup,
        iterations,
        || {
            let y = model
                .root
                .benchmark_conv_decoder_with_policy(&root_hidden, &root_state, baseline)?;
            std::hint::black_box(y);
            Ok(())
        },
        || {
            let y = model
                .root
                .benchmark_conv_decoder_with_policy(&root_hidden, &root_state, candidate)?;
            std::hint::black_box(y);
            Ok(())
        },
    )?;
    let vq_decoder = bench_aba(
        "vq.decoder_only",
        device,
        warmup,
        iterations,
        || {
            let y = model.decoder.benchmark_decoder_from_codebook_with_policy(
                &vq_state,
                &input.vqvae,
                baseline,
            )?;
            std::hint::black_box(y);
            Ok(())
        },
        || {
            let y = model.decoder.benchmark_decoder_from_codebook_with_policy(
                &vq_state,
                &input.vqvae,
                candidate,
            )?;
            std::hint::black_box(y);
            Ok(())
        },
    )?;
    let root_full = bench_aba(
        "root.full",
        device,
        warmup,
        iterations,
        || {
            let y = model.root.forward_fixed_16_with_policy(&input.root, baseline)?;
            std::hint::black_box(y);
            Ok(())
        },
        || {
            let y = model.root.forward_fixed_16_with_policy(&input.root, candidate)?;
            std::hint::black_box(y);
            Ok(())
        },
    )?;
    let vq_full = bench_aba(
        "vq.full",
        device,
        warmup,
        iterations,
        || {
            let y = model.decoder.forward_with_policy(&input.vqvae, baseline)?;
            std::hint::black_box(y);
            Ok(())
        },
        || {
            let y = model.decoder.forward_with_policy(&input.vqvae, candidate)?;
            std::hint::black_box(y);
            Ok(())
        },
    )?;
    let runtime_full = bench_aba(
        "runtime.full",
        device,
        warmup,
        iterations,
        || {
            let r = model.root.forward_fixed_16_with_policy(&input.root, baseline)?;
            let p = model.pose.forward_fixed_16(&input.pose)?;
            let v = model.decoder.forward_with_policy(&input.vqvae, baseline)?;
            std::hint::black_box((r, p, v));
            Ok(())
        },
        || {
            let r = model.root.forward_fixed_16_with_policy(&input.root, candidate)?;
            let p = model.pose.forward_fixed_16(&input.pose)?;
            let v = model.decoder.forward_with_policy(&input.vqvae, candidate)?;
            std::hint::black_box((r, p, v));
            Ok(())
        },
    )?;

    println!("=== V0.3.2-R2 A/B/A SUMMARY ===");
    for (name, r) in [
        ("root.decoder_only", root_decoder),
        ("vq.decoder_only", vq_decoder),
        ("root.full", root_full),
        ("vq.full", vq_full),
        ("runtime.full", runtime_full),
    ] {
        println!(
            "result={name} baseline_midpoint_us={:.3} candidate_us={:.3} speedup={:.3} control_drift_pct={:.2} candidate_beats_both={}",
            r.baseline_midpoint_us,
            r.candidate_us,
            r.speedup,
            r.control_drift_pct,
            r.candidate_beats_both,
        );
    }

    let decoder_perf_pass = root_decoder.candidate_beats_both && vq_decoder.candidate_beats_both;
    let runtime_perf_pass = runtime_full.candidate_beats_both;
    let promotion_pass = parity_pass && decoder_perf_pass && runtime_perf_pass;
    println!(
        "MOTIONBRICKS_V0_3_2_R2_DECODER_ABA_GATE={}",
        if decoder_perf_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "MOTIONBRICKS_V0_3_2_R2_RUNTIME_ABA_GATE={}",
        if runtime_perf_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "decision_next={}",
        if promotion_pass {
            "promote_motionbricks_local_sm61_dispatch"
        } else {
            "keep_baseline_and_inspect_failed_component"
        }
    );
    println!(
        "MOTIONBRICKS_V0_3_2_R2_PROMOTION_GATE={}",
        if promotion_pass { "PASS" } else { "FAIL" }
    );
    if !promotion_pass {
        bail!("v0.3.2-r2 A/B/A promotion gate failed")
    }
    Ok(())
}

fn print_share(name: &str, median_us: f64, runtime_median_us: f64) {
    let pct = if runtime_median_us > 0.0 {
        100.0 * median_us / runtime_median_us
    } else {
        0.0
    };
    println!("component={name} median_share_pct={pct:.2}");
}

fn calc_stats(samples: &[Duration]) -> BenchStats {
    let mut us = samples
        .iter()
        .map(|d| d.as_secs_f64() * 1e6)
        .collect::<Vec<_>>();
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = us.len();
    let min_us = us[0];
    let median_us = if n % 2 == 0 {
        (us[n / 2 - 1] + us[n / 2]) * 0.5
    } else {
        us[n / 2]
    };
    let mean_us = us.iter().sum::<f64>() / n as f64;
    let p95_us = us[((n as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1)];
    BenchStats {
        min_us,
        median_us,
        mean_us,
        p95_us,
    }
}
