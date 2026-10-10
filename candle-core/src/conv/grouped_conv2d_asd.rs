use crate::backend::BackendStorage;
use crate::builder_arg as barg;
use crate::conv::ParamsConv2D;
use crate::cuda_backend::{kernels, CudaStorage, CudaStorageSlice as S, WrapErr};
use crate::{Layout, Result};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

fn trace_enabled() -> bool {
    env_truthy("CANDLE_ASD_EXACT_TRACE")
}

fn exact_disabled() -> bool {
    env_truthy("CANDLE_ASD_EXACT_DISABLE")
}

pub(super) fn exact_dispatch_enabled() -> bool {
    !exact_disabled()
}

#[cfg(feature = "cudnn")]
fn real_dispatch_validation_enabled() -> bool {
    env_truthy("CANDLE_ASD_REAL_DISPATCH_VALIDATION")
}

#[cfg(feature = "cudnn")]
pub(super) fn require_cudnn_submission() -> bool {
    real_dispatch_validation_enabled() && env_truthy("CANDLE_ASD_REAL_DISPATCH_REQUIRE_CUDNN")
}

fn exact_call(
    input: &CudaStorage,
    input_l: &Layout,
    kernel_l: &Layout,
    p: &ParamsConv2D,
) -> candle_kernels::asd_exact::ExactOperationCall {
    let w = kernel_l.dims();
    candle_kernels::asd_exact::ExactOperationCall {
        op: candle_kernels::asd_exact::ExactOperation::Conv2d,
        dim: 2,
        batch: p.b_size,
        c_in: p.c_in,
        c_out: p.c_out,
        spatial0: p.i_h,
        spatial1: p.i_w,
        weight_rank: w.len(),
        weight0: w.first().copied().unwrap_or(0),
        weight1: w.get(1).copied().unwrap_or(0),
        weight2: w.get(2).copied().unwrap_or(0),
        weight3: w.get(3).copied().unwrap_or(0),
        groups: p.groups,
        kernel: p.k_h,
        stride: p.stride,
        padding: p.padding,
        output_padding: 0,
        dilation: p.dilation,
        dtype: input.dtype().as_str(),
        input_contiguous: input_l.is_contiguous(),
        input_start_offset: input_l.start_offset(),
        weight_contiguous: kernel_l.is_contiguous(),
        weight_start_offset: kernel_l.start_offset(),
    }
}

fn actual_gpu_uuid(context: &std::sync::Arc<cudarc::driver::CudaContext>) -> Result<String> {
    let uuid = context.uuid().map_err(|err| {
        crate::Error::msg(format!("ASD: unable to read CUDA device UUID: {err:?}"))
    })?;
    let hex = uuid
        .bytes
        .iter()
        .map(|byte| format!("{:02x}", *byte as u8))
        .collect::<String>();
    if hex.len() != 32 {
        crate::bail!("ASD: unexpected CUDA UUID length {}", hex.len())
    }
    Ok(format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    ))
}

fn launch_f32(
    input: &CudaSlice<f32>,
    kernel: &CudaSlice<f32>,
    p: &ParamsConv2D,
    dev: &crate::cuda_backend::CudaDevice,
) -> Result<CudaSlice<f32>> {
    if p.b_size != 1
        || p.c_in != p.c_out
        || p.groups != p.c_in
        || p.k_h != 5
        || p.k_w != 5
        || p.stride != 1
        || p.padding != 2
        || p.dilation != 1
        || p.out_h() != p.i_h
        || p.out_w() != p.i_w
    {
        crate::bail!("exact ASD DW5x5 launch received an out-of-domain Conv2D")
    }

    let dst_el = p.c_out * p.out_h() * p.out_w();
    let out = unsafe { dev.alloc::<f32>(dst_el)? };
    let func = dev.get_or_load_func("asd_dw5x5_conv2d_f32", &kernels::ASD_DW5X5)?;
    let cfg = LaunchConfig {
        grid_dim: (
            p.i_w.div_ceil(16) as u32,
            p.i_h.div_ceil(8) as u32,
            p.c_in as u32,
        ),
        block_dim: (16, 8, 1),
        shared_mem_bytes: 0,
    };
    let channels = p.c_in as u64;
    let height = p.i_h as u64;
    let width = p.i_w as u64;
    let mut builder = func.builder();
    barg!(builder, channels, height, width);
    builder.arg(input);
    builder.arg(kernel);
    builder.arg(&out);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(out)
}

pub(super) fn try_launch_exact(
    input: &CudaStorage,
    input_l: &Layout,
    kernel: &CudaStorage,
    kernel_l: &Layout,
    p: &ParamsConv2D,
) -> Result<Option<CudaStorage>> {
    if input.device.id() != kernel.device.id() {
        crate::bail!("exact ASD DW5x5 requires input and kernel on one CUDA device")
    }

    // Authenticate the real CUDA context before passing the device identity to
    // the single V2 lookup. Environment UUIDs never authorize runtime dispatch.
    let stream = input.device.cuda_stream();
    let context = stream.context();
    let (major, minor) = context.compute_capability().map_err(|err| {
        crate::Error::msg(format!(
            "ASD V2: unable to read CUDA compute capability: {err:?}"
        ))
    })?;
    if major * 10 + minor != candle_kernels::CUDA_BUILD_COMPUTE_CAP as i32 {
        crate::bail!("ASD V2 DW5x5 runtime GPU SM does not match build target")
    }

    let actual_uuid = actual_gpu_uuid(context)?;

    let call = exact_call(input, input_l, kernel_l, p);
    let Some(matched) = crate::asd_runtime_extensions::lookup_exact(call, Some(&actual_uuid))?
    else {
        return Ok(None);
    };

    if &*matched.state != "promoted"
        || matched.execution_provider
            != crate::asd_runtime_extensions::RuntimeExecutionProvider::RawCuda
        || &*matched.implementation_id != "candle.depthwise-conv2d-5x5.raw.v1"
        || &*matched.evidence_sha256
            != "84f3dc50433e225b1f63c92a08355b7b04f0afeec49694e5e8d5e040cf092a9a"
        || matched.min_integrated_speedup_x != Some(1.10)
    {
        crate::bail!("unexpected runtime Exact Profile match returned to DW5x5 consumer")
    }

    if trace_enabled() {
        eprintln!(
            "[candle grouped-conv2d] requested=auto sm={} selected=raw reason=exact_asd asd_profile={} asd_policy={} asd_decision={} asd_state={} asd_impl={} evidence={} min_integrated_speedup_x={:.8}",
            candle_kernels::CUDA_BUILD_COMPUTE_CAP,
            matched.profile_id,
            matched.profile_id,
            matched.decision_id,
            matched.state,
            matched.implementation_id,
            matched.evidence_sha256,
            matched.min_integrated_speedup_x.unwrap_or_default(),
        );
    }

    let dev = input.device.clone();
    let slice = match (&input.slice, &kernel.slice) {
        (S::F32(x), S::F32(k)) => S::F32(launch_f32(x, k, p, &dev)?),
        _ => crate::bail!("exact ASD DW5x5 raw v1 requires matching f32 input and kernel"),
    };
    if trace_enabled() {
        eprintln!(
            "[candle grouped-conv2d] submitted_backend=raw launch_submission=success asd_profile={} asd_policy={} asd_decision={} asd_state={} asd_impl={} evidence={} min_integrated_speedup_x={:.8}",
            matched.profile_id,
            matched.profile_id,
            matched.decision_id,
            matched.state,
            matched.implementation_id,
            matched.evidence_sha256,
            matched.min_integrated_speedup_x.unwrap_or_default(),
        );
    }
    Ok(Some(CudaStorage { slice, device: dev }))
}

pub(super) fn trace_current_selection() {
    if exact_dispatch_enabled() && trace_enabled() {
        let reason = if exact_disabled() {
            "asd_disabled"
        } else {
            "exact_miss"
        };
        eprintln!(
            "[candle grouped-conv2d] requested=auto selected=current reason={} asd_profile=none asd_policy=none asd_decision=none asd_state=none asd_impl=none",
            reason
        );
    }
}

pub(super) fn trace_current_submission(backend: &str) {
    if exact_dispatch_enabled() && trace_enabled() {
        eprintln!(
            "[candle grouped-conv2d] submitted_backend={} launch_submission=success asd_profile=none asd_policy=none asd_decision=none asd_state=none asd_impl=none",
            backend
        );
    }
}
