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
    if !candle_kernels::asd_exact_conv2d::POLICY_EMBEDDED {
        return false;
    }
    if candle_kernels::asd_exact_conv2d::VALIDATION_BUILD {
        return env_truthy("CANDLE_ASD_REAL_DISPATCH_VALIDATION");
    }
    true
}

pub(super) fn real_dispatch_validation_enabled() -> bool {
    candle_kernels::asd_exact_conv2d::VALIDATION_BUILD
        && env_truthy("CANDLE_ASD_REAL_DISPATCH_VALIDATION")
}

pub(super) fn require_cudnn_submission() -> bool {
    real_dispatch_validation_enabled() && env_truthy("CANDLE_ASD_REAL_DISPATCH_REQUIRE_CUDNN")
}

fn exact_call(
    input: &CudaStorage,
    input_l: &Layout,
    kernel_l: &Layout,
    p: &ParamsConv2D,
) -> candle_kernels::asd_exact_conv2d::ExactConv2dCall {
    let w = kernel_l.dims();
    candle_kernels::asd_exact_conv2d::ExactConv2dCall {
        batch: p.b_size,
        c_in: p.c_in,
        c_out: p.c_out,
        spatial0: p.i_h,
        spatial1: p.i_w,
        weight0: w.first().copied().unwrap_or(0),
        weight1: w.get(1).copied().unwrap_or(0),
        weight2: w.get(2).copied().unwrap_or(0),
        weight3: w.get(3).copied().unwrap_or(0),
        groups: p.groups,
        kernel: p.k_h,
        stride: p.stride,
        padding: p.padding,
        dilation: p.dilation,
        dtype: input.dtype().as_str(),
        input_contiguous: input_l.is_contiguous(),
        input_start_offset: input_l.start_offset(),
        weight_contiguous: kernel_l.is_contiguous(),
        weight_start_offset: kernel_l.start_offset(),
    }
}

// NVIDIA's GPU- UUID contains 16 hex octets, optionally separated by dashes.
// This parser intentionally does not consult environment variables or ordinals.
fn parse_policy_gpu_uuid(value: &str) -> Option<[u8; 16]> {
    let hex = value.strip_prefix("GPU-").unwrap_or(value);
    let compact: String = hex.chars().filter(|c| *c != '-').collect();
    if compact.len() != 32 || !compact.is_ascii() {
        return None;
    }
    let mut out = [0u8; 16];
    for (index, slot) in out.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&compact[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod device_scope_tests {
    use super::parse_policy_gpu_uuid;

    #[test]
    fn uuid_accepts_gpu_prefix_and_dashes() {
        let expected: Vec<u8> = (0..16).collect();
        assert_eq!(parse_policy_gpu_uuid("GPU-00010203-0405-0607-0809-0a0b0c0d0e0f").unwrap().to_vec(), expected);
    }
    #[test]
    fn uuid_rejects_missing_or_incomplete_identity() {
        assert!(parse_policy_gpu_uuid("").is_none());
        assert!(parse_policy_gpu_uuid("GPU-00010203").is_none());
        assert!(parse_policy_gpu_uuid("GPU-00010203-0405-0607-0809-0a0b0c0d0e0z").is_none());
    }
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
    let call = exact_call(input, input_l, kernel_l, p);
    let Some(matched) = candle_kernels::asd_exact_conv2d::lookup(call) else {
        return Ok(None);
    };
    if !candle_kernels::asd_exact_conv2d::VALIDATION_BUILD && matched.state != "promoted" {
        crate::bail!(
            "exact Conv2D ASD decision {} has state={} outside validation build",
            matched.decision_id,
            matched.state
        )
    }
    if matched.selected_backend != "raw_cuda" {
        crate::bail!(
            "unsupported exact Conv2D ASD selected backend {}",
            matched.selected_backend
        )
    }
    if input.device.id() != kernel.device.id() {
        crate::bail!("exact ASD DW5x5 requires input and kernel on one CUDA device")
    }
    // A compiled policy and a declared UUID do not authenticate the CUDA device
    // on which this process is currently executing. Fail closed before launch.
    let stream = input.device.cuda_stream();
    let context = stream.context();
    let (major, minor) = context.compute_capability()
        .map_err(|err| crate::Error::msg(format!(
        "ASD V2: unable to read CUDA compute capability: {err:?}"
    )))?;
    if major * 10 + minor != candle_kernels::CUDA_BUILD_COMPUTE_CAP as i32 {
        crate::bail!("ASD V2 DW5x5 runtime GPU SM does not match build target")
    }
    if let Some(expected) = candle_kernels::asd_exact_conv2d::TARGET_GPU_UUID {
        let expected_bytes = parse_policy_gpu_uuid(expected)
            .ok_or_else(|| crate::Error::Msg("invalid embedded ASD V2 device UUID".into()))?;
	let actual_bytes = context.uuid()
	    .map_err(|err| crate::Error::msg(format!(
		"ASD V2: unable to read CUDA device UUID: {err:?}"
	    )))?
	    .bytes;
        if !actual_bytes.iter().zip(expected_bytes.iter())
            .all(|(actual, expected)| *actual as u8 == *expected)
        {
            crate::bail!("ASD V2 device-scoped GPU UUID mismatch at runtime")
        }
    }
    if trace_enabled() {
        eprintln!(
            "[candle grouped-conv2d] requested=auto sm={} selected=raw reason=exact_asd asd_policy={} asd_decision={} asd_state={}",
            candle_kernels::CUDA_BUILD_COMPUTE_CAP,
            matched.policy_id,
            matched.decision_id,
            matched.state,
        );
    }

    // Exact V2 lookup already requires contiguous_zero_offset, so using the
    // original slices preserves the launch ABI and avoids converting them to
    // CudaView values.
    let dev = input.device.clone();
    let slice = match (&input.slice, &kernel.slice) {
        (S::F32(x), S::F32(k)) => S::F32(launch_f32(x, k, p, &dev)?),
        _ => crate::bail!("exact ASD DW5x5 raw v1 requires matching f32 input and kernel"),
    };
    if trace_enabled() {
        eprintln!(
            "[candle grouped-conv2d] submitted_backend=raw launch_submission=success asd_policy={} asd_decision={} asd_state={}",
            matched.policy_id,
            matched.decision_id,
            matched.state,
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
            "[candle grouped-conv2d] requested=auto selected=current reason={} asd_policy=none asd_decision=none asd_state=none",
            reason
        );
    }
}

pub(super) fn trace_current_submission(backend: &str) {
    if exact_dispatch_enabled() && trace_enabled() {
        eprintln!(
            "[candle grouped-conv2d] submitted_backend={} launch_submission=success asd_policy=none asd_decision=none asd_state=none",
            backend
        );
    }
}
