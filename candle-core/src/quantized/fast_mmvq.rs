//! CUDA fast path for GGUF matmul with BF16/F32 activations.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::cuda::{QCudaStorage, MATRIX_ROW_PADDING};
use super::GgmlDType;
use crate::cuda_backend::DeviceId;
use crate::{backend::BackendStorage, CudaDevice, CudaStorage, DType, Result, Shape};

use cudarc::driver::{CudaSlice, DevicePtr};

const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_TYPE_SIZE: usize = 36; // 2 halves (4 bytes) + QK8_1 int8 = 4 + 32 = 36

#[inline]
fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

/// Quant types supported by the fast MMVQ kernels.
fn supports(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    )
}

const MMVQ_MAX_BATCH: usize = 8;

// ---------------------------------------------------------------------------
// Per-device Q8_1 scratch workspace (grows-only, reused across calls).
// ---------------------------------------------------------------------------

struct WorkspaceSlot {
    slice: CudaSlice<u8>,
    cap: usize,
}

static WORKSPACE: OnceLock<Mutex<HashMap<DeviceId, WorkspaceSlot>>> = OnceLock::new();

/// Returns a device pointer to the scratch workspace, growing it if needed.
/// The returned `MutexGuard` must be held alive until the kernels using
/// this pointer have been launched (all launches are on the device's
/// default stream, so they are serialised).
fn workspace_ensure(
    dev: &CudaDevice,
    bytes: usize,
) -> Result<(
    u64,
    std::sync::MutexGuard<'static, HashMap<DeviceId, WorkspaceSlot>>,
)> {
    let map = WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let device_key = dev.id();
    let mut guard = map.lock().unwrap();
    let slot = match guard.entry(device_key) {
        std::collections::hash_map::Entry::Occupied(entry) => {
            let slot = entry.into_mut();
            if slot.cap < bytes {
                slot.slice = unsafe { dev.alloc::<u8>(bytes)? };
                slot.cap = bytes;
            }
            slot
        }
        std::collections::hash_map::Entry::Vacant(entry) => {
            let slice = unsafe { dev.alloc::<u8>(bytes)? };
            entry.insert(WorkspaceSlot { slice, cap: bytes })
        }
    };
    let ptr = slot.slice.device_ptr(slot.slice.stream()).0;
    Ok((ptr, guard))
}

// ---------------------------------------------------------------------------
// Launcher dispatch by weight dtype and output dtype.
// ---------------------------------------------------------------------------

type PlainLauncher = unsafe extern "C" fn(
    vx: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    stream: *mut std::ffi::c_void,
);

fn plain_launcher_bf16(dtype: GgmlDType) -> Option<PlainLauncher> {
    use candle_kernels::ffi;
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_bf16_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_bf16_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_bf16_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_bf16_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_bf16_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_bf16_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_bf16_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_bf16_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_bf16_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_bf16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f16(dtype: GgmlDType) -> Option<PlainLauncher> {
    use candle_kernels::ffi;
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_f16_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_f16_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_f16_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_f16_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_f16_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_f16_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_f16_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_f16_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_f16_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_f16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f32(dtype: GgmlDType) -> Option<PlainLauncher> {
    use candle_kernels::ffi;
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_f32_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_f32_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_f32_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_f32_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_f32_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_f32_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_f32_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_f32_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_f32_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_f32_plain,
        _ => return None,
    };
    Some(f)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

// S3.3: reuse the existing specialized device entries with the identical
// launch geometry and Q8_1 workspace. Resolve both before launching either.
// No host-side CUDA runtime launcher is needed on the selected path.
#[cfg(feature = "cuda-dkr-s31")]
#[allow(clippy::too_many_arguments)]
fn try_dkr_launch(
    dev: &CudaDevice, dtype: GgmlDType, input_dtype: DType,
    rhs: u64, scratch: u64, weight: u64, output: u64,
    k: usize, k_padded: usize, nrows: usize, batch: usize,
    stride_col_y: i32, stride_col_dst: i32,
) -> Result<bool> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use crate::cuda_backend::WrapErr;
    let codec = match dtype {
        GgmlDType::Q4_0 => "q4_0", GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0", GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0", GgmlDType::Q2K => "q2_k",
        GgmlDType::Q3K => "q3_k", GgmlDType::Q4K => "q4_k",
        GgmlDType::Q5K => "q5_k", GgmlDType::Q6K => "q6_k",
        _ => return Ok(false),
    };
    let tag = match input_dtype {
        DType::BF16 => "bf16", DType::F16 => "f16", DType::F32 => "f32",
        _ => return Ok(false),
    };
    let Some(quant) = dev.dkr_mmvq_func(&format!("mmvq_gguf_quantize_q8_1_{tag}"))? else {
        return Ok(false);
    };
    let mat = dev.dkr_mmvq_func(&format!("mmvq_gguf_{codec}_{tag}_plain_cuda{batch}"))?
        .ok_or_else(|| crate::Error::Msg("DKR MMVQ provider changed during launch".into()))?;
    if k == 0 || nrows == 0 || !(1..=8).contains(&batch)
        || [k, k_padded, nrows].iter().any(|n| *n > i32::MAX as usize)
        || k_padded.checked_mul(batch).is_none_or(|n| n > i32::MAX as usize)
        || nrows.checked_mul(batch).is_none_or(|n| n > i32::MAX as usize)
    {
        crate::bail!("DKR MMVQ geometry exceeds native ABI");
    }
    let k_i32 = k as i32;
    let padded_i32 = k_padded as i32;
    let rows_i32 = nrows as i32;
    let qcfg = LaunchConfig {
        grid_dim: (k_padded.div_ceil(256) as u32, batch as u32, 1),
        block_dim: (256, 1, 1), shared_mem_bytes: 0,
    };
    let mcfg = LaunchConfig {
        grid_dim: (nrows.div_ceil(if batch == 1 { 1 } else { 2 }) as u32, 1, 1),
        block_dim: (32, if batch <= 4 { 4 } else { 2 }, 1), shared_mem_bytes: 0,
    };
    let mut q = quant.builder();
    q.arg(&rhs).arg(&scratch).arg(&k_i32).arg(&padded_i32);
    unsafe { q.launch(qcfg).w()?; }
    quant.record_submission();
    let mut m = mat.builder();
    m.arg(&weight).arg(&scratch).arg(&output).arg(&k_i32).arg(&rows_i32)
        .arg(&stride_col_y).arg(&stride_col_dst);
    unsafe { m.launch(mcfg).w()?; }
    mat.record_submission();
    Ok(true)
}

/// Try the fast MMVQ path. Returns `Ok(None)` when the fast path is not applicable:
/// - unsupported quant dtype
/// - batch too large
/// - non-BF16/F32 input
pub fn try_fwd(
    qstorage: &QCudaStorage,
    self_shape: &Shape,
    rhs: &CudaStorage,
    rhs_l: &crate::Layout,
) -> Result<Option<(CudaStorage, Shape)>> {
    use candle_kernels::ffi;

    // Gate checks.
    let w_dtype = qstorage.dtype();
    if !supports(w_dtype) {
        return Ok(None);
    }
    let input_dtype = rhs.dtype();
    if !matches!(input_dtype, DType::BF16 | DType::F16 | DType::F32) {
        return Ok(None);
    }

    let (nrows, ncols) = self_shape.dims2()?;

    let (b_size, k) = match rhs_l.shape().dims() {
        [b, m, k] => (b * m, *k),
        [b, k] => (*b, *k),
        _ => return Ok(None),
    };
    if ncols != k {
        return Ok(None);
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        return Ok(None);
    }

    let (o1, o2) = match rhs_l.contiguous_offsets() {
        Some(offsets) => offsets,
        None => return Ok(None),
    };

    let dev = qstorage.device();
    let stream_ptr = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;

    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let (scratch_ptr, _workspace_guard) = workspace_ensure(dev, scratch_bytes)?;
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;
    let weight_ptr = qstorage.device_ptr()? as *const std::ffi::c_void;

    let mut out_shape = rhs_l.shape().dims().to_vec();
    out_shape.pop();
    out_shape.push(nrows);

    let stream = dev.cuda_stream();

    match input_dtype {
        DType::BF16 => {
            let rhs_slice = rhs.as_cuda_slice::<half::bf16>()?;
            let rhs_slice = rhs_slice.slice(o1..o2);
            let out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };

            let (rhs_addr, rhs_guard) = rhs_slice.device_ptr(&stream);
            let (out_addr, out_guard) = out.device_ptr(&stream);
            let rhs_ptr = rhs_addr as *const std::ffi::c_void;
            let out_ptr = out_addr as *mut std::ffi::c_void;

            #[cfg(feature = "cuda-dkr-s31")]
            let launched = try_dkr_launch(dev, w_dtype, input_dtype,
                rhs_ptr as u64, scratch_ptr as u64, weight_ptr as u64, out_ptr as u64,
                k, k_padded, nrows, b_size, stride_col_y, stride_col_dst)?;
            #[cfg(not(feature = "cuda-dkr-s31"))]
            let launched = false;
            if !launched {
                #[cfg(feature="cuda-dkr-s31")]
                dkr_kernel_registry::telemetry::handle(dkr_kernel_registry::NATIVE_ABI,"fast_mmvq","static_ffi","",0).record(dkr_kernel_registry::telemetry::Event::FallbackSelected);
                unsafe {
                ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                    rhs_ptr,
                    scratch_ptr,
                    k as i32,
                    k_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
                let launcher = plain_launcher_bf16(w_dtype).unwrap();
                launcher(
                    weight_ptr,
                    scratch_ptr as *const std::ffi::c_void,
                    out_ptr,
                    k as i32,
                    nrows as i32,
                    stride_col_y,
                    stride_col_dst,
                    b_size as i32,
                    stream_ptr,
                );
            } }

            drop(rhs_guard);
            drop(out_guard);
            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Some((out_storage, out_shape.into())))
        }
        DType::F16 => {
            let rhs_slice = rhs.as_cuda_slice::<half::f16>()?;
            let rhs_slice = rhs_slice.slice(o1..o2);
            let out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };

            let (rhs_addr, rhs_guard) = rhs_slice.device_ptr(&stream);
            let (out_addr, out_guard) = out.device_ptr(&stream);
            let rhs_ptr = rhs_addr as *const std::ffi::c_void;
            let out_ptr = out_addr as *mut std::ffi::c_void;

            #[cfg(feature = "cuda-dkr-s31")]
            let launched = try_dkr_launch(dev, w_dtype, input_dtype,
                rhs_ptr as u64, scratch_ptr as u64, weight_ptr as u64, out_ptr as u64,
                k, k_padded, nrows, b_size, stride_col_y, stride_col_dst)?;
            #[cfg(not(feature = "cuda-dkr-s31"))]
            let launched = false;
            if !launched {
                #[cfg(feature="cuda-dkr-s31")]
                dkr_kernel_registry::telemetry::handle(dkr_kernel_registry::NATIVE_ABI,"fast_mmvq","static_ffi","",0).record(dkr_kernel_registry::telemetry::Event::FallbackSelected);
                unsafe {
                ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                    rhs_ptr,
                    scratch_ptr,
                    k as i32,
                    k_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
                let launcher = plain_launcher_f16(w_dtype).unwrap();
                launcher(
                    weight_ptr,
                    scratch_ptr as *const std::ffi::c_void,
                    out_ptr,
                    k as i32,
                    nrows as i32,
                    stride_col_y,
                    stride_col_dst,
                    b_size as i32,
                    stream_ptr,
                );
            } }

            drop(rhs_guard);
            drop(out_guard);
            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Some((out_storage, out_shape.into())))
        }
        DType::F32 => {
            let rhs_slice = rhs.as_cuda_slice::<f32>()?;
            let rhs_slice = rhs_slice.slice(o1..o2);
            let out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

            let (rhs_addr, rhs_guard) = rhs_slice.device_ptr(&stream);
            let (out_addr, out_guard) = out.device_ptr(&stream);
            let rhs_ptr = rhs_addr as *const std::ffi::c_void;
            let out_ptr = out_addr as *mut std::ffi::c_void;

            #[cfg(feature = "cuda-dkr-s31")]
            let launched = try_dkr_launch(dev, w_dtype, input_dtype,
                rhs_ptr as u64, scratch_ptr as u64, weight_ptr as u64, out_ptr as u64,
                k, k_padded, nrows, b_size, stride_col_y, stride_col_dst)?;
            #[cfg(not(feature = "cuda-dkr-s31"))]
            let launched = false;
            if !launched {
                #[cfg(feature="cuda-dkr-s31")]
                dkr_kernel_registry::telemetry::handle(dkr_kernel_registry::NATIVE_ABI,"fast_mmvq","static_ffi","",0).record(dkr_kernel_registry::telemetry::Event::FallbackSelected);
                unsafe {
                ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                    rhs_ptr,
                    scratch_ptr,
                    k as i32,
                    k_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
                let launcher = plain_launcher_f32(w_dtype).unwrap();
                launcher(
                    weight_ptr,
                    scratch_ptr as *const std::ffi::c_void,
                    out_ptr,
                    k as i32,
                    nrows as i32,
                    stride_col_y,
                    stride_col_dst,
                    b_size as i32,
                    stream_ptr,
                );
            } }

            drop(rhs_guard);
            drop(out_guard);
            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Some((out_storage, out_shape.into())))
        }
        _ => Ok(None),
    }
}
