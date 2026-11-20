//! CUDA fast path for GGUF tiled matmul (prompt/prefill phase).
//! Handles batch > 8 (complement to fast_mmvq which handles batch 1-8).

use std::cell::RefCell;
use std::collections::{hash_map::Entry, HashMap};
use std::sync::{Mutex, OnceLock};
use std::thread::LocalKey;

use super::cuda::{QCudaStorage, MATRIX_ROW_PADDING};
use super::GgmlDType;
use crate::cuda_backend::{DeviceId, WrapErr};
use crate::{backend::BackendStorage, CudaDevice, CudaStorage, DType, Result, Shape};

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

const QK8_1: usize = 32;
const BLOCK_Q8_1_MMQ_SIZE: usize = 4 * QK8_1 + 4 * 4; // 128 qs + 16 scale bytes = 144

#[inline]
fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

/// Quant types supported by MMQ kernels (same as MMVQ).
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

/// qk (block quantization size) per dtype.
fn qk_for(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 => {
            32
        }
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => 256,
        _ => unreachable!(),
    }
}

// ds_layout mapping: which Q8_1_mmq scale layout to use per weight type.
enum DsLayout {
    D4,
    DS4,
    D2S6,
}

fn ds_layout_for(dtype: GgmlDType) -> DsLayout {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 => DsLayout::DS4,
        GgmlDType::Q5_0 => DsLayout::D4,
        GgmlDType::Q5_1 => DsLayout::DS4,
        GgmlDType::Q8_0 => DsLayout::D4,
        GgmlDType::Q2K => DsLayout::D2S6,
        GgmlDType::Q3K => DsLayout::D4,
        GgmlDType::Q4K | GgmlDType::Q5K => DsLayout::DS4,
        GgmlDType::Q6K => DsLayout::D4,
        _ => unreachable!(),
    }
}

type QuantizeLauncher = unsafe extern "C" fn(
    x: *const std::ffi::c_void,
    ids: *const i32,
    vy: *mut std::ffi::c_void,
    type_x: i32,
    ne00: i64,
    s01: i64,
    s02: i64,
    s03: i64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    stream: *mut std::ffi::c_void,
) -> i32;

fn quantize_launcher(layout: DsLayout) -> QuantizeLauncher {
    use candle_kernels::ffi;
    match layout {
        DsLayout::D4 => ffi::launch_mmq_quantize_q8_1_D4,
        DsLayout::DS4 => ffi::launch_mmq_quantize_q8_1_DS4,
        DsLayout::D2S6 => ffi::launch_mmq_quantize_q8_1_D2S6,
    }
}

type MmqLauncher = unsafe extern "C" fn(
    tmp_fixup: *mut std::ffi::c_void,
    x: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i64,
    nrows_x: i64,
    ncols_y: i64,
    stride_row_x: i64,
    stride_col_dst: i64,
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

fn mmq_launcher(dtype: GgmlDType) -> Option<MmqLauncher> {
    use candle_kernels::ffi;
    let f: MmqLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmq_gguf_q4_0,
        GgmlDType::Q4_1 => ffi::launch_mmq_gguf_q4_1,
        GgmlDType::Q5_0 => ffi::launch_mmq_gguf_q5_0,
        GgmlDType::Q5_1 => ffi::launch_mmq_gguf_q5_1,
        GgmlDType::Q8_0 => ffi::launch_mmq_gguf_q8_0,
        GgmlDType::Q2K => ffi::launch_mmq_gguf_q2_k,
        GgmlDType::Q3K => ffi::launch_mmq_gguf_q3_k,
        GgmlDType::Q4K => ffi::launch_mmq_gguf_q4_k,
        GgmlDType::Q5K => ffi::launch_mmq_gguf_q5_k,
        GgmlDType::Q6K => ffi::launch_mmq_gguf_q6_k,
        _ => return None,
    };
    Some(f)
}

// ---------------------------------------------------------------------------
// The CUDA per-thread stream has the same raw handle on every host thread.
// ---------------------------------------------------------------------------

type Workspaces = RefCell<HashMap<DeviceId, CudaSlice<u8>>>;

thread_local! {
    static MMQ_WORKSPACE: Workspaces = RefCell::new(HashMap::new());
    static FIXUP_WORKSPACE: Workspaces = RefCell::new(HashMap::new());
}

fn with_workspace<T>(
    workspaces: &'static LocalKey<Workspaces>,
    dev: &CudaDevice,
    stream: &CudaStream,
    bytes: usize,
    run: impl FnOnce(u64) -> Result<T>,
) -> Result<T> {
    workspaces.with(|cache| {
        let mut cache = cache.borrow_mut();
        let slice = match cache.entry(dev.id()) {
            Entry::Occupied(mut entry) => {
                if entry.get().len() < bytes {
                    entry.insert(unsafe { dev.alloc::<u8>(bytes)? });
                }
                entry.into_mut()
            }
            Entry::Vacant(entry) => entry.insert(unsafe { dev.alloc::<u8>(bytes)? }),
        };
        let (ptr, _guard) = slice.device_ptr_mut(stream);
        run(ptr)
    })
}

// ---------------------------------------------------------------------------
// Per-device info cache (compute capability, SM count, etc.)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct DeviceInfo {
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
}

static DEVICE_INFO: OnceLock<Mutex<HashMap<DeviceId, DeviceInfo>>> = OnceLock::new();

fn get_device_info(dev: &CudaDevice) -> DeviceInfo {
    use cudarc::driver::{result, sys};
    let map = DEVICE_INFO.get_or_init(|| Mutex::new(HashMap::new()));
    let key = dev.id();
    let mut guard = map.lock().unwrap();
    if let Some(info) = guard.get(&key) {
        return *info;
    }
    let cu_device = dev.cuda_stream().context().cu_device();
    let major = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .unwrap_or(8);
    let minor = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .unwrap_or(0);
    let nsm = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
    }
    .unwrap_or(1);
    let smpbo = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
    }
    .unwrap_or(49152);
    let warp_size = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        )
    }
    .unwrap_or(32);
    let info = DeviceInfo {
        cc: major * 100 + minor * 10,
        nsm,
        smpbo: smpbo as i64,
        warp_size,
    };
    guard.insert(key, info);
    info
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

// The catalogue selects the qualified native launch recipe. Retain a bounded
// thread-local plan cache, including resolved functions and launch configs.
// No static fixup buffer is needed for conventional tiling.
#[cfg(feature = "cuda-dkr-catalog")]
mod external {
    use super::*;
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use crate::cuda_backend::WrapErr;

    #[derive(Clone, Copy, PartialEq, Eq)]
    struct Key {
        device: DeviceId,
        dtype: GgmlDType,
        k: usize,
        padded: usize,
        rows: usize,
        batch: usize,
    }

    struct Plan {
        quant: crate::cuda_backend::CudaFunc,
        mat: crate::cuda_backend::CudaFunc,
        qcfg: LaunchConfig,
        mcfg: LaunchConfig,
    }

    const PLAN_CAPACITY: usize = 64;
    thread_local! {
        static PLANS: std::cell::RefCell<dkr_kernel_registry::PlanCache<Key,Plan,PLAN_CAPACITY>> =
            const { std::cell::RefCell::new(dkr_kernel_registry::PlanCache::new()) };
    }

    fn prepare(dev: &CudaDevice, key: Key) -> Result<Plan> {
        use cudarc::driver::{result, sys};
        let stream = dev.cuda_stream();
        let context = stream.context();
        let cu = context.cu_device();
        let limit = unsafe { result::device::get_attribute(cu,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK) }.w()?;
        let warp = unsafe { result::device::get_attribute(cu,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE) }.w()?;
        let codec = match key.dtype {
            GgmlDType::Q4_0 => "q4_0", GgmlDType::Q4_1 => "q4_1",
            GgmlDType::Q5_0 => "q5_0", GgmlDType::Q5_1 => "q5_1", GgmlDType::Q8_0 => "q8_0",
            GgmlDType::Q2K => "q2_k", GgmlDType::Q3K => "q3_k", GgmlDType::Q4K => "q4_k",
            GgmlDType::Q5K => "q5_k", GgmlDType::Q6K => "q6_k",
            _ => crate::bail!("unsupported MMQ format"),
        };
        let layout = match ds_layout_for(key.dtype) { DsLayout::D4 => "D4", DsLayout::DS4 => "DS4", DsLayout::D2S6 => "D2S6" };
        let (matrix_key, qcfg, mcfg) = dev.dkr_mmq_configs(codec, key.rows as u32,
            key.batch as u32, key.padded as u32, warp, limit)?;
        let quant = dev.dkr_mmq_func("mmq_quantize", layout)?;
        let mat = dev.dkr_mmq_func(&format!("mmq_instance_{codec}"), &matrix_key)?;
        Ok(Plan { quant, mat, qcfg, mcfg })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch(dev: &CudaDevice, dtype: GgmlDType, k: usize, padded: usize,
        rows: usize, batch: usize, rhs: u64, scratch: u64, weight: u64, output: u64) -> Result<()> {
        // Kernel indexes include signed int products. Constrain the native ABI
        // conservatively before any GPU launch (large shapes stay unsupported).
        if [k,padded,rows,batch].iter().any(|&v| v == 0 || v > i32::MAX as usize)
            || padded.div_ceil(512) > 65535
            || rows.checked_mul(batch).is_none_or(|n| n > i32::MAX as usize)
            || rows.checked_mul(k).is_none_or(|n| n > i32::MAX as usize)
            || padded.checked_mul(batch).is_none_or(|n| n > i32::MAX as usize)
        { crate::bail!("DKR MMQ geometry exceeds native ABI"); }
        let key = Key { device:dev.id(), dtype,k,padded,rows,batch };
        PLANS.with(|slot| -> Result<()> {
            slot.borrow_mut().with_plan_for(dkr_kernel_registry::NATIVE_ABI,"fast_mmq",key, || prepare(dev,key), |p| {
            let stream = dev.cuda_stream();
            let null = 0u64;
            let zero64 = 0i64;
            let k64 = k as i64;
            let padded64 = padded as i64;
            let batch32 = batch as i32;
            let one = 1i32;
            let zero = 0i32;
            let mut q = stream.launch_builder(&p.quant);
            // Original quantize_mmq_q8_1 native ABI (five i64, two i32).
            q.arg(&rhs).arg(&null).arg(&scratch).arg(&k64).arg(&k64)
                .arg(&zero64).arg(&zero64).arg(&padded64).arg(&batch32).arg(&one);
            unsafe { q.launch(p.qcfg.clone()).w()?; }
            p.quant.record_submission();
            let k32 = k as i32;
            let rows32 = rows as i32;
            let stride = (k / qk_for(dtype)) as i32;
            let mut m = stream.launch_builder(&p.mat);
            // Exact original dense MMQ ABI; no MoE ids, no Stream-K/fixup.
            m.arg(&weight).arg(&scratch).arg(&null).arg(&null).arg(&output).arg(&null)
                .arg(&k32).arg(&rows32).arg(&batch32).arg(&stride).arg(&batch32).arg(&rows32)
                .arg(&one).arg(&one).arg(&zero).arg(&zero).arg(&zero)
                .arg(&one).arg(&one).arg(&zero).arg(&zero).arg(&zero).arg(&batch32);
            unsafe { m.launch(p.mcfg.clone()).w()?; }
            p.mat.record_submission();
            Ok(())
            })
        })
    }
}

/// Try the fast MMQ path. Returns `Ok(None)` when the fast path is not applicable:
/// - unsupported quant dtype
/// - batch size 0
/// - non-BF16/F16/F32 input
/// - non-contiguous input
pub fn try_fwd(
    qstorage: &QCudaStorage,
    self_shape: &Shape,
    rhs: &CudaStorage,
    rhs_l: &crate::Layout,
) -> Result<Option<(CudaStorage, Shape)>> {
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
    if b_size == 0 {
        return Ok(None);
    }

    let qk = qk_for(w_dtype);
    if k % qk != 0 {
        return Ok(None);
    }

    let (o1, o2) = match rhs_l.contiguous_offsets() {
        Some(offsets) => offsets,
        None => return Ok(None),
    };

    let dev = qstorage.device();
    #[cfg(feature = "cuda-dkr-catalog")]
    let use_external = dev.dkr_mmq_enabled();
    if rhs.device().id() != dev.id() {
        crate::bail!("GGUF MMQ requires same-device inputs");
    }
    if k == 0
        || nrows == 0
        || nrows.checked_mul(k).is_none_or(|v| v > i32::MAX as usize)
        || nrows
            .checked_mul(b_size)
            .is_none_or(|v| v > i32::MAX as usize)
        || k.checked_mul(b_size).is_none_or(|v| v > i32::MAX as usize)
    {
        crate::bail!("GGUF MMQ geometry exceeds native ABI");
    }
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;

    // MMQ quantize expects f32 input. Convert if needed.
    let rhs_f32_storage: Option<CudaStorage> = if input_dtype != DType::F32 {
        let layout_for_cast =
            crate::Layout::contiguous_with_offset(rhs_l.shape(), rhs_l.start_offset());
        Some(rhs.to_dtype(&layout_for_cast, DType::F32)?)
    } else if !o1.is_multiple_of(4) {
        // The quantizer loads float4, even for an F32 view with a nonzero offset.
        let src = rhs.as_cuda_slice::<f32>()?.slice(o1..o2);
        let mut aligned = unsafe { dev.alloc::<f32>(src.len())? };
        dev.memcpy_dtod(&src, &mut aligned)?;
        Some(CudaStorage::wrap_cuda_slice(aligned, dev.clone()))
    } else {
        None
    };

    // Get the f32 slice (either original or converted)
    let rhs_f32_ref = rhs_f32_storage.as_ref().unwrap_or(rhs);
    let rhs_f32_slice = rhs_f32_ref.as_cuda_slice::<f32>()?;
    // For the converted case, offset is 0 (to_dtype returns contiguous from 0).
    // For the original f32 case, apply the original offsets.
    let rhs_f32_slice = if rhs_f32_storage.is_some() {
        rhs_f32_slice.slice(..b_size * k)
    } else {
        rhs_f32_slice.slice(o1..o2)
    };
    let (rhs_addr, rhs_guard) = rhs_f32_slice.device_ptr(&stream);
    let rhs_ptr = rhs_addr as *const std::ffi::c_void;

    // Compute padded dimensions
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    // Must also be multiple of 4*QK8_1 = 128 for block_q8_1_mmq
    let k_padded = pad(k_padded, 4 * QK8_1);

    // Workspace for block_q8_1_mmq quantized activations
    let blocks_per_row = k_padded / (4 * QK8_1);
    let workspace_main = b_size * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE;
    // Extra padding for mmq_x_max (128 for MMA path)
    let workspace_extra = 128 * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_bytes = workspace_main + workspace_extra;

    let (weight_ptr, weight_guard) = qstorage.device_ptr_with_guard(&stream)?;
    let stride_row_x = (k / qk) as i64;

    let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };
    let stride_col_dst = nrows as i64;

    let (out_addr, out_guard) = out.device_ptr_mut(&stream);
    let out_ptr = out_addr as *mut std::ffi::c_void;

    with_workspace(
        &MMQ_WORKSPACE,
        dev,
        &stream,
        workspace_bytes,
        |scratch_addr| {
            #[cfg(feature = "cuda-dkr-catalog")]
            if use_external {
                return external::launch(
                    dev,
                    w_dtype,
                    k,
                    k_padded,
                    nrows,
                    b_size,
                    rhs_addr,
                    scratch_addr,
                    weight_ptr as u64,
                    out_addr,
                );
            }

            let di = get_device_info(dev);
            stream.context().bind_to_thread().w()?;
            let scratch_ptr = scratch_addr as *mut std::ffi::c_void;
            let quantize = quantize_launcher(ds_layout_for(w_dtype));
            let status = unsafe {
                quantize(
                    rhs_ptr,
                    std::ptr::null(),
                    scratch_ptr,
                    0,
                    k as i64,
                    k as i64,
                    0,
                    0,
                    k_padded as i64,
                    b_size as i64,
                    1,
                    1,
                    stream_ptr,
                )
            };
            if status != 0 {
                crate::bail!(
                    "Candle static MMQ quantize {w_dtype:?} failed (cudaError_t={status})"
                );
            }

            let launcher = mmq_launcher(w_dtype).expect("supports() checked");
            let launch = |fixup_addr| {
                let status = unsafe {
                    launcher(
                        fixup_addr as *mut std::ffi::c_void,
                        weight_ptr.cast(),
                        scratch_ptr.cast_const(),
                        out_ptr,
                        k as i64,
                        nrows as i64,
                        b_size as i64,
                        stride_row_x,
                        stride_col_dst,
                        di.cc,
                        di.nsm,
                        di.smpbo,
                        di.warp_size,
                        stream_ptr,
                    )
                };
                if status != 0 {
                    crate::bail!(
                        "Candle static MMQ matmul {w_dtype:?} failed (cudaError_t={status})"
                    );
                }
                Ok(())
            };
            if di.cc >= 700 {
                const MMQ_X_MAX: usize = 128;
                const MMQ_Y_MAX: usize = 128;
                const MAX_SMS: usize = 256;
                let fixup_bytes = MAX_SMS * MMQ_X_MAX * MMQ_Y_MAX * std::mem::size_of::<f32>();
                with_workspace(&FIXUP_WORKSPACE, dev, &stream, fixup_bytes, launch)?;
            } else {
                launch(0)?;
            }

            #[cfg(feature = "cuda-dkr-catalog")]
            dkr_kernel_registry::telemetry::handle(
                dkr_kernel_registry::NATIVE_ABI,
                "fast_mmq",
                "static_ffi",
                "",
                0,
            )
            .record(dkr_kernel_registry::telemetry::Event::FallbackSelected);
            Ok(())
        },
    )?;
    drop(weight_guard);
    drop(rhs_guard);
    drop(out_guard);

    let mut out_shape = rhs_l.shape().dims().to_vec();
    out_shape.pop();
    out_shape.push(nrows);

    let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());

    if input_dtype == DType::F32 {
        Ok(Some((out_storage, out_shape.into())))
    } else {
        // Cast f32 output back to input dtype
        let out_layout = crate::Layout::contiguous(Shape::from(out_shape.clone()));
        let cast_storage = out_storage.to_dtype(&out_layout, input_dtype)?;
        Ok(Some((cast_storage, out_shape.into())))
    }
}
