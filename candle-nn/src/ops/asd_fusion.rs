//! Operation-scoped exact activation fusions extracted from ASD runner 0835f4a.
//! Fallbacks preserve generic Candle CPU behavior for unsupported signatures.
use candle::{DType, Result, Tensor};

// R4-R1-P1 production activation: exact F32 CUDA Conv2d bias+SiLU epilogue.
// The ten promoted shapes select this path by default. A kill switch is retained
// for diagnostics; unsupported devices/dtypes/layouts/backprop always fall back.
fn conv2d_bias_silu_disabled() -> bool {
    matches!(
        std::env::var("CANDLE_CUDA_BIAS_SILU_DISABLE")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

fn conv2d_bias_silu_trace_enabled() -> bool {
    matches!(
        std::env::var("CANDLE_CUDA_BIAS_SILU_TRACE").ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

#[cfg(feature = "cuda")]
fn conv2d_bias_silu_exact_shape(c: usize, h: usize, w: usize) -> bool {
    matches!(
        (c, h, w),
        (48, 128, 96)
            | (24, 128, 96)
            | (96, 64, 48)
            | (48, 64, 48)
            | (192, 32, 24)
            | (96, 32, 24)
            | (384, 16, 12)
            | (192, 16, 12)
            | (768, 8, 6)
            | (384, 8, 6)
    )
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
struct Conv2dBiasSiluF32;

#[cfg(feature = "cuda")]
impl candle::CustomOp2 for Conv2dBiasSiluF32 {
    fn name(&self) -> &'static str {
        "conv2d-bias-silu-f32-r4-r1"
    }

    fn cpu_fwd(
        &self,
        _src: &candle::CpuStorage,
        _src_l: &candle::Layout,
        _bias: &candle::CpuStorage,
        _bias_l: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("conv2d bias+SiLU CUDA candidate cannot execute on CPU")
    }

    fn cuda_fwd(
        &self,
        src: &candle::CudaStorage,
        src_l: &candle::Layout,
        bias: &candle::CudaStorage,
        bias_l: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::{cudarc, WrapErr};
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        if src.device().id() != bias.device().id() {
            candle::bail!("conv2d bias+SiLU candidate requires one CUDA device")
        }
        if src.dtype() != DType::F32 || bias.dtype() != DType::F32 {
            candle::bail!("conv2d bias+SiLU candidate requires f32")
        }
        if !src_l.is_contiguous()
            || src_l.start_offset() != 0
            || !bias_l.is_contiguous()
            || bias_l.start_offset() != 0
        {
            candle::bail!("conv2d bias+SiLU candidate requires contiguous zero-offset storage")
        }
        let dims = src_l.dims();
        if dims.len() != 4 || dims[0] != 1 {
            candle::bail!("conv2d bias+SiLU candidate expects NCHW batch=1, got {dims:?}")
        }
        let (channels, height, width) = (dims[1], dims[2], dims[3]);
        if !conv2d_bias_silu_exact_shape(channels, height, width) {
            candle::bail!("conv2d bias+SiLU candidate invoked outside exact dispatch domain")
        }
        if bias_l.dims() != [channels] {
            candle::bail!(
                "conv2d bias+SiLU candidate bias mismatch: src={dims:?} bias={:?}",
                bias_l.dims()
            )
        }

        let total = dims.iter().product::<usize>();
        let dev = src.device().clone();
        let src = src.as_cuda_slice::<f32>()?;
        let bias = bias.as_cuda_slice::<f32>()?;
        let out = unsafe { dev.alloc::<f32>(total)? };
        let func = dev.get_or_load_func(
            "conv2d_bias_silu_f32",
            &candle::cuda_backend::kernels::ASD_FUSIONS,
        )?;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let total_u64 = total as u64;
        let channels_u64 = channels as u64;
        let spatial_u64 = (height * width) as u64;
        let mut builder = func.builder();
        builder.arg(&total_u64);
        builder.arg(&channels_u64);
        builder.arg(&spatial_u64);
        builder.arg(src);
        builder.arg(bias);
        builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;
        Ok((
            candle::CudaStorage::wrap_cuda_slice(out, dev),
            candle::Shape::from((1, channels, height, width)),
        ))
    }
}

/// Applies a channel bias followed by SiLU to an NCHW Conv2d output.
///
/// R4-R1-P1 promotes the exact CUDA path for the ten RTMPose shapes that
/// passed the micro and E2E promotion frontier. Every other case uses the ordinary Candle `broadcast_add(...).silu()`
/// path. Backprop-tracked tensors always fall back.
pub fn conv2d_bias_silu(xs: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (_, channels, height, width) = xs.dims4()?;
    #[cfg(not(feature = "cuda"))]
    let _ = (height, width);
    let bias_channels = bias.dims1()?;
    if bias_channels != channels {
        candle::bail!(
            "conv2d bias+SiLU channel mismatch: output={:?} bias={:?}",
            xs.dims(),
            bias.dims()
        )
    }

    #[cfg(feature = "cuda")]
    {
        let exact_production = !conv2d_bias_silu_disabled()
            && xs.device().is_cuda()
            && xs.device().same_device(bias.device())
            && xs.dtype() == DType::F32
            && bias.dtype() == DType::F32
            && !xs.track_op()
            && !bias.track_op()
            && xs.is_contiguous()
            && bias.is_contiguous()
            && xs.layout().start_offset() == 0
            && bias.layout().start_offset() == 0
            && xs.dims()[0] == 1
            && conv2d_bias_silu_exact_shape(channels, height, width);
        if exact_production {
            let out = xs.apply_op2_no_bwd(bias, &Conv2dBiasSiluF32)?;
            if conv2d_bias_silu_trace_enabled() {
                eprintln!(
                    "[candle-bias-silu-r4-r1-p1] dispatch=selected kernel=conv2d_bias_silu_f32 shape=1x{channels}x{height}x{width} elems={}",
                    out.elem_count()
                );
            }
            return Ok(out);
        }
    }

    if conv2d_bias_silu_trace_enabled() {
        eprintln!(
            "[candle-bias-silu-r4-r1-p1] dispatch=fallback shape={:?} production_disabled={}",
            xs.dims(),
            conv2d_bias_silu_disabled()
        );
    }
    let bias = bias.reshape((1, channels, 1, 1))?;
    xs.broadcast_add(&bias)?.silu()
}

// v0.3.19-r2c production C3: exact RTMPose DW5x5 + bias + SiLU fusion.
//
// This retains the exact NVRTC kernel path measured by r2b. The four promoted
// F32 CUDA signatures select by default; a diagnostic kill switch restores the
// pre-C3 production path (DW5x5 producer followed by the C2 bias+SiLU epilogue).
fn dw5x5_bias_silu_disabled() -> bool {
    matches!(
        std::env::var("CANDLE_CUDA_DW5X5_BIAS_SILU_DISABLE")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

fn dw5x5_bias_silu_trace_enabled() -> bool {
    matches!(
        std::env::var("CANDLE_CUDA_DW5X5_BIAS_SILU_TRACE")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

#[cfg(feature = "cuda")]
fn dw5x5_bias_silu_exact_shape(c: usize, h: usize, w: usize) -> bool {
    matches!(
        (c, h, w),
        (48, 64, 48) | (96, 32, 24) | (192, 16, 12) | (384, 8, 6)
    )
}

#[cfg(feature = "cuda")]
fn dw5x5_bias_silu_device_is_sm61(input: &Tensor) -> bool {
    use candle::backend::BackendStorage;
    use candle::cuda_backend::cudarc::driver::{result, sys};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static CACHE: OnceLock<Mutex<HashMap<candle::cuda_backend::DeviceId, bool>>> = OnceLock::new();
    let (storage, _) = input.storage_and_layout();
    let candle::Storage::Cuda(storage) = &*storage else {
        return false;
    };
    let dev = storage.device();
    let key = dev.id();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().unwrap();
    if let Some(value) = cache.get(&key) {
        return *value;
    }
    let cu_device = dev.cuda_stream().context().cu_device();
    let major = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    };
    let minor = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    };
    let is_sm61 = matches!((major, minor), (Ok(6), Ok(1)));
    cache.insert(key, is_sm61);
    is_sm61
}

#[cfg(feature = "cuda")]
const DW5X5_BIAS_SILU_CUDA_MODULE: &str = "candle_dw5x5_bias_silu_v0319_r2c";
#[cfg(feature = "cuda")]
const DW5X5_BIAS_SILU_FN: &str = "flow_c3_rtmpose_dw5x5_bias_silu_f32";

#[cfg(feature = "cuda")]
const DW5X5_BIAS_SILU_CUDA: &str = r#"extern "C" __global__ void flow_c3_rtmpose_dw5x5_bias_silu_f32(
    const unsigned long long channels,
    const unsigned long long height,
    const unsigned long long width,
    const float *src,
    const float *weight,
    const float *bias,
    float *dst
) {
    constexpr int BX = 16;
    constexpr int BY = 8;
    constexpr int R = 2;
    constexpr int TW = BX + 2 * R;
    constexpr int TH = BY + 2 * R;
    __shared__ float tile[TH * TW];
    __shared__ float filter[25];

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;
    const int tid = ty * BX + tx;
    const unsigned long long channel = (unsigned long long)blockIdx.z;
    if (channel >= channels) return;
    const int base_y = (int)blockIdx.y * BY;
    const int base_x = (int)blockIdx.x * BX;

    for (int i = tid; i < TH * TW; i += BX * BY) {
        const int ly = i / TW;
        const int lx = i - ly * TW;
        const int iy = base_y + ly - R;
        const int ix = base_x + lx - R;
        float value = 0.0f;
        if ((unsigned)iy < (unsigned)height && (unsigned)ix < (unsigned)width) {
            const unsigned long long src_i =
                (channel * height + (unsigned long long)iy) * width +
                (unsigned long long)ix;
            value = __ldg(src + src_i);
        }
        tile[i] = value;
    }
    if (tid < 25) {
        filter[tid] = __ldg(weight + channel * 25ull + (unsigned long long)tid);
    }
    __syncthreads();

    const int oy = base_y + ty;
    const int ox = base_x + tx;
    if ((unsigned)oy >= (unsigned)height || (unsigned)ox >= (unsigned)width) return;

    float acc = 0.0f;
#pragma unroll
    for (int ky = 0; ky < 5; ++ky) {
#pragma unroll
        for (int kx = 0; kx < 5; ++kx) {
            acc += tile[(ty + ky) * TW + tx + kx] * filter[ky * 5 + kx];
        }
    }
    const float x = acc + __ldg(bias + channel);
    const unsigned long long dst_i =
        (channel * height + (unsigned long long)oy) * width +
        (unsigned long long)ox;
    dst[dst_i] = x / (1.0f + expf(-x));
}"#;

#[cfg(feature = "cuda")]
static DW5X5_BIAS_SILU_PTX: std::sync::LazyLock<std::sync::Arc<String>> =
    std::sync::LazyLock::new(|| {
        let ptx = candle::cuda_backend::cudarc::nvrtc::compile_ptx(DW5X5_BIAS_SILU_CUDA)
            .expect("v0.3.19-r2c C3 DW5x5+Bias+SiLU NVRTC compilation failed");
        std::sync::Arc::new(ptx.to_src())
    });

#[cfg(feature = "cuda")]
#[derive(Clone)]
struct Dw5x5BiasSiluF32 {
    ptx: std::sync::Arc<String>,
}

#[cfg(feature = "cuda")]
fn validate_dw5x5_bias_silu_cuda(
    input: &candle::CudaStorage,
    input_l: &candle::Layout,
    weight: &candle::CudaStorage,
    weight_l: &candle::Layout,
    bias: &candle::CudaStorage,
    bias_l: &candle::Layout,
) -> Result<(usize, usize, usize)> {
    use candle::backend::BackendStorage;

    if input.device().id() != weight.device().id() || input.device().id() != bias.device().id() {
        candle::bail!("C3 DW5x5+Bias+SiLU requires one CUDA device")
    }
    if input.dtype() != DType::F32 || weight.dtype() != DType::F32 || bias.dtype() != DType::F32 {
        candle::bail!("C3 DW5x5+Bias+SiLU requires f32")
    }
    if !input_l.is_contiguous()
        || input_l.start_offset() != 0
        || !weight_l.is_contiguous()
        || weight_l.start_offset() != 0
        || !bias_l.is_contiguous()
        || bias_l.start_offset() != 0
    {
        candle::bail!("C3 DW5x5+Bias+SiLU requires contiguous zero-offset storage")
    }
    let dims = input_l.dims();
    if dims.len() != 4 || dims[0] != 1 {
        candle::bail!("C3 DW5x5+Bias+SiLU expects NCHW batch=1, got {dims:?}")
    }
    let (channels, height, width) = (dims[1], dims[2], dims[3]);
    if weight_l.dims() != [channels, 1, 5, 5] || bias_l.dims() != [channels] {
        candle::bail!(
            "C3 DW5x5+Bias+SiLU weight/bias mismatch input={dims:?} weight={:?} bias={:?}",
            weight_l.dims(),
            bias_l.dims()
        )
    }
    if !dw5x5_bias_silu_exact_shape(channels, height, width) {
        candle::bail!(
            "C3 DW5x5+Bias+SiLU outside exact four-shape frontier c={channels} h={height} w={width}"
        )
    }
    Ok((channels, height, width))
}

#[cfg(feature = "cuda")]
impl candle::CustomOp3 for Dw5x5BiasSiluF32 {
    fn name(&self) -> &'static str {
        "candle-dw5x5-bias-silu-f32-v0319-r2c"
    }

    fn cpu_fwd(
        &self,
        _input: &candle::CpuStorage,
        _input_l: &candle::Layout,
        _weight: &candle::CpuStorage,
        _weight_l: &candle::Layout,
        _bias: &candle::CpuStorage,
        _bias_l: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("C3 DW5x5+Bias+SiLU is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &candle::CudaStorage,
        input_l: &candle::Layout,
        weight: &candle::CudaStorage,
        weight_l: &candle::Layout,
        bias: &candle::CudaStorage,
        bias_l: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        let (channels, height, width) =
            validate_dw5x5_bias_silu_cuda(input, input_l, weight, weight_l, bias, bias_l)?;
        let dev = input.device().clone();
        let src = input.as_cuda_slice::<f32>()?;
        let wgt = weight.as_cuda_slice::<f32>()?;
        let bs = bias.as_cuda_slice::<f32>()?;
        let out = unsafe { dev.alloc::<f32>(channels * height * width)? };
        let func = dev.get_or_load_custom_func(
            DW5X5_BIAS_SILU_FN,
            DW5X5_BIAS_SILU_CUDA_MODULE,
            self.ptx.as_str(),
        )?;
        let cfg = LaunchConfig {
            grid_dim: (
                width.div_ceil(16) as u32,
                height.div_ceil(8) as u32,
                channels as u32,
            ),
            block_dim: (16, 8, 1),
            shared_mem_bytes: 0,
        };
        let c = channels as u64;
        let h = height as u64;
        let w = width as u64;
        let mut builder = func.builder();
        builder.arg(&c);
        builder.arg(&h);
        builder.arg(&w);
        builder.arg(src);
        builder.arg(wgt);
        builder.arg(bs);
        builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;
        Ok((
            candle::CudaStorage::wrap_cuda_slice(out, dev),
            candle::Shape::from((1, channels, height, width)),
        ))
    }
}

/// Tries the promoted C3 exact DW5x5 + bias + SiLU CUDA fusion.
///
/// `None` means the caller must execute the pre-C3 production path. This
/// includes CPU, unsupported dtype/layout/shape, backprop, and the diagnostic
/// kill switch. The function never broadens the four-shape sm61 evidence domain.
pub fn dw5x5_bias_silu_exact(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<Option<Tensor>> {
    let dims = input.dims();
    let weight_dims = weight.dims();
    let bias_dims = bias.dims();

    #[cfg(feature = "cuda")]
    {
        let exact = !dw5x5_bias_silu_disabled()
            && input.device().is_cuda()
            && dw5x5_bias_silu_device_is_sm61(input)
            && input.device().same_device(weight.device())
            && input.device().same_device(bias.device())
            && input.dtype() == DType::F32
            && weight.dtype() == DType::F32
            && bias.dtype() == DType::F32
            && !input.track_op()
            && !weight.track_op()
            && !bias.track_op()
            && input.is_contiguous()
            && weight.is_contiguous()
            && bias.is_contiguous()
            && input.layout().start_offset() == 0
            && weight.layout().start_offset() == 0
            && bias.layout().start_offset() == 0
            && dims.len() == 4
            && dims[0] == 1
            && dw5x5_bias_silu_exact_shape(dims[1], dims[2], dims[3])
            && weight_dims == [dims[1], 1, 5, 5]
            && bias_dims == [dims[1]];
        if exact {
            let op = Dw5x5BiasSiluF32 {
                ptx: DW5X5_BIAS_SILU_PTX.clone(),
            };
            let out = input.apply_op3_no_bwd(weight, bias, &op)?;
            if dw5x5_bias_silu_trace_enabled() {
                eprintln!(
                    "[candle-dw5x5-bias-silu-v0319-r2c] dispatch=selected kernel={DW5X5_BIAS_SILU_FN} shape={:?} elems={}",
                    out.dims(),
                    out.elem_count()
                );
            }
            return Ok(Some(out));
        }
    }

    if dw5x5_bias_silu_trace_enabled() {
        eprintln!(
            "[candle-dw5x5-bias-silu-v0319-r2c] dispatch=fallback shape={dims:?} weight={weight_dims:?} bias={bias_dims:?} production_disabled={}",
            dw5x5_bias_silu_disabled()
        );
    }
    Ok(None)
}
