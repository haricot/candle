use crate::WithDType;
use cudarc;
use cudarc::cudnn::safe::{ConvForward, Cudnn};
use cudarc::driver::{CudaSlice, CudaView, DeviceRepr, ValidAsZeroBits};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// The cudnn handles are stored per thread here rather than on the CudaDevice as they are neither
// send nor sync.
thread_local! {
    static CUDNN: RefCell<HashMap<crate::cuda_backend::DeviceId, Arc<Cudnn>>> = HashMap::new().into();
}

impl From<cudarc::cudnn::CudnnError> for crate::Error {
    fn from(err: cudarc::cudnn::CudnnError) -> Self {
        crate::Error::wrap(err)
    }
}

impl From<cudarc::driver::DriverError> for crate::Error {
    fn from(err: cudarc::driver::DriverError) -> Self {
        crate::Error::wrap(err)
    }
}

pub(crate) fn launch_conv2d<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv2D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    use crate::conv::CudnnFwdAlgo as CandleAlgo;
    use cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t as A;

    let device_id = dev.id();
    let cudnn = CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return Ok(cudnn.clone());
        }
        let c = Cudnn::new(dev.cuda_stream());
        if let Ok(c) = &c {
            cudnn.borrow_mut().insert(device_id, c.clone());
        }
        c
    })?;
    let conv = cudnn.create_conv2d::<Y>(
        /* pad */ [params.padding as i32, params.padding as i32],
        /* stride */ [params.stride as i32, params.stride as i32],
        /* dilation */ [params.dilation as i32, params.dilation as i32],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    let x_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.i_h as i32,
        params.i_w as i32,
    ];
    // Note that `src` already starts at the proper offset.
    let x = if src_l.is_contiguous() {
        cudnn.create_4d_tensor::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            x_shape,
        )?
    } else {
        let s = src_l.stride();
        cudnn.create_4d_tensor_ex::<T>(
            x_shape,
            [s[0] as i32, s[1] as i32, s[2] as i32, s[3] as i32],
        )?
    };
    let w = cudnn.create_4d_filter::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.c_out as i32,
            params.c_in as i32,
            params.k_h as i32,
            params.k_w as i32,
        ],
    )?;
    let (w_out, h_out) = (params.out_w() as i32, params.out_h() as i32);
    let y = cudnn.create_4d_tensor::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [params.b_size as i32, params.c_out as i32, h_out, w_out],
    )?;
    let conv2d = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let alg = match params.cudnn_fwd_algo {
        None => conv2d.pick_algorithm()?,
        Some(CandleAlgo::ImplicitGemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        Some(CandleAlgo::ImplicitPrecompGemm) => {
            A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        }
        Some(CandleAlgo::Gemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        Some(CandleAlgo::Direct) => A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        Some(CandleAlgo::Fft) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        Some(CandleAlgo::FftTiling) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        Some(CandleAlgo::Winograd) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        Some(CandleAlgo::WinogradNonFused) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        Some(CandleAlgo::Count) => A::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
    };
    let workspace_size = conv2d.get_workspace_size(alg)?;
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        conv2d.launch::<CudaSlice<u8>, _, _, _>(
            alg,
            Some(&mut workspace),
            (T::one(), T::zero()),
            src,
            filter,
            dst,
        )?;
    }
    Ok(())
}

pub(crate) fn launch_conv1d<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv1D,
    dev: &crate::cuda_backend::CudaDevice,
) -> crate::Result<()> {
    use crate::conv::CudnnFwdAlgo as CandleAlgo;
    use cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t as A;

    let device_id = dev.id();
    let cudnn = CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return Ok(cudnn.clone());
        }
        let c = Cudnn::new(dev.cuda_stream());
        if let Ok(c) = &c {
            cudnn.borrow_mut().insert(device_id, c.clone());
        }
        c
    })?;
    let conv = cudnn.create_conv2d::<Y>(
        /* pad */ [params.padding as i32, 0],
        /* stride */ [params.stride as i32, 1],
        /* dilation */ [params.dilation as i32, 1],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    // https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html#cudnnsettensornddescriptor
    // > Tensors are restricted to having at least 4 dimensions, and at most CUDNN_DIM_MAX
    // > dimensions (defined in cudnn.h). When working with lower dimensional data, it is
    // > recommended that the user create a 4D tensor, and set the size along unused dimensions
    // > to 1.
    let x_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.l_in as i32,
        1,
    ];
    // Note that `src` already starts at the proper offset.
    let x = if src_l.is_contiguous() {
        cudnn.create_4d_tensor::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            x_shape,
        )?
    } else {
        let s = src_l.stride();
        cudnn.create_4d_tensor_ex::<T>(x_shape, [s[0] as i32, s[1] as i32, s[2] as i32, 1i32])?
    };
    let w = cudnn.create_4d_filter::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.c_out as i32,
            params.c_in as i32,
            params.k_size as i32,
            1,
        ],
    )?;
    let l_out = params.l_out() as i32;
    let y = cudnn.create_4d_tensor::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [params.b_size as i32, params.c_out as i32, l_out, 1],
    )?;
    let conv1d = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let alg = match params.cudnn_fwd_algo {
        None => conv1d.pick_algorithm()?,
        Some(CandleAlgo::ImplicitGemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        Some(CandleAlgo::ImplicitPrecompGemm) => {
            A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        }
        Some(CandleAlgo::Gemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        Some(CandleAlgo::Direct) => A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        Some(CandleAlgo::Fft) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        Some(CandleAlgo::FftTiling) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        Some(CandleAlgo::Winograd) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        Some(CandleAlgo::WinogradNonFused) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        Some(CandleAlgo::Count) => A::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
    };
    let workspace_size = conv1d.get_workspace_size(alg)?;
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        conv1d.launch::<CudaSlice<u8>, _, _, _>(
            alg,
            Some(&mut workspace),
            (T::one(), T::zero()),
            src,
            filter,
            dst,
        )?;
    }
    Ok(())
}

// Native grouped operations from historical runner 0835f4a, without its
// separate general cuDNN failure policy or FP8-specific dispatch.

fn launch_conv2d_with_groups<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv2D,
    dev: &crate::cuda_backend::CudaDevice,
    groups: usize,
) -> crate::Result<()> {
    use crate::conv::CudnnFwdAlgo as CandleAlgo;
    use cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t as A;

    if groups == 0 || params.c_in % groups != 0 || params.c_out % groups != 0 {
        crate::bail!(
            "invalid grouped conv2d channels: c_in={}, c_out={}, groups={groups}",
            params.c_in,
            params.c_out
        )
    }
    if groups > i32::MAX as usize {
        crate::bail!("grouped conv2d group count {groups} exceeds i32::MAX")
    }

    let device_id = dev.id();
    let cudnn = CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return Ok(cudnn.clone());
        }
        let c = Cudnn::new(dev.cuda_stream());
        if let Ok(c) = &c {
            cudnn.borrow_mut().insert(device_id, c.clone());
        }
        c
    })?;
    let mut conv = cudnn.create_conv2d::<Y>(
        /* pad */ [params.padding as i32, params.padding as i32],
        /* stride */ [params.stride as i32, params.stride as i32],
        /* dilation */ [params.dilation as i32, params.dilation as i32],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    if groups > 1 {
        conv.set_group_count(groups as i32)?;
    }

    let x_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.i_h as i32,
        params.i_w as i32,
    ];
    let x = if src_l.is_contiguous() {
        cudnn.create_4d_tensor::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            x_shape,
        )?
    } else {
        let s = src_l.stride();
        cudnn.create_4d_tensor_ex::<T>(
            x_shape,
            [s[0] as i32, s[1] as i32, s[2] as i32, s[3] as i32],
        )?
    };
    let w = cudnn.create_4d_filter::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.c_out as i32,
            (params.c_in / groups) as i32,
            params.k_h as i32,
            params.k_w as i32,
        ],
    )?;
    let (w_out, h_out) = (params.out_w() as i32, params.out_h() as i32);
    let y = cudnn.create_4d_tensor::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [params.b_size as i32, params.c_out as i32, h_out, w_out],
    )?;
    let conv2d = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let alg = match params.cudnn_fwd_algo {
        None => conv2d.pick_algorithm()?,
        Some(CandleAlgo::ImplicitGemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        Some(CandleAlgo::ImplicitPrecompGemm) => {
            A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        }
        Some(CandleAlgo::Gemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        Some(CandleAlgo::Direct) => A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        Some(CandleAlgo::Fft) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        Some(CandleAlgo::FftTiling) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        Some(CandleAlgo::Winograd) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        Some(CandleAlgo::WinogradNonFused) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        Some(CandleAlgo::Count) => A::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
    };
    let workspace_size = conv2d.get_workspace_size(alg)?;
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        conv2d.launch::<CudaSlice<u8>, _, _, _>(
            alg,
            Some(&mut workspace),
            (T::one(), T::zero()),
            src,
            filter,
            dst,
        )?;
    }
    Ok(())
}

pub(crate) fn launch_grouped_conv2d(
    input: &crate::cuda_backend::CudaStorage,
    input_l: &crate::Layout,
    kernel: &crate::cuda_backend::CudaStorage,
    kernel_l: &crate::Layout,
    params: &crate::conv::ParamsConv2D,
    groups: usize,
) -> crate::Result<crate::cuda_backend::CudaStorage> {
    use crate::cuda_backend::{CudaStorage, CudaStorageSlice as S};

    if !kernel_l.is_contiguous() {
        crate::bail!("native grouped cuDNN conv2d requires a contiguous kernel")
    }
    if input.device.id() != kernel.device.id() {
        crate::bail!("native grouped cuDNN conv2d requires input and kernel on the same device")
    }

    let device = input.device.clone();
    let dst_el = params.c_out * params.out_w() * params.out_h() * params.b_size;
    let slice = match (&input.slice, &kernel.slice) {
        (S::U8(inp), S::U8(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<u8>(dst_el)? };
            launch_conv2d_with_groups::<u8, u8>(
                inp, input_l, k, &mut out, params, &device, groups,
            )?;
            S::U8(out)
        }
        (S::BF16(inp), S::BF16(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<half::bf16>(dst_el)? };
            launch_conv2d_with_groups::<half::bf16, f32>(
                inp, input_l, k, &mut out, params, &device, groups,
            )?;
            S::BF16(out)
        }
        (S::F16(inp), S::F16(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<half::f16>(dst_el)? };
            launch_conv2d_with_groups::<half::f16, half::f16>(
                inp, input_l, k, &mut out, params, &device, groups,
            )?;
            S::F16(out)
        }
        (S::F32(inp), S::F32(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<f32>(dst_el)? };
            launch_conv2d_with_groups::<f32, f32>(
                inp, input_l, k, &mut out, params, &device, groups,
            )?;
            S::F32(out)
        }
        (S::F64(inp), S::F64(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<f64>(dst_el)? };
            launch_conv2d_with_groups::<f64, f64>(
                inp, input_l, k, &mut out, params, &device, groups,
            )?;
            S::F64(out)
        }
        (S::U32(_), S::U32(_)) => crate::bail!("grouped cuDNN conv2d does not support u32"),
        (S::I16(_), S::I16(_)) => crate::bail!("grouped cuDNN conv2d does not support i16"),
        (S::I32(_), S::I32(_)) => crate::bail!("grouped cuDNN conv2d does not support i32"),
        (S::I64(_), S::I64(_)) => crate::bail!("grouped cuDNN conv2d does not support i64"),
        _ => crate::bail!("dtype mismatch in native grouped cuDNN conv2d"),
    };
    Ok(CudaStorage { slice, device })
}

fn launch_conv1d_with_groups<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    Y: cudarc::cudnn::CudnnDataType,
>(
    src: &CudaView<T>,
    src_l: &crate::Layout,
    filter: &CudaView<T>,
    dst: &mut CudaSlice<T>,
    params: &crate::conv::ParamsConv1D,
    dev: &crate::cuda_backend::CudaDevice,
    groups: usize,
) -> crate::Result<()> {
    if groups == 0
        || groups != params.groups
        || params.c_in % groups != 0
        || params.c_out % groups != 0
    {
        crate::bail!("invalid native grouped Conv1D channel or group configuration")
    }
    if groups > i32::MAX as usize {
        crate::bail!("native grouped Conv1D groups exceed i32::MAX")
    }
    for value in [
        params.b_size,
        params.c_in,
        params.c_out,
        params.l_in,
        params.k_size,
        params.padding,
        params.stride,
        params.dilation,
        params.l_out(),
    ] {
        if value > i32::MAX as usize {
            crate::bail!("native grouped Conv1D descriptor dimension exceeds i32::MAX")
        }
    }
    use crate::conv::CudnnFwdAlgo as CandleAlgo;
    use cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t as A;

    let device_id = dev.id();
    let cudnn = CUDNN.with(|cudnn| {
        if let Some(cudnn) = cudnn.borrow().get(&device_id) {
            return Ok(cudnn.clone());
        }
        let c = Cudnn::new(dev.cuda_stream());
        if let Ok(c) = &c {
            cudnn.borrow_mut().insert(device_id, c.clone());
        }
        c
    })?;
    let mut conv = cudnn.create_conv2d::<Y>(
        /* pad */ [params.padding as i32, 0],
        /* stride */ [params.stride as i32, 1],
        /* dilation */ [params.dilation as i32, 1],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    if groups > 1 {
        conv.set_group_count(groups as i32)?;
    }
    // https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html#cudnnsettensornddescriptor
    // > Tensors are restricted to having at least 4 dimensions, and at most CUDNN_DIM_MAX
    // > dimensions (defined in cudnn.h). When working with lower dimensional data, it is
    // > recommended that the user create a 4D tensor, and set the size along unused dimensions
    // > to 1.
    let x_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.l_in as i32,
        1,
    ];
    // Note that `src` already starts at the proper offset.
    let x = if src_l.is_contiguous() {
        cudnn.create_4d_tensor::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            x_shape,
        )?
    } else {
        let s = src_l.stride();
        cudnn.create_4d_tensor_ex::<T>(x_shape, [s[0] as i32, s[1] as i32, s[2] as i32, 1i32])?
    };
    let w = cudnn.create_4d_filter::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.c_out as i32,
            (params.c_in / groups) as i32,
            params.k_size as i32,
            1,
        ],
    )?;
    let l_out = params.l_out() as i32;
    let y = cudnn.create_4d_tensor::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [params.b_size as i32, params.c_out as i32, l_out, 1],
    )?;
    let conv1d = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let alg = match params.cudnn_fwd_algo {
        None => conv1d.pick_algorithm()?,
        Some(CandleAlgo::ImplicitGemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        Some(CandleAlgo::ImplicitPrecompGemm) => {
            A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        }
        Some(CandleAlgo::Gemm) => A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        Some(CandleAlgo::Direct) => A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        Some(CandleAlgo::Fft) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        Some(CandleAlgo::FftTiling) => A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        Some(CandleAlgo::Winograd) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        Some(CandleAlgo::WinogradNonFused) => A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        Some(CandleAlgo::Count) => A::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
    };
    let workspace_size = conv1d.get_workspace_size(alg)?;
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        conv1d.launch::<CudaSlice<u8>, _, _, _>(
            alg,
            Some(&mut workspace),
            (T::one(), T::zero()),
            src,
            filter,
            dst,
        )?;
    }
    Ok(())
}

// Native grouped Conv1D, matching the existing grouped Conv2D cuDNN path.
// Use cuDNN's 4D representation [N,C,L,1] and filter [Cout,Cin/groups,K,1].

pub(crate) fn launch_grouped_conv1d(
    input: &crate::cuda_backend::CudaStorage,
    input_l: &crate::Layout,
    kernel: &crate::cuda_backend::CudaStorage,
    kernel_l: &crate::Layout,
    params: &crate::conv::ParamsConv1D,
    groups: usize,
) -> crate::Result<crate::cuda_backend::CudaStorage> {
    use crate::cuda_backend::{CudaStorage, CudaStorageSlice as S};

    if !kernel_l.is_contiguous() {
        crate::bail!("native grouped cuDNN conv1d requires a contiguous kernel")
    }
    if input.device.id() != kernel.device.id() {
        crate::bail!("native grouped cuDNN conv1d requires input and kernel on the same device")
    }
    let device = input.device.clone();
    let dst_el = params.c_out * params.l_out() * params.b_size;
    let slice = match (&input.slice, &kernel.slice) {
        (S::U8(inp), S::U8(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<u8>(dst_el)? };
            launch_conv1d_with_groups::<u8, u8>(
                inp, input_l, k, &mut out, params, &device, groups,
            )?;
            S::U8(out)
        }
        (S::BF16(inp), S::BF16(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<half::bf16>(dst_el)? };
            launch_conv1d_with_groups::<half::bf16, f32>(
                inp, input_l, k, &mut out, params, &device, groups,
            )?;
            S::BF16(out)
        }
        (S::F16(inp), S::F16(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<half::f16>(dst_el)? };
            launch_conv1d_with_groups::<half::f16, half::f16>(
                inp, input_l, k, &mut out, params, &device, groups,
            )?;
            S::F16(out)
        }
        (S::F32(inp), S::F32(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<f32>(dst_el)? };
            launch_conv1d_with_groups::<f32, f32>(
                inp, input_l, k, &mut out, params, &device, groups,
            )?;
            S::F32(out)
        }
        (S::F64(inp), S::F64(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<f64>(dst_el)? };
            launch_conv1d_with_groups::<f64, f64>(
                inp, input_l, k, &mut out, params, &device, groups,
            )?;
            S::F64(out)
        }
        _ => crate::bail!("unsupported or mismatched dtype in native grouped cuDNN conv1d"),
    };
    Ok(CudaStorage { slice, device })
}
