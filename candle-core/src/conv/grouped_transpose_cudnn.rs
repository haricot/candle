use crate::conv::{ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::cuda_backend::{CudaStorage, CudaStorageSlice as S};
use crate::{Layout, Result, WithDType};
use cudarc::cudnn::safe::{ConvBackwardData, Cudnn};
use cudarc::driver::{CudaSlice, CudaView, DeviceRepr, ValidAsZeroBits};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

thread_local! {
    static CUDNN_TRANSPOSE: RefCell<HashMap<crate::cuda_backend::DeviceId, Arc<Cudnn>>> =
        HashMap::new().into();
}

fn cudnn_for(dev: &crate::cuda_backend::CudaDevice) -> Result<Arc<Cudnn>> {
    let device_id = dev.id();
    CUDNN_TRANSPOSE.with(|handles| {
        if let Some(handle) = handles.borrow().get(&device_id) {
            return Ok(handle.clone());
        }
        let handle = Cudnn::new(dev.cuda_stream())?;
        handles.borrow_mut().insert(device_id, handle.clone());
        Ok(handle)
    })
}

fn validate_common(groups: usize, c_in: usize, c_out: usize) -> Result<()> {
    if groups == 0 || !c_in.is_multiple_of(groups) || !c_out.is_multiple_of(groups) {
        crate::bail!(
            "invalid grouped transpose convolution channels: c_in={c_in}, c_out={c_out}, groups={groups}"
        )
    }
    if groups > i32::MAX as usize {
        crate::bail!("grouped transpose convolution group count {groups} exceeds i32::MAX")
    }
    Ok(())
}

fn launch_transpose2d_t<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    C: cudarc::cudnn::CudnnDataType,
>(
    input: &CudaView<T>,
    input_l: &Layout,
    filter: &CudaView<T>,
    output: &mut CudaSlice<T>,
    params: &ParamsConvTranspose2D,
    dev: &crate::cuda_backend::CudaDevice,
) -> Result<()> {
    validate_common(params.groups, params.c_in, params.c_out)?;
    let cudnn = cudnn_for(dev)?;
    let mut conv = cudnn.create_conv2d::<C>(
        [params.padding as i32, params.padding as i32],
        [params.stride as i32, params.stride as i32],
        [params.dilation as i32, params.dilation as i32],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    if params.groups > 1 {
        conv.set_group_count(params.groups as i32)?;
    }

    let dx = cudnn.create_4d_tensor::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.b_size as i32,
            params.c_out as i32,
            params.out_h() as i32,
            params.out_w() as i32,
        ],
    )?;
    let w = cudnn.create_4d_filter::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.c_in as i32,
            (params.c_out / params.groups) as i32,
            params.k_h as i32,
            params.k_w as i32,
        ],
    )?;
    let dy_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.i_h as i32,
        params.i_w as i32,
    ];
    let dy = if input_l.is_contiguous() {
        cudnn.create_4d_tensor::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            dy_shape,
        )?
    } else {
        let s = input_l.stride();
        cudnn.create_4d_tensor_ex::<T>(
            dy_shape,
            [s[0] as i32, s[1] as i32, s[2] as i32, s[3] as i32],
        )?
    };

    let op = ConvBackwardData {
        conv: &conv,
        dx: &dx,
        w: &w,
        dy: &dy,
    };
    let algo = op.pick_algorithm()?;
    let workspace_size = op.get_workspace_size(algo)?;
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        op.launch::<CudaSlice<u8>, _, _, _>(
            algo,
            Some(&mut workspace),
            (T::one(), T::zero()),
            output,
            filter,
            input,
        )?;
    }
    Ok(())
}

fn launch_transpose1d_t<
    T: DeviceRepr + WithDType + ValidAsZeroBits + cudarc::cudnn::CudnnDataType,
    C: cudarc::cudnn::CudnnDataType,
>(
    input: &CudaView<T>,
    input_l: &Layout,
    filter: &CudaView<T>,
    output: &mut CudaSlice<T>,
    params: &ParamsConvTranspose1D,
    dev: &crate::cuda_backend::CudaDevice,
) -> Result<()> {
    validate_common(params.groups, params.c_in, params.c_out)?;
    let cudnn = cudnn_for(dev)?;
    let mut conv = cudnn.create_conv2d::<C>(
        [params.padding as i32, 0],
        [params.stride as i32, 1],
        [params.dilation as i32, 1],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    if params.groups > 1 {
        conv.set_group_count(params.groups as i32)?;
    }

    let dx = cudnn.create_4d_tensor::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.b_size as i32,
            params.c_out as i32,
            params.l_out() as i32,
            1,
        ],
    )?;
    let w = cudnn.create_4d_filter::<T>(
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [
            params.c_in as i32,
            (params.c_out / params.groups) as i32,
            params.k_size as i32,
            1,
        ],
    )?;
    let dy_shape = [
        params.b_size as i32,
        params.c_in as i32,
        params.l_in as i32,
        1,
    ];
    let dy = if input_l.is_contiguous() {
        cudnn.create_4d_tensor::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            dy_shape,
        )?
    } else {
        let s = input_l.stride();
        cudnn.create_4d_tensor_ex::<T>(dy_shape, [s[0] as i32, s[1] as i32, s[2] as i32, 1])?
    };

    let op = ConvBackwardData {
        conv: &conv,
        dx: &dx,
        w: &w,
        dy: &dy,
    };
    let algo = op.pick_algorithm()?;
    let workspace_size = op.get_workspace_size(algo)?;
    let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
    unsafe {
        op.launch::<CudaSlice<u8>, _, _, _>(
            algo,
            Some(&mut workspace),
            (T::one(), T::zero()),
            output,
            filter,
            input,
        )?;
    }
    Ok(())
}

pub(super) fn launch_grouped_conv_transpose2d(
    input: &CudaStorage,
    input_l: &Layout,
    kernel: &CudaStorage,
    kernel_l: &Layout,
    params: &ParamsConvTranspose2D,
) -> Result<CudaStorage> {
    if !kernel_l.is_contiguous() {
        crate::bail!("native grouped cuDNN conv_transpose2d requires a contiguous kernel")
    }
    if crate::cuda_backend::cudnn::convolution_is_disabled(input.device.id()) {
        crate::bail!("cuDNN convolution is disabled for this CUDA device")
    }
    if input.device.id() != kernel.device.id() {
        crate::bail!(
            "native grouped cuDNN conv_transpose2d requires input and kernel on the same device"
        )
    }

    let device = input.device.clone();
    let dst_el = params.b_size * params.c_out * params.out_h() * params.out_w();
    let slice = match (&input.slice, &kernel.slice) {
        (S::U8(inp), S::U8(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<u8>(dst_el)? };
            launch_transpose2d_t::<u8, u8>(inp, input_l, k, &mut out, params, &device)?;
            S::U8(out)
        }
        (S::BF16(inp), S::BF16(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<half::bf16>(dst_el)? };
            launch_transpose2d_t::<half::bf16, f32>(inp, input_l, k, &mut out, params, &device)?;
            S::BF16(out)
        }
        (S::F16(inp), S::F16(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<half::f16>(dst_el)? };
            launch_transpose2d_t::<half::f16, half::f16>(
                inp, input_l, k, &mut out, params, &device,
            )?;
            S::F16(out)
        }
        (S::F32(inp), S::F32(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<f32>(dst_el)? };
            launch_transpose2d_t::<f32, f32>(inp, input_l, k, &mut out, params, &device)?;
            S::F32(out)
        }
        (S::F64(inp), S::F64(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<f64>(dst_el)? };
            launch_transpose2d_t::<f64, f64>(inp, input_l, k, &mut out, params, &device)?;
            S::F64(out)
        }
        (S::U32(_), S::U32(_)) => {
            crate::bail!("grouped cuDNN conv_transpose2d does not support u32")
        }
        (S::I16(_), S::I16(_)) => {
            crate::bail!("grouped cuDNN conv_transpose2d does not support i16")
        }
        (S::I32(_), S::I32(_)) => {
            crate::bail!("grouped cuDNN conv_transpose2d does not support i32")
        }
        (S::I64(_), S::I64(_)) => {
            crate::bail!("grouped cuDNN conv_transpose2d does not support i64")
        }
        (S::F8E4M3(_), S::F8E4M3(_)) => {
            crate::bail!("grouped cuDNN conv_transpose2d does not support f8e4m3")
        }
        _ => crate::bail!("dtype mismatch in native grouped cuDNN conv_transpose2d"),
    };
    Ok(CudaStorage { slice, device })
}

pub(super) fn launch_grouped_conv_transpose1d(
    input: &CudaStorage,
    input_l: &Layout,
    kernel: &CudaStorage,
    kernel_l: &Layout,
    params: &ParamsConvTranspose1D,
) -> Result<CudaStorage> {
    if !kernel_l.is_contiguous() {
        crate::bail!("native grouped cuDNN conv_transpose1d requires a contiguous kernel")
    }
    if crate::cuda_backend::cudnn::convolution_is_disabled(input.device.id()) {
        crate::bail!("cuDNN convolution is disabled for this CUDA device")
    }
    if input.device.id() != kernel.device.id() {
        crate::bail!(
            "native grouped cuDNN conv_transpose1d requires input and kernel on the same device"
        )
    }

    let device = input.device.clone();
    let dst_el = params.b_size * params.c_out * params.l_out();
    let slice = match (&input.slice, &kernel.slice) {
        (S::U8(inp), S::U8(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<u8>(dst_el)? };
            launch_transpose1d_t::<u8, u8>(inp, input_l, k, &mut out, params, &device)?;
            S::U8(out)
        }
        (S::BF16(inp), S::BF16(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<half::bf16>(dst_el)? };
            launch_transpose1d_t::<half::bf16, f32>(inp, input_l, k, &mut out, params, &device)?;
            S::BF16(out)
        }
        (S::F16(inp), S::F16(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<half::f16>(dst_el)? };
            launch_transpose1d_t::<half::f16, half::f16>(
                inp, input_l, k, &mut out, params, &device,
            )?;
            S::F16(out)
        }
        (S::F32(inp), S::F32(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<f32>(dst_el)? };
            launch_transpose1d_t::<f32, f32>(inp, input_l, k, &mut out, params, &device)?;
            S::F32(out)
        }
        (S::F64(inp), S::F64(k)) => {
            let inp = &inp.slice(input_l.start_offset()..);
            let k = &k.slice(kernel_l.start_offset()..);
            let mut out = unsafe { device.alloc::<f64>(dst_el)? };
            launch_transpose1d_t::<f64, f64>(inp, input_l, k, &mut out, params, &device)?;
            S::F64(out)
        }
        (S::U32(_), S::U32(_)) => {
            crate::bail!("grouped cuDNN conv_transpose1d does not support u32")
        }
        (S::I16(_), S::I16(_)) => {
            crate::bail!("grouped cuDNN conv_transpose1d does not support i16")
        }
        (S::I32(_), S::I32(_)) => {
            crate::bail!("grouped cuDNN conv_transpose1d does not support i32")
        }
        (S::I64(_), S::I64(_)) => {
            crate::bail!("grouped cuDNN conv_transpose1d does not support i64")
        }
        (S::F8E4M3(_), S::F8E4M3(_)) => {
            crate::bail!("grouped cuDNN conv_transpose1d does not support f8e4m3")
        }
        _ => crate::bail!("dtype mismatch in native grouped cuDNN conv_transpose1d"),
    };
    Ok(CudaStorage { slice, device })
}
