use crate::builder_arg as barg;
use crate::conv::{ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::cuda_backend::{
    kernel_name, kernels, CudaStorage, CudaStorageSlice as S, SlicePtrOrNull, WrapErr,
};
use crate::{Layout, Result, WithDType};
use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};

fn launch1d_t<T: DeviceRepr + WithDType + ValidAsZeroBits>(
    input: &CudaSlice<T>,
    input_l: &Layout,
    kernel: &CudaSlice<T>,
    kernel_l: &Layout,
    p: &ParamsConvTranspose1D,
    dev: &crate::cuda_backend::CudaDevice,
) -> Result<CudaSlice<T>> {
    let l_out = p.l_out();
    let dst_el = p.b_size * p.c_out * l_out;
    let dims = input_l.dims();
    if dims.len() != 3 {
        crate::bail!("unexpected input shape for grouped conv_transpose1d {dims:?}")
    }
    let info = [dims, input_l.stride(), kernel_l.dims(), kernel_l.stride()].concat();
    let info = SlicePtrOrNull::params_from_vec(dev, info)?;
    let input = &input.slice(input_l.start_offset()..);
    let kernel = &kernel.slice(kernel_l.start_offset()..);
    let out = unsafe { dev.alloc::<T>(dst_el)? };
    let func = dev.get_or_load_func(
        &kernel_name::<T>("grouped_conv_transpose1d"),
        &kernels::GROUPED_TRANSPOSE,
    )?;
    let cfg = LaunchConfig::for_num_elems(dst_el as u32);
    let mut builder = func.builder();
    barg!(
        builder,
        l_out,
        p.stride,
        p.padding,
        p.output_padding,
        p.dilation,
        p.groups
    );
    info.builder_arg(&mut builder);
    builder.arg(input);
    builder.arg(kernel);
    builder.arg(&out);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(out)
}

fn launch2d_t<T: DeviceRepr + WithDType + ValidAsZeroBits>(
    input: &CudaSlice<T>,
    input_l: &Layout,
    kernel: &CudaSlice<T>,
    kernel_l: &Layout,
    p: &ParamsConvTranspose2D,
    dev: &crate::cuda_backend::CudaDevice,
) -> Result<CudaSlice<T>> {
    let out_h = p.out_h();
    let out_w = p.out_w();
    let dst_el = p.b_size * p.c_out * out_h * out_w;
    let dims = input_l.dims();
    if dims.len() != 4 {
        crate::bail!("unexpected input shape for grouped conv_transpose2d {dims:?}")
    }
    let info = [dims, input_l.stride(), kernel_l.dims(), kernel_l.stride()].concat();
    let info = SlicePtrOrNull::params_from_vec(dev, info)?;
    let input = &input.slice(input_l.start_offset()..);
    let kernel = &kernel.slice(kernel_l.start_offset()..);
    let out = unsafe { dev.alloc::<T>(dst_el)? };
    let func = dev.get_or_load_func(
        &kernel_name::<T>("grouped_conv_transpose2d"),
        &kernels::GROUPED_TRANSPOSE,
    )?;
    let cfg = LaunchConfig::for_num_elems(dst_el as u32);
    let mut builder = func.builder();
    barg!(
        builder,
        out_w,
        out_h,
        p.stride,
        p.padding,
        p.output_padding,
        p.dilation,
        p.groups
    );
    info.builder_arg(&mut builder);
    builder.arg(input);
    builder.arg(kernel);
    builder.arg(&out);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(out)
}

pub(super) fn launch1d(
    input: &CudaStorage,
    input_l: &Layout,
    kernel: &CudaStorage,
    kernel_l: &Layout,
    p: &ParamsConvTranspose1D,
) -> Result<CudaStorage> {
    if input.device.id() != kernel.device.id() {
        crate::bail!("native grouped CUDA conv_transpose1d requires one device")
    }
    let dev = input.device.clone();
    let slice = match (&input.slice, &kernel.slice) {
        (S::U8(x), S::U8(k)) => S::U8(launch1d_t::<u8>(x, input_l, k, kernel_l, p, &dev)?),
        (S::U32(x), S::U32(k)) => S::U32(launch1d_t::<u32>(x, input_l, k, kernel_l, p, &dev)?),
        (S::BF16(x), S::BF16(k)) => {
            S::BF16(launch1d_t::<half::bf16>(x, input_l, k, kernel_l, p, &dev)?)
        }
        (S::F16(x), S::F16(k)) => {
            S::F16(launch1d_t::<half::f16>(x, input_l, k, kernel_l, p, &dev)?)
        }
        (S::F32(x), S::F32(k)) => S::F32(launch1d_t::<f32>(x, input_l, k, kernel_l, p, &dev)?),
        (S::F64(x), S::F64(k)) => S::F64(launch1d_t::<f64>(x, input_l, k, kernel_l, p, &dev)?),
        _ => crate::bail!("native grouped CUDA conv_transpose1d dtype mismatch/unsupported"),
    };
    Ok(CudaStorage { slice, device: dev })
}

pub(super) fn launch2d(
    input: &CudaStorage,
    input_l: &Layout,
    kernel: &CudaStorage,
    kernel_l: &Layout,
    p: &ParamsConvTranspose2D,
) -> Result<CudaStorage> {
    if input.device.id() != kernel.device.id() {
        crate::bail!("native grouped CUDA conv_transpose2d requires one device")
    }
    let dev = input.device.clone();
    let slice = match (&input.slice, &kernel.slice) {
        (S::U8(x), S::U8(k)) => S::U8(launch2d_t::<u8>(x, input_l, k, kernel_l, p, &dev)?),
        (S::U32(x), S::U32(k)) => S::U32(launch2d_t::<u32>(x, input_l, k, kernel_l, p, &dev)?),
        (S::BF16(x), S::BF16(k)) => {
            S::BF16(launch2d_t::<half::bf16>(x, input_l, k, kernel_l, p, &dev)?)
        }
        (S::F16(x), S::F16(k)) => {
            S::F16(launch2d_t::<half::f16>(x, input_l, k, kernel_l, p, &dev)?)
        }
        (S::F32(x), S::F32(k)) => S::F32(launch2d_t::<f32>(x, input_l, k, kernel_l, p, &dev)?),
        (S::F64(x), S::F64(k)) => S::F64(launch2d_t::<f64>(x, input_l, k, kernel_l, p, &dev)?),
        _ => crate::bail!("native grouped CUDA conv_transpose2d dtype mismatch/unsupported"),
    };
    Ok(CudaStorage { slice, device: dev })
}
