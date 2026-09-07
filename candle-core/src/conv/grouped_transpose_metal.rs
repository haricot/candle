use crate::backend::BackendStorage;
use crate::conv::{ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::{DType, Layout, MetalStorage, Result};

fn kernel_name_1d(dtype: DType) -> Result<&'static str> {
    Ok(match dtype {
        DType::F32 => "grouped_conv_transpose1d_f32",
        DType::F16 => "grouped_conv_transpose1d_f16",
        DType::BF16 => "grouped_conv_transpose1d_bf16",
        DType::U8 => "grouped_conv_transpose1d_u8",
        DType::U32 => "grouped_conv_transpose1d_u32",
        _ => crate::bail!("native grouped Metal conv_transpose1d does not support {dtype:?}"),
    })
}

fn kernel_name_2d(dtype: DType) -> Result<&'static str> {
    Ok(match dtype {
        DType::F32 => "grouped_conv_transpose2d_f32",
        DType::F16 => "grouped_conv_transpose2d_f16",
        DType::BF16 => "grouped_conv_transpose2d_bf16",
        _ => crate::bail!("native grouped Metal conv_transpose2d does not support {dtype:?}"),
    })
}

pub(super) fn launch1d(
    input: &MetalStorage,
    input_l: &Layout,
    kernel: &MetalStorage,
    kernel_l: &Layout,
    p: &ParamsConvTranspose1D,
) -> Result<MetalStorage> {
    if input.dtype() != kernel.dtype() {
        crate::bail!("native grouped Metal conv_transpose1d dtype mismatch")
    }
    let dtype = input.dtype();
    let device = input.device().clone();
    let l_out = p.l_out();
    let dst_el = p.b_size * p.c_out * l_out;
    let output = device.new_buffer(dst_el, dtype, "grouped-conv-transpose1d")?;
    let encoder = device.command_encoder()?;
    candle_metal_kernels::call_grouped_conv_transpose1d(
        device.metal_device(),
        &encoder,
        device.kernels(),
        kernel_name_1d(dtype)?,
        l_out,
        p.stride,
        p.padding,
        p.output_padding,
        p.dilation,
        p.groups,
        input_l.dims(),
        input_l.stride(),
        kernel_l.dims(),
        kernel_l.stride(),
        input.buffer(),
        input_l.start_offset() * dtype.size_in_bytes(),
        kernel.buffer(),
        kernel_l.start_offset() * dtype.size_in_bytes(),
        &output,
    )
    .map_err(crate::metal_backend::MetalError::from)?;
    Ok(MetalStorage::new(output, device, dst_el, dtype))
}

pub(super) fn launch2d(
    input: &MetalStorage,
    input_l: &Layout,
    kernel: &MetalStorage,
    kernel_l: &Layout,
    p: &ParamsConvTranspose2D,
) -> Result<MetalStorage> {
    if input.dtype() != kernel.dtype() {
        crate::bail!("native grouped Metal conv_transpose2d dtype mismatch")
    }
    let dtype = input.dtype();
    let device = input.device().clone();
    let out_h = p.out_h();
    let out_w = p.out_w();
    let dst_el = p.b_size * p.c_out * out_h * out_w;
    let output = device.new_buffer(dst_el, dtype, "grouped-conv-transpose2d")?;
    let encoder = device.command_encoder()?;
    candle_metal_kernels::call_grouped_conv_transpose2d(
        device.metal_device(),
        &encoder,
        device.kernels(),
        kernel_name_2d(dtype)?,
        out_w,
        out_h,
        p.stride,
        p.padding,
        p.output_padding,
        p.dilation,
        p.groups,
        input_l.dims(),
        input_l.stride(),
        kernel_l.dims(),
        kernel_l.stride(),
        input.buffer(),
        input_l.start_offset() * dtype.size_in_bytes(),
        kernel.buffer(),
        kernel_l.start_offset() * dtype.size_in_bytes(),
        &output,
    )
    .map_err(crate::metal_backend::MetalError::from)?;
    Ok(MetalStorage::new(output, device, dst_el, dtype))
}
