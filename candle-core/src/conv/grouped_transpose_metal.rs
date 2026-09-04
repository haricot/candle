use crate::conv::{ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::{Layout, MetalStorage, Result};
use candle_metal_kernels::{BufferOffset, Output};

fn output_buffer(storage: &MetalStorage, layout: &Layout) -> BufferOffset<'_> {
    BufferOffset {
        buffer: &storage.buffer,
        offset_in_bytes: layout.start_offset() * storage.dtype.size_in_bytes(),
    }
}

pub(super) fn launch1d(
    input: &MetalStorage,
    input_l: &Layout,
    kernel: &MetalStorage,
    kernel_l: &Layout,
    p: &ParamsConvTranspose1D,
) -> Result<MetalStorage> {
    if input.device.id() != kernel.device.id() {
        crate::bail!("native grouped Metal conv_transpose1d requires one device")
    }
    let l_out = p.l_out();
    let dst_el = p.b_size * p.c_out * l_out;
    let dtype = input.dtype;
    if dtype != kernel.dtype {
        crate::bail!("native grouped Metal conv_transpose1d dtype mismatch")
    }
    let name = match dtype {
        crate::DType::F32 => "grouped_conv_transpose1d_f32",
        crate::DType::F16 => "grouped_conv_transpose1d_f16",
        crate::DType::BF16 => "grouped_conv_transpose1d_bf16",
        crate::DType::U32 => "grouped_conv_transpose1d_u32",
        crate::DType::U8 => "grouped_conv_transpose1d_u8",
        _ => crate::bail!("native grouped Metal conv_transpose1d {dtype:?} not implemented"),
    };
    let device = input.device.clone();
    let buffer = device
        .new_buffer_builder()
        .with_size_for(dst_el, dtype)
        .with_label("grouped-conv-transpose1d")
        .build()?;
    let encoder = device.command_encoder()?;
    candle_metal_kernels::call_grouped_conv_transpose1d(
        &device.device,
        &encoder,
        &device.kernels,
        name,
        p.stride,
        p.padding,
        p.output_padding,
        p.dilation,
        p.groups,
        l_out,
        input_l.dims(),
        input_l.stride(),
        kernel_l.dims(),
        kernel_l.stride(),
        output_buffer(input, input_l),
        output_buffer(kernel, kernel_l),
        Output::Buffer(&buffer),
    )?;
    Ok(MetalStorage::new(buffer, device, dst_el, dtype))
}

pub(super) fn launch2d(
    input: &MetalStorage,
    input_l: &Layout,
    kernel: &MetalStorage,
    kernel_l: &Layout,
    p: &ParamsConvTranspose2D,
) -> Result<MetalStorage> {
    if input.device.id() != kernel.device.id() {
        crate::bail!("native grouped Metal conv_transpose2d requires one device")
    }
    let out_h = p.out_h();
    let out_w = p.out_w();
    let dst_el = p.b_size * p.c_out * out_h * out_w;
    let dtype = input.dtype;
    if dtype != kernel.dtype {
        crate::bail!("native grouped Metal conv_transpose2d dtype mismatch")
    }
    let name = match dtype {
        crate::DType::F32 => "grouped_conv_transpose2d_f32",
        crate::DType::F16 => "grouped_conv_transpose2d_f16",
        crate::DType::BF16 => "grouped_conv_transpose2d_bf16",
        _ => crate::bail!("native grouped Metal conv_transpose2d {dtype:?} not implemented"),
    };
    let device = input.device.clone();
    let buffer = device
        .new_buffer_builder()
        .with_size_for(dst_el, dtype)
        .with_label("grouped-conv-transpose2d")
        .build()?;
    let encoder = device.command_encoder()?;
    candle_metal_kernels::call_grouped_conv_transpose2d(
        &device.device,
        &encoder,
        &device.kernels,
        name,
        p.stride,
        p.padding,
        p.output_padding,
        p.dilation,
        p.groups,
        out_w,
        out_h,
        input_l.dims(),
        input_l.stride(),
        kernel_l.dims(),
        kernel_l.stride(),
        output_buffer(input, input_l),
        output_buffer(kernel, kernel_l),
        Output::Buffer(&buffer),
    )?;
    Ok(MetalStorage::new(buffer, device, dst_el, dtype))
}
