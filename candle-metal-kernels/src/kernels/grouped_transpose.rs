use crate::utils::EncoderProvider;
use crate::{debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Output, Source};

#[allow(clippy::too_many_arguments)]
pub fn call_grouped_conv_transpose1d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    l_out: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
    input_dims: &[usize],
    input_stride: &[usize],
    kernel_dims: &[usize],
    kernel_stride: &[usize],
    input: &Buffer,
    input_offset: usize,
    kernel: &Buffer,
    kernel_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let c_out = kernel_dims[1] * groups;
    let dst_el = input_dims[0] * c_out * l_out;
    let pipeline = kernels.load_pipeline(device, Source::GroupedTranspose, name)?;
    let (thread_group_count, thread_group_size) = crate::linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "grouped_conv_transpose1d {name} groups={groups}");
    set_params!(
        encoder,
        (
            l_out,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            input_dims,
            input_stride,
            kernel_dims,
            kernel_stride,
            (input, input_offset),
            (kernel, kernel_offset),
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_grouped_conv_transpose2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    out_w: usize,
    out_h: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
    input_dims: &[usize],
    input_stride: &[usize],
    kernel_dims: &[usize],
    kernel_stride: &[usize],
    input: &Buffer,
    input_offset: usize,
    kernel: &Buffer,
    kernel_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let c_out = kernel_dims[1] * groups;
    let dst_el = input_dims[0] * c_out * out_w * out_h;
    let pipeline = kernels.load_pipeline(device, Source::GroupedTranspose, name)?;
    let (thread_group_count, thread_group_size) = crate::linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "grouped_conv_transpose2d {name} groups={groups}");
    set_params!(
        encoder,
        (
            out_w,
            out_h,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            input_dims,
            input_stride,
            kernel_dims,
            kernel_stride,
            (input, input_offset),
            (kernel, kernel_offset),
            Output::new(output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
