use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{
    ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D,
};
use crate::{CpuStorage, CudaStorage, CustomOp2, Layout, MetalStorage, Result, Shape, Tensor};

fn grouped_conv1d_fallback<S>(
    input: &S,
    input_l: &Layout,
    kernel: &S,
    kernel_l: &Layout,
    params: &ParamsConv1D,
) -> Result<S>
where
    S: BackendStorage,
    S::Device: BackendDevice<Storage = S>,
{
    let groups = params.groups;
    let c_in_group = params.c_in / groups;
    let c_out_group = params.c_out / groups;
    let group_params = ParamsConv1D {
        c_in: c_in_group,
        c_out: c_out_group,
        groups: 1,
        ..params.clone()
    };
    let out_l = params.l_out();
    let group_batch_el = c_out_group * out_l;
    let full_batch_el = params.c_out * out_l;
    let mut output = input
        .device()
        .zeros_impl(&Shape::from(params.out_dims()), input.dtype())?;

    for group in 0..groups {
        let input_group_l = input_l.narrow(1, group * c_in_group, c_in_group)?;
        let kernel_group_l = kernel_l.narrow(0, group * c_out_group, c_out_group)?;
        let group_output = input.conv1d(&input_group_l, kernel, &kernel_group_l, &group_params)?;
        group_output.copy2d(
            &mut output,
            params.b_size,
            group_batch_el,
            group_batch_el,
            full_batch_el,
            0,
            group * group_batch_el,
        )?;
    }
    Ok(output)
}

fn grouped_conv2d_fallback<S>(
    input: &S,
    input_l: &Layout,
    kernel: &S,
    kernel_l: &Layout,
    params: &ParamsConv2D,
) -> Result<S>
where
    S: BackendStorage,
    S::Device: BackendDevice<Storage = S>,
{
    let groups = params.groups;
    let c_in_group = params.c_in / groups;
    let c_out_group = params.c_out / groups;
    let group_params = ParamsConv2D {
        c_in: c_in_group,
        c_out: c_out_group,
        groups: 1,
        ..params.clone()
    };
    let spatial = params.out_h() * params.out_w();
    let group_batch_el = c_out_group * spatial;
    let full_batch_el = params.c_out * spatial;
    let mut output = input
        .device()
        .zeros_impl(&Shape::from(params.out_dims()), input.dtype())?;

    for group in 0..groups {
        let input_group_l = input_l.narrow(1, group * c_in_group, c_in_group)?;
        let kernel_group_l = kernel_l.narrow(0, group * c_out_group, c_out_group)?;
        let group_output = input.conv2d(&input_group_l, kernel, &kernel_group_l, &group_params)?;
        group_output.copy2d(
            &mut output,
            params.b_size,
            group_batch_el,
            group_batch_el,
            full_batch_el,
            0,
            group * group_batch_el,
        )?;
    }
    Ok(output)
}

fn grouped_conv_transpose1d_fallback<S>(
    input: &S,
    input_l: &Layout,
    kernel: &S,
    kernel_l: &Layout,
    params: &ParamsConvTranspose1D,
) -> Result<S>
where
    S: BackendStorage,
    S::Device: BackendDevice<Storage = S>,
{
    let groups = params.groups;
    let c_in_group = params.c_in / groups;
    let c_out_group = params.c_out / groups;
    let group_params = ParamsConvTranspose1D {
        c_in: c_in_group,
        c_out: c_out_group,
        groups: 1,
        ..params.clone()
    };
    let out_l = params.l_out();
    let group_batch_el = c_out_group * out_l;
    let full_batch_el = params.c_out * out_l;
    let mut output = input
        .device()
        .zeros_impl(&Shape::from(params.out_dims()), input.dtype())?;

    for group in 0..groups {
        let input_group_l = input_l.narrow(1, group * c_in_group, c_in_group)?;
        let kernel_group_l = kernel_l.narrow(0, group * c_in_group, c_in_group)?;
        let group_output = input.conv_transpose1d(
            &input_group_l,
            kernel,
            &kernel_group_l,
            &group_params,
        )?;
        group_output.copy2d(
            &mut output,
            params.b_size,
            group_batch_el,
            group_batch_el,
            full_batch_el,
            0,
            group * group_batch_el,
        )?;
    }
    Ok(output)
}

fn grouped_conv_transpose2d_fallback<S>(
    input: &S,
    input_l: &Layout,
    kernel: &S,
    kernel_l: &Layout,
    params: &ParamsConvTranspose2D,
) -> Result<S>
where
    S: BackendStorage,
    S::Device: BackendDevice<Storage = S>,
{
    let groups = params.groups;
    let c_in_group = params.c_in / groups;
    let c_out_group = params.c_out / groups;
    let group_params = ParamsConvTranspose2D {
        c_in: c_in_group,
        c_out: c_out_group,
        groups: 1,
        ..params.clone()
    };
    let spatial = params.out_h() * params.out_w();
    let group_batch_el = c_out_group * spatial;
    let full_batch_el = params.c_out * spatial;
    let mut output = input
        .device()
        .zeros_impl(&Shape::from(params.out_dims()), input.dtype())?;

    for group in 0..groups {
        let input_group_l = input_l.narrow(1, group * c_in_group, c_in_group)?;
        let kernel_group_l = kernel_l.narrow(0, group * c_in_group, c_in_group)?;
        let group_output = input.conv_transpose2d(
            &input_group_l,
            kernel,
            &kernel_group_l,
            &group_params,
        )?;
        group_output.copy2d(
            &mut output,
            params.b_size,
            group_batch_el,
            group_batch_el,
            full_batch_el,
            0,
            group * group_batch_el,
        )?;
    }
    Ok(output)
}

#[derive(Clone, Debug)]
pub(super) struct GroupedConv1D(pub(super) ParamsConv1D);

impl CustomOp2 for GroupedConv1D {
    fn name(&self) -> &'static str {
        "grouped-conv1d"
    }

    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        input_l: &Layout,
        kernel: &CpuStorage,
        kernel_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let out = grouped_conv1d_fallback(input, input_l, kernel, kernel_l, &self.0)?;
        Ok((out, Shape::from(self.0.out_dims())))
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_l: &Layout,
        kernel: &CudaStorage,
        kernel_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let out = grouped_conv1d_fallback(input, input_l, kernel, kernel_l, &self.0)?;
        Ok((out, Shape::from(self.0.out_dims())))
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_l: &Layout,
        kernel: &MetalStorage,
        kernel_l: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let out = grouped_conv1d_fallback(input, input_l, kernel, kernel_l, &self.0)?;
        Ok((out, Shape::from(self.0.out_dims())))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        kernel: &Tensor,
        _res: &Tensor,
        grad: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let p = &self.0;
        let groups = p.groups;
        let c_in_group = p.c_in / groups;
        let c_out_group = p.c_out / groups;
        let arg_groups = arg.chunk(groups, 1)?;
        let kernel_groups = kernel.chunk(groups, 0)?;
        let grad_groups = grad.chunk(groups, 1)?;

        let mut grad_args = Vec::with_capacity(groups);
        let mut grad_kernels = Vec::with_capacity(groups);
        for group in 0..groups {
            let arg_g = &arg_groups[group];
            let kernel_g = &kernel_groups[group];
            let grad_g = &grad_groups[group];

            let grad_l_in = grad_g.dim(2)?;
            let k_size = kernel_g.dim(2)?;
            let out_size =
                (grad_l_in - 1) * p.stride + p.dilation * (k_size - 1) + 1 - 2 * p.padding;
            let out_padding = arg_g.dim(2)? - out_size;
            grad_args.push(grad_g.conv_transpose1d(
                kernel_g,
                p.padding,
                out_padding,
                p.stride,
                p.dilation,
                1,
            )?);

            let grad_kernel = arg_g
                .transpose(0, 1)?
                .conv1d(&grad_g.transpose(0, 1)?, p.padding, p.dilation, p.stride, 1)?
                .transpose(0, 1)?;
            let (_, _, k0) = kernel_g.dims3()?;
            let (_, _, g_k0) = grad_kernel.dims3()?;
            grad_kernels.push(if g_k0 != k0 {
                grad_kernel.narrow(2, 0, k0)?
            } else {
                grad_kernel
            });
        }

        debug_assert_eq!(c_in_group * groups, p.c_in);
        debug_assert_eq!(c_out_group * groups, p.c_out);
        Ok((
            Some(Tensor::cat(&grad_args, 1)?),
            Some(Tensor::cat(&grad_kernels, 0)?),
        ))
    }
}

#[derive(Clone, Debug)]
pub(super) struct GroupedConv2D(pub(super) ParamsConv2D);

impl CustomOp2 for GroupedConv2D {
    fn name(&self) -> &'static str {
        "grouped-conv2d"
    }

    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        input_l: &Layout,
        kernel: &CpuStorage,
        kernel_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let out = grouped_conv2d_fallback(input, input_l, kernel, kernel_l, &self.0)?;
        Ok((out, Shape::from(self.0.out_dims())))
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_l: &Layout,
        kernel: &CudaStorage,
        kernel_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        #[cfg(feature = "cudnn")]
        {
            if kernel_l.is_contiguous() {
                if let Ok(out) = crate::cudnn::launch_grouped_conv2d(
                    input,
                    input_l,
                    kernel,
                    kernel_l,
                    &self.0,
                    self.0.groups,
                ) {
                    return Ok((out, Shape::from(self.0.out_dims())));
                }
            }
        }
        let out = grouped_conv2d_fallback(input, input_l, kernel, kernel_l, &self.0)?;
        Ok((out, Shape::from(self.0.out_dims())))
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_l: &Layout,
        kernel: &MetalStorage,
        kernel_l: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let out = grouped_conv2d_fallback(input, input_l, kernel, kernel_l, &self.0)?;
        Ok((out, Shape::from(self.0.out_dims())))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        kernel: &Tensor,
        _res: &Tensor,
        grad: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let p = &self.0;
        let groups = p.groups;
        let arg_groups = arg.chunk(groups, 1)?;
        let kernel_groups = kernel.chunk(groups, 0)?;
        let grad_groups = grad.chunk(groups, 1)?;

        let mut grad_args = Vec::with_capacity(groups);
        let mut grad_kernels = Vec::with_capacity(groups);
        for group in 0..groups {
            let arg_g = &arg_groups[group];
            let kernel_g = &kernel_groups[group];
            let grad_g = &grad_groups[group];

            let grad_h = grad_g.dim(2)?;
            let k_h = kernel_g.dim(2)?;
            let out_size = (grad_h - 1) * p.stride + p.dilation * (k_h - 1) + 1 - 2 * p.padding;
            let out_padding = arg_g.dim(2)? - out_size;
            grad_args.push(grad_g.conv_transpose2d(
                kernel_g,
                p.padding,
                out_padding,
                p.stride,
                p.dilation,
            )?);

            let grad_kernel = arg_g
                .transpose(0, 1)?
                .conv2d(&grad_g.transpose(0, 1)?, p.padding, p.dilation, p.stride, 1)?
                .transpose(0, 1)?;
            let (_, _, k0, k1) = kernel_g.dims4()?;
            let (_, _, g_k0, g_k1) = grad_kernel.dims4()?;
            grad_kernels.push(if g_k0 != k0 || g_k1 != k1 {
                grad_kernel.narrow(2, 0, k0)?.narrow(3, 0, k1)?
            } else {
                grad_kernel
            });
        }

        Ok((
            Some(Tensor::cat(&grad_args, 1)?),
            Some(Tensor::cat(&grad_kernels, 0)?),
        ))
    }
}

#[derive(Clone, Debug)]
pub(super) struct GroupedConvTranspose1D(pub(super) ParamsConvTranspose1D);

impl CustomOp2 for GroupedConvTranspose1D {
    fn name(&self) -> &'static str {
        "grouped-conv-transpose1d"
    }

    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        input_l: &Layout,
        kernel: &CpuStorage,
        kernel_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let out = grouped_conv_transpose1d_fallback(input, input_l, kernel, kernel_l, &self.0)?;
        Ok((out, Shape::from(self.0.out_dims())))
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_l: &Layout,
        kernel: &CudaStorage,
        kernel_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        #[cfg(feature = "cudnn")]
        if kernel_l.is_contiguous() {
            match crate::cudnn::launch_grouped_conv_transpose1d(
                input, input_l, kernel, kernel_l, &self.0,
            ) {
                Ok(out) => return Ok((out, Shape::from(self.0.out_dims()))),
                Err(err) if std::env::var_os("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT").is_some() => {
                    return Err(err)
                }
                Err(_) => {}
            }
        }
        let out = grouped_conv_transpose1d_fallback(input, input_l, kernel, kernel_l, &self.0)?;
        Ok((out, Shape::from(self.0.out_dims())))
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_l: &Layout,
        kernel: &MetalStorage,
        kernel_l: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let out = grouped_conv_transpose1d_fallback(input, input_l, kernel, kernel_l, &self.0)?;
        Ok((out, Shape::from(self.0.out_dims())))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        kernel: &Tensor,
        _res: &Tensor,
        grad: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let p = &self.0;
        let groups = p.groups;
        let grad_arg = grad.conv1d(kernel, p.padding, p.stride, p.dilation, groups)?;

        let arg_groups = arg.chunk(groups, 1)?;
        let grad_groups = grad.chunk(groups, 1)?;
        let kernel_groups = kernel.chunk(groups, 0)?;
        let mut grad_kernels = Vec::with_capacity(groups);
        for group in 0..groups {
            let arg_g = &arg_groups[group];
            let grad_g = &grad_groups[group];
            let kernel_g = &kernel_groups[group];
            let grad_kernel = grad_g
                .transpose(0, 1)?
                .conv1d(&arg_g.transpose(0, 1)?, p.padding, p.dilation, p.stride, 1)?
                .transpose(0, 1)?;
            let (_, _, k0) = kernel_g.dims3()?;
            let (_, _, g_k0) = grad_kernel.dims3()?;
            grad_kernels.push(if g_k0 != k0 {
                grad_kernel.narrow(2, 0, k0)?
            } else {
                grad_kernel
            });
        }

        Ok((Some(grad_arg), Some(Tensor::cat(&grad_kernels, 0)?)))
    }
}

#[derive(Clone, Debug)]
pub(super) struct GroupedConvTranspose2D(pub(super) ParamsConvTranspose2D);

impl CustomOp2 for GroupedConvTranspose2D {
    fn name(&self) -> &'static str {
        "grouped-conv-transpose2d"
    }

    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        input_l: &Layout,
        kernel: &CpuStorage,
        kernel_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let out = grouped_conv_transpose2d_fallback(input, input_l, kernel, kernel_l, &self.0)?;
        Ok((out, Shape::from(self.0.out_dims())))
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_l: &Layout,
        kernel: &CudaStorage,
        kernel_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        #[cfg(feature = "cudnn")]
        if kernel_l.is_contiguous() {
            match crate::cudnn::launch_grouped_conv_transpose2d(
                input, input_l, kernel, kernel_l, &self.0,
            ) {
                Ok(out) => return Ok((out, Shape::from(self.0.out_dims()))),
                Err(err) if std::env::var_os("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT").is_some() => {
                    return Err(err)
                }
                Err(_) => {}
            }
        }
        let out = grouped_conv_transpose2d_fallback(input, input_l, kernel, kernel_l, &self.0)?;
        Ok((out, Shape::from(self.0.out_dims())))
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_l: &Layout,
        kernel: &MetalStorage,
        kernel_l: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let out = grouped_conv_transpose2d_fallback(input, input_l, kernel, kernel_l, &self.0)?;
        Ok((out, Shape::from(self.0.out_dims())))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        kernel: &Tensor,
        _res: &Tensor,
        grad: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let p = &self.0;
        let groups = p.groups;
        let grad_arg = grad.conv2d(kernel, p.padding, p.stride, p.dilation, groups)?;

        let arg_groups = arg.chunk(groups, 1)?;
        let grad_groups = grad.chunk(groups, 1)?;
        let kernel_groups = kernel.chunk(groups, 0)?;
        let mut grad_kernels = Vec::with_capacity(groups);
        for group in 0..groups {
            let arg_g = &arg_groups[group];
            let grad_g = &grad_groups[group];
            let kernel_g = &kernel_groups[group];
            let grad_kernel = grad_g
                .transpose(0, 1)?
                .conv2d(&arg_g.transpose(0, 1)?, p.padding, p.dilation, p.stride, 1)?
                .transpose(0, 1)?;
            let (_, _, k0, k1) = kernel_g.dims4()?;
            let (_, _, g_k0, g_k1) = grad_kernel.dims4()?;
            grad_kernels.push(if g_k0 != k0 || g_k1 != k1 {
                grad_kernel.narrow(2, 0, k0)?.narrow(3, 0, k1)?
            } else {
                grad_kernel
            });
        }

        Ok((Some(grad_arg), Some(Tensor::cat(&grad_kernels, 0)?)))
    }
}
