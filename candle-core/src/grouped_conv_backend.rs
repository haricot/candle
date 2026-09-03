//! Backend-storage grouped convolution fallbacks.
//!
//! Group semantics live at the convolution-op level. Backends that do not yet expose a native
//! grouped primitive can use these helpers to execute one single-group convolution per group and
//! assemble the output directly in backend storage. This avoids building `chunk -> conv -> cat`
//! tensor subgraphs and keeps autograd attached to one grouped `Op::Conv*`.

use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D};
use crate::{Layout, Result, Shape};

pub(crate) fn conv1d<S: BackendStorage>(
    input: &S,
    input_l: &Layout,
    kernel: &S,
    kernel_l: &Layout,
    params: &ParamsConv1D,
) -> Result<S> {
    if params.groups == 1 {
        return input.conv1d(input_l, kernel, kernel_l, params);
    }

    let groups = params.groups;
    if groups == 0
        || !params.c_in.is_multiple_of(groups)
        || !params.c_out.is_multiple_of(groups)
    {
        crate::bail!(
            "invalid grouped conv1d channels: c_in={}, c_out={}, groups={groups}",
            params.c_in,
            params.c_out
        )
    }

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
        let group_output = input.conv1d(
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

pub(crate) fn conv2d<S: BackendStorage>(
    input: &S,
    input_l: &Layout,
    kernel: &S,
    kernel_l: &Layout,
    params: &ParamsConv2D,
) -> Result<S> {
    if params.groups == 1 {
        return input.conv2d(input_l, kernel, kernel_l, params);
    }

    let groups = params.groups;
    if groups == 0
        || !params.c_in.is_multiple_of(groups)
        || !params.c_out.is_multiple_of(groups)
    {
        crate::bail!(
            "invalid grouped conv2d channels: c_in={}, c_out={}, groups={groups}",
            params.c_in,
            params.c_out
        )
    }

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
        let group_output = input.conv2d(
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
