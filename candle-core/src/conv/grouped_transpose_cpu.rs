use crate::conv::{ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::{CpuStorage, Layout, Result, WithDType};
use rayon::prelude::*;

fn launch1d_t<T: WithDType>(
    input: &CpuStorage,
    input_l: &Layout,
    kernel: &CpuStorage,
    kernel_l: &Layout,
    p: &ParamsConvTranspose1D,
) -> Result<CpuStorage> {
    let input = T::cpu_storage_as_slice(input)?;
    let kernel = T::cpu_storage_as_slice(kernel)?;
    let input = &input[input_l.start_offset()..];
    let kernel = &kernel[kernel_l.start_offset()..];
    let s = input_l.stride();
    let ks = kernel_l.stride();
    let l_out = p.l_out();
    let c_in_group = p.c_in / p.groups;
    let c_out_group = p.c_out / p.groups;
    let dst_el = p.b_size * p.c_out * l_out;

    let out = (0..dst_el)
        .into_par_iter()
        .map(|dst_i| {
            let b_idx = dst_i / (p.c_out * l_out);
            let dst_c_idx = (dst_i / l_out) % p.c_out;
            let out_x = dst_i % l_out;
            let group = dst_c_idx / c_out_group;
            let dst_c_local = dst_c_idx % c_out_group;
            let src_c_begin = group * c_in_group;
            let src_c_end = src_c_begin + c_in_group;
            let mut d = 0f64;

            for k_x in 0..p.k_size {
                let inp_x_stride =
                    out_x as isize + p.padding as isize - (k_x * p.dilation) as isize;
                if inp_x_stride < 0 || inp_x_stride % p.stride as isize != 0 {
                    continue;
                }
                let inp_x = (inp_x_stride / p.stride as isize) as usize;
                if inp_x >= p.l_in {
                    continue;
                }
                for src_c_idx in src_c_begin..src_c_end {
                    let src_idx = b_idx * s[0] + src_c_idx * s[1] + inp_x * s[2];
                    let k_idx =
                        src_c_idx * ks[0] + dst_c_local * ks[1] + k_x * ks[2];
                    d += input[src_idx].to_f64() * kernel[k_idx].to_f64();
                }
            }
            T::from_f64(d)
        })
        .collect::<Vec<_>>();
    Ok(T::to_cpu_storage_owned(out))
}

fn launch2d_t<T: WithDType>(
    input: &CpuStorage,
    input_l: &Layout,
    kernel: &CpuStorage,
    kernel_l: &Layout,
    p: &ParamsConvTranspose2D,
) -> Result<CpuStorage> {
    let input = T::cpu_storage_as_slice(input)?;
    let kernel = T::cpu_storage_as_slice(kernel)?;
    let input = &input[input_l.start_offset()..];
    let kernel = &kernel[kernel_l.start_offset()..];
    let s = input_l.stride();
    let ks = kernel_l.stride();
    let out_h = p.out_h();
    let out_w = p.out_w();
    let c_in_group = p.c_in / p.groups;
    let c_out_group = p.c_out / p.groups;
    let dst_el = p.b_size * p.c_out * out_h * out_w;

    let out = (0..dst_el)
        .into_par_iter()
        .map(|dst_i| {
            let spatial = out_h * out_w;
            let b_idx = dst_i / (p.c_out * spatial);
            let dst_c_idx = (dst_i / spatial) % p.c_out;
            let out_y = (dst_i / out_w) % out_h;
            let out_x = dst_i % out_w;
            let group = dst_c_idx / c_out_group;
            let dst_c_local = dst_c_idx % c_out_group;
            let src_c_begin = group * c_in_group;
            let src_c_end = src_c_begin + c_in_group;
            let mut d = 0f64;

            for k_x in 0..p.k_w {
                let inp_x_stride =
                    out_x as isize + p.padding as isize - (k_x * p.dilation) as isize;
                if inp_x_stride < 0 || inp_x_stride % p.stride as isize != 0 {
                    continue;
                }
                let inp_x = (inp_x_stride / p.stride as isize) as usize;
                if inp_x >= p.i_w {
                    continue;
                }
                for k_y in 0..p.k_h {
                    let inp_y_stride =
                        out_y as isize + p.padding as isize - (k_y * p.dilation) as isize;
                    if inp_y_stride < 0 || inp_y_stride % p.stride as isize != 0 {
                        continue;
                    }
                    let inp_y = (inp_y_stride / p.stride as isize) as usize;
                    if inp_y >= p.i_h {
                        continue;
                    }
                    for src_c_idx in src_c_begin..src_c_end {
                        let src_idx =
                            b_idx * s[0] + src_c_idx * s[1] + inp_y * s[2] + inp_x * s[3];
                        let k_idx = src_c_idx * ks[0]
                            + dst_c_local * ks[1]
                            + k_y * ks[2]
                            + k_x * ks[3];
                        d += input[src_idx].to_f64() * kernel[k_idx].to_f64();
                    }
                }
            }
            T::from_f64(d)
        })
        .collect::<Vec<_>>();
    Ok(T::to_cpu_storage_owned(out))
}

pub(super) fn launch1d(
    input: &CpuStorage,
    input_l: &Layout,
    kernel: &CpuStorage,
    kernel_l: &Layout,
    p: &ParamsConvTranspose1D,
) -> Result<CpuStorage> {
    match (input, kernel) {
        (CpuStorage::F32(_), CpuStorage::F32(_)) => {
            launch1d_t::<f32>(input, input_l, kernel, kernel_l, p)
        }
        (CpuStorage::F64(_), CpuStorage::F64(_)) => {
            launch1d_t::<f64>(input, input_l, kernel, kernel_l, p)
        }
        (CpuStorage::F16(_), CpuStorage::F16(_)) => {
            launch1d_t::<half::f16>(input, input_l, kernel, kernel_l, p)
        }
        (CpuStorage::BF16(_), CpuStorage::BF16(_)) => {
            launch1d_t::<half::bf16>(input, input_l, kernel, kernel_l, p)
        }
        _ => crate::bail!(
            "native grouped CPU conv_transpose1d does not support {:?}",
            input.dtype()
        ),
    }
}

pub(super) fn launch2d(
    input: &CpuStorage,
    input_l: &Layout,
    kernel: &CpuStorage,
    kernel_l: &Layout,
    p: &ParamsConvTranspose2D,
) -> Result<CpuStorage> {
    match (input, kernel) {
        (CpuStorage::F32(_), CpuStorage::F32(_)) => {
            launch2d_t::<f32>(input, input_l, kernel, kernel_l, p)
        }
        (CpuStorage::F64(_), CpuStorage::F64(_)) => {
            launch2d_t::<f64>(input, input_l, kernel, kernel_l, p)
        }
        (CpuStorage::F16(_), CpuStorage::F16(_)) => {
            launch2d_t::<half::f16>(input, input_l, kernel, kernel_l, p)
        }
        (CpuStorage::BF16(_), CpuStorage::BF16(_)) => {
            launch2d_t::<half::bf16>(input, input_l, kernel, kernel_l, p)
        }
        _ => crate::bail!(
            "native grouped CPU conv_transpose2d does not support {:?}",
            input.dtype()
        ),
    }
}
