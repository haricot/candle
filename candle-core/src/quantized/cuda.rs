use super::{GgmlDType, QStorage};
use crate::quantized::k_quants::GgmlType;
use crate::{backend::BackendDevice, cuda_backend::WrapErr};
use crate::{builder_arg as barg, CudaDevice, CudaStorage, Result};
use half::f16;

use cudarc::driver::{CudaSlice, CudaStream, CudaView, DevicePtr, PushKernelArg, SyncOnDrop};

#[derive(Clone, Debug)]
struct PaddedCudaSlice {
    inner: CudaSlice<u8>,
    len: usize,
}

#[derive(Clone, Debug)]
pub struct QCudaStorage {
    data: PaddedCudaSlice,
    dtype: GgmlDType,
    device: CudaDevice,
}

pub(crate) static FORCE_DMMV: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
pub const GGML_CUDA_MMV_X: usize = 32;
pub const GGML_CUDA_MMV_Y: usize = 1;
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_GET_ROWS_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

fn quantize_q8_1(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = kx_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;

    const CHUNK_SIZE: usize = 65535; // gridDim.y limit
    let func = dev.get_or_load_func("quantize_q8_1", &candle_kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        // --- calculate the number of rows for this chunk ---
        let remaining_rows = total_rows - rows_processed;
        // This is our gridDim.y, now <= 65535
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        // --- slice the source (f32) tensor by elements ---
        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        // --- slice the destination (u8) tensor by bytes ---
        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&dst_chunk);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

fn dequantize_f32(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f32", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f32", false, 32, nb),
        GgmlDType::Mxfp4 => ("dequantize_block_mxfp4_f32", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f32", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f32", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f32", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f32", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f32", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f32", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f32", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_f16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f16", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f16", false, 32, nb),
        GgmlDType::Mxfp4 => ("dequantize_block_mxfp4_f16", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f16", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f16", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f16", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f16", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f16", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f16", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f16", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn get_rows(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    hidden: usize,
    ids: &CudaView<u32>,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let (kernel_name, block_dim, block_num_y, can_stride_y) = match dtype {
        GgmlDType::F32 => (
            "get_rows_f32",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::F16 => (
            "get_rows_f16",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::BF16 => (
            "get_rows_bf16",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q4_0 => (
            "get_rows_q4_0",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q4_1 => (
            "get_rows_q4_1",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q5_0 => (
            "get_rows_q5_0",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q5_1 => (
            "get_rows_q5_1",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q8_0 => (
            "get_rows_q8_0",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q2K => ("get_rows_q2_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q3K => ("get_rows_q3_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q4K => ("get_rows_q4_K", 32, hidden / dtype.block_size(), false),
        GgmlDType::Q5K => ("get_rows_q5_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q6K => ("get_rows_q6_K", 64, hidden / dtype.block_size(), false),
        _ => crate::bail!("unsupported dtype for CUDA quantized embedding {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let ids_len = ids.len();
    let dst = unsafe { dev.alloc::<f32>(ids_len * hidden)? };
    if ids_len == 0 {
        return Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()));
    }
    if !can_stride_y && block_num_y > u16::MAX as usize {
        crate::bail!("quantized embedding hidden size {hidden} exceeds CUDA grid y limit")
    }
    let grid_y = if can_stride_y {
        block_num_y.min(u16::MAX as usize)
    } else {
        block_num_y
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (ids_len as u32, grid_y as u32, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let row_stride = hidden * dtype.type_size() / dtype.block_size();

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(ids);
    builder.arg(&dst);
    barg!(builder, hidden as i64, row_stride);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_mul_mat_vec(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "dequantize_mul_mat_vec_q4_0_cuda",
        GgmlDType::Q4_1 => "dequantize_mul_mat_vec_q4_1_cuda",
        GgmlDType::Q5_0 => "dequantize_mul_mat_vec_q5_0_cuda",
        GgmlDType::Q5_1 => "dequantize_mul_mat_vec_q5_1_cuda",
        GgmlDType::Q8_0 => "dequantize_mul_mat_vec_q8_0_cuda",
        GgmlDType::Q2K => "dequantize_mul_mat_vec_q2_k",
        GgmlDType::Q3K => "dequantize_mul_mat_vec_q3_k",
        GgmlDType::Q4K => "dequantize_mul_mat_vec_q4_k",
        GgmlDType::Q5K => "dequantize_mul_mat_vec_q5_k",
        GgmlDType::Q6K => "dequantize_mul_mat_vec_q6_k",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows)? };
    let block_num_y = ceil_div(nrows, GGML_CUDA_MMV_Y);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (block_num_y as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, GGML_CUDA_MMV_Y as u32, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(y);
    builder.arg(&dst);
    barg!(builder, ncols as i32, nrows as i32);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn mul_mat_vec_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if b_size == 0 || b_size > 8 {
        crate::bail!("only bsize between 1 and 8 are supported, got {b_size}")
    }
    // Start by quantizing y
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = dev.alloc_zeros::<u8>(y_size_in_bytes)?;
    quantize_q8_1(y, &mut y_q8_1, ncols, b_size, dev)?;

    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_cuda",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_cuda",
        GgmlDType::Mxfp4 => "mul_mat_vec_mxfp4_q8_1_cuda",
        GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1_cuda",
        GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1_cuda",
        GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1_cuda",
        GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1_cuda",
        GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1_cuda",
        GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_cuda",
        GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_cuda",
        GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1_cuda",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let kernel_name = format!("{kernel_name}{b_size}");
    let func = dev.get_or_load_func(&kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = dev.alloc_zeros::<f32>(nrows * b_size)?;
    // https://github.com/ggerganov/llama.cpp/blob/facb8b56f8fd3bb10a693bf0943ae9d69d0828ef/ggml-cuda/mmvq.cu#L98
    let (nblocks, nwarps) = match b_size {
        1 => (nrows as u32, 4),
        2..=4 => ((nrows as u32).div_ceil(2), 4),
        5..=8 => ((nrows as u32).div_ceil(2), 2),
        _ => crate::bail!("unexpected bsize {b_size}"),
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, 1, 1),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(&y_q8_1);
    builder.arg(&dst);
    barg!(
        builder,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
fn mul_mat_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        crate::bail!("unexpected lhs size {}, {x_rows} {x_cols}", data_elems)
    }
    if y.len() != y_rows * y_cols {
        crate::bail!("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if x_cols != y_rows {
        crate::bail!("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    let k = x_cols;
    // Start by quantizing y
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        k_padded * y_cols * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = dev.alloc_zeros::<u8>(y_size_in_bytes)?;
    quantize_q8_1(y, &mut y_q8_1, k, y_cols, dev)?;

    let (kernel_name, mmq_x, mmq_y, nwarps) = match dtype {
        GgmlDType::Q4_0 => ("mul_mat_q4_0", 64, 128, 4),
        GgmlDType::Q4_1 => ("mul_mat_q4_1", 64, 128, 4),
        GgmlDType::Mxfp4 => ("mul_mat_mxfp4", 1, 1, 4),
        GgmlDType::Q5_0 => ("mul_mat_q5_0", 128, 64, 4),
        GgmlDType::Q5_1 => ("mul_mat_q5_1", 128, 64, 4),
        GgmlDType::Q8_0 => ("mul_mat_q8_0", 128, 64, 4),
        GgmlDType::Q2K => ("mul_mat_q2_K", 64, 128, 4),
        GgmlDType::Q3K => ("mul_mat_q3_K", 128, 128, 4),
        GgmlDType::Q4K => ("mul_mat_q4_K", 64, 128, 4),
        GgmlDType::Q5K => ("mul_mat_q5_K", 64, 128, 4),
        GgmlDType::Q6K => ("mul_mat_q6_K", 64, 64, 4),
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = dev.alloc_zeros::<f32>(x_rows * y_cols)?;
    let cfg = if dtype == GgmlDType::Mxfp4 {
        if y_cols > u16::MAX as usize {
            crate::bail!("MXFP4 SM61 prefill currently supports at most {} rows, got {y_cols}", u16::MAX)
        }
        cudarc::driver::LaunchConfig {
            grid_dim: (x_rows as u32, y_cols as u32, 1),
            block_dim: (WARP_SIZE as u32, 4, 1),
            shared_mem_bytes: 0,
        }
    } else {
        cudarc::driver::LaunchConfig {
            grid_dim: (
                ceil_div(x_rows, mmq_y) as u32,
                ceil_div(y_cols, mmq_x) as u32,
                1,
            ),
            block_dim: (WARP_SIZE as u32, nwarps, 1),
            shared_mem_bytes: 0,
        }
    };

    let mut builder = func.builder();
    builder.arg(/* vx */ &data.inner);
    builder.arg(/* vy */ &y_q8_1);
    builder.arg(/* dst */ &dst);
    barg!(
        builder,
        /* ncols_x */ x_cols as i32,
        /* nrows_x */ x_rows as i32,
        /* ncols_y */ y_cols as i32,
        /* nrows_y */ k_padded as i32,
        /* nrows_dst */ x_rows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
fn indexed_moe_forward_fused_q8_1_input(
    weight: &CudaView<u8>,
    w_shape: &crate::Shape, //[num_experts, n, k]
    w_dtype: GgmlDType,
    input: &CudaSlice<f32>,
    in_shape: &crate::Shape, //[batch, topk or 1, k]
    ids: &CudaView<u32>,
    idx_shape: &crate::Shape, //[batch, topk]
    dev: &CudaDevice,
) -> Result<(CudaStorage, crate::Shape)> {
    let (_, n, k) = w_shape.dims3()?;
    let in_dims = in_shape.dims();
    let (batch, input_dim1, in_k) = match in_dims {
        [batch, in_k] => (*batch, 1usize, *in_k),
        [batch, input_dim1, in_k] => (*batch, *input_dim1, *in_k),
        _ => crate::bail!("indexed_moe_forward expects input rank 2 or 3, got shape {in_dims:?}"),
    };
    if in_k != k {
        crate::bail!("indexed_moe_forward expects input k={k}, got {in_k}")
    }

    let topk = idx_shape.dims()[1];
    assert!(batch == idx_shape.dims()[0], "batch dim not match!");

    // Quantize input into q8_1.
    let total_rows = batch * input_dim1;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = k_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;
    let y_size_in_bytes = total_rows * dst_row_size_bytes;
    let mut input_quant = dev.alloc_zeros::<u8>(y_size_in_bytes)?;

    let input_view = input.slice(0..);
    quantize_q8_1(&input_view, &mut input_quant, k, total_rows, dev)?;

    // output buffer
    let outsize = batch * topk * n;
    let out = dev.alloc_zeros::<f32>(outsize)?;

    let kernel_name = match w_dtype {
        GgmlDType::Q2K => "indexed_moe_forward_q2k_q8_1",
        GgmlDType::Q3K => "indexed_moe_forward_q3k_q8_1",
        GgmlDType::Q4K => "indexed_moe_forward_q4k_q8_1",
        GgmlDType::Q5K => "indexed_moe_forward_q5k_q8_1",
        GgmlDType::Q6K => "indexed_moe_forward_q6k_q8_1",
        GgmlDType::Q8_0 => "indexed_moe_forward_q8_0_q8_1",
        GgmlDType::Mxfp4 => "indexed_moe_forward_mxfp4_q8_1",
        _ => crate::bail!("unsupported dtype for indexed_moe_forward {w_dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let (nblocks, nwarps) = (n as u32, 4);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, batch as u32, topk as u32),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(weight);
    builder.arg(&input_quant);
    builder.arg(ids);
    builder.arg(&out);

    barg!(
        builder,
        n as i32,
        k as i32,
        batch as i32,
        topk as i32,
        k_padded as i32,
        input_dim1 as i32
    );
    unsafe { builder.launch(cfg) }.w()?;

    Ok((
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        (batch, topk, n).into(),
    ))
}

impl QCudaStorage {
    pub fn indexed_moe_forward(
        &self,
        self_shape: &crate::Shape, //[num_experts, n, k]
        input: &CudaStorage,       //[batch, topk or 1, k]
        input_l: &crate::Layout,
        ids: &CudaStorage, //[batch, topk]
        ids_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        if matches!(
            self.dtype(),
            GgmlDType::Q8_0
                | GgmlDType::Mxfp4
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        ) {
            let input_storage = input.as_cuda_slice::<f32>()?;
            let ids_storage = ids.as_cuda_slice::<u32>()?;
            indexed_moe_forward_fused_q8_1_input(
                &self.data.inner.slice(0..),
                self_shape, //[num_experts, n, k]
                self.dtype(),
                input_storage,
                input_l.shape(), //[batch, topk or 1, k]
                &ids_storage.slice(0..),
                ids_l.shape(), //[batch, topk]
                &self.device,
            )
        } else {
            crate::bail!(
                "The given quantized dtype {:?} is not supported for indexed_moe_forward!",
                self.dtype()
            );
        }
    }

    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes)?;
        Ok(QCudaStorage {
            data: PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            },
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
        fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) {
            let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
            let vec = slice.to_vec();
            T::to_float(&vec, dst)
        }

        let fast_kernel = matches!(
            self.dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Mxfp4
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8K
        );
        if fast_kernel {
            return dequantize_f32(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.

        let buffer = self
            .device
            .clone_dtoh(&self.data.inner.slice(..self.data.len))?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => deq::<f32>(&buffer, block_len, &mut out),
            GgmlDType::F16 => deq::<half::f16>(&buffer, block_len, &mut out),
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, block_len, &mut out),
            GgmlDType::Q4_0 => deq::<crate::quantized::BlockQ4_0>(&buffer, block_len, &mut out),
            GgmlDType::Q4_1 => deq::<crate::quantized::BlockQ4_1>(&buffer, block_len, &mut out),
            GgmlDType::Mxfp4 => deq::<crate::quantized::BlockMxfp4>(&buffer, block_len, &mut out),
            GgmlDType::Q5_0 => deq::<crate::quantized::BlockQ5_0>(&buffer, block_len, &mut out),
            GgmlDType::Q5_1 => deq::<crate::quantized::BlockQ5_1>(&buffer, block_len, &mut out),
            GgmlDType::Q8_0 => deq::<crate::quantized::BlockQ8_0>(&buffer, block_len, &mut out),
            GgmlDType::Q8_1 => deq::<crate::quantized::BlockQ8_1>(&buffer, block_len, &mut out),
            GgmlDType::Q2K => deq::<crate::quantized::BlockQ2K>(&buffer, block_len, &mut out),
            GgmlDType::Q3K => deq::<crate::quantized::BlockQ3K>(&buffer, block_len, &mut out),
            GgmlDType::Q4K => deq::<crate::quantized::BlockQ4K>(&buffer, block_len, &mut out),
            GgmlDType::Q5K => deq::<crate::quantized::BlockQ5K>(&buffer, block_len, &mut out),
            GgmlDType::Q6K => deq::<crate::quantized::BlockQ6K>(&buffer, block_len, &mut out),
            GgmlDType::Q8K => deq::<crate::quantized::BlockQ8K>(&buffer, block_len, &mut out),
        }

        self.device
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<CudaStorage> {
        dequantize_f16(&self.data, self.dtype, elem_count, self.device())
    }

    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &CudaStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len
    }

    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &CudaStorage,
        ids_l: &crate::Layout,
    ) -> Result<CudaStorage> {
        if !ids_l.is_contiguous() {
            crate::bail!("quantized embedding requires contiguous ids")
        }
        if !hidden.is_multiple_of(self.dtype.block_size()) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                self.dtype.block_size()
            )
        }
        let expected_size = rows * hidden * self.dtype.type_size() / self.dtype.block_size();
        if self.storage_size_in_bytes() != expected_size {
            crate::bail!(
                "quantized tensor has {} bytes, expected {expected_size}",
                self.storage_size_in_bytes()
            )
        }
        let ids = ids.as_cuda_slice::<u32>()?;
        let ids = match ids_l.contiguous_offsets() {
            Some((o1, o2)) => ids.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous {
                op: "quantized-embedding",
            }
            .bt())?,
        };
        get_rows(&self.data, self.dtype, hidden, &ids, self.device())
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        // Optimized MMVQ and MMQ paths (support most paths: BF16/F16/F32, batch 1-8, all quant types, reuses per-device workspace).
        if !FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(result) = super::fast_mmvq::try_fwd(self, self_shape, storage, layout)? {
                return Ok(result);
            }
            if let Some(result) = super::fast_mmq::try_fwd(self, self_shape, storage, layout)? {
                return Ok(result);
            }
        }

        // Fallback
        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            8
        };
        let use_vec_kernel = match layout.shape().dims() {
            [b, m, _k] => b * m <= max_bm,
            [b, _k] => *b <= max_bm,
            _ => false,
        };
        if use_vec_kernel {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.data.len];
        self.device
            .memcpy_dtoh(&self.data.inner.slice(..self.data.len), &mut out)?;
        Ok(out)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Ok(self.data.inner.device_ptr(self.data.inner.stream()).0 as *const u8)
    }

    pub fn device_ptr_with_guard<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> Result<(*const u8, SyncOnDrop<'a>)> {
        let (ptr, guard) = self.data.inner.device_ptr(stream);
        Ok((ptr as *const u8, guard))
    }
}

impl QCudaStorage {
    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let rhs = rhs.as_cuda_slice::<f32>()?;
        let rhs = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt())?,
        };
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            dequantize_mul_mat_vec(&self.data, &rhs, self.dtype, ncols, nrows, self.device())?
        } else {
            mul_mat_vec_via_q8_1(
                &self.data,
                &rhs,
                self.dtype,
                ncols,
                nrows,
                b_size,
                self.device(),
            )?
        };
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }

    fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            let data_f32 = self.dequantize(n * k)?;
            let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
            storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?
        } else {
            let storage = storage.as_cuda_slice::<f32>()?;
            let storage = match layout.contiguous_offsets() {
                Some((o1, o2)) => storage.slice(o1..o2),
                None => Err(crate::Error::RequiresContiguous {
                    op: "quantized-matmul",
                }
                .bt())?,
            };
            mul_mat_via_q8_1(
                &self.data,
                &storage,
                self.dtype,
                /* x_rows */ n,
                /* x_cols */ k,
                /* y_rows */ k,
                /* y_cols */ b * m,
                self.device(),
            )?
        };
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = device.alloc_zeros::<u8>(padded_len)?;
    device.memcpy_htod(data, &mut inner.slice_mut(..data.len()))?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
    }))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn cuda_quantize_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let el = 256;
        let el_padded = pad(el, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
        let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        quantize_q8_1(&y.as_view(), &mut y_q8_1, el, 1, &dev)?;
        Ok(())
    }

    #[test]
    fn cuda_mmv_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_vec_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            /* b_size */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
        // Q8 means 1/256 precision.
        assert_eq!(vs[0], 5561664.5);

        let cuda_storage = dequantize_mul_mat_vec(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0], 5561851.0);
        Ok(())
    }

    #[test]
    fn cuda_mm_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ 4,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ 4,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;

        /*
           x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
           x @ x.t() / 16
        tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
                [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
                [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
                [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
                */
        assert_eq!(vs.len(), 16);
        assert_eq!(vs[0], 347604.0);
        assert_eq!(vs[1], 888153.06);
        assert_eq!(vs[4], 869780.7);
        assert_eq!(vs[5], 2483145.0);
        assert_eq!(vs[11], 9407368.0);
        assert_eq!(vs[14], 9470856.0);
        assert_eq!(vs[15], 13138824.0);
        Ok(())
    }

    #[test]
    fn cuda_nvfp4_legacy_lut_probe() -> Result<()> {
        let dev = CudaDevice::new(0)?;

        // Two NVFP4 blocks, 16 E2M1 values each.
        // Each byte holds two consecutive E2M1 nibbles.
        let packed = vec![
            0x10u8, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
        ];
        // E4M3FN: 0x38 = 1.0, 0x40 = 2.0.
        let scales = vec![0x38u8, 0x40];
        let global_scale = 1.25f32;

        let packed_d = dev.clone_htod(&packed)?;
        let scales_d = dev.clone_htod(&scales)?;
        let out = unsafe { dev.alloc::<f32>(32)? };

        let func =
            dev.get_or_load_func("nvfp4_experiment_dequant_f32", &candle_kernels::QUANTIZED)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&packed_d);
        builder.arg(&scales_d);
        barg!(builder, global_scale);
        builder.arg(&out);
        barg!(builder, 2i32);
        unsafe { builder.launch(cfg) }.w()?;

        let got = dev.clone_dtoh(&out.as_view())?;
        let e2m1 = [
            0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let mut expected = Vec::with_capacity(32);
        for (block, scale) in [(0usize, 1.0f32), (1usize, 2.0f32)] {
            for &byte in &packed[block * 8..block * 8 + 8] {
                expected.push(e2m1[(byte & 0x0f) as usize] * scale * global_scale);
                expected.push(e2m1[(byte >> 4) as usize] * scale * global_scale);
            }
        }

        assert_eq!(got, expected);
        Ok(())
    }

    fn nvfp4_test_reference(
        packed: &[u8],
        scales: &[u8],
        global_scale: f32,
        rows: usize,
        k: usize,
        activations: &[f32],
        batch: usize,
    ) -> Vec<f32> {
        const E2M1: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let blocks_per_row = k / 16;
        let packed_row_bytes = blocks_per_row * 8;
        let mut out = vec![0f32; rows * batch];

        for b in 0..batch {
            for row in 0..rows {
                let mut acc = 0f32;
                for block in 0..blocks_per_row {
                    let scale = e4m3fn_to_f32_test(scales[row * blocks_per_row + block]) * global_scale;
                    let p = &packed[
                        row * packed_row_bytes + block * 8
                            .. row * packed_row_bytes + block * 8 + 8
                    ];
                    for i in 0..8 {
                        let byte = p[i];
                        let w0 = E2M1[(byte & 0x0f) as usize] * scale;
                        let w1 = E2M1[(byte >> 4) as usize] * scale;
                        let col = block * 16 + 2 * i;
                        acc += w0 * activations[b * k + col];
                        acc += w1 * activations[b * k + col + 1];
                    }
                }
                out[b * rows + row] = acc;
            }
        }
        out
    }

    fn e4m3fn_to_f32_test(x: u8) -> f32 {
        let sign = (x >> 7) & 1;
        let exp = (x >> 3) & 0x0f;
        let mant = x & 0x07;
        let value = if exp == 0 {
            if mant == 0 { 0.0 } else { mant as f32 * 2f32.powi(-9) }
        } else if exp == 0x0f && mant == 0x07 {
            f32::NAN
        } else {
            (1.0 + mant as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
        };
        if sign != 0 { -value } else { value }
    }

    fn f32_to_e4m3fn_nearest_test(x: f32) -> u8 {
        if x.is_nan() {
            return 0x7f;
        }
        if x == 0.0 {
            return if x.is_sign_negative() { 0x80 } else { 0x00 };
        }

        let mut best = 0u8;
        let mut best_err = f32::INFINITY;
        for raw in 0u16..=255 {
            let raw = raw as u8;
            if raw & 0x7f == 0x7f {
                continue;
            }
            let value = e4m3fn_to_f32_test(raw);
            if !value.is_finite() {
                continue;
            }
            let err = (value - x).abs();
            if err < best_err || (err == best_err && raw < best) {
                best = raw;
                best_err = err;
            }
        }
        best
    }

    fn nearest_e2m1_test(x: f32) -> u8 {
        const E2M1: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let mut best = 0usize;
        let mut best_err = f32::INFINITY;
        for (i, &value) in E2M1.iter().enumerate() {
            let err = (value - x).abs();
            if err < best_err {
                best = i;
                best_err = err;
            }
        }
        best as u8
    }

    fn nvfp4_quantize_experimental_test(xs: &[f32]) -> (Vec<u8>, Vec<u8>, f32) {
        assert!(xs.len().is_multiple_of(16));

        let amax = xs.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let global_scale = if amax == 0.0 {
            1.0
        } else {
            amax / (6.0 * 448.0)
        };

        let mut packed = vec![0u8; xs.len() / 2];
        let mut scales = vec![0u8; xs.len() / 16];

        for (block_idx, block) in xs.chunks_exact(16).enumerate() {
            let block_amax = block.iter().fold(0f32, |m, &x| m.max(x.abs()));
            let desired_scale = if block_amax == 0.0 {
                0.0
            } else {
                (block_amax / (6.0 * global_scale)).min(448.0)
            };
            let scale_raw = f32_to_e4m3fn_nearest_test(desired_scale);
            scales[block_idx] = scale_raw;
            let scale = e4m3fn_to_f32_test(scale_raw) * global_scale;

            for i in 0..8 {
                let q0 = if scale == 0.0 {
                    0
                } else {
                    nearest_e2m1_test(block[2 * i] / scale)
                };
                let q1 = if scale == 0.0 {
                    0
                } else {
                    nearest_e2m1_test(block[2 * i + 1] / scale)
                };
                packed[block_idx * 8 + i] = q0 | (q1 << 4);
            }
        }

        (packed, scales, global_scale)
    }


    fn assert_nvfp4_close(expected: &[f32], got: &[f32], label: &str) {
        assert_eq!(expected.len(), got.len(), "{label}: length mismatch");
        let mut max_abs = 0f32;
        let mut mean_abs = 0f32;
        let mut dot = 0f64;
        let mut nr = 0f64;
        let mut ng = 0f64;
        for (&a, &b) in expected.iter().zip(got.iter()) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            mean_abs += d;
            dot += a as f64 * b as f64;
            nr += a as f64 * a as f64;
            ng += b as f64 * b as f64;
        }
        mean_abs /= got.len() as f32;
        let cosine = dot / (nr.sqrt() * ng.sqrt()).max(f64::MIN_POSITIVE);
        println!(
            "NVFP4_SM61_PARITY label={label} max_abs={max_abs:.6} mean_abs={mean_abs:.6} cosine={cosine:.8}"
        );
        assert!(max_abs < 0.08, "{label}: max_abs={max_abs}");
        assert!(mean_abs < 0.02, "{label}: mean_abs={mean_abs}");
        assert!(cosine >= 0.99999, "{label}: cosine={cosine}");
    }

    fn nvfp4_metrics(reference: &[f32], got: &[f32]) -> (f32, f32, f64) {
        assert_eq!(reference.len(), got.len());
        let mut max_abs = 0f32;
        let mut mean_abs = 0f32;
        let mut dot = 0f64;
        let mut nr = 0f64;
        let mut ng = 0f64;
        for (&a, &b) in reference.iter().zip(got.iter()) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            mean_abs += d;
            dot += a as f64 * b as f64;
            nr += a as f64 * a as f64;
            ng += b as f64 * b as f64;
        }
        mean_abs /= got.len() as f32;
        let cosine = dot / (nr.sqrt() * ng.sqrt()).max(f64::MIN_POSITIVE);
        (max_abs, mean_abs, cosine)
    }

    fn dequantize_cuda_q8_1_reference(
        bytes: &[u8],
        rows: usize,
        k: usize,
        k_padded: usize,
    ) -> Vec<f32> {
        let blocks_per_padded_row = k_padded / 32;
        let blocks_per_live_row = k / 32;
        let row_bytes = blocks_per_padded_row * 36;
        assert_eq!(bytes.len(), rows * row_bytes);

        let mut out = vec![0f32; rows * k];
        for row in 0..rows {
            for block in 0..blocks_per_live_row {
                let off = row * row_bytes + block * 36;
                let d_bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                let d = f16::from_bits(d_bits).to_f32();
                for j in 0..32 {
                    let q = bytes[off + 4 + j] as i8;
                    out[row * k + block * 32 + j] = d * q as f32;
                }
            }
        }
        out
    }


    #[test]
    fn cuda_nvfp4_sm61_compute_parity() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let rows = 37usize;
        let k = 256usize;
        let global_scale = 0.125f32;
        let blocks_per_row = k / 16;

        // Deterministic valid NVFP4 payload. E4M3FN scales cycle through
        // 0.5, 1.0 and 2.0; packed nibbles cover all E2M1 codes.
        let mut packed = vec![0u8; rows * blocks_per_row * 8];
        for (i, byte) in packed.iter_mut().enumerate() {
            let lo = (i % 16) as u8;
            let hi = ((i * 7 + 3) % 16) as u8;
            *byte = lo | (hi << 4);
        }
        let scale_codes = [0x30u8, 0x38, 0x40];
        let scales = (0..rows * blocks_per_row)
            .map(|i| scale_codes[i % scale_codes.len()])
            .collect::<Vec<_>>();

        let packed_d = dev.clone_htod(&packed)?;
        let scales_d = dev.clone_htod(&scales)?;

        for batch in 1usize..=8 {
            let activations = (0..batch * k)
                .map(|i| ((i as f32) * 0.013).sin() * 0.7 + ((i as f32) * 0.003).cos() * 0.2)
                .collect::<Vec<_>>();
            let expected_f32 = nvfp4_test_reference(
                &packed,
                &scales,
                global_scale,
                rows,
                k,
                &activations,
                batch,
            );

            let activations_d = dev.clone_htod(&activations)?;
            let k_padded = pad(k, MATRIX_ROW_PADDING);
            let y_size_in_bytes =
                batch * k_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
            let mut q8 = dev.alloc_zeros::<u8>(y_size_in_bytes)?;
            quantize_q8_1(&activations_d.as_view(), &mut q8, k, batch, &dev)?;

            // Kernel parity must compare against the exact Q8_1 activations
            // consumed by CUDA, not against the original F32 activations.
            let q8_host = dev.clone_dtoh(&q8.as_view())?;
            let activations_q8 =
                dequantize_cuda_q8_1_reference(&q8_host, batch, k, k_padded);
            let expected_kernel = nvfp4_test_reference(
                &packed,
                &scales,
                global_scale,
                rows,
                k,
                &activations_q8,
                batch,
            );

            let out = dev.alloc_zeros::<f32>(rows * batch)?;
            let kernel = format!("nvfp4_mat_vec_q8_1_cuda{batch}");
            let func = dev.get_or_load_func(&kernel, &candle_kernels::QUANTIZED)?;
            let nwarps = if batch <= 4 { 4 } else { 2 };
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (WARP_SIZE as u32, nwarps, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = func.builder();
            builder.arg(&packed_d);
            builder.arg(&scales_d);
            barg!(builder, global_scale);
            builder.arg(&q8);
            builder.arg(&out);
            barg!(
                builder,
                k as i32,
                rows as i32,
                k_padded as i32,
                rows as i32
            );
            unsafe { builder.launch(cfg) }.w()?;
            let got = dev.clone_dtoh(&out.as_view())?;

            assert_nvfp4_close(
                &expected_kernel,
                &got,
                &format!("kernel_decode_batch_{batch}"),
            );
            let e2e = nvfp4_metrics(&expected_f32, &got);
            println!(
                "NVFP4_A8_E2E label=decode_batch_{batch} max_abs={:.6} mean_abs={:.6} cosine={:.8}",
                e2e.0, e2e.1, e2e.2
            );
            assert!(e2e.2 >= 0.999, "decode_batch_{batch}: Q8_1 end-to-end cosine={}", e2e.2);
        }

        // Dynamic-batch prefill uses the same DP4A primitive.
        let batch = 17usize;
        let activations = (0..batch * k)
            .map(|i| ((i as f32) * 0.009).cos() * 0.65 - 0.05)
            .collect::<Vec<_>>();
        let expected_f32 = nvfp4_test_reference(
            &packed,
            &scales,
            global_scale,
            rows,
            k,
            &activations,
            batch,
        );
        let activations_d = dev.clone_htod(&activations)?;
        let k_padded = pad(k, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            batch * k_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut q8 = dev.alloc_zeros::<u8>(y_size_in_bytes)?;
        quantize_q8_1(&activations_d.as_view(), &mut q8, k, batch, &dev)?;

        let q8_host = dev.clone_dtoh(&q8.as_view())?;
        let activations_q8 =
            dequantize_cuda_q8_1_reference(&q8_host, batch, k, k_padded);
        let expected_kernel = nvfp4_test_reference(
            &packed,
            &scales,
            global_scale,
            rows,
            k,
            &activations_q8,
            batch,
        );

        let out = dev.alloc_zeros::<f32>(rows * batch)?;
        let func = dev.get_or_load_func("nvfp4_mat_mul_q8_1", &candle_kernels::QUANTIZED)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (rows as u32, batch as u32, 1),
            block_dim: (WARP_SIZE as u32, 4, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&packed_d);
        builder.arg(&scales_d);
        barg!(builder, global_scale);
        builder.arg(&q8);
        builder.arg(&out);
        barg!(
            builder,
            k as i32,
            rows as i32,
            batch as i32,
            k_padded as i32,
            rows as i32
        );
        unsafe { builder.launch(cfg) }.w()?;
        let got = dev.clone_dtoh(&out.as_view())?;
        assert_nvfp4_close(&expected_kernel, &got, "kernel_prefill_batch_17");
        let e2e = nvfp4_metrics(&expected_f32, &got);
        println!(
            "NVFP4_A8_E2E label=prefill_batch_17 max_abs={:.6} mean_abs={:.6} cosine={:.8}",
            e2e.0, e2e.1, e2e.2
        );
        assert!(e2e.2 >= 0.999, "prefill_batch_17: Q8_1 end-to-end cosine={}", e2e.2);

        // Indexed MoE correctness with the same packed NVFP4 representation.
        let num_experts = 4usize;
        let moe_rows = 19usize;
        let topk = 2usize;
        let moe_batch = 4usize;
        let moe_blocks_per_row = k / 16;
        let mut moe_packed =
            vec![0u8; num_experts * moe_rows * moe_blocks_per_row * 8];
        for (i, byte) in moe_packed.iter_mut().enumerate() {
            let lo = ((i * 3 + 1) % 16) as u8;
            let hi = ((i * 5 + 7) % 16) as u8;
            *byte = lo | (hi << 4);
        }
        let moe_scales = (0..num_experts * moe_rows * moe_blocks_per_row)
            .map(|i| scale_codes[(i * 7) % scale_codes.len()])
            .collect::<Vec<_>>();
        let moe_inputs = (0..moe_batch * k)
            .map(|i| ((i as f32) * 0.007).sin() * 0.55 + 0.08)
            .collect::<Vec<_>>();
        let ids = vec![0u32, 1, 2, 3, 3, 2, 1, 0];

        let moe_packed_d = dev.clone_htod(&moe_packed)?;
        let moe_scales_d = dev.clone_htod(&moe_scales)?;
        let moe_inputs_d = dev.clone_htod(&moe_inputs)?;
        let ids_d = dev.clone_htod(&ids)?;

        let moe_q8_size =
            moe_batch * k_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut moe_q8 = dev.alloc_zeros::<u8>(moe_q8_size)?;
        quantize_q8_1(&moe_inputs_d.as_view(), &mut moe_q8, k, moe_batch, &dev)?;
        let moe_q8_host = dev.clone_dtoh(&moe_q8.as_view())?;
        let moe_inputs_q8 =
            dequantize_cuda_q8_1_reference(&moe_q8_host, moe_batch, k, k_padded);

        let packed_row_bytes = moe_blocks_per_row * 8;
        let mut expected_moe = vec![0f32; moe_batch * topk * moe_rows];
        for b in 0..moe_batch {
            for t in 0..topk {
                let expert = ids[b * topk + t] as usize;
                let expert_packed_start = expert * moe_rows * packed_row_bytes;
                let expert_scale_start = expert * moe_rows * moe_blocks_per_row;
                let expert_expected = nvfp4_test_reference(
                    &moe_packed[
                        expert_packed_start
                            .. expert_packed_start + moe_rows * packed_row_bytes
                    ],
                    &moe_scales[
                        expert_scale_start
                            .. expert_scale_start + moe_rows * moe_blocks_per_row
                    ],
                    global_scale,
                    moe_rows,
                    k,
                    &moe_inputs_q8[b * k..(b + 1) * k],
                    1,
                );
                for row in 0..moe_rows {
                    expected_moe[(b * topk + t) * moe_rows + row] = expert_expected[row];
                }
            }
        }

        let moe_out = dev.alloc_zeros::<f32>(moe_batch * topk * moe_rows)?;
        let func = dev.get_or_load_func("nvfp4_indexed_moe_q8_1", &candle_kernels::QUANTIZED)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (moe_rows as u32, moe_batch as u32, topk as u32),
            block_dim: (WARP_SIZE as u32, 4, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&moe_packed_d);
        builder.arg(&moe_scales_d);
        barg!(builder, global_scale);
        builder.arg(&moe_q8);
        builder.arg(&ids_d);
        builder.arg(&moe_out);
        barg!(
            builder,
            moe_rows as i32,
            k as i32,
            moe_batch as i32,
            topk as i32,
            k_padded as i32
        );
        unsafe { builder.launch(cfg) }.w()?;
        let got_moe = dev.clone_dtoh(&moe_out.as_view())?;
        assert_nvfp4_close(&expected_moe, &got_moe, "kernel_indexed_moe");

        Ok(())
    }



    #[test]
    #[ignore = "release benchmark: NVFP4 4096x4096 on the same workload as mxfp4_sm61_benchmark_gate"]
    fn cuda_nvfp4_sm61_benchmark_4096() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (n, k) = (4096usize, 4096usize);
        let runs = 10usize;

        println!(
            "NVFP4_BENCH_CONFIG profile={} n={n} k={k}",
            if cfg!(debug_assertions) { "debug" } else { "release" }
        );

        let weights = (0..n * k)
            .map(|i| ((i as f32) * 0.0013).sin() * 0.75 + ((i as f32) * 0.0007).cos() * 0.25)
            .collect::<Vec<_>>();
        let activations = (0..k)
            .map(|i| ((i as f32) * 0.017).cos() * 0.5 + 0.1)
            .collect::<Vec<_>>();

        let (packed, scales, global_scale) = nvfp4_quantize_experimental_test(&weights);
        let bytes = packed.len() + scales.len() + std::mem::size_of::<f32>();
        let bits_per_weight = bytes as f64 * 8.0 / (n * k) as f64;

        let quant_reference =
            nvfp4_test_reference(&packed, &scales, global_scale, n, k, &activations, 1);

        let packed_d = dev.clone_htod(&packed)?;
        let scales_d = dev.clone_htod(&scales)?;
        let activations_d = dev.clone_htod(&activations)?;
        let k_padded = pad(k, MATRIX_ROW_PADDING);
        let q8_size =
            k_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();

        // Stage timings isolate the part that shared-A8 can actually
        // amortize from the NVFP4 weight/decode kernel itself.
        let stream = dev.cuda_stream();

        let mut q8_stage = dev.alloc_zeros::<u8>(q8_size)?;
        for _ in 0..3 {
            quantize_q8_1(&activations_d.as_view(), &mut q8_stage, k, 1, &dev)?;
        }
        stream.synchronize().w()?;
        let q8_start = std::time::Instant::now();
        for _ in 0..runs {
            quantize_q8_1(&activations_d.as_view(), &mut q8_stage, k, 1, &dev)?;
            stream.synchronize().w()?;
        }
        let q8_latency_us = q8_start.elapsed().as_secs_f64() * 1e6 / runs as f64;

        let kernel_stage_out = dev.alloc_zeros::<f32>(n)?;
        let kernel_stage_func =
            dev.get_or_load_func("nvfp4_mat_vec_q8_1_cuda1", &candle_kernels::QUANTIZED)?;
        let kernel_stage_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (WARP_SIZE as u32, 4, 1),
            shared_mem_bytes: 0,
        };

        for _ in 0..3 {
            let mut builder = kernel_stage_func.builder();
            builder.arg(&packed_d);
            builder.arg(&scales_d);
            barg!(builder, global_scale);
            builder.arg(&q8_stage);
            builder.arg(&kernel_stage_out);
            barg!(
                builder,
                k as i32,
                n as i32,
                k_padded as i32,
                n as i32
            );
            unsafe { builder.launch(kernel_stage_cfg) }.w()?;
        }
        stream.synchronize().w()?;
        let kernel_start = std::time::Instant::now();
        for _ in 0..runs {
            let mut builder = kernel_stage_func.builder();
            builder.arg(&packed_d);
            builder.arg(&scales_d);
            barg!(builder, global_scale);
            builder.arg(&q8_stage);
            builder.arg(&kernel_stage_out);
            barg!(
                builder,
                k as i32,
                n as i32,
                k_padded as i32,
                n as i32
            );
            unsafe { builder.launch(kernel_stage_cfg) }.w()?;
            stream.synchronize().w()?;
        }
        let kernel_latency_us =
            kernel_start.elapsed().as_secs_f64() * 1e6 / runs as f64;

        let run_once = || -> Result<Vec<f32>> {
            // Match Candle's quantized matvec path: activation quantization
            // and output allocation are part of the measured operation.
            let mut q8 = dev.alloc_zeros::<u8>(q8_size)?;
            quantize_q8_1(&activations_d.as_view(), &mut q8, k, 1, &dev)?;

            let out = dev.alloc_zeros::<f32>(n)?;
            let func =
                dev.get_or_load_func("nvfp4_mat_vec_q8_1_cuda1", &candle_kernels::QUANTIZED)?;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (n as u32, 1, 1),
                block_dim: (WARP_SIZE as u32, 4, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = func.builder();
            builder.arg(&packed_d);
            builder.arg(&scales_d);
            barg!(builder, global_scale);
            builder.arg(&q8);
            builder.arg(&out);
            barg!(
                builder,
                k as i32,
                n as i32,
                k_padded as i32,
                n as i32
            );
            unsafe { builder.launch(cfg) }.w()?;
            dev.clone_dtoh(&out.as_view()).map_err(Into::into)
        };

        for _ in 0..3 {
            std::hint::black_box(run_once()?);
        }
        let start = std::time::Instant::now();
        let mut got = Vec::new();
        for _ in 0..runs {
            got = run_once()?;
            std::hint::black_box(&got);
        }
        let latency_us = start.elapsed().as_secs_f64() * 1e6 / runs as f64;

        // Isolate kernel parity using the exact Q8_1 activation consumed by CUDA.
        let mut q8 = dev.alloc_zeros::<u8>(q8_size)?;
        quantize_q8_1(&activations_d.as_view(), &mut q8, k, 1, &dev)?;
        let q8_host = dev.clone_dtoh(&q8.as_view())?;
        let activations_q8 = dequantize_cuda_q8_1_reference(&q8_host, 1, k, k_padded);
        let kernel_reference =
            nvfp4_test_reference(&packed, &scales, global_scale, n, k, &activations_q8, 1);

        let kernel_metrics = nvfp4_metrics(&kernel_reference, &got);
        let a8_metrics = nvfp4_metrics(&quant_reference, &got);

        // F32 baseline output on CPU, matching v0.5's quality reference.
        let mut f32_reference = vec![0f32; n];
        for row in 0..n {
            let mut acc = 0f32;
            let w = &weights[row * k..(row + 1) * k];
            for col in 0..k {
                acc += w[col] * activations[col];
            }
            f32_reference[row] = acc;
        }
        let e2e_metrics = nvfp4_metrics(&f32_reference, &got);
        let quant_metrics = nvfp4_metrics(&f32_reference, &quant_reference);

        let stage_overhead_us = latency_us - q8_latency_us - kernel_latency_us;
        println!(
            "NVFP4_BENCH_STAGES q8_quantize_us={q8_latency_us:.3} kernel_us={kernel_latency_us:.3} overhead_us={stage_overhead_us:.3} q8_fraction={:.4} kernel_fraction={:.4} kernel_effective_gbps={:.3}",
            q8_latency_us / latency_us,
            kernel_latency_us / latency_us,
            bytes as f64 / (kernel_latency_us * 1000.0)
        );
        println!(
            "NVFP4_QUANT_QUALITY bits_per_weight={bits_per_weight:.4} max_abs={:.6} mean_abs={:.6} cosine={:.8} global_scale={global_scale:.9}",
            quant_metrics.0,
            quant_metrics.1,
            quant_metrics.2
        );
        println!(
            "NVFP4_KERNEL_PARITY max_abs={:.6} mean_abs={:.6} cosine={:.8}",
            kernel_metrics.0,
            kernel_metrics.1,
            kernel_metrics.2
        );
        println!(
            "NVFP4_A8_QUALITY max_abs={:.6} mean_abs={:.6} cosine={:.8}",
            a8_metrics.0,
            a8_metrics.1,
            a8_metrics.2
        );
        println!(
            "NVFP4_BENCH_RESULT bytes={bytes} bits_per_weight={bits_per_weight:.4} latency_us={latency_us:.3} effective_gbps={:.3} e2e_max_abs={:.6} e2e_mean_abs={:.6} e2e_cosine={:.8}",
            bytes as f64 / (latency_us * 1000.0),
            e2e_metrics.0,
            e2e_metrics.1,
            e2e_metrics.2
        );

        assert!(bits_per_weight > 4.49 && bits_per_weight < 4.51);
        assert!(kernel_metrics.2 >= 0.99999, "NVFP4 kernel cosine={}", kernel_metrics.2);
        assert!(kernel_metrics.1 < 0.02, "NVFP4 kernel mean_abs={}", kernel_metrics.1);
        assert!(a8_metrics.2 >= 0.999, "NVFP4 A8 cosine={}", a8_metrics.2);
        assert!(e2e_metrics.2.is_finite());

        Ok(())
    }

    // The following test used to fail under compute-sanitizer until #2526.
    #[test]
    fn cuda_mm_q8_1_pad() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (x_rows, ncols, y_cols) = (4, 16, 2048);
        let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ x_rows,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ y_cols,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let _vs = dev.clone_dtoh(&vs.as_view())?;
        Ok(())
    }
}
