//! Explicit software NVFP4 weights and CUDA execution for legacy GPUs.
//!
//! NVFP4 is **not** a GGML/GGUF dtype: each 16-element E2M1 weight block
//! has a separate E4M3FN scale, and the tensor has one multiplicative F32
//! global scale. Keep these three buffers separate instead of shoehorning
//! them into GgmlDType/QTensor.
//!
//! The quantized CUDA path uses F32 -> Q8_1 activation quantization and
//! software LUT/DP4A kernels; it does not require hardware FP4 instructions.

use crate::{Result, Shape};

#[cfg(feature = "cuda-legacy-fp4")]
use crate::{
    builder_arg as barg, cuda_backend::WrapErr, op::BackpropOp, tensor::from_storage, CudaDevice,
    CudaStorage, DType, Device, Storage, Tensor,
};
#[cfg(feature = "cuda-legacy-fp4")]
use cudarc::driver::{CudaSlice, DevicePtr, PushKernelArg};

/// A host-side, row-major NVFP4 weight tensor.
///
/// Dense weights have shape [n, k]; indexed MoE weights have shape
/// [num_experts, n, k]. k must be a positive multiple of 32 because the
/// legacy DP4A kernels consume pairs of 16-weight blocks.
///
/// Each block has 8 packed E2M1 bytes (low nibble = first weight), one
/// non-negative finite E4M3FN scale, and one positive, finite, *multiplicative*
/// global F32 scale shared by the tensor. Callers of checkpoints storing an
/// inverse global scale must invert it before constructing this type.
///
/// This type deliberately does not contain a quantizer. It accepts native
/// packed checkpoint data without assuming a particular ModelOpt calibration.
#[derive(Clone, Debug)]
pub struct NvFp4Weights {
    shape: Shape,
    packed_e2m1: Vec<u8>,
    scales_e4m3: Vec<u8>,
    global_scale: f32,
}

impl NvFp4Weights {
    /// Validate and retain packed native NVFP4 weights.
    pub fn new<S: Into<Shape>>(
        shape: S,
        packed_e2m1: Vec<u8>,
        scales_e4m3: Vec<u8>,
        global_scale: f32,
    ) -> Result<Self> {
        let shape = shape.into();
        let dims = shape.dims();
        let (experts, n, k) = match dims {
            [n, k] => (1usize, *n, *k),
            [experts, n, k] => (*experts, *n, *k),
            _ => crate::bail!("NVFP4 weights need [n, k] or [experts, n, k], got {dims:?}"),
        };

        if experts == 0 || n == 0 || k == 0 || k % 32 != 0 {
            crate::bail!("NVFP4 requires nonzero experts/rows and k divisible by 32, got {dims:?}")
        }
        // Existing kernels pass dimensions as signed 32-bit arguments and
        // pad the Q8_1 activation stride to a multiple of 512.
        if experts > i32::MAX as usize || n > i32::MAX as usize || k > i32::MAX as usize - 511 {
            crate::bail!("NVFP4 dimensions exceed the legacy CUDA kernel limits: {dims:?}")
        }
        let count = experts
            .checked_mul(n)
            .and_then(|x| x.checked_mul(k))
            .ok_or_else(|| crate::Error::Msg("NVFP4 shape element count overflow".into()))?;
        let bytes = count / 2;
        let scales = count / 16;
        if packed_e2m1.len() != bytes || scales_e4m3.len() != scales {
            crate::bail!(
                "NVFP4 size mismatch: shape {dims:?} needs {bytes} packed bytes and \
                 {scales} scales; got {} and {}",
                packed_e2m1.len(),
                scales_e4m3.len()
            )
        }
        if !global_scale.is_finite() || global_scale <= 0.0 {
            crate::bail!("NVFP4 global scale must be positive and finite, got {global_scale}")
        }
        // E4M3FN codes 0x7f and 0xff encode NaN. Block scales are
        // non-negative by construction, including zero for zero blocks.
        if let Some((index, &code)) = scales_e4m3
            .iter()
            .enumerate()
            .find(|(_, code)| (**code & 0x80) != 0 || (**code & 0x7f) == 0x7f)
        {
            crate::bail!("NVFP4 block scale {index} is negative or NaN: {code:#04x}")
        }
        Ok(Self {
            shape,
            packed_e2m1,
            scales_e4m3,
            global_scale,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn packed_e2m1(&self) -> &[u8] {
        &self.packed_e2m1
    }

    pub fn scales_e4m3(&self) -> &[u8] {
        &self.scales_e4m3
    }

    pub fn global_scale(&self) -> f32 {
        self.global_scale
    }

    /// Upload native packed weights, with no intermediate F16/F32 expansion.
    #[cfg(feature = "cuda-legacy-fp4")]
    pub fn upload(&self, device: &CudaDevice) -> Result<NvFp4Cuda> {
        Ok(NvFp4Cuda {
            shape: self.shape.clone(),
            packed_e2m1: device.clone_htod(&self.packed_e2m1)?,
            scales_e4m3: device.clone_htod(&self.scales_e4m3)?,
            global_scale: self.global_scale,
            device: device.clone(),
        })
    }
}

/// GPU-resident software NVFP4 weights. Requires cuda-legacy-fp4.
///
/// No backward implementation: this is an inference-only quantized path.
#[cfg(feature = "cuda-legacy-fp4")]
#[derive(Debug)]
pub struct NvFp4Cuda {
    shape: Shape,
    packed_e2m1: CudaSlice<u8>,
    scales_e4m3: CudaSlice<u8>,
    global_scale: f32,
    device: CudaDevice,
}

#[cfg(feature = "cuda-legacy-fp4")]
impl NvFp4Cuda {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    fn quantize_input(&self, xs: &Tensor, batch: usize, k: usize) -> Result<CudaSlice<u8>> {
        if !xs.device().same_device(&Device::Cuda(self.device.clone())) {
            crate::bail!("NVFP4 input and weights must be on the same CUDA device")
        }
        if !matches!(xs.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            crate::bail!(
                "NVFP4 expects F32, F16 or BF16 activations, got {:?}",
                xs.dtype()
            )
        }
        let xs = xs.to_dtype(DType::F32)?.contiguous()?;
        let (storage, layout) = xs.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            crate::bail!("NVFP4 expected CUDA activation storage")
        };
        let src = storage.as_cuda_slice::<f32>()?;
        let start = layout.start_offset();
        let len = batch
            .checked_mul(k)
            .ok_or_else(|| crate::Error::Msg("NVFP4 activation size overflow".into()))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| crate::Error::Msg("NVFP4 activation offset overflow".into()))?;
        let padded_k =
            k.div_ceil(super::cuda::MATRIX_ROW_PADDING) * super::cuda::MATRIX_ROW_PADDING;
        let row_bytes = padded_k / 32 * 36; // Q8_1: 32 int8 + two fp16 scales
        let total_bytes = batch
            .checked_mul(row_bytes)
            .ok_or_else(|| crate::Error::Msg("NVFP4 Q8_1 buffer size overflow".into()))?;
        let mut quantized = self.device.alloc_zeros::<u8>(total_bytes)?;
        super::cuda::quantize_q8_1(
            &src.slice(start..end),
            &mut quantized,
            k,
            batch,
            &self.device,
        )?;
        Ok(quantized)
    }

    /// Dense inference: [.., k] -> [.., n], returning F32.
    ///
    /// The trailing dimension must equal the packed weights' k. Up to eight
    /// input rows use decode/GEMV, larger batches use the prefill kernel.
    /// Non-contiguous inputs are materialized before Q8_1 quantization.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let [n, k] = self.shape.dims() else {
            crate::bail!(
                "NVFP4 dense forward needs [n, k] weights, got {:?}",
                self.shape
            )
        };
        if xs.rank() < 2 || xs.dims().last() != Some(k) {
            crate::bail!(
                "NVFP4 dense forward expects [.., {k}] with rank >= 2, got {:?}",
                xs.shape()
            )
        }
        let batch = xs.dims()[..xs.rank() - 1]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| crate::Error::Msg("NVFP4 batch size overflow".into()))?;
        if batch == 0 || batch > 65535 {
            crate::bail!("NVFP4 dense batch must be in 1..=65535, got {batch}")
        }
        let activation_q8 = self.quantize_input(xs, batch, *k)?;
        let k_padded =
            k.div_ceil(super::cuda::MATRIX_ROW_PADDING) * super::cuda::MATRIX_ROW_PADDING;
        let dst = self.device.alloc_zeros::<f32>(
            n.checked_mul(batch)
                .ok_or_else(|| crate::Error::Msg("NVFP4 output size overflow".into()))?,
        )?;
        let is_decode = batch <= 8;
        let name = if is_decode {
            format!("nvfp4_mat_vec_q8_1_cuda{batch}")
        } else {
            "nvfp4_mat_mul_q8_1".into()
        };
        let func = self
            .device
            .get_or_load_func(&name, &candle_kernels::QUANTIZED)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (*n as u32, if is_decode { 1 } else { batch as u32 }, 1),
            block_dim: (32, if is_decode && batch > 4 { 2 } else { 4 }, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&self.packed_e2m1);
        builder.arg(&self.scales_e4m3);
        barg!(builder, self.global_scale);
        builder.arg(&activation_q8);
        builder.arg(&dst);
        if is_decode {
            barg!(builder, *k as i32, *n as i32, k_padded as i32, *n as i32);
        } else {
            barg!(
                builder,
                *k as i32,
                *n as i32,
                batch as i32,
                k_padded as i32,
                *n as i32
            );
        }
        unsafe { builder.launch(cfg) }.w()?;

        let mut out_shape = xs.dims().to_vec();
        *out_shape.last_mut().unwrap() = *n;
        // Explicitly inference-only, matching QTensor::apply_op1_no_bwd.
        Ok(from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(dst, self.device.clone())),
            out_shape,
            BackpropOp::none(),
            false,
        ))
    }

    /// Indexed MoE inference: weights [experts, n, k], inputs [batch, k],
    /// ids [batch, topk] (U32); output [batch, topk, n], F32.
    ///
    /// Unlike the legacy kernel, this API validates every expert index
    /// *before* launch, preventing out-of-bounds device reads. This requires
    /// one D2H synchronization per call; CUDA graph capture is not yet supported.
    pub fn indexed_moe(&self, xs: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let [experts, n, k] = self.shape.dims() else {
            crate::bail!("NVFP4 indexed MoE needs [experts, n, k] weights")
        };
        let [batch, in_k] = xs.dims() else {
            crate::bail!(
                "NVFP4 indexed MoE expects input [batch, k], got {:?}",
                xs.shape()
            )
        };
        if *batch == 0 || *batch > 65535 || in_k != k {
            crate::bail!("NVFP4 indexed MoE invalid input shape {:?}", xs.shape())
        }
        let [idx_batch, topk] = ids.dims() else {
            crate::bail!(
                "NVFP4 indexed MoE expects indices [batch, topk], got {:?}",
                ids.shape()
            )
        };
        if idx_batch != batch || *topk == 0 || *topk > 65535 || ids.dtype() != DType::U32 {
            crate::bail!("NVFP4 indexed MoE requires nonempty U32 [batch, topk] indices")
        }
        if !ids.device().same_device(&Device::Cuda(self.device.clone())) {
            crate::bail!("NVFP4 indexed MoE indices and weights must share a CUDA device")
        }
        let ids = ids.contiguous()?;
        let checked_ids = ids
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        if let Some((pos, &id)) = checked_ids
            .iter()
            .enumerate()
            .find(|(_, id)| **id as usize >= *experts)
        {
            crate::bail!("NVFP4 expert id {id} at offset {pos} >= {experts}")
        }
        let (ids_storage, ids_layout) = ids.storage_and_layout();
        let Storage::Cuda(ids_storage) = &*ids_storage else {
            crate::bail!("NVFP4 expected CUDA index storage")
        };
        let ids_slice = ids_storage.as_cuda_slice::<u32>()?;
        let start = ids_layout.start_offset();
        let ids_view = ids_slice.slice(start..start + checked_ids.len());
        let q8 = self.quantize_input(xs, *batch, *k)?;
        let k_padded =
            k.div_ceil(super::cuda::MATRIX_ROW_PADDING) * super::cuda::MATRIX_ROW_PADDING;
        let out_count = batch
            .checked_mul(*topk)
            .and_then(|v| v.checked_mul(*n))
            .ok_or_else(|| crate::Error::Msg("NVFP4 MoE output size overflow".into()))?;
        let dst = self.device.alloc_zeros::<f32>(out_count)?;
        let func = self
            .device
            .get_or_load_func("nvfp4_indexed_moe_q8_1", &candle_kernels::QUANTIZED)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (*n as u32, *batch as u32, *topk as u32),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&self.packed_e2m1);
        builder.arg(&self.scales_e4m3);
        barg!(builder, self.global_scale);
        builder.arg(&q8);
        builder.arg(&ids_view);
        builder.arg(&dst);
        barg!(
            builder,
            *n as i32,
            *k as i32,
            *batch as i32,
            *topk as i32,
            k_padded as i32
        );
        unsafe { builder.launch(cfg) }.w()?;
        Ok(from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(dst, self.device.clone())),
            (*batch, *topk, *n),
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(feature = "cuda-legacy-fp4")]
impl crate::Module for NvFp4Cuda {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        NvFp4Cuda::forward(self, xs)
    }
}
