//! F8E-R4C: generic FP8 E4M3 fallback on Pascal SM61.
//! SIMT W8A16 is the fail-closed default; resident FP16/matmul is an optional
//! admission based on explicit shape/workload measurement and global VRAM budget.
//! A single instance holds ONE weight representation. DP4A is explicit research
//! only, never selected in the generic Auto dispatch.
use super::fp8_sm61_dispatch::{Fp8Backend, Fp8DispatchEntry};
use crate::cuda_backend::{
    cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg},
    WrapErr,
};
use crate::{CpuStorage, CudaDevice, CudaStorage, DType, Device, Layout, Result, Shape, Tensor};
use crate::{CustomOp1, CustomOp2};
use half::{bf16, f16};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8Sm61Mode {
    SimtW8A16,
    Dp4aW8A8,
    ExpandedF16,
    Auto,
}

use super::fp8_sm61_dispatch::{required_bytes, Fp8LinearSpec};
pub use super::fp8_sm61_format::e4m3fn_decode;
use super::fp8_sm61_format::{checked_f16_weight, decode_f8e_inverse_scales};

/// Dispatch selection fixed after the F8E v0.4 sm_61 GPU campaign.
/// The 8-token DP4A tile was slower in ALL tested prefill workloads; only
/// the 4-token tile is production-selectable in F8E-R1.
/// For M<4 the GEMV kernel handles one output row per grid-y coordinate.
pub fn dp4a_launch_for_tokens(tokens: usize) -> (&'static str, usize) {
    if tokens >= 4 {
        ("fp8_sm61_dp4a_prefill4", 4)
    } else {
        ("fp8_sm61_dp4a_gemv", 1)
    }
}

/// Compatibility entry point for the historical F8E Auto constructor.
/// R4-C intentionally disables shape-only auto-FP16/DP4A selection: a caller
/// MUST first provide both a model-wide residency dispatch plan and measured profiles.
/// Without these, SIMT is the single admissible generic backend.
pub fn fp8_sm61_select_mode(
    expected_m: usize,
    _n: usize,
    _k: usize,
    _group: usize,
    fp8_bytes: usize,
    _dp_bytes: usize,
    _f16_bytes: usize,
    budget: usize,
) -> Result<Fp8Sm61Mode> {
    if expected_m == 0 {
        crate::bail!("F8E expected_m must be nonzero")
    }
    if fp8_bytes > budget {
        crate::bail!("F8E SIMT representation exceeds supplied budget")
    }
    Ok(Fp8Sm61Mode::SimtW8A16)
}

// Enum prevents FP8 and INT8/FP16 copies from coexisting in the same Linear.
enum ResidentWeights {
    Fp8 {
        weights: CudaSlice<u8>,
        scales: CudaSlice<f32>,
    },
    Int8 {
        weights: CudaSlice<i8>,
        scales: CudaSlice<f32>,
        group: usize,
    },
    F16 {
        weights: Tensor,
    },
}

pub struct Fp8Sm61Linear {
    device: CudaDevice,
    resident: ResidentWeights,
    n: usize,
    k: usize,
    block_n: usize,
    block_k: usize,
    mode: Fp8Sm61Mode,
    resident_bytes: usize,
}

impl Fp8Sm61Linear {
    pub fn from_bytes(
        device: &CudaDevice,
        n: usize,
        k: usize,
        weights: &[u8],
        scales: &[f32],
        block_n: usize,
        block_k: usize,
        mode: Fp8Sm61Mode,
        dp4a_group: usize,
    ) -> Result<Self> {
        let spec = Fp8LinearSpec {
            out_features: n,
            in_features: k,
            block_n,
            block_k,
            resident_calls: 0,
            profile: None,
        };
        required_bytes(&spec).map_err(|e| crate::Error::Msg(e.to_string().into()))?;
        if weights.len() != n * k || scales.len() != n.div_ceil(block_n) * k.div_ceil(block_k) {
            crate::bail!("F8E invalid E4M3FN weight shape or scale block dimensions");
        }
        if weights.iter().any(|&c| c & 127 == 127)
            || scales.iter().any(|x| !x.is_finite() || *x < 0.)
        {
            crate::bail!("F8E NaN FP8 payload, negative or nonfinite dequantization scale")
        }
        if mode == Fp8Sm61Mode::Auto {
            crate::bail!("F8E Auto requires from_bytes_profiled(expected_m, memory_budget_bytes)")
        }
        let scale_at = |row: usize, col: usize| -> f32 {
            scales[(row / block_n) * k.div_ceil(block_k) + (col / block_k)]
        };
        for row in 0..n {
            for col in 0..k {
                if !(e4m3fn_decode(weights[row * k + col]) * scale_at(row, col)).is_finite() {
                    crate::bail!("F8E dequantized weight exceeds the finite F32 range");
                }
            }
        }
        let (resident, bytes) = match mode {
            Fp8Sm61Mode::SimtW8A16 => {
                let w = device.clone_htod(weights)?;
                let s = device.clone_htod(scales)?;
                (
                    ResidentWeights::Fp8 {
                        weights: w,
                        scales: s,
                    },
                    weights.len() + 4 * scales.len(),
                )
            }
            Fp8Sm61Mode::Dp4aW8A8 => {
                if !matches!(dp4a_group, 32 | 64 | 128) {
                    crate::bail!("F8E DP4A group must be 32,64,128")
                }
                let groups = k / dp4a_group;
                let mut packed = vec![0i8; n * k];
                let mut sq = vec![0f32; n * groups];
                for row in 0..n {
                    for g in 0..groups {
                        let start = row * k + g * dp4a_group;
                        let end = start + dp4a_group;
                        let mut peak = 0f32;
                        for (j, &v) in weights[start..end].iter().enumerate() {
                            let effective = e4m3fn_decode(v) * scale_at(row, g * dp4a_group + j);
                            if !effective.is_finite() {
                                crate::bail!("F8E effective weight overflow")
                            }
                            peak = peak.max(effective.abs());
                        }
                        let scale = if peak == 0. { 0. } else { peak / 127. };
                        if !scale.is_finite() || (peak > 0. && scale == 0.) {
                            crate::bail!("F8E invalid INT8 scale")
                        }
                        sq[row * groups + g] = scale;
                        for (j, &v) in weights[start..end].iter().enumerate() {
                            let effective = e4m3fn_decode(v) * scale_at(row, g * dp4a_group + j);
                            packed[start + j] = if scale == 0. {
                                0
                            } else {
                                (effective / scale).round_ties_even().clamp(-127., 127.) as i8
                            };
                        }
                    }
                }
                // Host temporary FP8 buffers die at caller; GPU only contains INT8 + INT8 scales.
                let w = device.clone_htod(&packed)?;
                let s = device.clone_htod(&sq)?;
                (
                    ResidentWeights::Int8 {
                        weights: w,
                        scales: s,
                        group: dp4a_group,
                    },
                    packed.len() + sq.len() * 4,
                )
            }
            Fp8Sm61Mode::ExpandedF16 => {
                let mut decoded = Vec::with_capacity(n * k);
                for row in 0..n {
                    for col in 0..k {
                        let f = e4m3fn_decode(weights[row * k + col]) * scale_at(row, col);
                        if !f.is_finite() {
                            crate::bail!("F8E expanded F16 weight overflow")
                        }
                        decoded.push(checked_f16_weight(f)?);
                    }
                }
                let wt = Tensor::from_vec(decoded, (n, k), &Device::Cuda(device.clone()))?;
                (ResidentWeights::F16 { weights: wt }, 2 * n * k)
            }
            Fp8Sm61Mode::Auto => unreachable!(),
        };
        Ok(Self {
            device: device.clone(),
            resident,
            n,
            k,
            block_n,
            block_k,
            mode,
            resident_bytes: bytes,
        })
    }

    /// Legacy Auto compatibility: always selects SIMT if budget permits.
    /// Use from_bytes_with_dispatch with an externally validated *model-wide*
    /// plan for opt-in resident FP16. Never silently use DP4A.
    pub fn from_bytes_profiled(
        device: &CudaDevice,
        n: usize,
        k: usize,
        weights: &[u8],
        scales: &[f32],
        bn: usize,
        bk: usize,
        expected_m: usize,
        memory_budget_bytes: usize,
        group: usize,
    ) -> Result<Self> {
        if expected_m == 0 {
            crate::bail!("F8E expected_m must be nonzero")
        }
        let fp8_bytes = weights.len().saturating_add(scales.len().saturating_mul(4));
        let dp_bytes = weights
            .len()
            .saturating_add(n.saturating_mul(k / group.max(1)).saturating_mul(4));
        let f16_bytes = n.saturating_mul(k).saturating_mul(2);
        let selected = fp8_sm61_select_mode(
            expected_m,
            n,
            k,
            group,
            fp8_bytes,
            dp_bytes,
            f16_bytes,
            memory_budget_bytes,
        )?;
        Self::from_bytes(device, n, k, weights, scales, bn, bk, selected, group)
    }

    /// Load exactly the representation selected by a generic residence plan.
    /// This function does not switch backends at runtime or duplicate weights.
    /// The application must reserve KV/workspace/other-model memory in the plan
    /// and arrange allocations consistent with all rows of the model-wide plan.
    pub fn from_bytes_with_dispatch(
        device: &CudaDevice,
        n: usize,
        k: usize,
        weights: &[u8],
        scales: &[f32],
        bn: usize,
        bk: usize,
        decision: &Fp8DispatchEntry,
    ) -> Result<Self> {
        if decision.spec.out_features != n
            || decision.spec.in_features != k
            || decision.spec.block_n != bn
            || decision.spec.block_k != bk
        {
            crate::bail!("F8E admission layer shape mismatch")
        }
        let mode = match decision.backend {
            Fp8Backend::SimtW8A16 => Fp8Sm61Mode::SimtW8A16,
            Fp8Backend::ExpandedF16 => Fp8Sm61Mode::ExpandedF16,
        };
        let linear = Self::from_bytes(device, n, k, weights, scales, bn, bk, mode, 128)?;
        if linear.resident_weight_bytes() != decision.resident_bytes {
            crate::bail!("F8E actual weight residency differs from admitted bytes")
        }
        Ok(linear)
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.n, self.k)
    }
    pub fn mode(&self) -> Fp8Sm61Mode {
        self.mode
    }
    pub fn resident_weight_bytes(&self) -> usize {
        self.resident_bytes
    }
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !matches!(input.dtype(), DType::F16 | DType::BF16)
            || input.dims().len() != 2 || input.dims()[1] != self.k {
            crate::bail!("F8E input requires rank-2 FP16 or BF16 [tokens,{}]", self.k)
        }
        if input.dtype() == DType::BF16
            && !matches!(self.resident, ResidentWeights::Fp8 { .. }) {
            crate::bail!("F8E BF16 activations require packed SIMT FP8 residency")
        }
        let m = input.dims()[0];
        if m == 0
            || m > 65535
            || m.checked_mul(self.k).is_none_or(|v| v > i32::MAX as usize)
            || m.checked_mul(self.n).is_none_or(|v| v > i32::MAX as usize)
        {
            crate::bail!("F8E batch/input/output exceeds CUDA signed 32-bit indexing");
        }
        if !input
            .device()
            .same_device(&Device::Cuda(self.device.clone()))
        {
            crate::bail!("F8E input and weights must use the same CUDA device");
        }
        match &self.resident {
            ResidentWeights::F16 { weights } => {
                input.apply_op2_no_bwd(weights, &ExpandedF16Op { linear: self })
            }
            _ => input.apply_op1_no_bwd(&ForwardOp {
                linear: self,
                bf16_input: input.dtype() == DType::BF16,
            }),
        }
    }

    /// Compatibility constructor for one projection. For full checkpoints prefer
    /// `fp8_sm61_loader::load_cuda`, which reads/deserializes shards once per load.
    /// Strict E4M3FN Safetensors single shard, using known inverse weight scales.
    /// The meaning of `weight_scale` varies across producers; do not silently assume
    /// it is an inverse/dequant multiplier. Unknown conventions are rejected.
    pub fn from_safetensors(
        device: &CudaDevice,
        path: impl AsRef<Path>,
        prefix: &str,
        mode: Fp8Sm61Mode,
        dp4a_group: usize,
    ) -> Result<Self> {
        let data = std::fs::read(path.as_ref())?;
        let st = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| crate::Error::Msg(format!("F8E invalid safetensors: {e}").into()))?;
        let wt = st
            .tensor(&format!("{prefix}.weight"))
            .map_err(|e| crate::Error::Msg(format!("F8E missing weight: {e}").into()))?;
        if !matches!(
            format!("{:?}", wt.dtype()).as_str(),
            "F8_E4M3" | "F8_E4M3FN"
        ) || wt.shape().len() != 2
        {
            crate::bail!("F8E requires actual FP8 E4M3 [n,k], not U8")
        }
        let (n, k) = (wt.shape()[0], wt.shape()[1]);
        let stensor = st
            .tensor(&format!("{prefix}.weight_scale_inv"))
            .map_err(|e| {
                crate::Error::Msg(
                    format!("F8E requires weight_scale_inv (not unqualified weight_scale): {e}")
                        .into(),
                )
            })?;
        let (scales, bn, bk) = decode_f8e_inverse_scales(n, k, &stensor)?;
        Self::from_bytes(device, n, k, wt.data(), &scales, bn, bk, mode, dp4a_group)
    }
}

struct ExpandedF16Op<'a> {
    linear: &'a Fp8Sm61Linear,
}

impl CustomOp2 for ExpandedF16Op<'_> {
    fn name(&self) -> &'static str {
        "haricot.fp8.sm61.expanded-f16-f32-gemm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        crate::bail!("Fp8Sm61Linear requires CUDA; no silent CPU fallback")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        weights: &CudaStorage,
        weight_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use crate::backend::BackendStorage;
        use crate::cuda_backend::cudarc::{
            cublas::{result, sys},
            driver::{DevicePtr, DevicePtrMut},
        };
        let op = self.linear;
        if input.device().id() != op.device.id() || weights.device().id() != op.device.id() {
            crate::bail!("F8E input and weights must use the same CUDA device");
        }
        if !input_layout.is_contiguous()
            || input_layout.start_offset() != 0
            || !weight_layout.is_contiguous()
            || weight_layout.start_offset() != 0
        {
            crate::bail!("F8E expanded GEMM requires contiguous zero-offset tensors");
        }
        let m = input_layout.dims()[0];
        let input = input.as_cuda_slice::<f16>()?;
        let weights = weights.as_cuda_slice::<f16>()?;
        if input.len() != m * op.k || weights.len() != op.n * op.k {
            crate::bail!("F8E expanded GEMM underlying slice length mismatch");
        }
        let mut output = op.device.alloc_zeros::<f32>(m * op.n)?;
        let stream = op.device.cuda_stream();
        let (a, _guard_a) = weights.device_ptr(&stream);
        let (b, _guard_b) = input.device_ptr(&stream);
        {
            let (c, _guard_c) = output.device_ptr_mut(&stream);
            let alpha = 1f32;
            let beta = 0f32;
            // X[M,K] * W[N,K]^T, expressed as column-major W^T * X.
            // Both operands stay resident in F16; accumulation AND output
            // are F32. Casting an F16 GEMM output to F32 afterwards loses
            // range and precision, including finite sums above 65504.
            unsafe {
                result::gemm_ex(
                    *op.device.blas.handle(),
                    sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    op.n as i32,
                    m as i32,
                    op.k as i32,
                    (&alpha as *const f32).cast(),
                    a as *const _,
                    sys::cudaDataType_t::CUDA_R_16F,
                    op.k as i32,
                    b as *const _,
                    sys::cudaDataType_t::CUDA_R_16F,
                    op.k as i32,
                    (&beta as *const f32).cast(),
                    c as *mut _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    op.n as i32,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
            }
            .w()?;
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, op.device.clone()),
            (m, op.n).into(),
        ))
    }
}

struct ForwardOp<'a> {
    linear: &'a Fp8Sm61Linear,
    bf16_input: bool,
}
impl CustomOp1 for ForwardOp<'_> {
    fn name(&self) -> &'static str {
        "haricot.fp8.sm61.f8e"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        crate::bail!("Fp8Sm61Linear requires CUDA; no silent CPU fallback")
    }
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let op = self.linear;
        use crate::backend::BackendStorage;
        if storage.device().id() != op.device.id() {
            crate::bail!("F8E input and weights must use the same CUDA device");
        }
        if !layout.is_contiguous()
            || layout.start_offset() != 0
            || layout.dims().len() != 2
            || layout.dims()[1] != op.k
        {
            crate::bail!("F8E requires contiguous zero-offset [tokens,k] input")
        }
        let m = layout.dims()[0];
        let inp_f16 = if self.bf16_input {
            None
        } else {
            Some(storage.as_cuda_slice::<f16>()?)
        };
        let inp_bf16 = if self.bf16_input {
            Some(storage.as_cuda_slice::<bf16>()?)
        } else {
            None
        };
        if inp_f16.map_or(false, |inp| inp.len() != m * op.k)
            || inp_bf16.map_or(false, |inp| inp.len() != m * op.k) {
            crate::bail!("F8E input underlying slice length mismatch")
        }
        let output = op.device.alloc_zeros::<f32>(m * op.n)?;
        let mi = m as i32;
        let ni = op.n as i32;
        let ki = op.k as i32;
        match &op.resident {
            ResidentWeights::Int8 {
                weights,
                scales,
                group,
            } => {
                let grp = *group as i32;
                let quant = op.device.alloc_zeros::<i8>(m * op.k)?;
                let act_scales = op.device.alloc_zeros::<f32>(m * (op.k / *group))?;
                let quant_func = op
                    .device
                    .get_or_load_func("fp8_sm61_quant_x", &candle_kernels::QUANTIZED)?;
                let qcfg = LaunchConfig {
                    grid_dim: ((m * op.k / *group) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut qb = quant_func.builder();
                qb.arg(inp_f16.ok_or_else(|| crate::Error::msg(
                    "F8E DP4A requires F16 input"
                ))?);
                qb.arg(&quant);
                qb.arg(&act_scales);
                qb.arg(&mi);
                qb.arg(&ki);
                qb.arg(&grp);
                unsafe { qb.launch(qcfg) }.w()?;
                let (name, tile) = dp4a_launch_for_tokens(m);
                let func = op
                    .device
                    .get_or_load_func(name, &candle_kernels::QUANTIZED)?;
                let cfg = LaunchConfig {
                    grid_dim: (op.n.div_ceil(4) as u32, m.div_ceil(tile) as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = func.builder();
                b.arg(weights);
                b.arg(scales);
                b.arg(&quant);
                b.arg(&act_scales);
                b.arg(&output);
                b.arg(&mi);
                b.arg(&ni);
                b.arg(&ki);
                b.arg(&grp);
                unsafe { b.launch(cfg) }.w()?;
            }
            ResidentWeights::Fp8 { weights, scales } => {
                let name = match (self.bf16_input, m >= 4) {
                    (true, true) => "fp8_sm61_w8abf16_prefill4",
                    (true, false) => "fp8_sm61_w8abf16_gemv",
                    (false, true) => "fp8_sm61_w8a16_prefill4",
                    (false, false) => "fp8_sm61_w8a16_gemv",
                };
                let func = op
                    .device
                    .get_or_load_func(name, &candle_kernels::QUANTIZED)?;
                let bn = op.block_n as i32;
                let bk = op.block_k as i32;
                let cfg = LaunchConfig {
                    grid_dim: (
                        op.n.div_ceil(4) as u32,
                        if m >= 4 {
                            m.div_ceil(4) as u32
                        } else {
                            m as u32
                        },
                        1,
                    ),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = func.builder();
                b.arg(weights);
                b.arg(scales);
                if let Some(inp) = inp_bf16 {
                    b.arg(inp);
                } else {
                    b.arg(inp_f16.ok_or_else(|| crate::Error::msg(
                        "F8E SIMT missing input"
                    ))?);
                }
                b.arg(&output);
                b.arg(&mi);
                b.arg(&ni);
                b.arg(&ki);
                b.arg(&bn);
                b.arg(&bk);
                unsafe { b.launch(cfg) }.w()?;
            }
            ResidentWeights::F16 { .. } => unreachable!("FP16 path handled by forward"),
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, op.device.clone()),
            (m, op.n).into(),
        ))
    }
}
