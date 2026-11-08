//! Adapters over existing providers. Numeric kernels and per-format APIs stay authoritative.
use super::contract::*;
use super::{
    select_bf16_linear, Architecture, Bf16LinearBackend, DeviceCapabilities, DispatchContext,
    LinearInventory, LinearMemory, LinearSpec, PreparedBf16Linear, RequireBackend, ResourceBudget,
};
use crate::quantized::fp8_sm61_dispatch::{
    dispatch_residence_for_activations, Fp8Activation, Fp8DispatchBudget, Fp8LinearSpec,
};
use crate::{DType, Device, Result, Tensor};
use std::sync::Arc;

pub const BF16_CPU_F32: ProviderId = ProviderId("candle.bf16.cpu-f32.v1");
pub const BF16_CUDA_F32: ProviderId = ProviderId("candle.bf16.cuda-f32.v1");
pub const BF16_CUDA_NATIVE: ProviderId = ProviderId("candle.bf16.cuda-native.v1");
pub const FP8_CUDA_PACKED: ProviderId = ProviderId("candle.fp8.cuda-packed-sm61.v1");
pub const NVFP4_CUDA_W4A16: ProviderId = ProviderId("candle.nvfp4.cuda-w4a16-sm61.v1");

#[derive(Clone, Copy, Debug)]
pub enum BuiltinLinearProvider {
    Bf16 {
        backend: Bf16LinearBackend,
        available: bool,
    },
    Fp8Packed {
        available: bool,
    },
    Nvfp4W4A16 {
        available: bool,
    },
}

impl BuiltinLinearProvider {
    /// Compile-time inventory only. Preparation also inspects the actual device and symbols.
    pub fn compiled() -> [Self; 5] {
        [
            Self::Bf16 {
                backend: Bf16LinearBackend::CudaBf16,
                available: cfg!(feature = "cuda"),
            },
            Self::Bf16 {
                backend: Bf16LinearBackend::CudaF32,
                available: cfg!(feature = "cuda"),
            },
            Self::Bf16 {
                backend: Bf16LinearBackend::CpuF32,
                available: true,
            },
            Self::Fp8Packed {
                available: cfg!(all(feature = "cuda", feature = "cuda-legacy-fp8")),
            },
            Self::Nvfp4W4A16 {
                available: cfg!(all(feature = "cuda", feature = "cuda-legacy-fp4")),
            },
        ]
    }
}

impl LinearProvider for BuiltinLinearProvider {
    fn id(&self) -> ProviderId {
        match self {
            Self::Bf16 {
                backend: Bf16LinearBackend::CpuF32,
                ..
            } => BF16_CPU_F32,
            Self::Bf16 {
                backend: Bf16LinearBackend::CudaF32,
                ..
            } => BF16_CUDA_F32,
            Self::Bf16 {
                backend: Bf16LinearBackend::CudaBf16,
                ..
            } => BF16_CUDA_NATIVE,
            Self::Fp8Packed { .. } => FP8_CUDA_PACKED,
            Self::Nvfp4W4A16 { .. } => NVFP4_CUDA_W4A16,
        }
    }
    fn device_kind(&self) -> DeviceKind {
        match self {
            Self::Bf16 {
                backend: Bf16LinearBackend::CpuF32,
                ..
            } => DeviceKind::Cpu,
            _ => DeviceKind::Cuda,
        }
    }
    fn assess(
        &self,
        request: &LinearRequest,
        caps: DeviceCapabilities,
    ) -> Result<ProviderAssessment> {
        match self.assess_builtin(request, caps) {
            Ok(candidate) => Ok(candidate),
            Err(error) => Ok(ProviderAssessment::rejected(
                self.id(),
                ProviderRejection::Constraint(error.to_string()),
            )),
        }
    }
}

impl BuiltinLinearProvider {
    fn assess_builtin(
        &self,
        request: &LinearRequest,
        caps: DeviceCapabilities,
    ) -> Result<ProviderAssessment> {
        request.validate()?;
        let id = self.id();
        let reject = |r| Ok(ProviderAssessment::rejected(id, r));
        let c = request.contract;
        let available = match self {
            Self::Bf16 { available, .. }
            | Self::Fp8Packed { available }
            | Self::Nvfp4W4A16 { available } => *available,
        };
        if !available {
            return reject(ProviderRejection::NotCompiled);
        }
        if c.accumulation != DType::F32 {
            return reject(ProviderRejection::NumericalContract);
        }
        let memory = match self {
            Self::Bf16 { backend, .. } => {
                if c.weights != WeightFormat::Bf16
                    || c.activations != DType::BF16
                    || c.output != DType::BF16
                    || !matches!(
                        c.bias,
                        BiasSemantics::None | BiasSemantics::AfterBf16GemmRounding
                    )
                {
                    return reject(ProviderRejection::NumericalContract);
                }
                let spec = bf16_spec(request);
                // Delegate architecture/provider/shape rules to the existing BF16 selector.
                match select_bf16_linear(
                    spec,
                    caps,
                    LinearInventory {
                        cpu_f32: *backend == Bf16LinearBackend::CpuF32,
                        cuda_f32: *backend == Bf16LinearBackend::CudaF32,
                        cuda_bf16: *backend == Bf16LinearBackend::CudaBf16,
                    },
                    ResourceBudget {
                        resident_bytes: usize::MAX,
                        workspace_bytes: usize::MAX,
                    },
                    Some(&RequireBackend(*backend)),
                ) {
                    Ok(decision) => decision.memory,
                    Err(e) => return reject(ProviderRejection::Constraint(e.to_string())),
                }
            }
            Self::Fp8Packed { .. } => {
                let WeightFormat::Fp8E4m3Fn { block_n, block_k } = c.weights else {
                    return reject(ProviderRejection::NumericalContract);
                };
                if !matches!(c.activations, DType::F16 | DType::BF16)
                    || c.output != DType::F32
                    || c.bias != BiasSemantics::None
                {
                    return reject(ProviderRejection::NumericalContract);
                }
                if caps.architecture != (Architecture::Cuda { major: 6, minor: 1 }) {
                    return reject(ProviderRejection::Architecture);
                }
                if request.input_layout != InputLayout::ContiguousZeroOffset {
                    return reject(ProviderRejection::Layout);
                }
                check_cuda_rows(request)?;
                let plan = dispatch_residence_for_activations(
                    &[fp8_spec(request, block_n, block_k)],
                    &[fp8_activation(c.activations)?],
                    Fp8DispatchBudget {
                        free_device_bytes: usize::MAX,
                        ..Default::default()
                    },
                )
                .map_err(crate::Error::msg)?;
                let bytes = plan.entries[0].resident_bytes;
                LinearMemory {
                    source_bytes: bytes,
                    resident_bytes: bytes,
                    workspace_bytes: output_bytes(request)?,
                }
            }
            Self::Nvfp4W4A16 { .. } => {
                if c.weights != WeightFormat::Nvfp4ModelOpt
                    || c.activations != DType::F16
                    || c.output != DType::F32
                    || c.bias != BiasSemantics::None
                {
                    return reject(ProviderRejection::NumericalContract);
                }
                if caps.architecture != (Architecture::Cuda { major: 6, minor: 1 }) {
                    return reject(ProviderRejection::Architecture);
                }
                if request.input_layout != InputLayout::ContiguousZeroOffset {
                    return reject(ProviderRejection::Layout);
                }
                crate::quantized::nvfp4_e7c_safetensors::checked_launch_shape(
                    request.out_features,
                    request.in_features,
                    request.max_rows,
                )?;
                let elements = request
                    .in_features
                    .checked_mul(request.out_features)
                    .unwrap();
                let bytes = (elements / 2)
                    .checked_add(elements / 16)
                    .ok_or_else(|| crate::Error::Msg("NVFP4 residency overflow".into()))?;
                LinearMemory {
                    source_bytes: bytes
                        .checked_add(4)
                        .ok_or_else(|| crate::Error::Msg("NVFP4 source overflow".into()))?,
                    resident_bytes: bytes,
                    workspace_bytes: output_bytes(request)?,
                }
            }
        };
        Ok(ProviderAssessment::eligible(id, memory))
    }
}

fn bf16_spec(r: &LinearRequest) -> LinearSpec {
    LinearSpec {
        in_features: r.in_features,
        out_features: r.out_features,
        max_rows: r.max_rows,
        bias: r.contract.bias != BiasSemantics::None,
    }
}
fn fp8_spec(r: &LinearRequest, block_n: usize, block_k: usize) -> Fp8LinearSpec {
    Fp8LinearSpec {
        in_features: r.in_features,
        out_features: r.out_features,
        block_n,
        block_k,
        resident_calls: 0,
        profile: None,
    }
}
fn fp8_activation(dtype: DType) -> Result<Fp8Activation> {
    match dtype {
        DType::F16 => Ok(Fp8Activation::F16),
        DType::BF16 => Ok(Fp8Activation::Bf16),
        _ => crate::bail!("Packed FP8 adapter requires F16 or BF16 activations"),
    }
}
fn check_cuda_rows(r: &LinearRequest) -> Result<()> {
    if r.max_rows > 65535
        || r.max_rows * r.in_features > i32::MAX as usize
        || r.max_rows * r.out_features > i32::MAX as usize
    {
        crate::bail!("Linear request exceeds CUDA launch/indexing limits")
    }
    Ok(())
}
fn output_bytes(r: &LinearRequest) -> Result<usize> {
    r.max_rows
        .checked_mul(r.out_features)
        .and_then(|x| x.checked_mul(4))
        .ok_or_else(|| crate::Error::Msg("Linear output memory overflow".into()))
}

/// Payloads keep their actual encodings. No MXFP4/NVFP4 reinterpretation is possible here.
pub enum LinearWeights<'a> {
    Bf16 {
        weight: &'a Tensor,
        bias: Option<&'a Tensor>,
    },
    Fp8E4m3Fn {
        packed: &'a [u8],
        scales: &'a [f32],
    },
    Nvfp4ModelOpt {
        packed: &'a [u8],
        scales_e4m3: &'a [u8],
        multiplier: f32,
    },
}

enum PreparedProvider {
    Bf16(PreparedBf16Linear),
    #[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
    Fp8(crate::quantized::fp8_linear_sm61::Fp8Sm61Linear),
    #[cfg(all(feature = "cuda", feature = "cuda-legacy-fp4"))]
    Nvfp4(crate::quantized::fp4_e7c_runtime::Fp4E7CLinear),
}

/// Unified rank-two execution facade over the existing providers.
pub struct PreparedLinear {
    device: Device,
    request: LinearRequest,
    selection: LinearSelection,
    provider: PreparedProvider,
}

impl PreparedLinear {
    /// Budget is a dedicated per-layer partition, not shared model-wide admission.
    /// Source tensors, retained outputs and backend/allocator reserves are external.
    /// Runtime availability is rebuilt here; a caller-supplied planning result is never trusted.
    pub fn prepare(
        request: LinearRequest,
        device: &Device,
        weights: LinearWeights<'_>,
        budget: ResourceBudget,
        preference: Option<&dyn LinearPreference>,
    ) -> Result<Self> {
        let caps = DeviceCapabilities::inspect(device)?;
        let providers = BuiltinLinearProvider::compiled();
        let refs: Vec<&dyn LinearProvider> =
            providers.iter().map(|p| p as &dyn LinearProvider).collect();
        let selection = select_linear(request, caps, budget, &refs, preference)?;
        let selected = selection.selected.provider;
        let provider = match (request.contract.weights, weights) {
            (WeightFormat::Bf16, LinearWeights::Bf16 { weight, bias }) => {
                if weight.dims() != [request.out_features, request.in_features]
                    || bias.is_some() != (request.contract.bias != BiasSemantics::None)
                {
                    crate::bail!("BF16 source does not match linear request")
                }
                let backend = if selected == BF16_CPU_F32 {
                    Bf16LinearBackend::CpuF32
                } else if selected == BF16_CUDA_F32 {
                    Bf16LinearBackend::CudaF32
                } else if selected == BF16_CUDA_NATIVE {
                    Bf16LinearBackend::CudaBf16
                } else {
                    crate::bail!("Unexpected BF16 provider identity")
                };
                let context =
                    DispatchContext::new(device, budget, Some(Arc::new(RequireBackend(backend))))?;
                PreparedProvider::Bf16(context.prepare_bf16_linear(
                    weight,
                    bias,
                    request.max_rows,
                )?)
            }
            #[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
            (
                WeightFormat::Fp8E4m3Fn { block_n, block_k },
                LinearWeights::Fp8E4m3Fn { packed, scales },
            ) => {
                if selected != FP8_CUDA_PACKED {
                    crate::bail!("Unexpected FP8 provider identity")
                }
                let plan = dispatch_residence_for_activations(
                    &[fp8_spec(&request, block_n, block_k)],
                    &[fp8_activation(request.contract.activations)?],
                    Fp8DispatchBudget {
                        free_device_bytes: budget.resident_bytes,
                        ..Default::default()
                    },
                )
                .map_err(crate::Error::msg)?;
                let cuda = device.as_cuda_device()?;
                for symbol in [
                    "fp8_sm61_w8a16_gemv",
                    "fp8_sm61_w8a16_prefill4",
                    "fp8_sm61_w8abf16_gemv",
                    "fp8_sm61_w8abf16_prefill4",
                ] {
                    cuda.get_or_load_func(symbol, &candle_kernels::QUANTIZED)?;
                }
                PreparedProvider::Fp8(crate::quantized::fp8_linear_sm61::Fp8Sm61Linear::from_bytes_with_activation_dispatch(
                    cuda, request.out_features, request.in_features, packed, scales, block_n, block_k,
                    &plan.entries[0], fp8_activation(request.contract.activations)?)?)
            }
            #[cfg(all(feature = "cuda", feature = "cuda-legacy-fp4"))]
            (
                WeightFormat::Nvfp4ModelOpt,
                LinearWeights::Nvfp4ModelOpt {
                    packed,
                    scales_e4m3,
                    multiplier,
                },
            ) => {
                if selected != NVFP4_CUDA_W4A16 {
                    crate::bail!("Unexpected NVFP4 provider identity")
                }
                let cuda = device.as_cuda_device()?;
                cuda.get_or_load_func("nvfp4_arch_e6_w4a16_scalar", &candle_kernels::QUANTIZED)?;
                PreparedProvider::Nvfp4(
                    crate::quantized::fp4_e7c_runtime::Fp4E7CLinear::from_modelopt_bytes(
                        cuda,
                        request.out_features,
                        request.in_features,
                        packed,
                        scales_e4m3,
                        multiplier,
                    )?,
                )
            }
            _ => crate::bail!(
                "Linear source encoding does not match request or provider is not compiled"
            ),
        };
        Ok(Self {
            device: device.clone(),
            request,
            selection,
            provider,
        })
    }

    pub fn request(&self) -> LinearRequest {
        self.request
    }
    pub fn selection(&self) -> &LinearSelection {
        &self.selection
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let r = self.request;
        let (rows, k) = input.dims2()?;
        if rows == 0
            || rows > r.max_rows
            || k != r.in_features
            || input.dtype() != r.contract.activations
            || !input.device().same_device(&self.device)
        {
            crate::bail!("Input violates prepared linear shape/dtype/device contract")
        }
        if r.input_layout == InputLayout::ContiguousZeroOffset
            && (!input.is_contiguous() || input.layout().start_offset() != 0)
        {
            crate::bail!("Input violates contiguous zero-offset linear contract")
        }
        let output = match &self.provider {
            PreparedProvider::Bf16(layer) => layer.forward(input)?,
            #[cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
            PreparedProvider::Fp8(layer) => layer.forward(input)?,
            #[cfg(all(feature = "cuda", feature = "cuda-legacy-fp4"))]
            PreparedProvider::Nvfp4(layer) => layer.forward(
                input,
                crate::quantized::fp4_e7c_runtime::Fp4Execution::Nvfp4W4A16Reference,
            )?,
        };
        if output.dtype() != r.contract.output || output.dims() != [rows, r.out_features] {
            crate::bail!("Provider violated prepared linear output contract")
        }
        Ok(output)
    }
}

impl crate::Module for PreparedLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        PreparedLinear::forward(self, input)
    }
}
