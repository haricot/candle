//! ARCH-E7C: explicitly typed FP4 projection selection.
//!
//! MXFP4 GGUF remains a GgmlDType::Mxfp4/QMatMul. ModelOpt NVFP4
//! is NOT a GGUF dtype and must be constructed from its own packed payload.
//! No silent MXFP4 -> NVFP4 conversion or W4A16 -> W4A8 dispatch.
use crate::{CudaDevice, Module, Result, Tensor};
use crate::quantized::{GgmlDType, QMatMul, QTensor};
use crate::quantized::nvfp4_arch_e7::{E7Mode, Nvfp4E7Linear};
use crate::quantized::nvfp4_e7c_safetensors::ModelOptNvfp4Shards;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp4Format {
    Mxfp4Gguf,
    Nvfp4ModelOpt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp4Execution {
    Mxfp4Existing,
    Nvfp4W4A16Reference,
    Nvfp4W4A8Reference,
    Nvfp4W4A8Dp4a,
}

pub enum Fp4E7CLinear {
    Mxfp4(QMatMul),
    Nvfp4(Nvfp4E7Linear),
}

impl Fp4E7CLinear {
    pub fn from_mxfp4(q: QTensor) -> Result<Self> {
        if q.dtype() != GgmlDType::Mxfp4 {
            crate::bail!("E7C requires actual GGUF MXFP4 QTensor, got {:?}", q.dtype());
        }
        Ok(Self::Mxfp4(QMatMul::from_qtensor(q)?))
    }

    pub fn from_modelopt_bytes(
        device: &CudaDevice, n: usize, k: usize,
        packed: &[u8], e4m3_scales: &[u8], multiplier: f32,
    ) -> Result<Self> {
        Ok(Self::Nvfp4(Nvfp4E7Linear::from_modelopt_bytes(
            device, n, k, packed, e4m3_scales, multiplier,
        )?))
    }

    pub fn from_modelopt_shards(
        device: &CudaDevice, n: usize, k: usize, prefix: &str,
        sources: &ModelOptNvfp4Shards,
    ) -> Result<Self> {
        let (packed, scales, multiplier) = sources.load(n, k, prefix)?;
        Self::from_modelopt_bytes(device, n, k, &packed, &scales, multiplier)
    }

    pub fn format(&self) -> Fp4Format {
        match self {
            Self::Mxfp4(_) => Fp4Format::Mxfp4Gguf,
            Self::Nvfp4(_) => Fp4Format::Nvfp4ModelOpt,
        }
    }

    pub fn forward(&self, input: &Tensor, execution: Fp4Execution) -> Result<Tensor> {
        match (self, execution) {
            (Self::Mxfp4(mm), Fp4Execution::Mxfp4Existing) => mm.forward(input),
            (Self::Nvfp4(op), Fp4Execution::Nvfp4W4A16Reference) =>
                op.forward(input, E7Mode::W4A16Reference),
            (Self::Nvfp4(op), Fp4Execution::Nvfp4W4A8Reference) =>
                op.forward(input, E7Mode::W4A8Reference),
            (Self::Nvfp4(op), Fp4Execution::Nvfp4W4A8Dp4a) =>
                op.forward(input, E7Mode::W4A8Dp4a),
            (Self::Mxfp4(_), _) => crate::bail!("E7C refuses NVFP4 execution on GGUF MXFP4"),
            (Self::Nvfp4(_), _) => crate::bail!("E7C refuses MXFP4 execution on ModelOpt NVFP4"),
        }
    }
}
