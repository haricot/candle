use super::*;
use crate::{DType, Tensor};

struct Prepared {
    reservation: Reservation,
    spec: LinearSpec,
    decision: LinearDecision,
    weight: Tensor,
    bias: Option<Tensor>,
}

/// Inference-only private weight snapshot. Input BF16 GEMM rounds before bias addition.
#[derive(Clone)]
pub struct PreparedBf16Linear(Arc<Prepared>);

impl PreparedBf16Linear {
    pub(super) fn prepare(
        context: &DispatchContext,
        weight: &Tensor,
        bias: Option<&Tensor>,
        max_rows: usize,
    ) -> Result<Self> {
        let (out_features, in_features) = weight.dims2()?;
        let spec = LinearSpec {
            in_features,
            out_features,
            bias: bias.is_some(),
            max_rows,
        };
        if weight.dtype() != DType::BF16 || !weight.device().same_device(&context.0.device) {
            crate::bail!("Dispatch requires BF16 weights on the session device")
        }
        if let Some(bias) = bias {
            if bias.dims() != [out_features]
                || bias.dtype() != DType::BF16
                || !bias.device().same_device(&context.0.device)
            {
                crate::bail!("Dispatch bias must be BF16 [out_features] on the session device")
            }
        }
        let used = *context.0.reserved.lock().unwrap();
        let budget = ResourceBudget {
            resident_bytes: context.0.budget.resident_bytes - used,
            ..context.0.budget
        };
        let decision = select_bf16_linear(
            spec,
            context.0.caps,
            context.0.inventory,
            budget,
            context.0.policy.as_deref(),
        )?;
        let mut used = context.0.reserved.lock().unwrap();
        if decision.memory.resident_bytes > context.0.budget.resident_bytes - *used {
            crate::bail!("Dispatch resident budget changed during preparation; retry admission");
        }
        *used += decision.memory.resident_bytes;
        let reservation = Reservation {
            session: context.0.clone(),
            bytes: decision.memory.resident_bytes,
        };
        drop(used);
        let dtype = if decision.backend == Bf16LinearBackend::CudaBf16 {
            DType::BF16
        } else {
            DType::F32
        };
        // Conversion allocates a new F32 buffer; same-dtype BF16 requires an explicit compact snapshot.
        let weight = if dtype == DType::BF16 {
            weight.detach().force_contiguous()?
        } else {
            weight.detach().to_dtype(dtype)?
        };
        let bias = bias.map(|b| b.detach().to_dtype(DType::F32)).transpose()?;
        Ok(Self(Arc::new(Prepared {
            reservation,
            spec,
            decision,
            weight,
            bias,
        })))
    }

    pub fn decision(&self) -> &LinearDecision {
        &self.0.decision
    }
    pub fn spec(&self) -> LinearSpec {
        self.0.spec
    }
    pub fn context_id(&self) -> u64 {
        self.0.reservation.session.id
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let prepared = &self.0;
        let spec = prepared.spec;
        if input.dtype() != DType::BF16
            || !input
                .device()
                .same_device(&prepared.reservation.session.device)
        {
            crate::bail!("Dispatch input must be BF16 on the prepared session device")
        }
        if input.dims().last() != Some(&spec.in_features) {
            crate::bail!(
                "Dispatch input last dimension must equal {}",
                spec.in_features
            )
        }
        let rows = input.elem_count() / spec.in_features;
        if rows > spec.max_rows {
            crate::bail!("Dispatch input exceeds prepared max_rows={}", spec.max_rows)
        }
        let mut shape = input.dims().to_vec();
        *shape.last_mut().unwrap() = spec.out_features;
        if rows == 0 {
            return Tensor::zeros(shape, DType::BF16, input.device());
        }
        let dtype = prepared.weight.dtype();
        let input = if dtype == DType::BF16 {
            input.detach().force_contiguous()?
        } else {
            input.detach().to_dtype(DType::F32)?
        }
        .reshape((rows, spec.in_features))?;
        let weight = prepared.weight.t()?;
        let output = match prepared.decision.backend {
            Bf16LinearBackend::CpuF32 => input.matmul(&weight)?,
            #[cfg(feature = "cuda")]
            Bf16LinearBackend::CudaBf16 | Bf16LinearBackend::CudaF32 => {
                input.apply_op2_no_bwd(&weight, &StrictGemm)?
            }
            #[cfg(not(feature = "cuda"))]
            _ => crate::bail!("Dispatch CUDA provider is not compiled"),
        }
        .to_dtype(DType::BF16)?;
        let output = match &prepared.bias {
            Some(bias) => output
                .to_dtype(DType::F32)?
                .broadcast_add(bias)?
                .to_dtype(DType::BF16)?,
            None => output,
        };
        output.reshape(shape)
    }
}

#[cfg(feature = "cuda")]
struct StrictGemm;

#[cfg(feature = "cuda")]
impl crate::CustomOp2 for StrictGemm {
    fn name(&self) -> &'static str {
        "candle.dispatch.bf16-linear.f32-accum.v1"
    }
    fn cpu_fwd(
        &self,
        _: &crate::CpuStorage,
        _: &crate::Layout,
        _: &crate::CpuStorage,
        _: &crate::Layout,
    ) -> Result<(crate::CpuStorage, crate::Shape)> {
        crate::bail!("Dispatch CUDA GEMM requires CUDA storage")
    }
    fn cuda_fwd(
        &self,
        lhs: &crate::CudaStorage,
        lhs_l: &crate::Layout,
        rhs: &crate::CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(crate::CudaStorage, crate::Shape)> {
        let (m, k) = lhs_l.shape().dims2()?;
        let (rk, n) = rhs_l.shape().dims2()?;
        if k != rk {
            crate::bail!("Dispatch GEMM K mismatch")
        }
        Ok((
            lhs.matmul_dispatch_f32_accum(rhs, (1, m, n, k), lhs_l, rhs_l)?,
            (m, n).into(),
        ))
    }
}

impl crate::Module for PreparedBf16Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        PreparedBf16Linear::forward(self, input)
    }
}
