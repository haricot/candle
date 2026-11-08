//! Format-neutral planning boundary. A policy may prefer only an eligible provider.
//!
//! Weights, activation dtype, accumulation and output are independent contracts.
//! Provider IDs name implementations, never an implicit dtype conversion.
use super::{Architecture, DeviceCapabilities, LinearMemory, ResourceBudget};
use crate::{DType, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WeightFormat {
    F16,
    Bf16,
    Fp8E4m3Fn { block_n: usize, block_k: usize },
    Nvfp4ModelOpt,
    Mxfp4Gguf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiasSemantics {
    None,
    AfterAccumulation,
    /// Round the GEMM to BF16, add BF16 bias in F32, then round to BF16 again.
    AfterBf16GemmRounding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Approximation {
    /// No additional weight/activation quantization beyond the supplied representation.
    PreserveStoredValues,
    AllowAdditionalQuantization,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearContract {
    pub weights: WeightFormat,
    pub activations: DType,
    pub accumulation: DType,
    pub output: DType,
    pub bias: BiasSemantics,
    pub approximation: Approximation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputLayout {
    ContiguousZeroOffset,
    Strided,
}

/// Rank-two X[rows, in_features] times W[out_features, in_features].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearRequest {
    pub contract: LinearContract,
    pub in_features: usize,
    pub out_features: usize,
    pub max_rows: usize,
    pub input_layout: InputLayout,
}

impl LinearRequest {
    pub fn validate(self) -> Result<()> {
        if self.in_features == 0 || self.out_features == 0 || self.max_rows == 0 {
            crate::bail!("Linear contract dimensions and max_rows must be positive")
        }
        for elements in [
            self.in_features.checked_mul(self.out_features),
            self.max_rows.checked_mul(self.in_features),
            self.max_rows.checked_mul(self.out_features),
        ] {
            if elements.is_none() {
                crate::bail!("Linear contract dimension overflow")
            }
        }
        Ok(())
    }
}

/// Stable implementation identity; adding a provider needs no new dtype/backend enum variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProviderId(pub &'static str);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderRejection {
    DeviceKind,
    NotCompiled,
    NumericalContract,
    Architecture,
    Layout,
    ResidentBudget,
    WorkspaceBudget,
    Constraint(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderAssessment {
    pub provider: ProviderId,
    pub memory: Option<LinearMemory>,
    pub rejection: Option<ProviderRejection>,
}

impl ProviderAssessment {
    pub fn eligible(provider: ProviderId, memory: LinearMemory) -> Self {
        Self {
            provider,
            memory: Some(memory),
            rejection: None,
        }
    }
    pub fn rejected(provider: ProviderId, reason: ProviderRejection) -> Self {
        Self {
            provider,
            memory: None,
            rejection: Some(reason),
        }
    }
}

/// A planning adapter, not proof that a particular binary/device can execute a kernel.
/// Runtime preparation must inspect the actual device and validate its source payload.
pub trait LinearProvider {
    fn id(&self) -> ProviderId;
    fn device_kind(&self) -> DeviceKind;
    fn assess(
        &self,
        request: &LinearRequest,
        caps: DeviceCapabilities,
    ) -> Result<ProviderAssessment>;
}

/// Future ASD policies implement this preference boundary; eligibility stays in Candle.
pub trait LinearPreference: Send + Sync {
    fn select(
        &self,
        request: &LinearRequest,
        eligible: &[ProviderAssessment],
    ) -> Option<ProviderId>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearSelection {
    pub selected: ProviderAssessment,
    pub candidates: Vec<ProviderAssessment>,
    pub policy_selected: bool,
}

/// Pure planning with caller-declared capabilities. Budget is one caller-owned partition;
/// this function does not reserve memory or coordinate concurrent/per-format planners.
pub fn select_linear(
    request: LinearRequest,
    caps: DeviceCapabilities,
    budget: ResourceBudget,
    providers: &[&dyn LinearProvider],
    preference: Option<&dyn LinearPreference>,
) -> Result<LinearSelection> {
    request.validate()?;
    let kind = match caps.architecture {
        Architecture::Cpu => DeviceKind::Cpu,
        Architecture::Cuda { .. } => DeviceKind::Cuda,
    };
    let mut candidates: Vec<ProviderAssessment> = Vec::new();
    for provider in providers {
        let id = provider.id();
        if candidates.iter().any(|c| c.provider == id) {
            crate::bail!("Duplicate linear provider ID: {id:?}")
        }
        let mut candidate = if provider.device_kind() != kind {
            ProviderAssessment::rejected(id, ProviderRejection::DeviceKind)
        } else {
            provider.assess(&request, caps)?
        };
        if candidate.provider != id {
            crate::bail!("Linear provider assessment changed its identity")
        }
        if candidate.rejection.is_none() {
            let memory = candidate.memory.ok_or_else(|| {
                crate::Error::Msg("Eligible linear provider omitted its memory estimate".into())
            })?;
            candidate.rejection = if memory.resident_bytes > budget.resident_bytes {
                Some(ProviderRejection::ResidentBudget)
            } else if memory.workspace_bytes > budget.workspace_bytes {
                Some(ProviderRejection::WorkspaceBudget)
            } else {
                None
            };
        }
        candidates.push(candidate);
    }
    let eligible: Vec<_> = candidates
        .iter()
        .filter(|c| c.rejection.is_none())
        .cloned()
        .collect();
    // Never ask a policy to rescue an empty eligible set.
    if eligible.is_empty() {
        crate::bail!("No eligible linear provider: {candidates:?}")
    }
    let preferred = preference.and_then(|p| p.select(&request, &eligible));
    let selected = match preferred {
        Some(id) => eligible.iter().find(|c| c.provider == id).ok_or_else(|| {
            crate::Error::Msg(format!(
                "Linear policy requested an ineligible provider {id:?}: {candidates:?}"
            ))
        })?,
        None => &eligible[0],
    }
    .clone();
    Ok(LinearSelection {
        selected,
        candidates,
        policy_selected: preferred.is_some(),
    })
}
