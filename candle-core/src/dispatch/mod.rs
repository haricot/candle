//! Prepared inference dispatch. Hardware eligibility is separate from policy preference.

mod linear;
pub use linear::PreparedBf16Linear;

use crate::{Device, Result};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

static NEXT_CONTEXT: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    Cpu,
    Cuda { major: i32, minor: i32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceCapabilities {
    pub architecture: Architecture,
}

impl DeviceCapabilities {
    /// Declarative facts for pure planning; execution always inspects the actual device.
    pub fn native_bf16_gemm(self) -> bool {
        matches!(self.architecture, Architecture::Cuda { major: 8..=12, .. })
    }

    fn inspect(device: &Device) -> Result<Self> {
        let architecture = match device {
            Device::Cpu => Architecture::Cpu,
            #[cfg(feature = "cuda")]
            Device::Cuda(cuda) => {
                let (major, minor) = cuda
                    .cuda_stream()
                    .context()
                    .compute_capability()
                    .map_err(crate::Error::msg)?;
                Architecture::Cuda { major, minor }
            }
            _ => crate::bail!("Dispatch S0 supports CPU and CUDA devices"),
        };
        Ok(Self { architecture })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LinearBackend {
    CpuF32,
    CudaF32,
    CudaBf16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearSpec {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
    pub max_rows: usize,
}

impl LinearSpec {
    pub fn validate(self) -> Result<()> {
        if self.in_features == 0 || self.out_features == 0 || self.max_rows == 0 {
            crate::bail!("Dispatch linear dimensions and max_rows must be positive")
        }
        if [self.in_features, self.out_features, self.max_rows]
            .iter()
            .any(|&d| d > i32::MAX as usize)
        {
            crate::bail!("Dispatch linear dimensions exceed the CUDA i32 contract")
        }
        Ok(())
    }

    pub fn memory(self, backend: LinearBackend) -> Result<LinearMemory> {
        self.validate()?;
        let checked = || -> Option<LinearMemory> {
            let weights = self.in_features.checked_mul(self.out_features)?;
            let source_bytes = weights.checked_mul(2)?.checked_add(if self.bias {
                self.out_features.checked_mul(2)?
            } else {
                0
            })?;
            let resident_bytes = weights
                .checked_mul(if backend == LinearBackend::CudaBf16 {
                    2
                } else {
                    4
                })?
                .checked_add(if self.bias {
                    self.out_features.checked_mul(4)?
                } else {
                    0
                })?;
            // Logical tensor buffers only; retained outputs and library/allocator reserves are external.
            let workspace_bytes = self
                .max_rows
                .checked_mul(self.in_features)?
                .checked_mul(4)?
                .checked_add(
                    self.max_rows
                        .checked_mul(self.out_features)?
                        .checked_mul(16)?,
                )?;
            Some(LinearMemory {
                source_bytes,
                resident_bytes,
                workspace_bytes,
            })
        };
        checked().ok_or_else(|| crate::Error::Msg("Dispatch linear memory overflow".into()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearMemory {
    /// Already-loaded source tensors, accounted for before choosing the session budget.
    pub source_bytes: usize,
    /// New private snapshots, released when the last prepared-layer clone is dropped.
    pub resident_bytes: usize,
    /// Conservative per-call logical buffer estimate, excluding backend-private allocations.
    pub workspace_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceBudget {
    pub resident_bytes: usize,
    pub workspace_bytes: usize,
}

impl ResourceBudget {
    /// Partition available memory after external weights, KV, library workspaces and headroom.
    pub fn from_available(
        available: usize,
        reserved: usize,
        headroom: usize,
        workspace: usize,
    ) -> Result<Self> {
        let resident_bytes = available
            .checked_sub(reserved)
            .and_then(|v| v.checked_sub(headroom))
            .and_then(|v| v.checked_sub(workspace))
            .ok_or_else(|| {
                crate::Error::Msg("Dispatch memory reservations exceed available memory".into())
            })?;
        Ok(Self {
            resident_bytes,
            workspace_bytes: workspace,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectionReason {
    DeviceKind,
    NativeBf16Unavailable,
    ProviderUnavailable,
    ResidentBudget,
    WorkspaceBudget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateAssessment {
    pub backend: LinearBackend,
    pub rejection: Option<RejectionReason>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearDecision {
    pub backend: LinearBackend,
    pub candidates: Vec<CandidateAssessment>,
    pub memory: LinearMemory,
    pub policy_selected: bool,
}

/// A preference among eligible providers, evaluated only while preparing a layer.
pub trait SelectionPolicy: Send + Sync {
    fn select(
        &self,
        spec: &LinearSpec,
        caps: &DeviceCapabilities,
        eligible: &[LinearBackend],
    ) -> Option<LinearBackend>;
}

#[derive(Clone, Copy, Debug)]
pub struct RequireBackend(pub LinearBackend);

impl SelectionPolicy for RequireBackend {
    fn select(
        &self,
        _: &LinearSpec,
        _: &DeviceCapabilities,
        _: &[LinearBackend],
    ) -> Option<LinearBackend> {
        Some(self.0)
    }
}

/// Actual provider availability is established by the session, not by a policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearInventory {
    pub cpu_f32: bool,
    pub cuda_f32: bool,
    pub cuda_bf16: bool,
}

pub fn select_bf16_linear(
    spec: LinearSpec,
    caps: DeviceCapabilities,
    inventory: LinearInventory,
    budget: ResourceBudget,
    policy: Option<&dyn SelectionPolicy>,
) -> Result<LinearDecision> {
    spec.validate()?;
    let mut candidates = Vec::new();
    let mut eligible = Vec::new();
    for backend in [
        LinearBackend::CudaBf16,
        LinearBackend::CudaF32,
        LinearBackend::CpuF32,
    ] {
        let memory = spec.memory(backend)?;
        let available = match backend {
            LinearBackend::CpuF32 => inventory.cpu_f32,
            LinearBackend::CudaF32 => inventory.cuda_f32,
            LinearBackend::CudaBf16 => inventory.cuda_bf16,
        };
        let device_matches = match backend {
            LinearBackend::CpuF32 => caps.architecture == Architecture::Cpu,
            _ => matches!(caps.architecture, Architecture::Cuda { .. }),
        };
        let rejection = if !device_matches {
            Some(RejectionReason::DeviceKind)
        } else if backend == LinearBackend::CudaBf16 && !caps.native_bf16_gemm() {
            Some(RejectionReason::NativeBf16Unavailable)
        } else if !available {
            Some(RejectionReason::ProviderUnavailable)
        } else if memory.resident_bytes > budget.resident_bytes {
            Some(RejectionReason::ResidentBudget)
        } else if memory.workspace_bytes > budget.workspace_bytes {
            Some(RejectionReason::WorkspaceBudget)
        } else {
            None
        };
        if rejection.is_none() {
            eligible.push(backend);
        }
        candidates.push(CandidateAssessment { backend, rejection });
    }
    let preferred = policy.and_then(|p| p.select(&spec, &caps, &eligible));
    let backend = match preferred {
        Some(backend) if eligible.contains(&backend) => backend,
        Some(_) => crate::bail!(
            "Dispatch policy requested an ineligible BF16 linear provider: {candidates:?}"
        ),
        None => *eligible.first().ok_or_else(|| {
            crate::Error::Msg(format!(
                "Dispatch has no eligible BF16 linear provider: {candidates:?}"
            ))
        })?,
    };
    Ok(LinearDecision {
        backend,
        candidates,
        memory: spec.memory(backend)?,
        policy_selected: preferred.is_some(),
    })
}

struct Session {
    id: u64,
    device: Device,
    caps: DeviceCapabilities,
    inventory: LinearInventory,
    budget: ResourceBudget,
    reserved: Mutex<usize>,
    policy: Option<Arc<dyn SelectionPolicy>>,
}

/// One device/stream and one resource partition. Clones share resident admission.
#[derive(Clone)]
pub struct DispatchContext(Arc<Session>);

impl DispatchContext {
    pub fn new(
        device: &Device,
        budget: ResourceBudget,
        policy: Option<Arc<dyn SelectionPolicy>>,
    ) -> Result<Self> {
        let caps = DeviceCapabilities::inspect(device)?;
        let is_cuda = matches!(caps.architecture, Architecture::Cuda { .. });
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = device {
            use crate::cuda_backend::kernels;
            cuda.get_or_load_func("cast_bf16_f32", &kernels::CAST)?;
            cuda.get_or_load_func("cast_f32_bf16", &kernels::CAST)?;
            cuda.get_or_load_func("ucopy_f32", &kernels::UNARY)?;
            cuda.get_or_load_func("badd_f32", &kernels::BINARY)?;
            if caps.native_bf16_gemm() {
                cuda.get_or_load_func("ucopy_bf16", &kernels::UNARY)?;
            }
        }

        Ok(Self(Arc::new(Session {
            id: NEXT_CONTEXT.fetch_add(1, Ordering::Relaxed),
            device: device.clone(),
            caps,
            inventory: LinearInventory {
                cpu_f32: device.is_cpu(),
                cuda_f32: is_cuda,
                cuda_bf16: is_cuda,
            },
            budget,
            reserved: Mutex::new(0),
            policy,
        })))
    }

    pub fn capabilities(&self) -> DeviceCapabilities {
        self.0.caps
    }
    pub fn id(&self) -> u64 {
        self.0.id
    }
    pub fn reserved_resident_bytes(&self) -> usize {
        *self.0.reserved.lock().unwrap()
    }

    pub fn prepare_bf16_linear(
        &self,
        weight: &crate::Tensor,
        bias: Option<&crate::Tensor>,
        max_rows: usize,
    ) -> Result<PreparedBf16Linear> {
        PreparedBf16Linear::prepare(self, weight, bias, max_rows)
    }
}

struct Reservation {
    session: Arc<Session>,
    bytes: usize,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        *self.session.reserved.lock().unwrap() -= self.bytes;
    }
}
