//! Generic manifest-driven ASD exact CUDA executor.
//!
//! Decision authority lives in the runtime Exact Profile. Executable kernel ABI,
//! entry symbol, launch geometry and artifact identity live in the external manifest.
//! This executor is architecture-agnostic: device/manifest compatibility is enforced
//! by the CUDA module registry during cold resolution.
use super::{ParamsConv1D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::backend::BackendStorage;
use crate::cuda_backend::{
    AsdCudaImplementation, CudaStorage, CudaStorageSlice as S, DeviceId, WrapErr,
};
use crate::{DType, Layout, Result};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use std::cell::RefCell;
use std::sync::Arc;

pub(crate) fn refresh_control_plane_flags() {
    super::grouped_transpose_dispatch::refresh_runtime_policy_snapshot();
}

fn layout_ok(l: &Layout) -> bool {
    l.is_contiguous() && l.start_offset() == 0
}
fn eligible(
    input: &CudaStorage,
    input_l: &Layout,
    kernel: &CudaStorage,
    kernel_l: &Layout,
) -> bool {
    input.dtype() == DType::F32
        && kernel.dtype() == DType::F32
        && layout_ok(input_l)
        && layout_ok(kernel_l)
        && input.device.id() == kernel.device.id()
}
pub(super) fn actual_uuid(input: &CudaStorage) -> Option<String> {
    let u = input.device.cuda_stream().context().uuid().ok()?;
    let hex = u
        .bytes
        .iter()
        .map(|b| format!("{:02x}", *b as u8))
        .collect::<String>();
    if hex.len() != 32 {
        return None;
    }
    Some(format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    ))
}
fn launch_plan(
    plan: &ResolvedAsdPlan,
    input: &CudaStorage,
    kernel: &CudaStorage,
) -> Result<Option<CudaStorage>> {
    let dev = input.device.clone();
    let implementation = &plan.implementation;
    let func = implementation.function();
    let out = unsafe { dev.alloc::<f32>(plan.output_count)? };
    let cfg = plan.launch_config.clone();
    let slice = match (&input.slice, &kernel.slice) {
        (S::F32(x), S::F32(w)) => {
            let mut b = func.builder();
            b.arg(x).arg(w).arg(&out);
            unsafe { b.launch(cfg) }.w()?;
            S::F32(out)
        }
        _ => crate::bail!("ASD exact CUDA executor dtype mismatch"),
    };
    if plan.trace_enabled {
        eprintln!(
            "[candle asd exact-cuda] submitted_backend=raw_exact launch_submission=success implementation={} candidate={} provider={} proof_status={} artifact_sha256={}",
            plan.selected.implementation_id(),
            implementation.candidate_id(),
            implementation.provider_name(),
            implementation.proof_status(),
            implementation.artifact_sha256().unwrap_or("none"),
        )
    }
    Ok(Some(CudaStorage { slice, device: dev }))
}

fn call(
    op: candle_kernels::asd_exact::ExactOperation,
    dim: u8,
    b: usize,
    ci: usize,
    co: usize,
    s0: usize,
    s1: usize,
    w: &[usize],
    g: usize,
    k: usize,
    st: usize,
    pad: usize,
    opad: usize,
    dil: usize,
    input_l: &Layout,
    kernel_l: &Layout,
) -> candle_kernels::asd_exact::ExactOperationCall {
    candle_kernels::asd_exact::ExactOperationCall {
        op,
        dim,
        batch: b,
        c_in: ci,
        c_out: co,
        spatial0: s0,
        spatial1: s1,
        weight_rank: w.len(),
        weight0: w.first().copied().unwrap_or(0),
        weight1: w.get(1).copied().unwrap_or(0),
        weight2: w.get(2).copied().unwrap_or(0),
        weight3: w.get(3).copied().unwrap_or(0),
        groups: g,
        kernel: k,
        stride: st,
        padding: pad,
        output_padding: opad,
        dilation: dil,
        dtype: "f32",
        input_contiguous: input_l.is_contiguous(),
        input_start_offset: input_l.start_offset(),
        weight_contiguous: kernel_l.is_contiguous(),
        weight_start_offset: kernel_l.start_offset(),
    }
}
#[derive(Clone)]
struct SelectedExact(Arc<crate::asd_runtime_extensions::RuntimeExactMatch>);

struct ResolvedAsdPlan {
    selected: SelectedExact,
    implementation: AsdCudaImplementation,
    output_count: usize,
    launch_config: LaunchConfig,
    trace_enabled: bool,
}

struct PlanCacheEntry {
    device_id: DeviceId,
    generation: u64,
    call: candle_kernels::asd_exact::ExactOperationCall,
    plan: Option<Arc<ResolvedAsdPlan>>,
}

thread_local! {
    static LAST_PLAN: RefCell<Option<PlanCacheEntry>> = const { RefCell::new(None) };
}

impl SelectedExact {
    fn profile_id(&self) -> &str {
        &self.0.profile_id
    }

    fn decision_id(&self) -> &str {
        &self.0.decision_id
    }

    fn decision_identity_sha256(&self) -> &str {
        &self.0.decision_identity_sha256
    }

    fn state(&self) -> &str {
        &self.0.state
    }

    fn implementation_id(&self) -> &str {
        &self.0.implementation_id
    }

    fn evidence_sha256(&self) -> &str {
        &self.0.evidence_sha256
    }

    fn source(&self) -> &'static str {
        "runtime_profile"
    }
}

fn selected(
    input: &CudaStorage,
    c: candle_kernels::asd_exact::ExactOperationCall,
    specialized_disabled: bool,
) -> Result<Option<SelectedExact>> {
    if specialized_disabled {
        return Ok(None);
    }

    let Some(uuid) = actual_uuid(input) else {
        return Ok(None);
    };

    Ok(crate::asd_runtime_extensions::lookup_exact(c, Some(&uuid))?
        .filter(|matched| {
            matched.execution_provider
                == crate::asd_runtime_extensions::RuntimeExecutionProvider::RawCuda
        })
        .map(SelectedExact))
}

fn resolved_plan(
    input: &CudaStorage,
    call: candle_kernels::asd_exact::ExactOperationCall,
) -> Result<Option<Arc<ResolvedAsdPlan>>> {
    let generation = input.device.asd_runtime_generation();
    let device_id = input.device.id();

    if let Some(hit) = LAST_PLAN.with(|cache| {
        let cache = cache.borrow();
        cache
            .as_ref()
            .filter(|entry| {
                entry.device_id == device_id && entry.generation == generation && entry.call == call
            })
            .map(|entry| entry.plan.clone())
    }) {
        return Ok(hit);
    }

    let specialized_disabled = super::grouped_transpose_dispatch::specialized_cuda_disabled();
    let exact_disabled = super::grouped_transpose_dispatch::exact_disabled();
    let trace_enabled = super::grouped_transpose_dispatch::asd_exact_cuda_trace_enabled();

    let plan = if exact_disabled {
        None
    } else if let Some(selected) = selected(input, call, specialized_disabled)? {
        let implementation_id = selected.implementation_id().to_owned();
        let decision_identity_sha256 = selected.decision_identity_sha256().to_owned();
        match input
            .device
            .asd_modules()
            .resolve_optional_bound(&implementation_id, &decision_identity_sha256)?
        {
            Some(implementation) => {
                let output_count = implementation.output_count();
                let launch_config = implementation.launch_config();
                Some(Arc::new(ResolvedAsdPlan {
                    selected,
                    implementation,
                    output_count,
                    launch_config,
                    trace_enabled,
                }))
            }
            None => {
                if trace_enabled {
                    eprintln!(
                        "[candle asd exact-cuda] submitted_backend=raw_exact launch_submission=unavailable implementation={} reason=external_artifact_missing",
                        implementation_id,
                    );
                }
                None
            }
        }
    } else {
        None
    };

    LAST_PLAN.with(|cache| {
        *cache.borrow_mut() = Some(PlanCacheEntry {
            device_id,
            generation,
            call,
            plan: plan.clone(),
        });
    });

    Ok(plan)
}

fn trace_selected(m: &SelectedExact) {
    eprintln!(
        "[candle asd-v3] execution_provider=raw_cuda selected_backend=raw_cuda reason=exact_asd profile_source={} asd_profile={} asd_policy={} asd_decision={} decision_identity_sha256={} asd_state={} asd_impl={} evidence={}",
        m.source(),
        m.profile_id(),
        m.profile_id(),
        m.decision_id(),
        m.decision_identity_sha256(),
        m.state(),
        m.implementation_id(),
        m.evidence_sha256(),
    )
}

#[derive(Clone, Debug)]
pub(super) struct QualifiedFallbackContext {
    pub(super) decision_id: String,
    pub(super) decision_identity_sha256: String,
    pub(super) profile_id: String,
}

pub(super) fn conv1d_qualified_fallback_context(
    input: &CudaStorage,
    input_l: &Layout,
    kernel: &CudaStorage,
    kernel_l: &Layout,
    p: &ParamsConv1D,
) -> Result<Option<QualifiedFallbackContext>> {
    if !eligible(input, input_l, kernel, kernel_l)
        || super::grouped_transpose_dispatch::exact_disabled()
    {
        return Ok(None);
    }

    let c = call(
        candle_kernels::asd_exact::ExactOperation::Conv1d,
        1,
        p.b_size,
        p.c_in,
        p.c_out,
        p.l_in,
        0,
        kernel_l.dims(),
        p.groups,
        p.k_size,
        p.stride,
        p.padding,
        0,
        p.dilation,
        input_l,
        kernel_l,
    );
    let specialized_disabled = super::grouped_transpose_dispatch::specialized_cuda_disabled();
    let Some(selected) = selected(input, c, specialized_disabled)? else {
        return Ok(None);
    };

    Ok(Some(QualifiedFallbackContext {
        decision_id: selected.decision_id().to_owned(),
        decision_identity_sha256: selected.decision_identity_sha256().to_owned(),
        profile_id: selected.profile_id().to_owned(),
    }))
}

pub(super) fn ct1d_qualified_fallback_context(
    input: &CudaStorage,
    input_l: &Layout,
    kernel: &CudaStorage,
    kernel_l: &Layout,
    p: &ParamsConvTranspose1D,
) -> Result<Option<QualifiedFallbackContext>> {
    if !eligible(input, input_l, kernel, kernel_l)
        || super::grouped_transpose_dispatch::exact_disabled()
    {
        return Ok(None);
    }

    let c = call(
        candle_kernels::asd_exact::ExactOperation::ConvTranspose1d,
        1,
        p.b_size,
        p.c_in,
        p.c_out,
        p.l_in,
        0,
        kernel_l.dims(),
        p.groups,
        p.k_size,
        p.stride,
        p.padding,
        p.output_padding,
        p.dilation,
        input_l,
        kernel_l,
    );

    let specialized_disabled = super::grouped_transpose_dispatch::specialized_cuda_disabled();

    let Some(selected) = selected(input, c, specialized_disabled)? else {
        return Ok(None);
    };

    Ok(Some(QualifiedFallbackContext {
        decision_id: selected.decision_id().to_owned(),
        decision_identity_sha256: selected.decision_identity_sha256().to_owned(),
        profile_id: selected.profile_id().to_owned(),
    }))
}

pub(super) fn ct2d_qualified_fallback_context(
    input: &CudaStorage,
    input_l: &Layout,
    kernel: &CudaStorage,
    kernel_l: &Layout,
    p: &ParamsConvTranspose2D,
) -> Result<Option<QualifiedFallbackContext>> {
    if !eligible(input, input_l, kernel, kernel_l)
        || super::grouped_transpose_dispatch::exact_disabled()
    {
        return Ok(None);
    }

    let c = call(
        candle_kernels::asd_exact::ExactOperation::ConvTranspose2d,
        2,
        p.b_size,
        p.c_in,
        p.c_out,
        p.i_h,
        p.i_w,
        kernel_l.dims(),
        p.groups,
        p.k_h,
        p.stride,
        p.padding,
        p.output_padding,
        p.dilation,
        input_l,
        kernel_l,
    );

    let specialized_disabled = super::grouped_transpose_dispatch::specialized_cuda_disabled();

    let Some(selected) = selected(input, c, specialized_disabled)? else {
        return Ok(None);
    };

    Ok(Some(QualifiedFallbackContext {
        decision_id: selected.decision_id().to_owned(),
        decision_identity_sha256: selected.decision_identity_sha256().to_owned(),
        profile_id: selected.profile_id().to_owned(),
    }))
}

pub(super) fn try_launch_conv1d(
    input: &CudaStorage,
    input_l: &Layout,
    kernel: &CudaStorage,
    kernel_l: &Layout,
    p: &ParamsConv1D,
) -> Result<Option<CudaStorage>> {
    if !eligible(input, input_l, kernel, kernel_l) {
        return Ok(None);
    }
    let c = call(
        candle_kernels::asd_exact::ExactOperation::Conv1d,
        1,
        p.b_size,
        p.c_in,
        p.c_out,
        p.l_in,
        0,
        kernel_l.dims(),
        p.groups,
        p.k_size,
        p.stride,
        p.padding,
        0,
        p.dilation,
        input_l,
        kernel_l,
    );
    let Some(plan) = resolved_plan(input, c)? else {
        return Ok(None);
    };
    if plan.trace_enabled {
        trace_selected(&plan.selected);
    }
    launch_plan(&plan, input, kernel)
}
pub(super) fn try_launch_ct1d(
    input: &CudaStorage,
    input_l: &Layout,
    kernel: &CudaStorage,
    kernel_l: &Layout,
    p: &ParamsConvTranspose1D,
) -> Result<Option<CudaStorage>> {
    if !eligible(input, input_l, kernel, kernel_l) {
        return Ok(None);
    }
    let c = call(
        candle_kernels::asd_exact::ExactOperation::ConvTranspose1d,
        1,
        p.b_size,
        p.c_in,
        p.c_out,
        p.l_in,
        0,
        kernel_l.dims(),
        p.groups,
        p.k_size,
        p.stride,
        p.padding,
        p.output_padding,
        p.dilation,
        input_l,
        kernel_l,
    );
    let Some(plan) = resolved_plan(input, c)? else {
        return Ok(None);
    };
    if plan.trace_enabled {
        trace_selected(&plan.selected);
    }
    launch_plan(&plan, input, kernel)
}
pub(super) fn try_launch_ct2d(
    input: &CudaStorage,
    input_l: &Layout,
    kernel: &CudaStorage,
    kernel_l: &Layout,
    p: &ParamsConvTranspose2D,
) -> Result<Option<CudaStorage>> {
    if !eligible(input, input_l, kernel, kernel_l) {
        return Ok(None);
    }
    let c = call(
        candle_kernels::asd_exact::ExactOperation::ConvTranspose2d,
        2,
        p.b_size,
        p.c_in,
        p.c_out,
        p.i_h,
        p.i_w,
        kernel_l.dims(),
        p.groups,
        p.k_h,
        p.stride,
        p.padding,
        p.output_padding,
        p.dilation,
        input_l,
        kernel_l,
    );
    let Some(plan) = resolved_plan(input, c)? else {
        return Ok(None);
    };
    if plan.trace_enabled {
        trace_selected(&plan.selected);
    }
    launch_plan(&plan, input, kernel)
}
