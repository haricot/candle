use crate::{DType, Layout};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use super::{ParamsConvTranspose1D, ParamsConvTranspose2D};

const POLICY_REQUEST_MASK: u32 = 0b11;
const POLICY_REQUEST_AUTO: u32 = 0;
const POLICY_REQUEST_RAW: u32 = 1;
const POLICY_REQUEST_CUDNN: u32 = 2;
const POLICY_REQUEST_INVALID: u32 = 3;
const POLICY_FORCE_RAW: u32 = 1 << 2;
const POLICY_CUDNN_STRICT: u32 = 1 << 3;
const POLICY_RAW_STRICT: u32 = 1 << 4;
const POLICY_EXACT_DISABLED: u32 = 1 << 5;
const POLICY_SPECIALIZED_CUDA_DISABLED: u32 = 1 << 6;
const POLICY_TRACE_GROUPED: u32 = 1 << 7;
const POLICY_TRACE_ASD: u32 = 1 << 8;
const POLICY_TRACE_ASD_EXACT_CUDA: u32 = 1 << 9;
const POLICY_QUALIFIED_FALLBACK_REQUIRED: u32 = 1 << 10;
const POLICY_INITIALIZED: u32 = 1 << 31;

static RUNTIME_POLICY: AtomicU32 = AtomicU32::new(0);

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

fn read_runtime_policy_from_env() -> u32 {
    let request = match std::env::var("CANDLE_GROUPED_TRANSPOSE_DISPATCH")
        .ok()
        .as_deref()
    {
        None | Some("auto") => POLICY_REQUEST_AUTO,
        Some("raw") => POLICY_REQUEST_RAW,
        Some("cudnn") => POLICY_REQUEST_CUDNN,
        Some(_) => POLICY_REQUEST_INVALID,
    };
    let mut bits = POLICY_INITIALIZED | request;
    if std::env::var_os("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL").is_some() {
        bits |= POLICY_FORCE_RAW;
    }
    if std::env::var_os("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT").is_some() {
        bits |= POLICY_CUDNN_STRICT;
    }
    if std::env::var_os("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT").is_some() {
        bits |= POLICY_RAW_STRICT;
    }
    if env_truthy("CANDLE_ASD_EXACT_DISABLE") {
        bits |= POLICY_EXACT_DISABLED;
    }
    if env_truthy("CANDLE_ASD_EXACT_CUDA_DISABLE")
        || env_truthy("CANDLE_SM61_EXACT_GROUPED_DISABLE")
    {
        bits |= POLICY_SPECIALIZED_CUDA_DISABLED;
    }
    if env_truthy("CANDLE_GROUPED_TRANSPOSE_TRACE") {
        bits |= POLICY_TRACE_GROUPED;
    }
    if env_truthy("CANDLE_ASD_EXACT_TRACE") {
        bits |= POLICY_TRACE_ASD;
    }
    if env_truthy("CANDLE_ASD_EXACT_CUDA_TRACE") || env_truthy("CANDLE_SM61_EXACT_GROUPED_TRACE") {
        bits |= POLICY_TRACE_ASD_EXACT_CUDA;
    }
    if env_truthy("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED") {
        bits |= POLICY_QUALIFIED_FALLBACK_REQUIRED;
    }
    bits
}

pub(crate) fn refresh_runtime_policy_snapshot() {
    RUNTIME_POLICY.store(read_runtime_policy_from_env(), Ordering::Release);
}

fn runtime_policy() -> u32 {
    let current = RUNTIME_POLICY.load(Ordering::Acquire);
    if current & POLICY_INITIALIZED != 0 {
        return current;
    }
    let loaded = read_runtime_policy_from_env();
    RUNTIME_POLICY.store(loaded, Ordering::Release);
    loaded
}

fn request_from_policy(bits: u32) -> GroupedTransposeDispatchRequest {
    match bits & POLICY_REQUEST_MASK {
        POLICY_REQUEST_AUTO => GroupedTransposeDispatchRequest::Auto,
        POLICY_REQUEST_RAW => GroupedTransposeDispatchRequest::Raw,
        POLICY_REQUEST_CUDNN => GroupedTransposeDispatchRequest::Cudnn,
        _ => GroupedTransposeDispatchRequest::Invalid,
    }
}

pub(super) fn specialized_cuda_disabled() -> bool {
    runtime_policy() & POLICY_SPECIALIZED_CUDA_DISABLED != 0
}

pub(super) fn exact_disabled() -> bool {
    runtime_policy() & POLICY_EXACT_DISABLED != 0
}

pub(super) fn asd_exact_cuda_trace_enabled() -> bool {
    runtime_policy() & (POLICY_TRACE_ASD_EXACT_CUDA | POLICY_TRACE_ASD) != 0
}

pub(super) fn qualified_fallback_required() -> bool {
    runtime_policy() & POLICY_QUALIFIED_FALLBACK_REQUIRED != 0
}

#[derive(Clone, Copy)]
pub(super) struct GroupedTransposeHotPolicy {
    bits: u32,
}

impl GroupedTransposeHotPolicy {
    pub(super) fn explicit_cudnn_required(self) -> crate::Result<bool> {
        let requested = request_from_policy(self.bits);
        check_backend_requests(
            Some(requested.as_str()),
            self.bits & POLICY_CUDNN_STRICT != 0,
            self.bits & POLICY_FORCE_RAW != 0,
            self.bits & POLICY_RAW_STRICT != 0,
        )
    }

    pub(super) fn automatic_exact_eligible(self) -> bool {
        request_from_policy(self.bits) == GroupedTransposeDispatchRequest::Auto
            && self.bits & POLICY_FORCE_RAW == 0
            && self.bits & POLICY_EXACT_DISABLED == 0
    }
}

pub(super) fn hot_policy() -> GroupedTransposeHotPolicy {
    GroupedTransposeHotPolicy {
        bits: runtime_policy(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GroupedTransposeDim {
    D1,
    D2,
}

impl GroupedTransposeDim {
    const fn as_str(self) -> &'static str {
        match self {
            Self::D1 => "1d",
            Self::D2 => "2d",
        }
    }
}

// Resolve explicit requirements before any specialized SM61 launch.
// A contradictory request is an error, never an undocumented preference.
fn check_backend_requests(
    requested: Option<&str>,
    cudnn_strict: bool,
    force_raw: bool,
    raw_strict: bool,
) -> crate::Result<bool> {
    let cudnn = cudnn_strict || requested == Some("cudnn");
    let raw = force_raw || raw_strict || requested == Some("raw");
    if cudnn && raw {
        crate::bail!("grouped ConvTranspose conflicting explicit CUDA and cuDNN backend requests")
    }
    Ok(cudnn)
}

pub(super) fn explicit_cudnn_required() -> crate::Result<bool> {
    hot_policy().explicit_cudnn_required()
}

fn auto_request(requested: Option<&str>, force_raw: bool, asd_disabled: bool) -> bool {
    matches!(requested, None | Some("auto")) && !force_raw && !asd_disabled
}

pub(super) fn automatic_exact_eligible() -> bool {
    hot_policy().automatic_exact_eligible()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GroupedTransposeCudaRule {
    raw_min_groups_1d: Option<usize>,
    raw_min_groups_2d: Option<usize>,
}

const fn dispatch_rule_for_sm(_sm: u32) -> GroupedTransposeCudaRule {
    // Generalized thresholds remain intentionally unset. V2 exact-domain
    // decisions are matched before this legacy rule and never imply groups>=N.
    GroupedTransposeCudaRule {
        raw_min_groups_1d: None,
        raw_min_groups_2d: None,
    }
}

fn auto_prefers_raw(dim: GroupedTransposeDim, groups: usize, sm: u32) -> bool {
    let rule = dispatch_rule_for_sm(sm);
    let threshold = match dim {
        GroupedTransposeDim::D1 => rule.raw_min_groups_1d,
        GroupedTransposeDim::D2 => rule.raw_min_groups_2d,
    };
    threshold.is_some_and(|min_groups| groups >= min_groups)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupedTransposeDispatchRequest {
    Auto,
    Raw,
    Cudnn,
    Invalid,
}

impl GroupedTransposeDispatchRequest {
    fn parse(value: Option<&str>) -> Self {
        match value {
            Some("raw") => Self::Raw,
            Some("cudnn") => Self::Cudnn,
            Some("auto") | None => Self::Auto,
            Some(_) => Self::Invalid,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Raw => "raw",
            Self::Cudnn => "cudnn",
            Self::Invalid => "invalid",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupedTransposeDispatchPath {
    Raw,
    Cudnn,
}

impl GroupedTransposeDispatchPath {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Cudnn => "cudnn",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupedTransposeDispatchReason {
    ForceKernelOverride,
    ExplicitRaw,
    ExplicitCudnn,
    ExactAsd,
    AutoRule,
    InvalidFallsBackToAuto,
}

impl GroupedTransposeDispatchReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ForceKernelOverride => "force_kernel_override",
            Self::ExplicitRaw => "explicit_raw",
            Self::ExplicitCudnn => "explicit_cudnn",
            Self::ExactAsd => "exact_asd",
            Self::AutoRule => "auto_rule",
            Self::InvalidFallsBackToAuto => "invalid_falls_back_to_auto",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct GroupedTransposeDispatchDecision {
    requested: GroupedTransposeDispatchRequest,
    selected: GroupedTransposeDispatchPath,
    reason: GroupedTransposeDispatchReason,
    asd_profile_id: Option<Arc<str>>,
    asd_decision_id: Option<Arc<str>>,
    asd_decision_identity_sha256: Option<Arc<str>>,
    asd_state: Option<Arc<str>>,
    asd_impl: Option<Arc<str>>,
}

impl GroupedTransposeDispatchDecision {
    #[allow(dead_code)]
    pub(super) fn prefers_raw(&self) -> bool {
        self.selected == GroupedTransposeDispatchPath::Raw
    }

    pub(super) fn is_exact_asd(&self) -> bool {
        self.reason == GroupedTransposeDispatchReason::ExactAsd
    }

    pub(super) fn exact_requires_cudnn(&self) -> bool {
        self.is_exact_asd() && self.selected == GroupedTransposeDispatchPath::Cudnn
    }

    pub(super) fn exact_requires_raw(&self) -> bool {
        self.is_exact_asd() && self.selected == GroupedTransposeDispatchPath::Raw
    }

    pub(super) fn exact_decision_id(&self) -> Option<&str> {
        self.is_exact_asd()
            .then(|| self.asd_decision_id.as_deref())
            .flatten()
    }

    pub(super) fn exact_decision_identity_sha256(&self) -> Option<&str> {
        self.is_exact_asd()
            .then(|| self.asd_decision_identity_sha256.as_deref())
            .flatten()
    }

    pub(super) fn exact_profile_id(&self) -> Option<&str> {
        self.is_exact_asd()
            .then(|| self.asd_profile_id.as_deref())
            .flatten()
    }

    pub(super) fn trace_submission(&self, backend: &str) {
        if !grouped_transpose_trace_enabled() && !asd_exact_trace_enabled() {
            return;
        }
        eprintln!(
            "[candle grouped-conv-transpose] submitted_backend={} launch_submission=success asd_profile={} asd_policy={} asd_decision={} asd_decision_identity_sha256={} asd_state={} asd_impl={}",
            backend,
            self.asd_profile_id.as_deref().unwrap_or("none"),
            self.asd_profile_id.as_deref().unwrap_or("none"),
            self.asd_decision_id.as_deref().unwrap_or("none"),
            self.asd_decision_identity_sha256.as_deref().unwrap_or("none"),
            self.asd_state.as_deref().unwrap_or("none"),
            self.asd_impl.as_deref().unwrap_or("none"),
        );
    }
}

fn resolve_grouped_transpose_dispatch(
    dim: GroupedTransposeDim,
    groups: usize,
    sm: u32,
    force_kernel: bool,
    requested: Option<&str>,
    exact_asd: Option<Arc<crate::asd_runtime_extensions::RuntimeExactMatch>>,
) -> GroupedTransposeDispatchDecision {
    let requested = GroupedTransposeDispatchRequest::parse(requested);

    if force_kernel {
        return GroupedTransposeDispatchDecision {
            requested,
            selected: GroupedTransposeDispatchPath::Raw,
            reason: GroupedTransposeDispatchReason::ForceKernelOverride,
            asd_profile_id: None,
            asd_decision_id: None,
            asd_decision_identity_sha256: None,
            asd_state: None,
            asd_impl: None,
        };
    }

    match requested {
        GroupedTransposeDispatchRequest::Raw => GroupedTransposeDispatchDecision {
            requested,
            selected: GroupedTransposeDispatchPath::Raw,
            reason: GroupedTransposeDispatchReason::ExplicitRaw,
            asd_profile_id: None,
            asd_decision_id: None,
            asd_decision_identity_sha256: None,
            asd_state: None,
            asd_impl: None,
        },
        GroupedTransposeDispatchRequest::Cudnn => GroupedTransposeDispatchDecision {
            requested,
            selected: GroupedTransposeDispatchPath::Cudnn,
            reason: GroupedTransposeDispatchReason::ExplicitCudnn,
            asd_profile_id: None,
            asd_decision_id: None,
            asd_decision_identity_sha256: None,
            asd_state: None,
            asd_impl: None,
        },
        GroupedTransposeDispatchRequest::Auto | GroupedTransposeDispatchRequest::Invalid => {
            if let Some(asd) = exact_asd {
                let selected = match asd.execution_provider {
                    crate::asd_runtime_extensions::RuntimeExecutionProvider::RawCuda => {
                        GroupedTransposeDispatchPath::Raw
                    }
                    crate::asd_runtime_extensions::RuntimeExecutionProvider::Cudnn => {
                        GroupedTransposeDispatchPath::Cudnn
                    }
                    // The Exact Profile contract reserves Native, but the Stage2F
                    // adapter does not emit promoted native decisions yet.
                    crate::asd_runtime_extensions::RuntimeExecutionProvider::Native => unreachable!(
                        "promoted native Exact Profile decision reached grouped-transpose before native executor identity was enabled"
                    ),
                };
                return GroupedTransposeDispatchDecision {
                    requested,
                    selected,
                    reason: GroupedTransposeDispatchReason::ExactAsd,
                    asd_profile_id: Some(asd.profile_id.clone()),
                    asd_decision_id: Some(asd.decision_id.clone()),
                    asd_decision_identity_sha256: Some(asd.decision_identity_sha256.clone()),
                    asd_state: Some(asd.state.clone()),
                    asd_impl: Some(asd.implementation_id.clone()),
                };
            }

            let selected = if auto_prefers_raw(dim, groups, sm) {
                GroupedTransposeDispatchPath::Raw
            } else {
                GroupedTransposeDispatchPath::Cudnn
            };
            let reason = match requested {
                GroupedTransposeDispatchRequest::Auto => GroupedTransposeDispatchReason::AutoRule,
                GroupedTransposeDispatchRequest::Invalid => {
                    GroupedTransposeDispatchReason::InvalidFallsBackToAuto
                }
                GroupedTransposeDispatchRequest::Raw | GroupedTransposeDispatchRequest::Cudnn => {
                    unreachable!()
                }
            };
            GroupedTransposeDispatchDecision {
                requested,
                selected,
                reason,
                asd_profile_id: None,
                asd_decision_id: None,
                asd_decision_identity_sha256: None,
                asd_state: None,
                asd_impl: None,
            }
        }
    }
}

fn grouped_transpose_trace_enabled() -> bool {
    runtime_policy() & POLICY_TRACE_GROUPED != 0
}

fn asd_exact_trace_enabled() -> bool {
    runtime_policy() & POLICY_TRACE_ASD != 0
}

fn trace_decision(
    dim: GroupedTransposeDim,
    groups: usize,
    sm: u32,
    decision: &GroupedTransposeDispatchDecision,
) {
    if grouped_transpose_trace_enabled() || asd_exact_trace_enabled() {
        eprintln!(
            "[candle grouped-conv-transpose] requested={} sm={} dim={} groups={} selected={} reason={} asd_profile={} asd_policy={} asd_decision={} asd_decision_identity_sha256={} asd_state={}",
            decision.requested.as_str(),
            sm,
            dim.as_str(),
            groups,
            decision.selected.as_str(),
            decision.reason.as_str(),
            decision.asd_profile_id.as_deref().unwrap_or("none"),
            decision.asd_profile_id.as_deref().unwrap_or("none"),
            decision.asd_decision_id.as_deref().unwrap_or("none"),
            decision.asd_decision_identity_sha256.as_deref().unwrap_or("none"),
            decision.asd_state.as_deref().unwrap_or("none"),
        );
    }
}

struct ExactCallArgs<'a> {
    dim: GroupedTransposeDim,
    batch: usize,
    c_in: usize,
    c_out: usize,
    spatial0: usize,
    spatial1: usize,
    groups: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    input_l: &'a Layout,
    kernel_l: &'a Layout,
    dtype: DType,
}

fn exact_call(args: ExactCallArgs<'_>) -> candle_kernels::asd_exact::ExactOperationCall {
    let weight = args.kernel_l.dims();
    candle_kernels::asd_exact::ExactOperationCall {
        op: match args.dim {
            GroupedTransposeDim::D1 => candle_kernels::asd_exact::ExactOperation::ConvTranspose1d,
            GroupedTransposeDim::D2 => candle_kernels::asd_exact::ExactOperation::ConvTranspose2d,
        },
        dim: match args.dim {
            GroupedTransposeDim::D1 => 1,
            GroupedTransposeDim::D2 => 2,
        },
        batch: args.batch,
        c_in: args.c_in,
        c_out: args.c_out,
        spatial0: args.spatial0,
        spatial1: args.spatial1,
        weight_rank: weight.len(),
        weight0: weight.first().copied().unwrap_or(0),
        weight1: weight.get(1).copied().unwrap_or(0),
        weight2: weight.get(2).copied().unwrap_or(0),
        weight3: weight.get(3).copied().unwrap_or(0),
        groups: args.groups,
        kernel: args.kernel,
        stride: args.stride,
        padding: args.padding,
        output_padding: args.output_padding,
        dilation: args.dilation,
        dtype: args.dtype.as_str(),
        input_contiguous: args.input_l.is_contiguous(),
        input_start_offset: args.input_l.start_offset(),
        weight_contiguous: args.kernel_l.is_contiguous(),
        weight_start_offset: args.kernel_l.start_offset(),
    }
}

fn resolve_runtime(
    dim: GroupedTransposeDim,
    groups: usize,
    exact_call: candle_kernels::asd_exact::ExactOperationCall,
    actual_uuid: Option<&str>,
) -> crate::Result<GroupedTransposeDispatchDecision> {
    let bits = runtime_policy();
    let force_kernel = bits & POLICY_FORCE_RAW != 0;
    let requested = request_from_policy(bits);
    let sm = candle_kernels::CUDA_BUILD_COMPUTE_CAP;
    let exact_asd = crate::asd_runtime_extensions::lookup_exact(exact_call, actual_uuid)?;
    let decision = resolve_grouped_transpose_dispatch(
        dim,
        groups,
        sm,
        force_kernel,
        Some(requested.as_str()),
        exact_asd,
    );
    trace_decision(dim, groups, sm, &decision);
    Ok(decision)
}

fn actual_profile_uuid(input: &crate::cuda_backend::CudaStorage) -> Option<String> {
    // A device-scoped Exact Profile must authenticate the real CUDA context.
    // Provider selection is irrelevant here: raw CUDA and cuDNN decisions use
    // the same exact-signature authority and the same device identity.
    let stream = input.device.cuda_stream();
    let context = stream.context();
    let (major, minor) = context.compute_capability().ok()?;
    if major * 10 + minor != candle_kernels::CUDA_BUILD_COMPUTE_CAP as i32 {
        return None;
    }

    let uuid = context.uuid().ok()?;
    let hex = uuid
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

pub(super) fn decision_1d(
    input: &crate::cuda_backend::CudaStorage,
    p: &ParamsConvTranspose1D,
    input_l: &Layout,
    kernel_l: &Layout,
    dtype: DType,
) -> crate::Result<GroupedTransposeDispatchDecision> {
    let call = exact_call(ExactCallArgs {
        dim: GroupedTransposeDim::D1,
        batch: p.b_size,
        c_in: p.c_in,
        c_out: p.c_out,
        spatial0: p.l_in,
        spatial1: 0,
        groups: p.groups,
        kernel: p.k_size,
        stride: p.stride,
        padding: p.padding,
        output_padding: p.output_padding,
        dilation: p.dilation,
        input_l,
        kernel_l,
        dtype,
    });
    // UUID comes from the real CUDA context, never CANDLE_ASD_TARGET_GPU_UUID.
    // There is no legacy CT1D opt-in: any device-scoped Exact Profile decision,
    // regardless of provider, must be able to resolve through the same lookup.
    // An unreadable or mismatched device identity fails closed.
    let actual_uuid = actual_profile_uuid(input);
    resolve_runtime(
        GroupedTransposeDim::D1,
        p.groups,
        call,
        actual_uuid.as_deref(),
    )
}

pub(super) fn decision_2d(
    input: &crate::cuda_backend::CudaStorage,
    p: &ParamsConvTranspose2D,
    input_l: &Layout,
    kernel_l: &Layout,
    dtype: DType,
) -> crate::Result<GroupedTransposeDispatchDecision> {
    let call = exact_call(ExactCallArgs {
        dim: GroupedTransposeDim::D2,
        batch: p.b_size,
        c_in: p.c_in,
        c_out: p.c_out,
        spatial0: p.i_h,
        spatial1: p.i_w,
        groups: p.groups,
        kernel: p.k_h,
        stride: p.stride,
        padding: p.padding,
        output_padding: p.output_padding,
        dilation: p.dilation,
        input_l,
        kernel_l,
        dtype,
    });
    // CT2D is device-scoped exactly like CT1D: authenticate the real CUDA
    // context before consulting the Exact Profile. Passing None here would
    // intentionally fail closed for device-scoped profiles and silently fall
    // back to the generic auto rule.
    let actual_uuid = actual_profile_uuid(input);
    resolve_runtime(
        GroupedTransposeDim::D2,
        p.groups,
        call,
        actual_uuid.as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolve(
        dim: GroupedTransposeDim,
        groups: usize,
        requested: Option<&str>,
    ) -> GroupedTransposeDispatchDecision {
        resolve_grouped_transpose_dispatch(dim, groups, 61, false, requested, None)
    }

    #[test]
    fn dispatch_auto_sm61_keeps_unpromoted_cudnn_rule() {
        let decision = resolve(GroupedTransposeDim::D2, 4, Some("auto"));
        assert_eq!(decision.requested, GroupedTransposeDispatchRequest::Auto);
        assert!(!decision.prefers_raw());
        assert_eq!(decision.reason, GroupedTransposeDispatchReason::AutoRule);
    }

    #[test]
    fn dispatch_explicit_raw_sm61_selects_raw() {
        let decision = resolve(GroupedTransposeDim::D2, 4, Some("raw"));
        assert_eq!(decision.requested, GroupedTransposeDispatchRequest::Raw);
        assert!(decision.prefers_raw());
        assert_eq!(decision.reason, GroupedTransposeDispatchReason::ExplicitRaw);
    }

    #[test]
    fn dispatch_explicit_cudnn_sm61_selects_cudnn() {
        let decision = resolve(GroupedTransposeDim::D2, 4, Some("cudnn"));
        assert_eq!(decision.requested, GroupedTransposeDispatchRequest::Cudnn);
        assert!(!decision.prefers_raw());
        assert_eq!(
            decision.reason,
            GroupedTransposeDispatchReason::ExplicitCudnn
        );
    }

    #[test]
    fn explicit_cudnn_conflicts_with_force_raw_in_both_dimensions() {
        for _dim in [GroupedTransposeDim::D1, GroupedTransposeDim::D2] {
            assert!(check_backend_requests(Some("cudnn"), false, true, false).is_err());
            assert!(check_backend_requests(Some("raw"), true, false, false).is_err());
            assert!(check_backend_requests(Some("cudnn"), false, false, true).is_err());
            assert_eq!(
                check_backend_requests(Some("cudnn"), false, false, false).unwrap(),
                true
            );
            assert_eq!(
                check_backend_requests(Some("auto"), false, false, false).unwrap(),
                false
            );
        }
    }

    #[test]
    fn only_auto_may_select_an_exact_asd_kernel() {
        assert!(auto_request(None, false, false));
        assert!(auto_request(Some("auto"), false, false));
        for req in [Some("cudnn"), Some("raw"), Some("invalid")] {
            assert!(!auto_request(req, false, false));
        }
        assert!(!auto_request(Some("auto"), true, false));
        assert!(!auto_request(Some("auto"), false, true));
    }

    #[test]
    fn dispatch_invalid_value_preserves_auto_fallback() {
        let decision = resolve(GroupedTransposeDim::D1, 2, Some("unexpected"));
        assert_eq!(decision.requested, GroupedTransposeDispatchRequest::Invalid);
        assert!(!decision.prefers_raw());
        assert_eq!(
            decision.reason,
            GroupedTransposeDispatchReason::InvalidFallsBackToAuto
        );
    }

    #[test]
    fn exact_asd_precedes_unpromoted_general_rule() {
        let exact = Arc::new(crate::asd_runtime_extensions::RuntimeExactMatch {
            profile_id: Arc::from("test-profile"),
            decision_id: Arc::from("ct1d-g2"),
            decision_identity_sha256: Arc::from(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ),
            state: Arc::from("promoted"),
            execution_provider: crate::asd_runtime_extensions::RuntimeExecutionProvider::RawCuda,
            evidence_sha256: Arc::from("test"),
            implementation_id: Arc::from("candle.grouped-transpose.raw.v1"),
            min_integrated_speedup_x: Some(1.10),
        });
        let decision = resolve_grouped_transpose_dispatch(
            GroupedTransposeDim::D1,
            2,
            61,
            false,
            Some("auto"),
            Some(exact),
        );
        assert!(decision.prefers_raw());
        assert!(decision.is_exact_asd());
        assert_eq!(decision.asd_decision_id.as_deref(), Some("ct1d-g2"));
    }

    #[test]
    fn exact_profile_cudnn_provider_precedes_unpromoted_general_rule() {
        let exact = Arc::new(crate::asd_runtime_extensions::RuntimeExactMatch {
            profile_id: Arc::from("test-profile"),
            decision_id: Arc::from("ct1d-g2-cudnn"),
            decision_identity_sha256: Arc::from(
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            state: Arc::from("promoted"),
            execution_provider: crate::asd_runtime_extensions::RuntimeExecutionProvider::Cudnn,
            evidence_sha256: Arc::from("test"),
            implementation_id: Arc::from("candle.cudnn.grouped-transpose.v1"),
            min_integrated_speedup_x: Some(1.01),
        });
        let decision = resolve_grouped_transpose_dispatch(
            GroupedTransposeDim::D1,
            2,
            61,
            false,
            Some("auto"),
            Some(exact),
        );
        assert!(!decision.prefers_raw());
        assert!(decision.is_exact_asd());
        assert!(decision.exact_requires_cudnn());
        assert_eq!(decision.asd_decision_id.as_deref(), Some("ct1d-g2-cudnn"));
    }
}
