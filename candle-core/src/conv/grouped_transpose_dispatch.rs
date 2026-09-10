use crate::{DType, Layout};

use super::{ParamsConvTranspose1D, ParamsConvTranspose2D};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GroupedTransposeCudaRule {
    raw_min_groups_1d: Option<usize>,
    raw_min_groups_2d: Option<usize>,
}

const fn dispatch_rule_for_sm(_sm: u32) -> GroupedTransposeCudaRule {
    // Generalized thresholds remain intentionally unset. ASD v1 exact-domain
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GroupedTransposeDispatchDecision {
    requested: GroupedTransposeDispatchRequest,
    selected: GroupedTransposeDispatchPath,
    reason: GroupedTransposeDispatchReason,
    asd_policy_id: Option<&'static str>,
    asd_decision_id: Option<&'static str>,
    asd_state: Option<&'static str>,
}

impl GroupedTransposeDispatchDecision {
    pub(super) fn prefers_raw(self) -> bool {
        self.selected == GroupedTransposeDispatchPath::Raw
    }

    pub(super) fn is_exact_asd(self) -> bool {
        self.reason == GroupedTransposeDispatchReason::ExactAsd
    }

    pub(super) fn trace_submission(self, backend: &str) {
        if !grouped_transpose_trace_enabled() && !asd_exact_trace_enabled() {
            return;
        }
        eprintln!(
            "[candle grouped-conv-transpose] submitted_backend={} launch_submission=success asd_policy={} asd_decision={} asd_state={}",
            backend,
            self.asd_policy_id.unwrap_or("none"),
            self.asd_decision_id.unwrap_or("none"),
            self.asd_state.unwrap_or("none"),
        );
    }
}

fn resolve_grouped_transpose_dispatch(
    dim: GroupedTransposeDim,
    groups: usize,
    sm: u32,
    force_kernel: bool,
    requested: Option<&str>,
    exact_asd: Option<candle_kernels::asd_exact::ExactAsdMatch>,
) -> GroupedTransposeDispatchDecision {
    let requested = GroupedTransposeDispatchRequest::parse(requested);

    if force_kernel {
        return GroupedTransposeDispatchDecision {
            requested,
            selected: GroupedTransposeDispatchPath::Raw,
            reason: GroupedTransposeDispatchReason::ForceKernelOverride,
            asd_policy_id: None,
            asd_decision_id: None,
            asd_state: None,
        };
    }

    match requested {
        GroupedTransposeDispatchRequest::Raw => GroupedTransposeDispatchDecision {
            requested,
            selected: GroupedTransposeDispatchPath::Raw,
            reason: GroupedTransposeDispatchReason::ExplicitRaw,
            asd_policy_id: None,
            asd_decision_id: None,
            asd_state: None,
        },
        GroupedTransposeDispatchRequest::Cudnn => GroupedTransposeDispatchDecision {
            requested,
            selected: GroupedTransposeDispatchPath::Cudnn,
            reason: GroupedTransposeDispatchReason::ExplicitCudnn,
            asd_policy_id: None,
            asd_decision_id: None,
            asd_state: None,
        },
        GroupedTransposeDispatchRequest::Auto | GroupedTransposeDispatchRequest::Invalid => {
            if let Some(asd) = exact_asd {
                let selected = match asd.selected_backend {
                    "raw_cuda" => GroupedTransposeDispatchPath::Raw,
                    // ASD-v1-fix2 build adapter currently rejects any other backend.
                    _ => GroupedTransposeDispatchPath::Cudnn,
                };
                return GroupedTransposeDispatchDecision {
                    requested,
                    selected,
                    reason: GroupedTransposeDispatchReason::ExactAsd,
                    asd_policy_id: Some(asd.policy_id),
                    asd_decision_id: Some(asd.decision_id),
                    asd_state: Some(asd.state),
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
                asd_policy_id: None,
                asd_decision_id: None,
                asd_state: None,
            }
        }
    }
}

fn grouped_transpose_trace_enabled() -> bool {
    matches!(
        std::env::var("CANDLE_GROUPED_TRANSPOSE_TRACE")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

fn asd_exact_trace_enabled() -> bool {
    matches!(
        std::env::var("CANDLE_ASD_EXACT_TRACE").ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

fn trace_decision(
    dim: GroupedTransposeDim,
    groups: usize,
    sm: u32,
    decision: GroupedTransposeDispatchDecision,
) {
    if grouped_transpose_trace_enabled() || asd_exact_trace_enabled() {
        eprintln!(
            "[candle grouped-conv-transpose] requested={} sm={} dim={} groups={} selected={} reason={} asd_policy={} asd_decision={} asd_state={}",
            decision.requested.as_str(),
            sm,
            dim.as_str(),
            groups,
            decision.selected.as_str(),
            decision.reason.as_str(),
            decision.asd_policy_id.unwrap_or("none"),
            decision.asd_decision_id.unwrap_or("none"),
            decision.asd_state.unwrap_or("none"),
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

fn exact_call(args: ExactCallArgs<'_>) -> candle_kernels::asd_exact::ExactConvTransposeCall {
    let weight = args.kernel_l.dims();
    candle_kernels::asd_exact::ExactConvTransposeCall {
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
    exact_call: candle_kernels::asd_exact::ExactConvTransposeCall,
) -> GroupedTransposeDispatchDecision {
    let force_kernel = std::env::var_os("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL").is_some();
    let requested = std::env::var("CANDLE_GROUPED_TRANSPOSE_DISPATCH").ok();
    let sm = candle_kernels::CUDA_BUILD_COMPUTE_CAP;
    let exact_asd = candle_kernels::asd_exact::lookup(exact_call);
    let decision = resolve_grouped_transpose_dispatch(
        dim,
        groups,
        sm,
        force_kernel,
        requested.as_deref(),
        exact_asd,
    );
    trace_decision(dim, groups, sm, decision);
    decision
}

pub(super) fn decision_1d(
    p: &ParamsConvTranspose1D,
    input_l: &Layout,
    kernel_l: &Layout,
    dtype: DType,
) -> GroupedTransposeDispatchDecision {
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
    resolve_runtime(GroupedTransposeDim::D1, p.groups, call)
}

pub(super) fn decision_2d(
    p: &ParamsConvTranspose2D,
    input_l: &Layout,
    kernel_l: &Layout,
    dtype: DType,
) -> GroupedTransposeDispatchDecision {
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
    resolve_runtime(GroupedTransposeDim::D2, p.groups, call)
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
    fn dispatch_force_kernel_override_wins_over_cudnn() {
        let decision = resolve_grouped_transpose_dispatch(
            GroupedTransposeDim::D1,
            2,
            61,
            true,
            Some("cudnn"),
            None,
        );
        assert_eq!(decision.requested, GroupedTransposeDispatchRequest::Cudnn);
        assert!(decision.prefers_raw());
        assert_eq!(
            decision.reason,
            GroupedTransposeDispatchReason::ForceKernelOverride
        );
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
        let exact = candle_kernels::asd_exact::ExactAsdMatch {
            policy_id: "test-policy",
            decision_id: "ct1d-g2",
            state: "tuner_candidate",
            selected_backend: "raw_cuda",
            evidence_sha256: "test",
            min_integrated_speedup_x: 1.10,
        };
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
        assert_eq!(decision.asd_decision_id, Some("ct1d-g2"));
    }
}
