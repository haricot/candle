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
struct GroupedTransposeCudaPolicy {
    raw_min_groups_1d: Option<usize>,
    raw_min_groups_2d: Option<usize>,
}

const fn policy_for_sm(_sm: u32) -> GroupedTransposeCudaPolicy {
    // Thresholds are intentionally unset in the scaffold. V4-B2 will fill
    // architecture-specific crossovers from measured dispatch frontiers.
    GroupedTransposeCudaPolicy {
        raw_min_groups_1d: None,
        raw_min_groups_2d: None,
    }
}

fn auto_prefers_raw(dim: GroupedTransposeDim, groups: usize, sm: u32) -> bool {
    let policy = policy_for_sm(sm);
    let threshold = match dim {
        GroupedTransposeDim::D1 => policy.raw_min_groups_1d,
        GroupedTransposeDim::D2 => policy.raw_min_groups_2d,
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
    AutoPolicy,
    InvalidFallsBackToAuto,
}

impl GroupedTransposeDispatchReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ForceKernelOverride => "force_kernel_override",
            Self::ExplicitRaw => "explicit_raw",
            Self::ExplicitCudnn => "explicit_cudnn",
            Self::AutoPolicy => "auto_policy",
            Self::InvalidFallsBackToAuto => "invalid_falls_back_to_auto",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GroupedTransposeDispatchDecision {
    requested: GroupedTransposeDispatchRequest,
    selected: GroupedTransposeDispatchPath,
    reason: GroupedTransposeDispatchReason,
}

fn resolve_grouped_transpose_dispatch(
    dim: GroupedTransposeDim,
    groups: usize,
    sm: u32,
    force_kernel: bool,
    requested: Option<&str>,
) -> GroupedTransposeDispatchDecision {
    let requested = GroupedTransposeDispatchRequest::parse(requested);

    if force_kernel {
        return GroupedTransposeDispatchDecision {
            requested,
            selected: GroupedTransposeDispatchPath::Raw,
            reason: GroupedTransposeDispatchReason::ForceKernelOverride,
        };
    }

    match requested {
        GroupedTransposeDispatchRequest::Raw => GroupedTransposeDispatchDecision {
            requested,
            selected: GroupedTransposeDispatchPath::Raw,
            reason: GroupedTransposeDispatchReason::ExplicitRaw,
        },
        GroupedTransposeDispatchRequest::Cudnn => GroupedTransposeDispatchDecision {
            requested,
            selected: GroupedTransposeDispatchPath::Cudnn,
            reason: GroupedTransposeDispatchReason::ExplicitCudnn,
        },
        GroupedTransposeDispatchRequest::Auto | GroupedTransposeDispatchRequest::Invalid => {
            let selected = if auto_prefers_raw(dim, groups, sm) {
                GroupedTransposeDispatchPath::Raw
            } else {
                GroupedTransposeDispatchPath::Cudnn
            };
            let reason = match requested {
                GroupedTransposeDispatchRequest::Auto => GroupedTransposeDispatchReason::AutoPolicy,
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

pub(super) fn prefers_raw_cuda(dim: GroupedTransposeDim, groups: usize) -> bool {
    let force_kernel = std::env::var_os("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL").is_some();
    let requested = std::env::var("CANDLE_GROUPED_TRANSPOSE_DISPATCH").ok();
    let sm = candle_kernels::CUDA_BUILD_COMPUTE_CAP;
    let decision =
        resolve_grouped_transpose_dispatch(dim, groups, sm, force_kernel, requested.as_deref());

    if grouped_transpose_trace_enabled() {
        eprintln!(
            "[candle grouped-conv-transpose] requested={} sm={} dim={} groups={} selected={} reason={}",
            decision.requested.as_str(),
            sm,
            dim.as_str(),
            groups,
            decision.selected.as_str(),
            decision.reason.as_str(),
        );
    }

    decision.selected == GroupedTransposeDispatchPath::Raw
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_auto_sm61_keeps_unpromoted_cudnn_policy() {
        let decision =
            resolve_grouped_transpose_dispatch(GroupedTransposeDim::D2, 4, 61, false, Some("auto"));
        assert_eq!(decision.requested, GroupedTransposeDispatchRequest::Auto);
        assert_eq!(decision.selected, GroupedTransposeDispatchPath::Cudnn);
        assert_eq!(decision.reason, GroupedTransposeDispatchReason::AutoPolicy);
    }

    #[test]
    fn dispatch_explicit_raw_sm61_selects_raw() {
        let decision =
            resolve_grouped_transpose_dispatch(GroupedTransposeDim::D2, 4, 61, false, Some("raw"));
        assert_eq!(decision.requested, GroupedTransposeDispatchRequest::Raw);
        assert_eq!(decision.selected, GroupedTransposeDispatchPath::Raw);
        assert_eq!(decision.reason, GroupedTransposeDispatchReason::ExplicitRaw);
    }

    #[test]
    fn dispatch_explicit_cudnn_sm61_selects_cudnn() {
        let decision = resolve_grouped_transpose_dispatch(
            GroupedTransposeDim::D2,
            4,
            61,
            false,
            Some("cudnn"),
        );
        assert_eq!(decision.requested, GroupedTransposeDispatchRequest::Cudnn);
        assert_eq!(decision.selected, GroupedTransposeDispatchPath::Cudnn);
        assert_eq!(
            decision.reason,
            GroupedTransposeDispatchReason::ExplicitCudnn
        );
    }

    #[test]
    fn dispatch_force_kernel_override_wins_over_cudnn() {
        let decision =
            resolve_grouped_transpose_dispatch(GroupedTransposeDim::D1, 2, 61, true, Some("cudnn"));
        assert_eq!(decision.requested, GroupedTransposeDispatchRequest::Cudnn);
        assert_eq!(decision.selected, GroupedTransposeDispatchPath::Raw);
        assert_eq!(
            decision.reason,
            GroupedTransposeDispatchReason::ForceKernelOverride
        );
    }

    #[test]
    fn dispatch_invalid_value_preserves_auto_fallback() {
        let decision = resolve_grouped_transpose_dispatch(
            GroupedTransposeDim::D1,
            2,
            61,
            false,
            Some("unexpected"),
        );
        assert_eq!(decision.requested, GroupedTransposeDispatchRequest::Invalid);
        assert_eq!(decision.selected, GroupedTransposeDispatchPath::Cudnn);
        assert_eq!(
            decision.reason,
            GroupedTransposeDispatchReason::InvalidFallsBackToAuto
        );
    }
}
