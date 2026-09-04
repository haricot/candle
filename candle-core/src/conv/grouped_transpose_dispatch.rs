#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GroupedTransposeDim {
    D1,
    D2,
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

fn auto_prefers_raw(dim: GroupedTransposeDim, groups: usize) -> bool {
    let policy = policy_for_sm(candle_kernels::CUDA_BUILD_COMPUTE_CAP);
    let threshold = match dim {
        GroupedTransposeDim::D1 => policy.raw_min_groups_1d,
        GroupedTransposeDim::D2 => policy.raw_min_groups_2d,
    };
    threshold.is_some_and(|min_groups| groups >= min_groups)
}

pub(super) fn prefers_raw_cuda(dim: GroupedTransposeDim, groups: usize) -> bool {
    // Keep the existing low-level diagnostic override as the strongest signal.
    if std::env::var_os("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL").is_some() {
        return true;
    }

    match std::env::var("CANDLE_GROUPED_TRANSPOSE_DISPATCH").ok().as_deref() {
        Some("raw") => true,
        Some("cudnn") => false,
        Some("auto") | None => auto_prefers_raw(dim, groups),
        Some(_) => auto_prefers_raw(dim, groups),
    }
}
