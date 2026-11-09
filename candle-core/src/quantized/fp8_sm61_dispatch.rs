//! Generic FP8 E4M3 memory-aware residence dispatcher for CUDA sm_61.
//!
//! This module is pure Rust: it neither depends on a model architecture nor
//! queries the driver. The caller supplies a conservative *device-wide* free
//! memory snapshot, reservations for KV-cache/workspaces/headroom, and optional
//! hardware-measured shape/workload profiles. No matching profile => SIMT.
//!
//! Decisions are made ONCE before construction, and each Fp8Sm61Linear keeps
//! exactly one resident representation. The dispatch is an admission estimate;
//! actual CUDA allocation can still fail, and must fail closed.
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fp8Backend {
    SimtW8A16,
    ExpandedF16,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fp8MeasuredProfile {
    /// Steady-state complete call latency measured for this *shape and workload*.
    pub simt_ms: f64,
    pub expanded_f16_ms: f64,
    /// Additional one-time construction cost to materialize the FP16 weights.
    pub extra_expand_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fp8LinearSpec {
    pub out_features: usize,
    pub in_features: usize,
    pub block_n: usize,
    pub block_k: usize,
    /// Planned number of forwards while this instance stays resident.
    pub resident_calls: u64,
    pub profile: Option<Fp8MeasuredProfile>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fp8DispatchBudget {
    /// Conservative free VRAM *before* allocating this group of linears.
    pub free_device_bytes: usize,
    /// Reserve non-FP8 weights, KV cache, activations, CUDA contexts etc.
    pub workload_reserve_bytes: usize,
    /// Additional non-negotiable safety margin for allocator fragmentation.
    pub safety_headroom_bytes: usize,
    /// Require this much steady-state performance gain for FP16 admission.
    pub min_steady_speedup: f64,
    /// Explicit opt-in; false makes the result exclusively SIMT.
    pub allow_auto_f16: bool,
}

impl Default for Fp8DispatchBudget {
    fn default() -> Self {
        Self {
            free_device_bytes: 0,
            workload_reserve_bytes: 0,
            safety_headroom_bytes: 0,
            min_steady_speedup: 1.03,
            allow_auto_f16: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fp8DispatchEntry {
    pub spec: Fp8LinearSpec,
    pub backend: Fp8Backend,
    pub resident_bytes: usize,
    pub simt_bytes: usize,
    pub expanded_f16_bytes: usize,
    pub predicted_net_saving_ms: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fp8DispatchPlan {
    pub entries: Vec<Fp8DispatchEntry>,
    pub simt_total_bytes: usize,
    pub selected_total_bytes: usize,
    pub budget_for_fp8_bytes: usize,
    pub expanded_f16_count: usize,
    pub estimated_net_savings_ms: f64,
    /// Number of independently accounted tensor entries.
    pub projection_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fp8DispatchError(pub &'static str);
impl fmt::Display for Fp8DispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for Fp8DispatchError {}
type Outcome<T> = std::result::Result<T, Fp8DispatchError>;

pub fn required_bytes(s: &Fp8LinearSpec) -> Outcome<(usize, usize)> {
    if s.out_features == 0 || s.in_features == 0 || s.block_n == 0 || s.block_k == 0 {
        return Err(Fp8DispatchError("invalid FP8 tensor/block geometry"));
    }
    // CUDA F8E kernel requires K divisible by 128; fail early and deterministically.
    if !s.in_features.is_multiple_of(128) {
        return Err(Fp8DispatchError("FP8 sm61 requires K divisible by 128"));
    }
    if s.out_features > i32::MAX as usize
        || s.in_features > i32::MAX as usize
        || s.block_n > i32::MAX as usize
        || s.block_k > i32::MAX as usize
    {
        return Err(Fp8DispatchError("FP8 dimensions exceed CUDA i32 ABI"));
    }
    let elements = s
        .out_features
        .checked_mul(s.in_features)
        .ok_or(Fp8DispatchError("FP8 tensor byte count overflow"))?;
    if elements > i32::MAX as usize {
        return Err(Fp8DispatchError(
            "FP8 linear dimensions exceed CUDA launch contract",
        ));
    }
    let nr = s.out_features.div_ceil(s.block_n);
    let kr = s.in_features.div_ceil(s.block_k);
    let scale_count = nr
        .checked_mul(kr)
        .ok_or(Fp8DispatchError("FP8 scales overflow"))?;
    // R3 resident scales are F32 even when checkpoint scales were BF16/F16.
    let fp8 = elements
        .checked_add(
            scale_count
                .checked_mul(4)
                .ok_or(Fp8DispatchError("FP8 scale bytes overflow"))?,
        )
        .ok_or(Fp8DispatchError("FP8 residency overflow"))?;
    let f16 = elements
        .checked_mul(2)
        .ok_or(Fp8DispatchError("FP16 residency overflow"))?;
    Ok((fp8, f16))
}

/// Greedy deterministic admission by expected saved milliseconds per extra VRAM byte.
/// Does not claim globally optimal knapsack packing. Layer identities are not inputs.
/// Any missing, invalid, or unqualified measurement => SIMT for that layer.
/// Failure to admit the all-SIMT baseline is a hard error, not an OOM fallback.
pub fn dispatch_residence(
    specs: &[Fp8LinearSpec],
    env: Fp8DispatchBudget,
) -> Outcome<Fp8DispatchPlan> {
    if specs.is_empty() {
        return Err(Fp8DispatchError("no FP8 linears supplied"));
    }
    if !env.min_steady_speedup.is_finite() || env.min_steady_speedup < 1.0 {
        return Err(Fp8DispatchError("invalid minimum speedup"));
    }
    let budget = env
        .free_device_bytes
        .checked_sub(env.workload_reserve_bytes)
        .and_then(|v| v.checked_sub(env.safety_headroom_bytes))
        .ok_or(Fp8DispatchError(
            "VRAM reserves exceed available device memory",
        ))?;
    let mut baseline = 0usize;
    let mut entries = Vec::with_capacity(specs.len());
    for &spec in specs {
        let (simt, expanded) = required_bytes(&spec)?;
        baseline = baseline
            .checked_add(simt)
            .ok_or(Fp8DispatchError("model FP8 total overflow"))?;
        entries.push(Fp8DispatchEntry {
            spec,
            backend: Fp8Backend::SimtW8A16,
            resident_bytes: simt,
            simt_bytes: simt,
            expanded_f16_bytes: expanded,
            predicted_net_saving_ms: 0.0,
        });
    }
    if baseline > budget {
        return Err(Fp8DispatchError(
            "SIMT baseline exceeds total FP8 VRAM budget",
        ));
    }
    let mut selected = baseline;
    let mut count = 0usize;
    let mut net_total = 0f64;
    if env.allow_auto_f16 {
        let mut candidates: Vec<(usize, f64, usize, f64)> = Vec::new();
        for (i, a) in entries.iter().enumerate() {
            let Some(p) = a.spec.profile else { continue };
            if a.spec.resident_calls == 0
                || !p.simt_ms.is_finite()
                || !p.expanded_f16_ms.is_finite()
                || !p.extra_expand_ms.is_finite()
                || p.simt_ms <= 0.0
                || p.expanded_f16_ms <= 0.0
                || p.extra_expand_ms < 0.0
                || p.simt_ms / p.expanded_f16_ms < env.min_steady_speedup
            {
                continue;
            };
            let benefit = (p.simt_ms - p.expanded_f16_ms) * (a.spec.resident_calls as f64)
                - p.extra_expand_ms;
            if !benefit.is_finite() || benefit <= 0.0 {
                continue;
            };
            let extra = a.expanded_f16_bytes.saturating_sub(a.simt_bytes);
            let score = if extra == 0 {
                f64::MAX
            } else {
                benefit / (extra as f64)
            };
            candidates.push((i, score, extra, benefit));
        }
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for (index, _, extra, benefit) in candidates {
            let Some(next) = selected.checked_add(extra) else {
                continue;
            };
            if next > budget {
                continue;
            };
            entries[index].backend = Fp8Backend::ExpandedF16;
            // Actual selected residency may be smaller than FP8 for an unusual shape.
            selected = next
                - (entries[index]
                    .simt_bytes
                    .saturating_sub(entries[index].expanded_f16_bytes));
            entries[index].resident_bytes = entries[index].expanded_f16_bytes;
            entries[index].predicted_net_saving_ms = benefit;
            net_total += benefit;
            count += 1;
        }
    }
    Ok(Fp8DispatchPlan {
        entries,
        simt_total_bytes: baseline,
        selected_total_bytes: selected,
        budget_for_fp8_bytes: budget,
        expanded_f16_count: count,
        estimated_net_savings_ms: net_total,
        projection_count: specs.len(),
    })
}

/// A measured *whole-model* workload, used for all-or-nothing FP16 admission.
/// Useful before per-projection profiles exist. Never extrapolate this result
/// from one model/context to unrelated shapes without rebenchmarking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fp8WorkloadProfile {
    pub simt_request_ms: f64,
    pub expanded_request_ms: f64,
    pub extra_initialization_ms: f64,
    pub planned_requests_while_resident: u64,
}

/// A conservative dual-backend dispatch: either all FP8 projections are SIMT,
/// or all are persistently expanded to FP16 at construction. Uses no layer names,
/// layer order, or knowledge of transformer families. If measurements are missing
/// or the residence estimate is too short, SIMT is retained.
pub fn dispatch_uniform_residence(
    specs: &[Fp8LinearSpec],
    env: Fp8DispatchBudget,
    profile: Option<Fp8WorkloadProfile>,
) -> Outcome<Fp8DispatchPlan> {
    let mut baseline_env = env;
    baseline_env.allow_auto_f16 = false;
    let mut plan = dispatch_residence(specs, baseline_env)?;
    if !env.allow_auto_f16 {
        return Ok(plan);
    }
    let Some(profile) = profile else {
        return Ok(plan);
    };
    if !profile.simt_request_ms.is_finite()
        || !profile.expanded_request_ms.is_finite()
        || !profile.extra_initialization_ms.is_finite()
        || profile.simt_request_ms <= 0.0
        || profile.expanded_request_ms <= 0.0
        || profile.extra_initialization_ms < 0.0
        || profile.planned_requests_while_resident == 0
        || profile.simt_request_ms / profile.expanded_request_ms < env.min_steady_speedup
    {
        return Ok(plan);
    }
    let net = (profile.simt_request_ms - profile.expanded_request_ms)
        * (profile.planned_requests_while_resident as f64)
        - profile.extra_initialization_ms;
    if !net.is_finite() || net <= 0.0 {
        return Ok(plan);
    }
    let mut expanded_total = 0usize;
    for row in &plan.entries {
        expanded_total = expanded_total
            .checked_add(row.expanded_f16_bytes)
            .ok_or(Fp8DispatchError("expanded FP16 model byte count overflow"))?;
    }
    if expanded_total > plan.budget_for_fp8_bytes {
        return Ok(plan);
    }
    plan.selected_total_bytes = expanded_total;
    plan.expanded_f16_count = plan.entries.len();
    plan.estimated_net_savings_ms = net;
    for row in &mut plan.entries {
        row.backend = Fp8Backend::ExpandedF16;
        row.resident_bytes = row.expanded_f16_bytes;
        // Full-model gain cannot safely be attributed to an individual layer.
        row.predicted_net_saving_ms = 0.0;
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(calls: u64, profile: Option<Fp8MeasuredProfile>) -> Fp8LinearSpec {
        Fp8LinearSpec {
            out_features: 1024,
            in_features: 1024,
            block_n: 128,
            block_k: 128,
            resident_calls: calls,
            profile,
        }
    }
    fn p() -> Fp8MeasuredProfile {
        Fp8MeasuredProfile {
            simt_ms: 10.0,
            expanded_f16_ms: 4.0,
            extra_expand_ms: 100.0,
        }
    }
    fn env(free: usize) -> Fp8DispatchBudget {
        Fp8DispatchBudget {
            free_device_bytes: free,
            workload_reserve_bytes: 1024,
            safety_headroom_bytes: 1024,
            min_steady_speedup: 1.03,
            allow_auto_f16: true,
        }
    }
    #[test]
    fn no_profile_stays_simt() {
        let v = dispatch_residence(&[s(100, None)], env(4 << 20)).unwrap();
        assert_eq!(v.entries[0].backend, Fp8Backend::SimtW8A16);
    }
    #[test]
    fn opt_in_required() {
        let mut e = env(4 << 20);
        e.allow_auto_f16 = false;
        let v = dispatch_residence(&[s(100, Some(p()))], e).unwrap();
        assert_eq!(v.expanded_f16_count, 0);
    }
    #[test]
    fn one_shot_never_amortizes() {
        let v = dispatch_residence(&[s(1, Some(p()))], env(4 << 20)).unwrap();
        assert_eq!(v.expanded_f16_count, 0);
    }
    #[test]
    fn enough_residency_admits_f16() {
        let v = dispatch_residence(&[s(100, Some(p()))], env(4 << 20)).unwrap();
        assert_eq!(v.expanded_f16_count, 1);
        assert!(v.estimated_net_savings_ms > 0.0);
    }
    #[test]
    fn tight_budget_keeps_simt() {
        let (b, _) = required_bytes(&s(100, None)).unwrap();
        let v = dispatch_residence(&[s(100, Some(p()))], env(b + 2048)).unwrap();
        assert_eq!(v.expanded_f16_count, 0);
    }
    #[test]
    fn baseline_oom_is_fail_closed() {
        assert!(dispatch_residence(&[s(100, None)], env(1024)).is_err());
    }
    #[test]
    fn bad_profile_never_selects() {
        let bad = Fp8MeasuredProfile {
            simt_ms: f64::NAN,
            ..p()
        };
        let v = dispatch_residence(&[s(100, Some(bad))], env(4 << 20)).unwrap();
        assert_eq!(v.expanded_f16_count, 0);
    }
    #[test]
    fn model_budget_considers_all_linears() {
        let (b, _) = required_bytes(&s(100, None)).unwrap();
        let v =
            dispatch_residence(&[s(100, Some(p())), s(100, Some(p()))], env(b * 2 + 2048)).unwrap();
        assert_eq!(v.expanded_f16_count, 0);
    }
    #[test]
    fn uniform_profile_amortizes() {
        let profile = Fp8WorkloadProfile {
            simt_request_ms: 900.0,
            expanded_request_ms: 225.0,
            extra_initialization_ms: 12000.0,
            planned_requests_while_resident: 20,
        };
        let v = dispatch_uniform_residence(&[s(0, None)], env(4 << 20), Some(profile)).unwrap();
        assert_eq!(v.expanded_f16_count, 1);
        let v = dispatch_uniform_residence(
            &[s(0, None)],
            env(4 << 20),
            Some(Fp8WorkloadProfile {
                planned_requests_while_resident: 1,
                ..profile
            }),
        )
        .unwrap();
        assert_eq!(v.expanded_f16_count, 0);
    }
    #[test]
    fn no_architecture_specific_names() {
        let x = dispatch_residence(&[s(100, Some(p()))], env(4 << 20)).unwrap();
        assert_eq!(x.entries.len(), 1);
    }
    #[test]
    fn mismatched_budget_keeps_simt() {
        let spec = s(100, Some(p()));
        let (base, _) = required_bytes(&spec).unwrap();
        let result = dispatch_residence(&[spec, spec], env(2 * base + 2048)).unwrap();
        assert_eq!(result.projection_count, 2);
        assert_eq!(result.selected_total_bytes, 2 * base);
    }
    #[test]
    fn zero_blocks_rejected() {
        assert!(required_bytes(&Fp8LinearSpec {
            block_n: 0,
            ..s(10, None)
        })
        .is_err());
    }
    #[test]
    fn overflow_protected() {
        assert!(required_bytes(&Fp8LinearSpec {
            out_features: usize::MAX,
            ..s(1, None)
        })
        .is_err());
    }
}
