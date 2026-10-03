//! Historical provider/performance lineage for ASD Exact Profile decisions.
//!
//! This module is descriptive memory only. It is deliberately not consulted by
//! production dispatch, profile lookup, promotion gates, or Flow-Adaptive
//! scheduling. Current execution authority remains the ASD Exact Profile and
//! its bound promotion evidence.
//!
//! Historical gains are useful because they preserve *why* a decision exists:
//! which implementation/provider it replaced and the order-of-magnitude effect
//! observed at the time. They must never be multiplied across generations.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistoricalState {
    /// Informative measurement from an earlier implementation/provider generation.
    HistoricalOnly,
    /// Historical record of the integrated measurement that led to the current
    /// promoted implementation. Dispatch still uses the Exact Profile, not this table.
    PromotionLineage,
}

impl HistoricalState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HistoricalOnly => "historical_only",
            Self::PromotionLineage => "promotion_lineage",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoricalGain {
    pub transition_id: &'static str,
    pub decision_id: &'static str,
    pub state: HistoricalState,
    /// Zero-based order inside the recorded lineage for this exact decision.
    pub generation: u32,
    /// Explicit edge to the immediately preceding recorded transition.
    pub predecessor_transition_id: Option<&'static str>,
    pub exact_signature: &'static str,
    pub from_provider: &'static str,
    pub from_implementation: &'static str,
    pub to_provider: &'static str,
    pub to_implementation: &'static str,
    pub speedup_x: f64,
    /// Optional earlier observation of the same transition, kept only as context.
    pub earlier_observed_speedup_x: Option<f64>,
    /// Commit/artifact that records the measurement value.
    pub source_commit: &'static str,
    pub source_artifact: &'static str,
    /// Commit that introduced the relevant implementation, when distinct.
    pub implementation_commit: Option<&'static str>,
    /// Commit that carried the benchmark harness used for the historical probe,
    /// when distinct from the measurement record.
    pub benchmark_commit: Option<&'static str>,
    pub evidence_sha256: Option<&'static str>,
    pub note: &'static str,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderChallengeConsensus {
    StableReject,
    PromotionCandidate,
    Inconclusive,
}

impl ProviderChallengeConsensus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StableReject => "stable_reject",
            Self::PromotionCandidate => "promotion_candidate",
            Self::Inconclusive => "inconclusive",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProviderChallengeReplication {
    pub replication_id: &'static str,
    pub evidence_sha256: &'static str,
    pub incumbent_median_us: f64,
    pub challenger_median_us: f64,
    pub incumbent_drift_pct: f64,
    pub challenger_drift_pct: f64,
    pub incumbent_p90_us: f64,
    pub challenger_p90_us: f64,
    pub decision: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProviderChallenge {
    pub challenge_id: &'static str,
    pub decision_id: &'static str,
    pub protocol: &'static str,
    pub measurement_plane: &'static str,
    pub harness_revision: &'static str,
    pub exact_signature: &'static str,
    pub incumbent_provider: &'static str,
    pub incumbent_implementation: &'static str,
    pub challenger_provider: &'static str,
    pub challenger_implementation: &'static str,
    pub challenger_identity: &'static str,
    pub replications: &'static [ProviderChallengeReplication],
    pub consensus: ProviderChallengeConsensus,
    pub incumbent_retained: bool,
    pub profile_change: bool,
    pub note: &'static str,
}

const DW48: &str = "op=conv2d,dim=2,batch=1,c_in=48,c_out=48,spatial=64x48,weight_shape=48x1x5x5,groups=48,kernel=5,stride=1,padding=2,output_padding=none,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset";
const DW96: &str = "op=conv2d,dim=2,batch=1,c_in=96,c_out=96,spatial=32x24,weight_shape=96x1x5x5,groups=96,kernel=5,stride=1,padding=2,output_padding=none,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset";
const DW192: &str = "op=conv2d,dim=2,batch=1,c_in=192,c_out=192,spatial=16x12,weight_shape=192x1x5x5,groups=192,kernel=5,stride=1,padding=2,output_padding=none,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset";
const DW384: &str = "op=conv2d,dim=2,batch=1,c_in=384,c_out=384,spatial=8x6,weight_shape=384x1x5x5,groups=384,kernel=5,stride=1,padding=2,output_padding=none,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset";

const CT1D_G2: &str = "op=conv_transpose1d,dim=1,batch=1,c_in=128,c_out=128,spatial=32,weight_shape=128x64x3,groups=2,kernel=3,stride=2,padding=1,output_padding=1,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset";

const CT1D_G2_CUDNN_IDENTITY: &str = "cudnn:runtime_version=91002:operation=conv_backward_data:algorithm_id=0:workspace_bytes=0:dtype=f32:n=1:ci=128:co=128:l=32:groups=2:k=3:s=2:p=1:op=1:d=1";

const CT1D_G2_PROVIDER_REPLICATIONS: [ProviderChallengeReplication; 2] = [
    ProviderChallengeReplication {
        replication_id: "r1",
        evidence_sha256: "1d86ff6da849ffc29dc30ec4eee69f9ab91330314d74221db5fd82bed11dc07b",
        incumbent_median_us: 10.507469,
        challenger_median_us: 113.392875,
        incumbent_drift_pct: 0.299,
        challenger_drift_pct: 0.349,
        incumbent_p90_us: 11.651000,
        challenger_p90_us: 119.087344,
        decision: "REJECT_CHALLENGER",
    },
    ProviderChallengeReplication {
        replication_id: "r2",
        evidence_sha256: "7e2d4d08f2210b0418261df07921c2ef33b0738d157cfbc834f7c7746f20e367",
        incumbent_median_us: 10.341531,
        challenger_median_us: 115.327250,
        incumbent_drift_pct: 0.256,
        challenger_drift_pct: 0.342,
        incumbent_p90_us: 10.464031,
        challenger_p90_us: 117.098250,
        decision: "REJECT_CHALLENGER",
    },
];

const PROMOTION_EVIDENCE: &str =
    "84f3dc50433e225b1f63c92a08355b7b04f0afeec49694e5e8d5e040cf092a9a";


/// Provider/implementation lineage retained for human and tooling context.
///
/// The 402.470x C384 transition is the historically memorable result: native
/// cuDNN grouped convolution replaced Candle's previous groups=N decomposition
/// into N single-group convolutions followed by concatenation. An earlier
/// isolated C384 observation was 445.265x.
///
/// The later raw ASD DW5x5 results compare against the then-current/cuDNN path
/// under the stricter time-equivalent integrated protocol. These are separate
/// generations and their speedups must not be multiplied.
pub const GAINS: [HistoricalGain; 5] = [
    HistoricalGain {
        transition_id: "dw5x5-c384-legacy-chunked-to-cudnn-grouped",
        decision_id: "conv2d-dw5x5-f32-b1-c384-h8-w6-g384-s1-p2-d1-raw",
        state: HistoricalState::HistoricalOnly,
        generation: 0,
        predecessor_transition_id: None,
        exact_signature: DW384,
        from_provider: "native",
        from_implementation: "candle.legacy-grouped-conv2d.chunk-per-group-cat",
        to_provider: "cudnn",
        to_implementation: "candle.cudnn.grouped-conv2d.native.v1",
        speedup_x: 402.470,
        earlier_observed_speedup_x: Some(445.265),
        source_commit: "fec68ced96355e7816e9ad65f2a2f88c583ffe5b",
        source_artifact: "haricot-pose-rtmpose-m-body26-candle/README.md",
        implementation_commit: Some("933f598ae50c5b8f1948c5a20d5ceb88a997f785"),
        benchmark_commit: Some("19e061d1c9615a21cb3a60870a6c2a746f2e985b"),
        evidence_sha256: None,
        note: "Historical grouped-convolution probe; exact F32 parity. Preserved as lineage, not promotion evidence.",
    },
    HistoricalGain {
        transition_id: "dw5x5-c48-cudnn-current-to-raw-asd",
        decision_id: "conv2d-dw5x5-f32-b1-c48-h64-w48-g48-s1-p2-d1-raw",
        state: HistoricalState::PromotionLineage,
        generation: 0,
        predecessor_transition_id: None,
        exact_signature: DW48,
        from_provider: "cudnn",
        from_implementation: "candle.cudnn.grouped-conv2d.native.v1",
        to_provider: "raw_cuda",
        to_implementation: "candle.depthwise-conv2d-5x5.raw.v1",
        speedup_x: 14.497987,
        earlier_observed_speedup_x: None,
        source_commit: "c2cb20de0abde86f82f7803964580c859402d428",
        source_artifact: "candle-core/examples/ASD_DW5X5_EXACT_INTEGRATED_V0.md",
        implementation_commit: None,
        benchmark_commit: None,
        evidence_sha256: Some(PROMOTION_EVIDENCE),
        note: "Time-equivalent integrated validation; exact parity, drift and p90 gates passed.",
    },
    HistoricalGain {
        transition_id: "dw5x5-c96-cudnn-current-to-raw-asd",
        decision_id: "conv2d-dw5x5-f32-b1-c96-h32-w24-g96-s1-p2-d1-raw",
        state: HistoricalState::PromotionLineage,
        generation: 0,
        predecessor_transition_id: None,
        exact_signature: DW96,
        from_provider: "cudnn",
        from_implementation: "candle.cudnn.grouped-conv2d.native.v1",
        to_provider: "raw_cuda",
        to_implementation: "candle.depthwise-conv2d-5x5.raw.v1",
        speedup_x: 22.132121,
        earlier_observed_speedup_x: None,
        source_commit: "c2cb20de0abde86f82f7803964580c859402d428",
        source_artifact: "candle-core/examples/ASD_DW5X5_EXACT_INTEGRATED_V0.md",
        implementation_commit: None,
        benchmark_commit: None,
        evidence_sha256: Some(PROMOTION_EVIDENCE),
        note: "Time-equivalent integrated validation; exact parity, drift and p90 gates passed.",
    },
    HistoricalGain {
        transition_id: "dw5x5-c192-cudnn-current-to-raw-asd",
        decision_id: "conv2d-dw5x5-f32-b1-c192-h16-w12-g192-s1-p2-d1-raw",
        state: HistoricalState::PromotionLineage,
        generation: 0,
        predecessor_transition_id: None,
        exact_signature: DW192,
        from_provider: "cudnn",
        from_implementation: "candle.cudnn.grouped-conv2d.native.v1",
        to_provider: "raw_cuda",
        to_implementation: "candle.depthwise-conv2d-5x5.raw.v1",
        speedup_x: 37.356473,
        earlier_observed_speedup_x: None,
        source_commit: "c2cb20de0abde86f82f7803964580c859402d428",
        source_artifact: "candle-core/examples/ASD_DW5X5_EXACT_INTEGRATED_V0.md",
        implementation_commit: None,
        benchmark_commit: None,
        evidence_sha256: Some(PROMOTION_EVIDENCE),
        note: "Time-equivalent integrated validation; exact parity, drift and p90 gates passed.",
    },
    HistoricalGain {
        transition_id: "dw5x5-c384-cudnn-current-to-raw-asd",
        decision_id: "conv2d-dw5x5-f32-b1-c384-h8-w6-g384-s1-p2-d1-raw",
        state: HistoricalState::PromotionLineage,
        generation: 1,
        predecessor_transition_id: Some("dw5x5-c384-legacy-chunked-to-cudnn-grouped"),
        exact_signature: DW384,
        from_provider: "cudnn",
        from_implementation: "candle.cudnn.grouped-conv2d.native.v1",
        to_provider: "raw_cuda",
        to_implementation: "candle.depthwise-conv2d-5x5.raw.v1",
        speedup_x: 38.313419,
        earlier_observed_speedup_x: None,
        source_commit: "c2cb20de0abde86f82f7803964580c859402d428",
        source_artifact: "candle-core/examples/ASD_DW5X5_EXACT_INTEGRATED_V0.md",
        implementation_commit: None,
        benchmark_commit: None,
        evidence_sha256: Some(PROMOTION_EVIDENCE),
        note: "Time-equivalent integrated validation; exact parity, drift and p90 gates passed.",
    },
];

/// Provider challenges are descriptive historical memory, not execution authority.
///
/// A challenge records an evaluated alternative provider/implementation that did
/// not necessarily become a lineage transition. Stable rejects remain separate
/// from `GAINS` so tooling never invents a provider transition that did not occur.
pub const PROVIDER_CHALLENGES: [ProviderChallenge; 1] = [ProviderChallenge {
    challenge_id: "ct1d-s32-g2-raw-vs-cudnn-v1",
    decision_id: "ct1d-sm61-s32-g2-raw-exact",
    protocol: "provider-evidence-v1",
    measurement_plane: "production_path",
    harness_revision: "provider-evidence-v1-no-hot-trace-r2",
    exact_signature: CT1D_G2,
    incumbent_provider: "raw_cuda",
    incumbent_implementation: "candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256",
    challenger_provider: "cudnn",
    challenger_implementation: "candle.cudnn.grouped-transpose.v1",
    challenger_identity: CT1D_G2_CUDNN_IDENTITY,
    replications: &CT1D_G2_PROVIDER_REPLICATIONS,
    consensus: ProviderChallengeConsensus::StableReject,
    incumbent_retained: true,
    profile_change: false,
    note: "Two independent trace-clean V1 production-path replications rejected the cuDNN challenger with exact parity and stable drift. V1 evidence is preserved as-is; future provider duels use Provider Evidence V2.",
}];

pub fn provider_challenges_for_decision(
    decision_id: &str,
) -> Vec<&'static ProviderChallenge> {
    PROVIDER_CHALLENGES
        .iter()
        .filter(|challenge| challenge.decision_id == decision_id)
        .collect()
}

pub fn history_for_decision(decision_id: &str) -> Vec<&'static HistoricalGain> {
    let mut gains = GAINS
        .iter()
        .filter(|gain| gain.decision_id == decision_id)
        .collect::<Vec<_>>();
    gains.sort_by_key(|gain| gain.generation);
    gains
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c384_preserves_two_generation_lineage() {
        let gains = history_for_decision(
            "conv2d-dw5x5-f32-b1-c384-h8-w6-g384-s1-p2-d1-raw",
        );
        assert_eq!(gains.len(), 2);
        assert_eq!(gains[0].generation, 0);
        assert_eq!(gains[0].predecessor_transition_id, None);
        assert_eq!(gains[0].speedup_x, 402.470);
        assert_eq!(gains[0].earlier_observed_speedup_x, Some(445.265));
        assert_eq!(gains[1].generation, 1);
        assert_eq!(
            gains[1].predecessor_transition_id,
            Some("dw5x5-c384-legacy-chunked-to-cudnn-grouped")
        );
        assert_eq!(gains[1].speedup_x, 38.313419);
        assert_eq!(gains[1].evidence_sha256, Some(PROMOTION_EVIDENCE));
    }

    #[test]
    fn history_is_not_a_generalized_depthwise_rule() {
        assert!(history_for_decision("unknown").is_empty());
        assert_eq!(
            GAINS
                .iter()
                .filter(|gain| gain.state == HistoricalState::PromotionLineage)
                .count(),
            4
        );
    }


    #[test]
    fn ct1d_g2_provider_challenge_preserves_v1_stable_reject() {
        let challenges = provider_challenges_for_decision("ct1d-sm61-s32-g2-raw-exact");
        assert_eq!(challenges.len(), 1);
        let challenge = challenges[0];
        assert_eq!(challenge.protocol, "provider-evidence-v1");
        assert_eq!(challenge.consensus, ProviderChallengeConsensus::StableReject);
        assert!(challenge.incumbent_retained);
        assert!(!challenge.profile_change);
        assert_eq!(challenge.replications.len(), 2);
        assert_eq!(
            challenge.replications[0].evidence_sha256,
            "1d86ff6da849ffc29dc30ec4eee69f9ab91330314d74221db5fd82bed11dc07b"
        );
        assert_eq!(
            challenge.replications[1].evidence_sha256,
            "7e2d4d08f2210b0418261df07921c2ef33b0738d157cfbc834f7c7746f20e367"
        );
        assert!(challenge
            .replications
            .iter()
            .all(|replication| replication.decision == "REJECT_CHALLENGER"));
    }

    #[test]
    fn stable_provider_reject_is_not_a_historical_gain() {
        assert!(history_for_decision("ct1d-sm61-s32-g2-raw-exact").is_empty());
        assert_eq!(
            provider_challenges_for_decision("ct1d-sm61-s32-g2-raw-exact").len(),
            1
        );
    }
}
