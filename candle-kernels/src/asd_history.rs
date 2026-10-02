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
    pub exact_signature: &'static str,
    pub from_provider: &'static str,
    pub from_implementation: &'static str,
    pub to_provider: &'static str,
    pub to_implementation: &'static str,
    pub speedup_x: f64,
    /// Optional earlier observation of the same transition, kept only as context.
    pub earlier_observed_speedup_x: Option<f64>,
    pub source_commit: &'static str,
    pub source_artifact: &'static str,
    pub evidence_sha256: Option<&'static str>,
    pub note: &'static str,
}

const DW48: &str = "op=conv2d,dim=2,batch=1,c_in=48,c_out=48,spatial=64x48,weight_shape=48x1x5x5,groups=48,kernel=5,stride=1,padding=2,output_padding=none,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset";
const DW96: &str = "op=conv2d,dim=2,batch=1,c_in=96,c_out=96,spatial=32x24,weight_shape=96x1x5x5,groups=96,kernel=5,stride=1,padding=2,output_padding=none,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset";
const DW192: &str = "op=conv2d,dim=2,batch=1,c_in=192,c_out=192,spatial=16x12,weight_shape=192x1x5x5,groups=192,kernel=5,stride=1,padding=2,output_padding=none,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset";
const DW384: &str = "op=conv2d,dim=2,batch=1,c_in=384,c_out=384,spatial=8x6,weight_shape=384x1x5x5,groups=384,kernel=5,stride=1,padding=2,output_padding=none,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset";

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
        exact_signature: DW384,
        from_provider: "native",
        from_implementation: "candle.legacy-grouped-conv2d.chunk-per-group-cat",
        to_provider: "cudnn",
        to_implementation: "candle.cudnn.grouped-conv2d.native.v1",
        speedup_x: 402.470,
        earlier_observed_speedup_x: Some(445.265),
        source_commit: "fec68ced96355e7816e9ad65f2a2f88c583ffe5b",
        source_artifact: "haricot-pose-rtmpose-m-body26-candle/README.md",
        evidence_sha256: None,
        note: "Historical grouped-convolution probe; exact F32 parity. Preserved as lineage, not promotion evidence.",
    },
    HistoricalGain {
        transition_id: "dw5x5-c48-cudnn-current-to-raw-asd",
        decision_id: "conv2d-dw5x5-f32-b1-c48-h64-w48-g48-s1-p2-d1-raw",
        state: HistoricalState::PromotionLineage,
        exact_signature: DW48,
        from_provider: "cudnn",
        from_implementation: "candle.current-grouped-conv2d",
        to_provider: "raw_cuda",
        to_implementation: "candle.depthwise-conv2d-5x5.raw.v1",
        speedup_x: 14.497987,
        earlier_observed_speedup_x: None,
        source_commit: "c2cb20de0abde86f82f7803964580c859402d428",
        source_artifact: "candle-core/examples/ASD_DW5X5_EXACT_INTEGRATED_V0.md",
        evidence_sha256: Some(PROMOTION_EVIDENCE),
        note: "Time-equivalent integrated validation; exact parity, drift and p90 gates passed.",
    },
    HistoricalGain {
        transition_id: "dw5x5-c96-cudnn-current-to-raw-asd",
        decision_id: "conv2d-dw5x5-f32-b1-c96-h32-w24-g96-s1-p2-d1-raw",
        state: HistoricalState::PromotionLineage,
        exact_signature: DW96,
        from_provider: "cudnn",
        from_implementation: "candle.current-grouped-conv2d",
        to_provider: "raw_cuda",
        to_implementation: "candle.depthwise-conv2d-5x5.raw.v1",
        speedup_x: 22.132121,
        earlier_observed_speedup_x: None,
        source_commit: "c2cb20de0abde86f82f7803964580c859402d428",
        source_artifact: "candle-core/examples/ASD_DW5X5_EXACT_INTEGRATED_V0.md",
        evidence_sha256: Some(PROMOTION_EVIDENCE),
        note: "Time-equivalent integrated validation; exact parity, drift and p90 gates passed.",
    },
    HistoricalGain {
        transition_id: "dw5x5-c192-cudnn-current-to-raw-asd",
        decision_id: "conv2d-dw5x5-f32-b1-c192-h16-w12-g192-s1-p2-d1-raw",
        state: HistoricalState::PromotionLineage,
        exact_signature: DW192,
        from_provider: "cudnn",
        from_implementation: "candle.current-grouped-conv2d",
        to_provider: "raw_cuda",
        to_implementation: "candle.depthwise-conv2d-5x5.raw.v1",
        speedup_x: 37.356473,
        earlier_observed_speedup_x: None,
        source_commit: "c2cb20de0abde86f82f7803964580c859402d428",
        source_artifact: "candle-core/examples/ASD_DW5X5_EXACT_INTEGRATED_V0.md",
        evidence_sha256: Some(PROMOTION_EVIDENCE),
        note: "Time-equivalent integrated validation; exact parity, drift and p90 gates passed.",
    },
    HistoricalGain {
        transition_id: "dw5x5-c384-cudnn-current-to-raw-asd",
        decision_id: "conv2d-dw5x5-f32-b1-c384-h8-w6-g384-s1-p2-d1-raw",
        state: HistoricalState::PromotionLineage,
        exact_signature: DW384,
        from_provider: "cudnn",
        from_implementation: "candle.current-grouped-conv2d",
        to_provider: "raw_cuda",
        to_implementation: "candle.depthwise-conv2d-5x5.raw.v1",
        speedup_x: 38.313419,
        earlier_observed_speedup_x: None,
        source_commit: "c2cb20de0abde86f82f7803964580c859402d428",
        source_artifact: "candle-core/examples/ASD_DW5X5_EXACT_INTEGRATED_V0.md",
        evidence_sha256: Some(PROMOTION_EVIDENCE),
        note: "Time-equivalent integrated validation; exact parity, drift and p90 gates passed.",
    },
];

pub fn history_for_decision(
    decision_id: &str,
) -> impl Iterator<Item = &'static HistoricalGain> {
    GAINS.iter().filter(move |gain| gain.decision_id == decision_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c384_preserves_two_generation_lineage() {
        let gains = history_for_decision(
            "conv2d-dw5x5-f32-b1-c384-h8-w6-g384-s1-p2-d1-raw",
        )
        .collect::<Vec<_>>();
        assert_eq!(gains.len(), 2);
        assert_eq!(gains[0].speedup_x, 402.470);
        assert_eq!(gains[0].earlier_observed_speedup_x, Some(445.265));
        assert_eq!(gains[1].speedup_x, 38.313419);
        assert_eq!(gains[1].evidence_sha256, Some(PROMOTION_EVIDENCE));
    }

    #[test]
    fn history_is_not_a_generalized_depthwise_rule() {
        assert!(history_for_decision("unknown").next().is_none());
        assert_eq!(
            GAINS
                .iter()
                .filter(|gain| gain.state == HistoricalState::PromotionLineage)
                .count(),
            4
        );
    }
}
