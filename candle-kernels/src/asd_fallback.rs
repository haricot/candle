//! Runtime-qualified provider fallbacks for ASD Exact Profile decisions.
//!
//! This is production fallback authority, not historical memory. Entries are
//! admitted only after correctness and stability evidence demonstrates that the
//! fallback can execute the exact signature. Being a qualified fallback does
//! not mean the provider won promotion.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualifiedFallbackProvider {
    Cudnn,
    Native,
}

impl QualifiedFallbackProvider {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cudnn => "cudnn",
            Self::Native => "native",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QualifiedFallback {
    pub decision_id: &'static str,
    pub rank: u32,
    pub provider: QualifiedFallbackProvider,
    pub implementation_id: &'static str,
    pub protocol: &'static str,
    pub evidence_sha256: &'static [&'static str],
    pub required_cudnn_version_raw: Option<usize>,
    pub qualification: &'static str,
    pub note: &'static str,
}

const CT1D_G2_V1_EVIDENCE: [&str; 2] = [
    "1d86ff6da849ffc29dc30ec4eee69f9ab91330314d74221db5fd82bed11dc07b",
    "7e2d4d08f2210b0418261df07921c2ef33b0738d157cfbc834f7c7746f20e367",
];

const CT1D_G4_V2_EVIDENCE: [&str; 2] = [
    "8258710d0fbae354bd4b95505e0caf03d13cdb73c7016c82c5bc53228c2c69c2",
    "f190a815d690c2b9e665edf985c067dac0d97419c3d237558f87bf5581e9dd8f",
];

pub const QUALIFIED_FALLBACKS: [QualifiedFallback; 2] = [
    QualifiedFallback {
        decision_id: "ct1d-sm61-s32-g2-raw-exact",
        rank: 1,
        provider: QualifiedFallbackProvider::Cudnn,
        implementation_id: "candle.cudnn.grouped-transpose.v1",
        protocol: "provider-evidence-v1",
        evidence_sha256: &CT1D_G2_V1_EVIDENCE,
        required_cudnn_version_raw: Some(91002),
        qualification: "exact_parity+stable_execution",
        note: "Rejected for promotion on performance, but qualified as a resilience fallback when the promoted raw artifact is unavailable.",
    },
    QualifiedFallback {
        decision_id: "ct1d-sm61-s32-g4-raw-exact",
        rank: 1,
        provider: QualifiedFallbackProvider::Cudnn,
        implementation_id: "candle.cudnn.grouped-transpose.v1",
        protocol: "provider-evidence-v2",
        evidence_sha256: &CT1D_G4_V2_EVIDENCE,
        required_cudnn_version_raw: Some(91002),
        qualification: "exact_parity+stable_execution",
        note: "Two authoritative fallback-qualification replications passed exact parity and drift; cuDNN was rejected for promotion on performance but qualified for resilience.",
    },
];

pub fn qualified_fallbacks_for_decision(
    decision_id: &str,
) -> Vec<&'static QualifiedFallback> {
    let mut fallbacks = QUALIFIED_FALLBACKS
        .iter()
        .filter(|fallback| fallback.decision_id == decision_id)
        .collect::<Vec<_>>();
    fallbacks.sort_by_key(|fallback| fallback.rank);
    fallbacks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct1d_g2_cudnn_is_fallback_not_primary() {
        let fallbacks = qualified_fallbacks_for_decision("ct1d-sm61-s32-g2-raw-exact");
        assert_eq!(fallbacks.len(), 1);
        assert_eq!(fallbacks[0].rank, 1);
        assert_eq!(fallbacks[0].provider, QualifiedFallbackProvider::Cudnn);
        assert_eq!(fallbacks[0].protocol, "provider-evidence-v1");
        assert_eq!(fallbacks[0].evidence_sha256.len(), 2);
        assert_eq!(fallbacks[0].required_cudnn_version_raw, Some(91002));
    }

    #[test]
    fn ct1d_g4_cudnn_is_qualified_from_two_v2_replications() {
        let fallbacks = qualified_fallbacks_for_decision("ct1d-sm61-s32-g4-raw-exact");
        assert_eq!(fallbacks.len(), 1);
        assert_eq!(fallbacks[0].rank, 1);
        assert_eq!(fallbacks[0].provider, QualifiedFallbackProvider::Cudnn);
        assert_eq!(fallbacks[0].protocol, "provider-evidence-v2");
        assert_eq!(
            fallbacks[0].evidence_sha256,
            &[
                "8258710d0fbae354bd4b95505e0caf03d13cdb73c7016c82c5bc53228c2c69c2",
                "f190a815d690c2b9e665edf985c067dac0d97419c3d237558f87bf5581e9dd8f",
            ]
        );
        assert_eq!(fallbacks[0].required_cudnn_version_raw, Some(91002));
    }
}
