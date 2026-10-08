//! ASD exact-operation compatibility types.
//!
//! Phase D2 deliberately contains no embedded Exact Profile. Runtime authority
//! lives in candle-core's runtime profile loader. These public types remain here
//! temporarily so existing callers/examples compile while the engine migrates.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactOperation {
    Conv1d,
    Conv2d,
    ConvTranspose1d,
    ConvTranspose2d,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactExecutionProvider {
    RawCuda,
    Cudnn,
    Native,
}

impl ExactExecutionProvider {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RawCuda => "raw_cuda",
            Self::Cudnn => "cudnn",
            Self::Native => "native",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactOperationCall {
    pub op: ExactOperation,
    pub dim: u8,
    pub batch: usize,
    pub c_in: usize,
    pub c_out: usize,
    pub spatial0: usize,
    pub spatial1: usize,
    pub weight_rank: usize,
    pub weight0: usize,
    pub weight1: usize,
    pub weight2: usize,
    pub weight3: usize,
    pub groups: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub dilation: usize,
    pub dtype: &'static str,
    pub input_contiguous: bool,
    pub input_start_offset: usize,
    pub weight_contiguous: bool,
    pub weight_start_offset: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactAsdMatch {
    pub profile_id: &'static str,
    pub policy_id: &'static str,
    pub decision_id: &'static str,
    pub decision_identity_sha256: &'static str,
    pub state: &'static str,
    pub execution_provider: ExactExecutionProvider,
    pub selected_backend: &'static str,
    pub implementation_id: &'static str,
    pub evidence_sha256: &'static str,
    pub min_integrated_speedup_x: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactUnprovenMatch {
    pub profile_id: &'static str,
    pub policy_id: &'static str,
    pub decision_id: &'static str,
    pub decision_identity_sha256: &'static str,
    pub state: &'static str,
    pub execution_provider: ExactExecutionProvider,
    pub selected_backend: &'static str,
    pub implementation_id: &'static str,
    pub proof_status: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExactMatch {
    Proven(ExactAsdMatch),
    Unproven(ExactUnprovenMatch),
}

// Compatibility constants: D2 guarantees there is no compiled Exact Profile.
pub const PROFILE_ID: Option<&str> = None;
pub const POLICY_ID: Option<&str> = None;
pub const TARGET_GPU_UUID: Option<&str> = None;
pub const VALIDATION_BUILD: bool = false;

// Compatibility lookup: all production authority is runtime-only in D2.
pub fn lookup(_call: ExactOperationCall, _actual_uuid: Option<&str>) -> Option<ExactMatch> {
    None
}
