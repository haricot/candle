//! Opt-in, operation-keyed Conv1D ASD candidate validation for candle-nn.
//!
//! This is NOT a promoted universal policy. A/B/A gains from MotionBricks are
//! attributable only to the original *atomic combination*, not these rules.
//! No policy is auto-loaded; no ASD code runs without `with_candidate_profile`.
//! Hardware declarations are not attestation. See the accompanying README.

use crate::conv::Conv1d;
use candle::{conv::CudnnFwdAlgo, DType, Device, Result, Tensor};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

const SCHEMA: &str = "flow.candle.asd.conv1d-candidates.v0.1";

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Route {
    K1Gemm,
    CudnnDirect,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Rule {
    id: String,
    op: String,
    input: [usize; 3],
    weight: [usize; 3],
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    dtype: String,
    input_contiguous: bool,
    weight_contiguous: bool,
    bias_present: bool,
    bias_contiguous: bool,
    route: Route,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyFile {
    schema: String,
    status: String,
    declared_sm: u32,
    declared_cudnn: String,
    evidence_scope: String,
    individual_route_speedups_validated: bool,
    rules: Vec<Rule>,
}

/// Candidate-only policy: its type deliberately cannot express `promoted`.
#[derive(Debug, Clone)]
pub struct AsdCandidateProfile {
    rules: Vec<Rule>,
    sha256: String,
}

impl AsdCandidateProfile {
    /// Read an operation-keyed candidate profile. `sm` and `cudnn` are
    /// *caller declarations*, not independently verified GPU/library identity.
    /// A mismatch or unsupported status is an error, never a silent promotion.
    pub fn from_path(path: &Path, declared_sm: u32, declared_cudnn: &str) -> Result<Self> {
        let metadata = fs::symlink_metadata(path)
            .map_err(|e| candle::Error::msg(format!("ASD candidate file {}: {e}", path.display())))?;
        if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 128 * 1024 {
            candle::bail!("ASD candidate policy must be a regular file <= 128 KiB");
        }
        let bytes = fs::read(path)
            .map_err(|e| candle::Error::msg(format!("ASD candidate read: {e}")))?;
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        let doc: PolicyFile = serde_json::from_slice(&bytes)
            .map_err(|e| candle::Error::msg(format!("ASD candidate JSON: {e}")))?;
        if doc.schema != SCHEMA || doc.status != "candidate_validation_only"
            || doc.evidence_scope != "atomic_bundle_only"
            || doc.individual_route_speedups_validated
            || doc.declared_sm != declared_sm || doc.declared_cudnn != declared_cudnn
            || declared_sm != 61 || declared_cudnn != "9.1.0.2" {
            candle::bail!("ASD profile schema/status/evidence/hardware mismatch; no promotion");
        }
        if doc.rules.is_empty() || doc.rules.len() > 64 {
            candle::bail!("ASD candidate requires 1..=64 exact rules");
        }
        let mut ids = HashSet::new();
        let mut keys = HashSet::new();
        for rule in &doc.rules {
            if rule.id.is_empty() || !ids.insert(rule.id.clone()) || rule.op != "conv1d"
                || rule.dtype != "F32" || rule.input.iter().any(|&n| n == 0)
                || rule.weight.iter().any(|&n| n == 0)
                || rule.input[1] != rule.weight[1] || rule.groups != 1
                || rule.padding > 1024 || rule.stride != 1 || rule.dilation != 1
                || !rule.input_contiguous || !rule.weight_contiguous
                || !rule.bias_present || !rule.bias_contiguous {
                candle::bail!("invalid/duplicate/noncontiguous ASD candidate rule {}", rule.id);
            }
            if !matches!((rule.route, rule.weight[2], rule.padding),
                    (Route::K1Gemm, 1, 0) | (Route::CudnnDirect, 3, 1)) {
                candle::bail!("route and Conv1D geometry mismatch for {}", rule.id);
            }
            let key = (rule.input, rule.weight, rule.padding, rule.stride,
                       rule.dilation, rule.groups);
            if !keys.insert(key) {
                candle::bail!("ambiguous duplicate ASD Conv1D rule {}", rule.id);
            }
        }
        Ok(Self { rules: doc.rules, sha256 })
    }

    pub fn sha256(&self) -> &str { &self.sha256 }
}

thread_local! {
    static ACTIVE: RefCell<Option<AsdCandidateProfile>> = const { RefCell::new(None) };
}

struct RestoreProfile(Option<AsdCandidateProfile>);
impl Drop for RestoreProfile {
    fn drop(&mut self) {
        ACTIVE.with(|slot| { slot.replace(self.0.take()); });
    }
}

/// Apply an explicit *candidate* profile to synchronous calls on this thread.
/// Nested calls and panics restore the previous profile; worker threads do
/// not inherit it. Do not use this for production until per-route promotion.
pub fn with_candidate_profile<R>(profile: &AsdCandidateProfile, run: impl FnOnce() -> R) -> R {
    let old = ACTIVE.with(|slot| slot.replace(Some(profile.clone())));
    let _restore = RestoreProfile(old);
    run()
}

#[cfg(feature = "cuda")]
fn runtime_sm61(x: &Tensor) -> bool {
    match x.device() {
        Device::Cuda(device) =>
            device.cuda_stream().context().compute_capability().ok() == Some((6, 1)),
        _ => false,
    }
}

#[cfg(not(feature = "cuda"))]
fn runtime_sm61(_x: &Tensor) -> bool { false }

fn applicable(rule: &Rule, conv: &Conv1d, x: &Tensor) -> bool {
    let Some(bias) = conv.bias() else { return false; };
    let cfg = conv.config();
    matches!(x.device(), Device::Cuda(_))
        && runtime_sm61(x)
        && x.device().same_device(conv.weight().device())
        && x.device().same_device(bias.device())
        && x.dtype() == DType::F32 && conv.weight().dtype() == DType::F32
        && bias.dtype() == DType::F32 && x.is_contiguous()
        && conv.weight().is_contiguous() && bias.is_contiguous()
        && x.dims() == rule.input && conv.weight().dims() == rule.weight
        && bias.dims() == [rule.weight[0]]
        && cfg.padding == rule.padding && cfg.stride == rule.stride
        && cfg.dilation == rule.dilation && cfg.groups == rule.groups
        && cfg.cudnn_fwd_algo.is_none() // Never override an explicit Candle algo.
}

fn run_k1_gemm(conv: &Conv1d, x: &Tensor) -> Result<Tensor> {
    let (batch, cin, len) = x.dims3()?;
    let (cout, wcin, kernel) = conv.weight().dims3()?;
    if kernel != 1 || cin != wcin { candle::bail!("ASD K1 GEMM shape mismatch"); }
    let a = x.transpose(1, 2)?.contiguous()?.reshape((batch * len, cin))?;
    let b = conv.weight().reshape((cout, cin))?.transpose(0, 1)?.contiguous()?;
    let y = a.matmul(&b)?;
    let bias = conv.bias().expect("applicability checked");
    y.broadcast_add(&bias.reshape((1, cout))?)?
        .reshape((batch, len, cout))?.transpose(1, 2)?.contiguous()
}

#[cfg(feature = "cudnn")]
fn run_cudnn_direct(conv: &Conv1d, x: &Tensor) -> Result<Tensor> {
    let cfg = conv.config();
    let y = x.conv1d_with_algo(conv.weight(), cfg.padding, cfg.stride,
                               cfg.dilation, cfg.groups, Some(CudnnFwdAlgo::Direct))?;
    let bias = conv.bias().expect("applicability checked");
    y.broadcast_add(&bias.reshape((1, bias.dims1()?, 1))?)
}

#[cfg(not(feature = "cudnn"))]
fn run_cudnn_direct(_conv: &Conv1d, _x: &Tensor) -> Result<Tensor> {
    candle::bail!("ASD cuDNN Direct selected without candle-nn/cudnn feature");
}

/// Returns None for default Candle (no profile, CPU, unsupported layouts/shapes).
/// A selected candidate route returns Some(Result); failures never silently
/// fall back to an unmeasured alternate kernel.
pub(crate) fn maybe_forward(conv: &Conv1d, x: &Tensor) -> Option<Result<Tensor>> {
    ACTIVE.with(|slot| {
        let profile = slot.borrow();
        let p = profile.as_ref()?;
        // The same operation-key matcher selects grouped and ungrouped routes.
        let input: [usize; 3] = x.dims().try_into().ok()?;
        let weight: [usize; 3] = conv.weight().dims().try_into().ok()?;
        let cfg = conv.config();
        let key = candle::asd_registry::OperationKey::conv1d(
            input, weight, cfg.padding, cfg.stride, cfg.dilation, cfg.groups,
        );
        let rule = candle::asd_registry::first_exact(&key, &p.rules, |rule| {
            Some(candle::asd_registry::OperationKey::conv1d(
                rule.input, rule.weight, rule.padding, rule.stride,
                rule.dilation, rule.groups,
            ))
        })?;
        if !applicable(rule, conv, x) { return None; }
        let result = match rule.route {
            Route::K1Gemm => run_k1_gemm(conv, x),
            Route::CudnnDirect => run_cudnn_direct(conv, x),
        };
        if result.is_ok() {
            eprintln!("CANDLE_ASD_ROUTE=EXECUTED status=CANDIDATE_VALIDATION_ONLY rule_id={} route={:?} profile_sha256={} hardware_identity=CALLER_DECLARED_NOT_ATTESTED", rule.id, rule.route, p.sha256);
        }
        Some(result)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn opt_in_scope_restores_after_nested_calls_and_panic() {
        assert!(ACTIVE.with(|s| s.borrow().is_none()));
        let a = AsdCandidateProfile { rules: vec![], sha256: "a".into() };
        let b = AsdCandidateProfile { rules: vec![], sha256: "b".into() };
        with_candidate_profile(&a, || {
            assert_eq!(ACTIVE.with(|s| s.borrow().as_ref().unwrap().sha256.clone()), "a");
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                with_candidate_profile(&b, || { panic!("restore on panic") });
            }));
            assert_eq!(ACTIVE.with(|s| s.borrow().as_ref().unwrap().sha256.clone()), "a");
        });
        assert!(ACTIVE.with(|s| s.borrow().is_none()));
    }

    #[test]
    fn baseline_without_profile_cannot_dispatch() {
        let x = Tensor::zeros((1, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let w = Tensor::zeros((2, 2, 1), DType::F32, &Device::Cpu).unwrap();
        let conv = Conv1d::new(w, None, Default::default());
        assert!(maybe_forward(&conv, &x).is_none());
    }
}
