//! Runtime ASD Exact Profile authority.
//!
//! Exact decisions are loaded from the runtime profile, never compiled into Candle.
//! The default source is `profiles/current.asd`; `CANDLE_ASD_RUNTIME_PROFILE`
//! may explicitly override it. The profile is parsed once into an in-memory exact
//! key index. Explicit refresh invalidates both the global profile cache and the
//! thread-local lookup cache; steady-state lookup performs no filesystem I/O.

use crate::{Error, Result};
use candle_kernels::asd_exact::{ExactOperation, ExactOperationCall};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

const LEGACY_POLICY_HEADER_V2: &str = "ASD-EXACT-POLICY-V2";
const PROFILE_HEADER_V3: &str = "ASD-EXACT-PROFILE-V3";
const EXTENSIONS_HEADER_V1: &str = "ASD-EXACT-EXTENSIONS-V1";

const DECISION_SIGNATURE_KEYS: &[&str] = &[
    "op",
    "dim",
    "batch",
    "c_in",
    "c_out",
    "spatial",
    "weight_shape",
    "groups",
    "kernel",
    "stride",
    "padding",
    "output_padding",
    "dilation",
    "dtype",
    "input_layout",
    "weight_layout",
];

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct ExactKey {
    op: u8,
    dim: u8,
    batch: usize,
    c_in: usize,
    c_out: usize,
    spatial0: usize,
    spatial1: usize,
    weight_rank: usize,
    weight0: usize,
    weight1: usize,
    weight2: usize,
    weight3: usize,
    groups: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RuntimeExecutionProvider {
    RawCuda,
    Cudnn,
    Native,
}

impl RuntimeExecutionProvider {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::RawCuda => "raw_cuda",
            Self::Cudnn => "cudnn",
            Self::Native => "native",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RuntimeExactMatch {
    pub profile_id: Arc<str>,
    pub decision_id: Arc<str>,
    pub decision_identity_sha256: Arc<str>,
    pub state: Arc<str>,
    pub execution_provider: RuntimeExecutionProvider,
    pub implementation_id: Arc<str>,
    pub evidence_sha256: Arc<str>,
    pub min_integrated_speedup_x: Option<f64>,
}

#[derive(Debug)]
struct RuntimeProfile {
    target_gpu_uuid: String,
    decisions: HashMap<ExactKey, Arc<RuntimeExactMatch>>,
}

#[derive(Debug)]
enum CacheState {
    Unresolved,
    Unavailable,
    Loaded(Arc<RuntimeProfile>),
    Invalid(String),
}

static CACHE: OnceLock<RwLock<CacheState>> = OnceLock::new();
// 0 = unresolved, 1 = no runtime profile/decisions, 2 = authoritative profile loaded.
static PROFILE_MODE: AtomicU8 = AtomicU8::new(0);
// Explicit refresh invalidates thread-local hits without adding a global lock to
// steady-state selection.
static GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct LastLookup {
    generation: u64,
    key: ExactKey,
    result: Option<Arc<RuntimeExactMatch>>,
}

thread_local! {
    static LAST_LOOKUP: RefCell<Option<LastLookup>> = const { RefCell::new(None) };
}

fn cache() -> &'static RwLock<CacheState> {
    CACHE.get_or_init(|| RwLock::new(CacheState::Unresolved))
}

fn profile_path() -> Option<PathBuf> {
    std::env::var_os("CANDLE_ASD_RUNTIME_PROFILE")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(candle_kernels::asd_paths::current_profile_path)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_usize(fields: &BTreeMap<String, String>, key: &str) -> Result<usize> {
    fields
        .get(key)
        .ok_or_else(|| Error::Msg(format!("runtime ASD decision missing {key}")))?
        .parse::<usize>()
        .map_err(|err| Error::Msg(format!("runtime ASD decision invalid {key}: {err}")))
}

fn dims(value: &str) -> Result<Vec<usize>> {
    value
        .split('x')
        .map(|part| {
            part.parse::<usize>()
                .map_err(|err| Error::Msg(format!("runtime ASD invalid dimension {part:?}: {err}")))
        })
        .collect()
}

fn op_code(op: &str) -> Result<u8> {
    Ok(match op {
        "conv1d" => 1,
        "conv2d" => 2,
        "conv_transpose1d" => 3,
        "conv_transpose2d" => 4,
        _ => return Err(Error::Msg(format!("runtime ASD unsupported operation {op:?}"))),
    })
}

fn call_op_code(op: ExactOperation) -> u8 {
    match op {
        ExactOperation::Conv1d => 1,
        ExactOperation::Conv2d => 2,
        ExactOperation::ConvTranspose1d => 3,
        ExactOperation::ConvTranspose2d => 4,
    }
}

fn key_from_signature(fields: &BTreeMap<String, String>) -> Result<ExactKey> {
    if fields.get("dtype").map(String::as_str) != Some("f32")
        || fields.get("input_layout").map(String::as_str) != Some("contiguous_zero_offset")
        || fields.get("weight_layout").map(String::as_str) != Some("contiguous_zero_offset")
    {
        return Err(Error::Msg(
            "runtime ASD exact decisions require f32 contiguous_zero_offset".into(),
        ));
    }

    let spatial = dims(
        fields
            .get("spatial")
            .ok_or_else(|| Error::Msg("runtime ASD decision missing spatial".into()))?,
    )?;
    let weight = dims(
        fields
            .get("weight_shape")
            .ok_or_else(|| Error::Msg("runtime ASD decision missing weight_shape".into()))?,
    )?;
    if spatial.is_empty() || spatial.len() > 2 || weight.len() < 3 || weight.len() > 4 {
        return Err(Error::Msg("runtime ASD decision has invalid tensor rank".into()));
    }

    let output_padding = match fields.get("output_padding").map(String::as_str) {
        Some("none") => 0,
        Some(value) => value
            .parse::<usize>()
            .map_err(|err| Error::Msg(format!("invalid output_padding: {err}")))?,
        None => return Err(Error::Msg("runtime ASD decision missing output_padding".into())),
    };

    Ok(ExactKey {
        op: op_code(
            fields
                .get("op")
                .ok_or_else(|| Error::Msg("runtime ASD decision missing op".into()))?,
        )?,
        dim: parse_usize(fields, "dim")? as u8,
        batch: parse_usize(fields, "batch")?,
        c_in: parse_usize(fields, "c_in")?,
        c_out: parse_usize(fields, "c_out")?,
        spatial0: spatial.first().copied().unwrap_or(0),
        spatial1: spatial.get(1).copied().unwrap_or(0),
        weight_rank: weight.len(),
        weight0: weight.first().copied().unwrap_or(0),
        weight1: weight.get(1).copied().unwrap_or(0),
        weight2: weight.get(2).copied().unwrap_or(0),
        weight3: weight.get(3).copied().unwrap_or(0),
        groups: parse_usize(fields, "groups")?,
        kernel: parse_usize(fields, "kernel")?,
        stride: parse_usize(fields, "stride")?,
        padding: parse_usize(fields, "padding")?,
        output_padding,
        dilation: parse_usize(fields, "dilation")?,
    })
}

fn key_from_call(call: ExactOperationCall) -> Option<ExactKey> {
    if call.dtype != "f32"
        || !call.input_contiguous
        || call.input_start_offset != 0
        || !call.weight_contiguous
        || call.weight_start_offset != 0
    {
        return None;
    }

    Some(ExactKey {
        op: call_op_code(call.op),
        dim: call.dim,
        batch: call.batch,
        c_in: call.c_in,
        c_out: call.c_out,
        spatial0: call.spatial0,
        spatial1: call.spatial1,
        weight_rank: call.weight_rank,
        weight0: call.weight0,
        weight1: call.weight1,
        weight2: call.weight2,
        weight3: call.weight3,
        groups: call.groups,
        kernel: call.kernel,
        stride: call.stride,
        padding: call.padding,
        output_padding: call.output_padding,
        dilation: call.dilation,
    })
}

fn signature_map(signature: &str) -> Result<BTreeMap<String, String>> {
    let mut fields = BTreeMap::new();
    for part in signature.split(',') {
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| Error::Msg(format!("malformed runtime ASD signature field {part:?}")))?;
        if key.is_empty()
            || value.is_empty()
            || fields.insert(key.to_owned(), value.to_owned()).is_some()
        {
            return Err(Error::Msg(format!(
                "duplicate/empty runtime ASD signature field {key:?}"
            )));
        }
    }

    if fields.len() != DECISION_SIGNATURE_KEYS.len()
        || DECISION_SIGNATURE_KEYS
            .iter()
            .any(|key| !fields.contains_key(*key))
    {
        return Err(Error::Msg(
            "runtime ASD signature fields differ from ASD-DECISION-V1".into(),
        ));
    }
    Ok(fields)
}

fn canonical_decision(
    id: &str,
    state: &str,
    provider: &str,
    signature: &BTreeMap<String, String>,
    implementation_id: &str,
    evidence: &str,
    min_speedup: &str,
) -> Result<String> {
    let mut out = String::from("ASD-DECISION-V1\n");
    out.push_str(&format!("id={id}\n"));
    out.push_str(&format!("state={state}\n"));
    out.push_str(&format!("provider={provider}\n"));
    for key in DECISION_SIGNATURE_KEYS {
        let value = signature
            .get(*key)
            .ok_or_else(|| Error::Msg(format!("runtime ASD decision missing {key}")))?;
        out.push_str(&format!("{key}={value}\n"));
    }
    out.push_str(&format!("implementation_id={implementation_id}\n"));
    out.push_str(&format!("evidence_sha256={evidence}\n"));
    out.push_str(&format!("min_integrated_speedup_x={min_speedup}\n"));
    Ok(out)
}

fn decision_identity(canonical: &str) -> String {
    Sha256::digest(canonical.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn execution_provider(value: &str) -> Result<RuntimeExecutionProvider> {
    match value {
        "raw_cuda" => Ok(RuntimeExecutionProvider::RawCuda),
        "cudnn" => Ok(RuntimeExecutionProvider::Cudnn),
        "native" => Ok(RuntimeExecutionProvider::Native),
        _ => Err(Error::Msg(format!(
            "runtime ASD decision uses unsupported provider {value:?}"
        ))),
    }
}

enum ProfileWireKind {
    Full,
    Extension,
}

fn profile_identity(
    kind: &ProfileWireKind,
    fields: &BTreeMap<String, String>,
) -> Result<String> {
    match kind {
        ProfileWireKind::Full => match (fields.get("profile_id"), fields.get("policy_id")) {
            (Some(profile), None) => Ok(profile.clone()),
            (None, Some(policy)) => Ok(policy.clone()),
            (Some(_), Some(_)) => Err(Error::Msg(
                "runtime Exact Profile cannot contain both profile_id and policy_id".into(),
            )),
            (None, None) => Err(Error::Msg("runtime Exact Profile missing identity".into())),
        },
        ProfileWireKind::Extension => fields
            .get("base_profile_id")
            .cloned()
            .ok_or_else(|| Error::Msg("runtime ASD extension missing base_profile_id".into())),
    }
}

fn validate_target(
    kind: &ProfileWireKind,
    fields: &BTreeMap<String, String>,
) -> Result<String> {
    let architecture = fields
        .get("target.architecture")
        .ok_or_else(|| Error::Msg("runtime ASD profile missing target.architecture".into()))?;
    let target_gpu_uuid = fields
        .get("target.gpu_uuid")
        .ok_or_else(|| Error::Msg("runtime ASD profile missing target.gpu_uuid".into()))?
        .clone();

    let expected_architecture = format!("sm{}", candle_kernels::CUDA_BUILD_COMPUTE_CAP);
    if architecture != &expected_architecture {
        return Err(Error::Msg(format!(
            "runtime ASD profile architecture mismatch: profile={architecture} build={expected_architecture}"
        )));
    }
    if !target_gpu_uuid.starts_with("GPU-") {
        return Err(Error::Msg(
            "runtime ASD profile target.gpu_uuid must be device scoped".into(),
        ));
    }

    if matches!(kind, ProfileWireKind::Full) {
        if fields.get("target.vendor").map(String::as_str) != Some("nvidia")
            || fields.get("target.scope").map(String::as_str) != Some("device")
        {
            return Err(Error::Msg(
                "runtime Exact Profile must target an NVIDIA device scope".into(),
            ));
        }
        let target_sm = fields
            .get("target.sm")
            .ok_or_else(|| Error::Msg("runtime Exact Profile missing target.sm".into()))?
            .parse::<u32>()
            .map_err(|err| Error::Msg(format!("invalid runtime target.sm: {err}")))?;
        if target_sm != candle_kernels::CUDA_BUILD_COMPUTE_CAP {
            return Err(Error::Msg(format!(
                "runtime Exact Profile SM mismatch: profile={target_sm} build={}",
                candle_kernels::CUDA_BUILD_COMPUTE_CAP
            )));
        }
    }

    Ok(target_gpu_uuid)
}

fn parse_profile_source(source: &str) -> Result<Option<Arc<RuntimeProfile>>> {
    let mut lines = source.lines();
    let header = lines
        .next()
        .ok_or_else(|| Error::Msg("empty runtime ASD profile".into()))?;
    let kind = match header {
        LEGACY_POLICY_HEADER_V2 | PROFILE_HEADER_V3 => ProfileWireKind::Full,
        EXTENSIONS_HEADER_V1 => ProfileWireKind::Extension,
        _ => {
            return Err(Error::Msg(format!(
                "unsupported runtime ASD profile header {header:?}"
            )))
        }
    };

    let mut profile_fields = BTreeMap::<String, String>::new();
    let mut rows = Vec::new();
    for raw in lines {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("decision|") {
            rows.push(line.to_owned());
        } else {
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| Error::Msg(format!("invalid runtime ASD profile line {line:?}")))?;
            if profile_fields
                .insert(key.to_owned(), value.to_owned())
                .is_some()
            {
                return Err(Error::Msg(format!(
                    "duplicate runtime ASD profile field {key:?}"
                )));
            }
        }
    }

    let profile_id = profile_identity(&kind, &profile_fields)?;
    let target_gpu_uuid = validate_target(&kind, &profile_fields)?;

    let mut decisions = HashMap::new();
    for row in rows {
        let parts = row.split('|').collect::<Vec<_>>();
        if parts.len() != 8 {
            return Err(Error::Msg(format!(
                "runtime ASD decision requires eight fields: {row:?}"
            )));
        }

        let id = parts[1];
        let state = parts[2];
        if state != "promoted" {
            continue;
        }

        let provider_name = parts[3];
        let provider = execution_provider(provider_name)?;
        let signature = signature_map(parts[4])?;
        let implementation_id = parts[5]
            .strip_prefix("impl=")
            .ok_or_else(|| Error::Msg("runtime ASD decision missing impl=".into()))?;
        let evidence = parts[6]
            .strip_prefix("evidence=")
            .ok_or_else(|| Error::Msg("runtime ASD decision missing evidence=".into()))?;
        let min_speedup = parts[7]
            .strip_prefix("min_integrated_speedup_x=")
            .ok_or_else(|| Error::Msg("runtime ASD decision missing speedup field".into()))?;

        if !is_sha256(evidence) {
            return Err(Error::Msg(format!(
                "runtime ASD decision {id} has invalid evidence SHA-256"
            )));
        }

        let min_integrated_speedup_x = if min_speedup == "none" {
            None
        } else {
            let value = min_speedup.parse::<f64>().map_err(|err| {
                Error::Msg(format!(
                    "runtime ASD decision {id} has invalid speedup threshold: {err}"
                ))
            })?;
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::Msg(format!(
                    "runtime ASD decision {id} has non-positive speedup threshold"
                )));
            }
            Some(value)
        };

        let key = key_from_signature(&signature)?;
        let canonical = canonical_decision(
            id,
            state,
            provider_name,
            &signature,
            implementation_id,
            evidence,
            min_speedup,
        )?;
        let decision = Arc::new(RuntimeExactMatch {
            profile_id: Arc::from(profile_id.as_str()),
            decision_id: Arc::from(id),
            decision_identity_sha256: Arc::from(decision_identity(&canonical)),
            state: Arc::from(state),
            execution_provider: provider,
            implementation_id: Arc::from(implementation_id),
            evidence_sha256: Arc::from(evidence),
            min_integrated_speedup_x,
        });

        if decisions.insert(key, decision).is_some() {
            return Err(Error::Msg(format!(
                "runtime ASD profile contains duplicate exact signature for decision {id}"
            )));
        }
    }

    if decisions.is_empty() {
        return Ok(None);
    }

    Ok(Some(Arc::new(RuntimeProfile {
        target_gpu_uuid,
        decisions,
    })))
}

fn load_profile() -> Result<Option<Arc<RuntimeProfile>>> {
    let Some(path) = profile_path() else {
        return Ok(None);
    };
    if !path.is_file() {
        return Ok(None);
    }

    let source = std::fs::read_to_string(&path).map_err(|err| {
        Error::Msg(format!(
            "failed to read runtime ASD profile {}: {err}",
            path.display()
        ))
    })?;
    parse_profile_source(&source)
}

fn resolved_profile() -> Result<Option<Arc<RuntimeProfile>>> {
    {
        let state = cache().read().unwrap();
        match &*state {
            CacheState::Loaded(profile) => return Ok(Some(profile.clone())),
            CacheState::Unavailable => return Ok(None),
            CacheState::Invalid(message) => return Err(Error::Msg(message.clone())),
            CacheState::Unresolved => {}
        }
    }

    let mut state = cache().write().unwrap();
    match &*state {
        CacheState::Loaded(profile) => return Ok(Some(profile.clone())),
        CacheState::Unavailable => return Ok(None),
        CacheState::Invalid(message) => return Err(Error::Msg(message.clone())),
        CacheState::Unresolved => {}
    }

    match load_profile() {
        Ok(Some(profile)) => {
            *state = CacheState::Loaded(profile.clone());
            PROFILE_MODE.store(2, Ordering::Release);
            Ok(Some(profile))
        }
        Ok(None) => {
            *state = CacheState::Unavailable;
            PROFILE_MODE.store(1, Ordering::Release);
            Ok(None)
        }
        Err(err) => {
            let message = err.to_string();
            *state = CacheState::Invalid(message.clone());
            Err(Error::Msg(message))
        }
    }
}

pub(crate) fn lookup_exact(
    call: ExactOperationCall,
    actual_uuid: Option<&str>,
) -> Result<Option<Arc<RuntimeExactMatch>>> {
    if PROFILE_MODE.load(Ordering::Acquire) == 1 {
        return Ok(None);
    }

    let Some(key) = key_from_call(call) else {
        return Ok(None);
    };

    let generation = GENERATION.load(Ordering::Acquire);
    if let Some(result) = LAST_LOOKUP.with(|cache| {
        cache
            .borrow()
            .as_ref()
            .filter(|hit| hit.generation == generation && hit.key == key)
            .map(|hit| hit.result.clone())
    }) {
        return Ok(result);
    }

    let result = match resolved_profile()? {
        Some(profile) if actual_uuid == Some(profile.target_gpu_uuid.as_str()) => {
            profile.decisions.get(&key).cloned()
        }
        Some(_) | None => None,
    };

    LAST_LOOKUP.with(|cache| {
        *cache.borrow_mut() = Some(LastLookup {
            generation,
            key,
            result: result.clone(),
        });
    });

    Ok(result)
}


pub(crate) fn refresh() {
    *cache().write().unwrap() = CacheState::Unresolved;
    PROFILE_MODE.store(0, Ordering::Release);
    GENERATION.fetch_add(1, Ordering::AcqRel);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIG: &str = "op=conv_transpose1d,dim=1,batch=1,c_in=128,c_out=128,spatial=33,weight_shape=128x64x3,groups=2,kernel=3,stride=2,padding=1,output_padding=1,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset";

    #[test]
    fn full_profile_is_authoritative_without_embedded_profile() {
        let source = format!(
            "{PROFILE_HEADER_V3}\nprofile_id=test-runtime\ntarget.vendor=nvidia\ntarget.architecture=sm{}\ntarget.scope=device\ntarget.sm={}\ntarget.gpu_uuid=GPU-test\ndecision|runtime-s33|promoted|raw_cuda|{SIG}|impl=candle.runtime.s33|evidence={}|min_integrated_speedup_x=none\n",
            candle_kernels::CUDA_BUILD_COMPUTE_CAP,
            candle_kernels::CUDA_BUILD_COMPUTE_CAP,
            "a".repeat(64),
        );
        let profile = parse_profile_source(&source).unwrap().unwrap();
        assert_eq!(profile.target_gpu_uuid, "GPU-test");
        assert_eq!(profile.decisions.len(), 1);
        let matched = profile.decisions.values().next().unwrap();
        assert_eq!(&*matched.profile_id, "test-runtime");
        assert_eq!(&*matched.decision_id, "runtime-s33");
        assert_eq!(matched.execution_provider, RuntimeExecutionProvider::RawCuda);
    }

    #[test]
    fn extension_override_remains_accepted_during_d2_migration() {
        let source = format!(
            "{EXTENSIONS_HEADER_V1}\nbase_profile_id=test-runtime\ntarget.architecture=sm{}\ntarget.gpu_uuid=GPU-test\ndecision|runtime-s33|promoted|raw_cuda|{SIG}|impl=candle.runtime.s33|evidence={}|min_integrated_speedup_x=none\n",
            candle_kernels::CUDA_BUILD_COMPUTE_CAP,
            "b".repeat(64),
        );
        assert!(parse_profile_source(&source).unwrap().is_some());
    }
}
