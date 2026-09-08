//! Dependency-free build adapter for ASD exact-domain policies.
//!
//! v1 intentionally consumes one explicit `ASD-EXACT-POLICY-V1` file. It does
//! not implement policy layering or generalized conditions.
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::Write,
    path::{Path, PathBuf},
};

const HEADER: &str = "ASD-EXACT-POLICY-V1";
const RAW_IMPL_ID: &str = "candle.grouped-transpose.raw.v1";

#[derive(Debug, Clone)]
struct Decision {
    id: String,
    state: String,
    selected: String,
    signature: BTreeMap<String, String>,
    impl_id: String,
    evidence_sha256: String,
    min_integrated_speedup_x: f64,
}

#[derive(Debug, Clone)]
struct Policy {
    fields: BTreeMap<String, String>,
    decisions: Vec<Decision>,
}

pub fn materialize_for_candle_build(
    build_sm: u32,
) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").ok_or("OUT_DIR is missing")?);
    let output = out_dir.join("asd_exact_dispatch.rs");

    for name in [
        "CANDLE_ASD_EXACT_POLICY",
        "CANDLE_ASD_VALIDATION",
        "CANDLE_ASD_TARGET_GPU_UUID",
        "CANDLE_ASD_EXACT_DISABLE",
        "CANDLE_ASD_EXACT_TRACE",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    let Some(policy_path) = env::var_os("CANDLE_ASD_EXACT_POLICY") else {
        fs::write(&output, fallback_native())?;
        return Ok(None);
    };
    let policy_path = PathBuf::from(policy_path);
    println!("cargo:rerun-if-changed={}", policy_path.display());

    let policy = parse(&fs::read_to_string(&policy_path)?)?;
    validate(&policy, build_sm)?;
    let validation_mode = env_truthy("CANDLE_ASD_VALIDATION");
    let has_non_promoted = policy
        .decisions
        .iter()
        .any(|d| d.state != "promoted");
    if has_non_promoted && !validation_mode {
        return Err(
            "candidate/integrated ASD exact policy requires CANDLE_ASD_VALIDATION=1".into(),
        );
    }

    let generated = render_native(&policy, validation_mode)?;
    let mut file = fs::File::create(&output)?;
    file.write_all(generated.as_bytes())?;
    file.sync_all()?;

    println!(
        "cargo:warning=ASD exact policy {} decisions={} validation_mode={}",
        policy.fields["policy_id"],
        policy.decisions.len(),
        validation_mode
    );
    Ok(Some(output))
}

fn env_truthy(name: &str) -> bool {
    matches!(
        env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

fn parse(text: &str) -> Result<Policy, Box<dyn std::error::Error>> {
    let mut lines = text.lines();
    if lines.next() != Some(HEADER) {
        return Err("invalid ASD exact wire header".into());
    }

    let mut fields = BTreeMap::new();
    let mut decisions = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if line.starts_with("decision|") {
            let parts = line.split('|').collect::<Vec<_>>();
            if parts.len() != 8 {
                return Err("invalid ASD exact decision record".into());
            }
            let signature = parse_kv_list(parts[4], ',')?;
            let impl_id = value_after(parts[5], "impl=")?.to_owned();
            let evidence_sha256 = value_after(parts[6], "evidence=")?.to_owned();
            let min_integrated_speedup_x = value_after(parts[7], "min_integrated_speedup_x=")?
                .parse::<f64>()?;
            decisions.push(Decision {
                id: parts[1].to_owned(),
                state: parts[2].to_owned(),
                selected: parts[3].to_owned(),
                signature,
                impl_id,
                evidence_sha256,
                min_integrated_speedup_x,
            });
        } else {
            let (key, value) = line
                .split_once('=')
                .ok_or("invalid ASD exact policy field")?;
            if fields.insert(key.to_owned(), value.to_owned()).is_some() {
                return Err(format!("duplicate ASD exact policy field {key}").into());
            }
        }
    }
    Ok(Policy { fields, decisions })
}

fn parse_kv_list(
    input: &str,
    separator: char,
) -> Result<BTreeMap<String, String>, Box<dyn std::error::Error>> {
    let mut out = BTreeMap::new();
    for token in input.split(separator) {
        let (key, value) = token
            .split_once('=')
            .ok_or("invalid ASD exact signature field")?;
        if out.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(format!("duplicate ASD exact signature field {key}").into());
        }
    }
    Ok(out)
}

fn value_after<'a>(input: &'a str, prefix: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    input
        .strip_prefix(prefix)
        .ok_or_else(|| format!("invalid ASD exact field, expected {prefix}").into())
}

fn required<'a>(
    map: &'a BTreeMap<String, String>,
    key: &str,
) -> Result<&'a str, Box<dyn std::error::Error>> {
    map.get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("missing ASD exact field {key}").into())
}

fn parse_usize(
    map: &BTreeMap<String, String>,
    key: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(required(map, key)?.parse()?)
}

fn parse_dims(input: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let dims = input
        .split('x')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    if dims.is_empty() || dims.iter().any(|v| *v == 0) {
        return Err("invalid ASD exact dimensions".into());
    }
    Ok(dims)
}

fn validate(policy: &Policy, build_sm: u32) -> Result<(), Box<dyn std::error::Error>> {
    if required(&policy.fields, "target.vendor")? != "nvidia" {
        return Err("ASD exact Candle consumer supports NVIDIA only".into());
    }
    if required(&policy.fields, "target.sm")?.parse::<u32>()? != build_sm {
        return Err(format!(
            "ASD exact policy sm{} does not match Candle build sm{}",
            required(&policy.fields, "target.sm")?,
            build_sm
        )
        .into());
    }
    let _ = required(&policy.fields, "policy_id")?;
    let scope = required(&policy.fields, "target.scope")?;
    if scope == "device" {
        let expected = required(&policy.fields, "target.gpu_uuid")?;
        let actual = env::var("CANDLE_ASD_TARGET_GPU_UUID").map_err(|_| {
            "device-scoped ASD exact policy requires CANDLE_ASD_TARGET_GPU_UUID"
        })?;
        if actual != expected {
            return Err("ASD exact GPU UUID does not match build target".into());
        }
    } else if !["architecture", "device_class"].contains(&scope) {
        return Err("invalid ASD exact target scope".into());
    }

    if policy.decisions.is_empty() {
        return Err("ASD exact policy has no decisions".into());
    }
    let mut ids = BTreeSet::new();
    let mut signatures = BTreeSet::new();
    for d in &policy.decisions {
        if !ids.insert(d.id.clone()) {
            return Err("duplicate ASD exact decision id".into());
        }
        if !["tuner_candidate", "integrated_validated", "promoted"].contains(&d.state.as_str()) {
            return Err("invalid ASD exact decision state".into());
        }
        if d.selected != "raw_cuda" {
            return Err("ASD-v1-fix2 consumes raw_cuda exact decisions only".into());
        }
        if d.impl_id != RAW_IMPL_ID {
            return Err(format!(
                "ASD exact implementation id {} does not match {}",
                d.impl_id, RAW_IMPL_ID
            )
            .into());
        }
        if d.evidence_sha256.len() != 64
            || !d.evidence_sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("invalid ASD exact evidence SHA-256".into());
        }
        if !d.min_integrated_speedup_x.is_finite() || d.min_integrated_speedup_x <= 1.0 {
            return Err("invalid ASD exact integrated speedup contract".into());
        }
        validate_signature(&d.signature)?;
        let signature_key = d
            .signature
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");
        if !signatures.insert(signature_key) {
            return Err("conflicting duplicate ASD exact call signature".into());
        }
    }
    Ok(())
}

fn validate_signature(
    s: &BTreeMap<String, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let expected = [
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
    if s.len() != expected.len() || expected.iter().any(|key| !s.contains_key(*key)) {
        return Err("ASD exact signature fields differ from v1 contract".into());
    }
    let dim = parse_usize(s, "dim")?;
    if !(1..=2).contains(&dim) {
        return Err("ASD exact ConvTranspose dim must be 1 or 2".into());
    }
    let spatial = parse_dims(required(s, "spatial")?)?;
    let weight = parse_dims(required(s, "weight_shape")?)?;
    if spatial.len() != dim || weight.len() != dim + 2 {
        return Err("ASD exact rank mismatch".into());
    }
    let batch = parse_usize(s, "batch")?;
    let c_in = parse_usize(s, "c_in")?;
    let c_out = parse_usize(s, "c_out")?;
    let groups = parse_usize(s, "groups")?;
    let kernel = parse_usize(s, "kernel")?;
    let stride = parse_usize(s, "stride")?;
    let dilation = parse_usize(s, "dilation")?;
    if [batch, c_in, c_out, groups, kernel, stride, dilation]
        .into_iter()
        .any(|v| v == 0)
    {
        return Err("zero dimension in ASD exact signature".into());
    }
    if c_in % groups != 0 || c_out % groups != 0 {
        return Err("ASD exact channels are not divisible by groups".into());
    }
    if weight[0] != c_in
        || weight[1] != c_out / groups
        || weight[2..].iter().any(|v| *v != kernel)
    {
        return Err("ASD exact weight shape mismatch".into());
    }
    if required(s, "input_layout")? != "contiguous_zero_offset"
        || required(s, "weight_layout")? != "contiguous_zero_offset"
    {
        return Err("ASD-v1-fix2 supports contiguous zero-offset layouts only".into());
    }
    if required(s, "dtype")?.is_empty() {
        return Err("ASD exact dtype is empty".into());
    }
    Ok(())
}

fn bool_literal(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn render_native(
    policy: &Policy,
    validation_mode: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut decisions = String::new();
    for d in &policy.decisions {
        let s = &d.signature;
        let spatial = parse_dims(required(s, "spatial")?)?;
        let weight = parse_dims(required(s, "weight_shape")?)?;
        let spatial0 = spatial[0];
        let spatial1 = spatial.get(1).copied().unwrap_or(0);
        let weight0 = weight[0];
        let weight1 = weight[1];
        let weight2 = weight[2];
        let weight3 = weight.get(3).copied().unwrap_or(0);
        decisions.push_str(&format!(
            "    ExactDecision {{ id: {:?}, state: {:?}, selected_backend: {:?}, evidence_sha256: {:?}, min_integrated_speedup_x: {:.8}, dim: {}, batch: {}, c_in: {}, c_out: {}, spatial0: {}, spatial1: {}, weight_rank: {}, weight0: {}, weight1: {}, weight2: {}, weight3: {}, groups: {}, kernel: {}, stride: {}, padding: {}, output_padding: {}, dilation: {}, dtype: {:?} }},\n",
            d.id,
            d.state,
            d.selected,
            d.evidence_sha256,
            d.min_integrated_speedup_x,
            parse_usize(s, "dim")?,
            parse_usize(s, "batch")?,
            parse_usize(s, "c_in")?,
            parse_usize(s, "c_out")?,
            spatial0,
            spatial1,
            weight.len(),
            weight0,
            weight1,
            weight2,
            weight3,
            parse_usize(s, "groups")?,
            parse_usize(s, "kernel")?,
            parse_usize(s, "stride")?,
            parse_usize(s, "padding")?,
            parse_usize(s, "output_padding")?,
            parse_usize(s, "dilation")?,
            required(s, "dtype")?,
        ));
    }
    let policy_id = required(&policy.fields, "policy_id")?;
    Ok(format!(
        r#"// @generated by candle-kernels/asd_exact_build_adapter.rs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactConvTransposeCall {{
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
}}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactAsdMatch {{
    pub policy_id: &'static str,
    pub decision_id: &'static str,
    pub state: &'static str,
    pub selected_backend: &'static str,
    pub evidence_sha256: &'static str,
    pub min_integrated_speedup_x: f64,
}}

#[derive(Clone, Copy)]
struct ExactDecision {{
    id: &'static str,
    state: &'static str,
    selected_backend: &'static str,
    evidence_sha256: &'static str,
    min_integrated_speedup_x: f64,
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
    dtype: &'static str,
}}

pub const POLICY_ID: Option<&str> = Some({policy_id:?});
pub const VALIDATION_BUILD: bool = {validation_mode};
const DECISIONS: &[ExactDecision] = &[
{decisions}];

fn runtime_disabled() -> bool {{
    matches!(
        std::env::var("CANDLE_ASD_EXACT_DISABLE").ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}}

pub fn lookup(call: ExactConvTransposeCall) -> Option<ExactAsdMatch> {{
    if runtime_disabled()
        || !call.input_contiguous
        || call.input_start_offset != 0
        || !call.weight_contiguous
        || call.weight_start_offset != 0
    {{
        return None;
    }}
    for d in DECISIONS {{
        if d.dim == call.dim
            && d.batch == call.batch
            && d.c_in == call.c_in
            && d.c_out == call.c_out
            && d.spatial0 == call.spatial0
            && d.spatial1 == call.spatial1
            && d.weight_rank == call.weight_rank
            && d.weight0 == call.weight0
            && d.weight1 == call.weight1
            && d.weight2 == call.weight2
            && d.weight3 == call.weight3
            && d.groups == call.groups
            && d.kernel == call.kernel
            && d.stride == call.stride
            && d.padding == call.padding
            && d.output_padding == call.output_padding
            && d.dilation == call.dilation
            && d.dtype == call.dtype
        {{
            return Some(ExactAsdMatch {{
                policy_id: {policy_id:?},
                decision_id: d.id,
                state: d.state,
                selected_backend: d.selected_backend,
                evidence_sha256: d.evidence_sha256,
                min_integrated_speedup_x: d.min_integrated_speedup_x,
            }});
        }}
    }}
    None
}}
"#,
        policy_id = policy_id,
        validation_mode = bool_literal(validation_mode),
        decisions = decisions
    ))
}

fn fallback_native() -> &'static str {
    r#"// @generated fallback: no ASD exact policy
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactConvTransposeCall {
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
    pub policy_id: &'static str,
    pub decision_id: &'static str,
    pub state: &'static str,
    pub selected_backend: &'static str,
    pub evidence_sha256: &'static str,
    pub min_integrated_speedup_x: f64,
}

pub const POLICY_ID: Option<&str> = None;
pub const VALIDATION_BUILD: bool = false;

pub fn lookup(_call: ExactConvTransposeCall) -> Option<ExactAsdMatch> {
    None
}
"#
}

#[allow(dead_code)]
fn _assert_path(_path: &Path) {}
