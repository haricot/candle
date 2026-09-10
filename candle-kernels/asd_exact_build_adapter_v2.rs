//! Dependency-free build adapter for ASD exact-domain V2 policies.
//!
//! V2 is intentionally isolated from the existing V1 adapter. The first
//! supported V2 consumer is the exact F32 NCHW depthwise Conv2D 5x5 raw CUDA
//! implementation used by the validation-only DW5x5 lane.
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::Write,
    path::PathBuf,
};

pub const HEADER: &str = "ASD-EXACT-POLICY-V2";
const RAW_DW5X5_IMPL_ID: &str = "candle.depthwise-conv2d-5x5.raw.v1";

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

    let policy_path = PathBuf::from(
        env::var_os("CANDLE_ASD_EXACT_POLICY")
            .ok_or("ASD exact V2 adapter requires CANDLE_ASD_EXACT_POLICY")?,
    );
    println!("cargo:rerun-if-changed={}", policy_path.display());

    let policy = parse(&fs::read_to_string(&policy_path)?)?;
    validate(&policy, build_sm)?;
    let validation_mode = env_truthy("CANDLE_ASD_VALIDATION");
    let has_non_promoted = policy.decisions.iter().any(|d| d.state != "promoted");
    if has_non_promoted && !validation_mode {
        return Err(
            "candidate/integrated ASD exact V2 policy requires CANDLE_ASD_VALIDATION=1".into(),
        );
    }

    let generated = render_native(&policy, validation_mode)?;
    let mut file = fs::File::create(&output)?;
    file.write_all(generated.as_bytes())?;
    file.sync_all()?;

    println!(
        "cargo:warning=ASD exact V2 policy {} decisions={} validation_mode={}",
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
        return Err("invalid ASD exact V2 wire header".into());
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
                return Err("invalid ASD exact V2 decision record".into());
            }
            let signature = parse_kv_list(parts[4], ',')?;
            let impl_id = value_after(parts[5], "impl=")?.to_owned();
            let evidence_sha256 = value_after(parts[6], "evidence=")?.to_owned();
            let min_integrated_speedup_x =
                value_after(parts[7], "min_integrated_speedup_x=")?.parse::<f64>()?;
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
                .ok_or("invalid ASD exact V2 policy field")?;
            if fields.insert(key.to_owned(), value.to_owned()).is_some() {
                return Err(format!("duplicate ASD exact V2 policy field {key}").into());
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
            .ok_or("invalid ASD exact V2 signature field")?;
        if out.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(format!("duplicate ASD exact V2 signature field {key}").into());
        }
    }
    Ok(out)
}

fn value_after<'a>(input: &'a str, prefix: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    input
        .strip_prefix(prefix)
        .ok_or_else(|| format!("invalid ASD exact V2 field, expected {prefix}").into())
}

fn required<'a>(
    map: &'a BTreeMap<String, String>,
    key: &str,
) -> Result<&'a str, Box<dyn std::error::Error>> {
    map.get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("missing ASD exact V2 field {key}").into())
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
    if dims.is_empty() || dims.contains(&0) {
        return Err("invalid ASD exact V2 dimensions".into());
    }
    Ok(dims)
}

fn validate(policy: &Policy, build_sm: u32) -> Result<(), Box<dyn std::error::Error>> {
    if required(&policy.fields, "target.vendor")? != "nvidia" {
        return Err("ASD exact V2 Candle consumer supports NVIDIA only".into());
    }
    if required(&policy.fields, "target.sm")?.parse::<u32>()? != build_sm {
        return Err(format!(
            "ASD exact V2 policy sm{} does not match Candle build sm{}",
            required(&policy.fields, "target.sm")?,
            build_sm
        )
        .into());
    }
    let _ = required(&policy.fields, "policy_id")?;
    let scope = required(&policy.fields, "target.scope")?;
    if scope == "device" {
        let expected = required(&policy.fields, "target.gpu_uuid")?;
        let actual = env::var("CANDLE_ASD_TARGET_GPU_UUID")
            .map_err(|_| "device-scoped ASD exact V2 policy requires CANDLE_ASD_TARGET_GPU_UUID")?;
        if actual != expected {
            return Err("ASD exact V2 GPU UUID does not match build target".into());
        }
    } else if !["architecture", "device_class"].contains(&scope) {
        return Err("invalid ASD exact V2 target scope".into());
    }

    if policy.decisions.is_empty() {
        return Err("ASD exact V2 policy has no decisions".into());
    }
    let mut ids = BTreeSet::new();
    let mut signatures = BTreeSet::new();
    for d in &policy.decisions {
        if !ids.insert(d.id.clone()) {
            return Err("duplicate ASD exact V2 decision id".into());
        }
        if !["tuner_candidate", "integrated_validated", "promoted"].contains(&d.state.as_str()) {
            return Err("invalid ASD exact V2 decision state".into());
        }
        if d.selected != "raw_cuda" {
            return Err("DW5x5 ASD V2 consumer accepts raw_cuda decisions only".into());
        }
        if d.impl_id != RAW_DW5X5_IMPL_ID {
            return Err(format!(
                "ASD exact V2 implementation id {} does not match {}",
                d.impl_id, RAW_DW5X5_IMPL_ID
            )
            .into());
        }
        if d.evidence_sha256.len() != 64
            || !d.evidence_sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("invalid ASD exact V2 evidence SHA-256".into());
        }
        if !d.min_integrated_speedup_x.is_finite() || d.min_integrated_speedup_x <= 1.0 {
            return Err("invalid ASD exact V2 integrated speedup contract".into());
        }
        validate_signature(&d.signature)?;
        let signature_key = d
            .signature
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");
        if !signatures.insert(signature_key) {
            return Err("conflicting duplicate ASD exact V2 call signature".into());
        }
    }
    Ok(())
}

fn validate_signature(s: &BTreeMap<String, String>) -> Result<(), Box<dyn std::error::Error>> {
    let expected = [
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
    if s.len() != expected.len() || expected.iter().any(|key| !s.contains_key(*key)) {
        return Err("ASD exact V2 signature fields differ from the Conv2D contract".into());
    }
    if required(s, "op")? != "conv2d" || parse_usize(s, "dim")? != 2 {
        return Err("DW5x5 ASD V2 consumer requires op=conv2d,dim=2".into());
    }
    if required(s, "output_padding")? != "none" {
        return Err("Conv2D ASD V2 output_padding must be none".into());
    }

    let spatial = parse_dims(required(s, "spatial")?)?;
    let weight = parse_dims(required(s, "weight_shape")?)?;
    if spatial.len() != 2 || weight.len() != 4 {
        return Err("DW5x5 ASD V2 rank mismatch".into());
    }
    let batch = parse_usize(s, "batch")?;
    let c_in = parse_usize(s, "c_in")?;
    let c_out = parse_usize(s, "c_out")?;
    let groups = parse_usize(s, "groups")?;
    let kernel = parse_usize(s, "kernel")?;
    let stride = parse_usize(s, "stride")?;
    let padding = parse_usize(s, "padding")?;
    let dilation = parse_usize(s, "dilation")?;
    if batch != 1
        || c_in == 0
        || c_out != c_in
        || groups != c_in
        || kernel != 5
        || stride != 1
        || padding != 2
        || dilation != 1
    {
        return Err("signature is outside exact DW5x5 F32 B1 depthwise domain".into());
    }
    if weight != [c_out, 1, 5, 5] {
        return Err("DW5x5 ASD V2 weight shape mismatch".into());
    }
    if required(s, "dtype")? != "f32" {
        return Err("DW5x5 ASD V2 consumer requires f32".into());
    }
    if required(s, "input_layout")? != "contiguous_zero_offset"
        || required(s, "weight_layout")? != "contiguous_zero_offset"
    {
        return Err("DW5x5 ASD V2 supports contiguous zero-offset layouts only".into());
    }
    Ok(())
}

fn bool_literal(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
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
        decisions.push_str(&format!(
            "    ExactConv2dDecision {{ id: {:?}, state: {:?}, selected_backend: {:?}, evidence_sha256: {:?}, min_integrated_speedup_x: {:.8}, batch: {}, c_in: {}, c_out: {}, spatial0: {}, spatial1: {}, weight0: {}, weight1: {}, weight2: {}, weight3: {}, groups: {}, kernel: {}, stride: {}, padding: {}, dilation: {}, dtype: {:?} }},\n",
            d.id,
            d.state,
            d.selected,
            d.evidence_sha256,
            d.min_integrated_speedup_x,
            parse_usize(s, "batch")?,
            parse_usize(s, "c_in")?,
            parse_usize(s, "c_out")?,
            spatial[0],
            spatial[1],
            weight[0],
            weight[1],
            weight[2],
            weight[3],
            parse_usize(s, "groups")?,
            parse_usize(s, "kernel")?,
            parse_usize(s, "stride")?,
            parse_usize(s, "padding")?,
            parse_usize(s, "dilation")?,
            required(s, "dtype")?,
        ));
    }
    let policy_id = required(&policy.fields, "policy_id")?;
    Ok(format!(
        r#"// @generated by candle-kernels/asd_exact_build_adapter_v2.rs
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactConv2dCall {{
    pub batch: usize,
    pub c_in: usize,
    pub c_out: usize,
    pub spatial0: usize,
    pub spatial1: usize,
    pub weight0: usize,
    pub weight1: usize,
    pub weight2: usize,
    pub weight3: usize,
    pub groups: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub dtype: &'static str,
    pub input_contiguous: bool,
    pub input_start_offset: usize,
    pub weight_contiguous: bool,
    pub weight_start_offset: usize,
}}

#[derive(Clone, Copy)]
struct ExactConv2dDecision {{
    id: &'static str,
    state: &'static str,
    selected_backend: &'static str,
    evidence_sha256: &'static str,
    min_integrated_speedup_x: f64,
    batch: usize,
    c_in: usize,
    c_out: usize,
    spatial0: usize,
    spatial1: usize,
    weight0: usize,
    weight1: usize,
    weight2: usize,
    weight3: usize,
    groups: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    dtype: &'static str,
}}

pub const POLICY_ID: Option<&str> = Some({policy_id:?});
pub const VALIDATION_BUILD: bool = {validation_mode};
const CONV2D_DECISIONS: &[ExactConv2dDecision] = &[
{decisions}];

fn runtime_disabled() -> bool {{
    matches!(
        std::env::var("CANDLE_ASD_EXACT_DISABLE").ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}}

// V2 Conv2D policies do not layer over V1 ConvTranspose policies. Keeping this
// V1-compatible symbol as an exact miss preserves the existing consumer ABI
// while this candidate is validation-only.
pub fn lookup(_call: ExactConvTransposeCall) -> Option<ExactAsdMatch> {{
    None
}}

pub fn lookup_conv2d(call: ExactConv2dCall) -> Option<ExactAsdMatch> {{
    if runtime_disabled()
        || !call.input_contiguous
        || call.input_start_offset != 0
        || !call.weight_contiguous
        || call.weight_start_offset != 0
    {{
        return None;
    }}
    for d in CONV2D_DECISIONS {{
        if d.batch == call.batch
            && d.c_in == call.c_in
            && d.c_out == call.c_out
            && d.spatial0 == call.spatial0
            && d.spatial1 == call.spatial1
            && d.weight0 == call.weight0
            && d.weight1 == call.weight1
            && d.weight2 == call.weight2
            && d.weight3 == call.weight3
            && d.groups == call.groups
            && d.kernel == call.kernel
            && d.stride == call.stride
            && d.padding == call.padding
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
