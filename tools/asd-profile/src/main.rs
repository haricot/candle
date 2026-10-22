#[path = "../../../candle-kernels/src/asd_paths.rs"]
mod asd_paths;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const FALLBACK_SEED: &str = include_str!("../fallbacks.seed.v1.json");

#[derive(Debug)]
struct Decision {
    id: String,
    state: String,
    provider: String,
    signature: String,
    implementation: String,
    evidence: String,
    min_speedup: String,
}

#[derive(Debug)]
struct Profile {
    fields: BTreeMap<String, String>,
    decisions: Vec<Decision>,
}

#[derive(Debug, Deserialize)]
struct FallbackSeedStore {
    schema: String,
    fallbacks: Vec<FallbackInput>,
}

#[derive(Debug, Deserialize)]
struct FallbackInput {
    decision_id: String,
    rank: u32,
    provider: String,
    implementation_id: String,
    protocol: String,
    evidence_sha256: Vec<String>,
    required_cudnn_version_raw: Option<usize>,
    qualification: String,
    note: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct FallbackRecord {
    decision_id: String,
    decision_identity_sha256: String,
    rank: u32,
    provider: String,
    implementation_id: String,
    protocol: String,
    evidence_sha256: Vec<String>,
    required_cudnn_version_raw: Option<usize>,
    qualification: String,
    note: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct FallbackStore {
    schema: String,
    profile_id: String,
    target_gpu_uuid: String,
    fallbacks: Vec<FallbackRecord>,
}

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

fn signature_map(signature: &str) -> Result<BTreeMap<&str, &str>, String> {
    let mut fields = BTreeMap::new();
    for item in signature.split(',') {
        let (key, value) = item
            .split_once('=')
            .ok_or_else(|| format!("malformed signature field {item:?}"))?;
        if fields.insert(key, value).is_some() {
            return Err(format!("duplicate signature field {key:?}"));
        }
    }
    Ok(fields)
}

fn canonical_decision(decision: &Decision) -> Result<String, String> {
    let signature = signature_map(&decision.signature)?;
    let mut out = String::from("ASD-DECISION-V1\n");
    out.push_str(&format!("id={}\n", decision.id));
    out.push_str(&format!("state={}\n", decision.state));
    out.push_str(&format!("provider={}\n", decision.provider));
    for key in DECISION_SIGNATURE_KEYS {
        let value = signature
            .get(key)
            .ok_or_else(|| format!("decision {} missing signature field {key}", decision.id))?;
        out.push_str(&format!("{key}={value}\n"));
    }
    out.push_str(&format!("implementation_id={}\n", decision.implementation));
    out.push_str(&format!("evidence_sha256={}\n", decision.evidence));
    out.push_str("min_integrated_speedup_x=");
    if decision.min_speedup == "none" {
        out.push_str("none");
    } else {
        let value = decision
            .min_speedup
            .parse::<f64>()
            .map_err(|err| format!("invalid min speedup {:?}: {err}", decision.min_speedup))?;
        out.push_str(&value.to_string());
    }
    out.push('\n');
    Ok(out)
}

fn decision_identity_sha256(decision: &Decision) -> Result<String, String> {
    let digest = Sha256::digest(canonical_decision(decision)?.as_bytes());
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn parse_profile(source: &str) -> Result<Profile, String> {
    let mut lines = source.lines();
    let header = lines.next().ok_or("empty profile")?;
    if header != "ASD-EXACT-POLICY-V2" && header != "ASD-EXACT-PROFILE-V3" {
        return Err(format!("unsupported ASD profile header {header:?}"));
    }

    let mut fields = BTreeMap::new();
    let mut decisions = Vec::new();
    for raw in lines {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("decision|") {
            let parts = line.split('|').collect::<Vec<_>>();
            if parts.len() != 8 {
                return Err(format!("invalid decision field count in {line:?}"));
            }
            decisions.push(Decision {
                id: parts[1].to_owned(),
                state: parts[2].to_owned(),
                provider: parts[3].to_owned(),
                signature: parts[4].to_owned(),
                implementation: parts[5]
                    .strip_prefix("impl=")
                    .ok_or("decision missing impl=")?
                    .to_owned(),
                evidence: parts[6]
                    .strip_prefix("evidence=")
                    .ok_or("decision missing evidence=")?
                    .to_owned(),
                min_speedup: parts[7]
                    .strip_prefix("min_integrated_speedup_x=")
                    .ok_or("decision missing min_integrated_speedup_x=")?
                    .to_owned(),
            });
        } else {
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("invalid profile header line {line:?}"))?;
            fields.insert(key.to_owned(), value.to_owned());
        }
    }
    Ok(Profile { fields, decisions })
}

fn arg_value(flag: &str) -> Option<PathBuf> {
    let args = std::env::args().collect::<Vec<_>>();
    args.windows(2)
        .find_map(|pair| (pair[0] == flag).then(|| PathBuf::from(&pair[1])))
}

fn flag(name: &str) -> bool {
    std::env::args().any(|arg| arg == name)
}

fn decision_filter() -> Option<String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut skip = false;
    for arg in args {
        if skip {
            skip = false;
            continue;
        }
        if arg == "--profile" || arg == "--home" || arg == "--merge-fallback-fragment" {
            skip = true;
            continue;
        }
        if arg == "--layout"
            || arg == "--install-fallback-seed"
            || arg == "--merge-fallback-seed"
        {
            continue;
        }
        if !arg.starts_with('-') {
            return Some(arg);
        }
    }
    None
}

fn asd_home() -> Option<PathBuf> {
    arg_value("--home").or_else(asd_paths::user_data_home)
}

fn profile_path(home: Option<&Path>) -> Option<PathBuf> {
    arg_value("--profile")
        .or_else(|| std::env::var_os("CANDLE_ASD_EXACT_POLICY").map(PathBuf::from))
        .or_else(|| home.map(|root| root.join("profiles/current.asd")))
}

fn fallback_path(home: &Path) -> PathBuf {
    home.join("fallbacks/current.json")
}

fn profile_id(profile: &Profile) -> Result<&str, String> {
    profile
        .fields
        .get("profile_id")
        .or_else(|| profile.fields.get("policy_id"))
        .map(String::as_str)
        .ok_or_else(|| "profile missing profile_id/policy_id".to_owned())
}

fn target_gpu_uuid(profile: &Profile) -> Result<&str, String> {
    profile
        .fields
        .get("target.gpu_uuid")
        .map(String::as_str)
        .ok_or_else(|| "profile missing target.gpu_uuid".to_owned())
}

fn validate_fallback_input(fallback: &FallbackInput) -> Result<(), String> {
    if fallback.decision_id.is_empty()
        || fallback.rank == 0
        || !matches!(fallback.provider.as_str(), "cudnn" | "native")
        || fallback.implementation_id.is_empty()
        || fallback.protocol.is_empty()
        || fallback.qualification.is_empty()
        || fallback.evidence_sha256.is_empty()
        || fallback
            .evidence_sha256
            .iter()
            .any(|evidence| !is_sha256(evidence))
    {
        return Err(format!(
            "invalid fallback input for decision {:?}",
            fallback.decision_id
        ));
    }
    Ok(())
}

fn fallback_record_from_input(
    fallback: FallbackInput,
    profile: &Profile,
) -> Result<FallbackRecord, Box<dyn std::error::Error>> {
    validate_fallback_input(&fallback)?;
    let decision = profile
        .decisions
        .iter()
        .find(|decision| decision.id == fallback.decision_id)
        .ok_or_else(|| {
            format!(
                "fallback decision {} is not present in current profile",
                fallback.decision_id
            )
        })?;
    if decision.state != "promoted" {
        return Err(format!(
            "fallback decision {} is not promoted in current profile",
            fallback.decision_id
        )
        .into());
    }
    Ok(FallbackRecord {
        decision_id: fallback.decision_id,
        decision_identity_sha256: decision_identity_sha256(decision)?,
        rank: fallback.rank,
        provider: fallback.provider,
        implementation_id: fallback.implementation_id,
        protocol: fallback.protocol,
        evidence_sha256: fallback.evidence_sha256,
        required_cudnn_version_raw: fallback.required_cudnn_version_raw,
        qualification: fallback.qualification,
        note: fallback.note,
    })
}

fn fallback_store_from_seed(
    profile: &Profile,
) -> Result<FallbackStore, Box<dyn std::error::Error>> {
    let seed: FallbackSeedStore = serde_json::from_str(FALLBACK_SEED)?;
    if seed.schema != "ASD-FALLBACK-SEED-V1" {
        return Err(format!("unsupported fallback seed schema {:?}", seed.schema).into());
    }

    let mut fallbacks = Vec::with_capacity(seed.fallbacks.len());
    for fallback in seed.fallbacks {
        fallbacks.push(fallback_record_from_input(fallback, profile)?);
    }

    Ok(FallbackStore {
        schema: "ASD-FALLBACKS-V1".to_owned(),
        profile_id: profile_id(profile)?.to_owned(),
        target_gpu_uuid: target_gpu_uuid(profile)?.to_owned(),
        fallbacks,
    })
}

fn merge_fallback_record(store: &mut FallbackStore, record: FallbackRecord) -> bool {
    let key = (record.decision_id.clone(), record.rank);
    let replaced = if let Some(index) = store
        .fallbacks
        .iter()
        .position(|fallback| fallback.decision_id == key.0 && fallback.rank == key.1)
    {
        store.fallbacks[index] = record;
        true
    } else {
        store.fallbacks.push(record);
        false
    };
    store.fallbacks.sort_by(|left, right| {
        left.decision_id
            .cmp(&right.decision_id)
            .then(left.rank.cmp(&right.rank))
    });
    replaced
}

fn write_fallback_store_atomic(
    home: &Path,
    store: &FallbackStore,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = fallback_path(home);
    let parent = path
        .parent()
        .ok_or_else(|| format!("fallback store path {} has no parent", path.display()))?;
    std::fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(".current.json.tmp-{}", std::process::id()));
    let encoded = serde_json::to_string_pretty(store)? + "\n";
    std::fs::write(&tmp, encoded)?;
    if let Err(err) = std::fs::rename(&tmp, &path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!(
            "failed to atomically replace fallback store {}: {err}",
            path.display()
        )
        .into());
    }
    Ok(path)
}

fn install_fallback_seed(home: &Path, profile: &Profile) -> Result<(), Box<dyn std::error::Error>> {
    let store = fallback_store_from_seed(profile)?;
    let path = write_fallback_store_atomic(home, &store)?;
    println!("ASD_FALLBACKS_INSTALLED={}", path.display());
    println!("schema={}", store.schema);
    println!("profile_id={}", store.profile_id);
    println!("target_gpu_uuid={}", store.target_gpu_uuid);
    println!("fallbacks={}", store.fallbacks.len());
    Ok(())
}

fn merge_fallback_seed(home: &Path, profile: &Profile) -> Result<(), Box<dyn std::error::Error>> {
    let seed = fallback_store_from_seed(profile)?;
    let mut store = load_fallback_store(home, profile)?.unwrap_or(FallbackStore {
        schema: "ASD-FALLBACKS-V1".to_owned(),
        profile_id: profile_id(profile)?.to_owned(),
        target_gpu_uuid: target_gpu_uuid(profile)?.to_owned(),
        fallbacks: Vec::new(),
    });
    let mut replaced = 0usize;
    let mut inserted = 0usize;
    for record in seed.fallbacks {
        if merge_fallback_record(&mut store, record) {
            replaced += 1;
        } else {
            inserted += 1;
        }
    }
    let path = write_fallback_store_atomic(home, &store)?;
    println!("ASD_FALLBACK_SEED_MERGED={}", path.display());
    println!("inserted={inserted}");
    println!("replaced={replaced}");
    println!("fallbacks={}", store.fallbacks.len());
    Ok(())
}

fn merge_fallback_fragment(
    home: &Path,
    profile: &Profile,
    fragment_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(fragment_path).map_err(|err| {
        format!(
            "failed to read fallback fragment {}: {err}",
            fragment_path.display()
        )
    })?;
    let fragment: FallbackInput = serde_json::from_str(&source)?;
    let record = fallback_record_from_input(fragment, profile)?;
    let decision_id = record.decision_id.clone();
    let decision_identity_sha256 = record.decision_identity_sha256.clone();
    let rank = record.rank;

    // If the runtime store does not exist yet, bootstrap it from the frozen Git
    // seed once. Existing stores are always preserved and merged in-place.
    let mut store = match load_fallback_store(home, profile)? {
        Some(store) => store,
        None => fallback_store_from_seed(profile)?,
    };
    let replaced = merge_fallback_record(&mut store, record);
    let path = write_fallback_store_atomic(home, &store)?;
    println!("ASD_FALLBACK_MERGED={}", path.display());
    println!("fragment={}", fragment_path.display());
    println!("decision_id={decision_id}");
    println!("decision_identity_sha256={decision_identity_sha256}");
    println!("rank={rank}");
    println!("merge_action={}", if replaced { "replaced" } else { "inserted" });
    println!("fallbacks={}", store.fallbacks.len());
    Ok(())
}

fn load_fallback_store(
    home: &Path,
    profile: &Profile,
) -> Result<Option<FallbackStore>, Box<dyn std::error::Error>> {
    let path = fallback_path(home);
    if !path.is_file() {
        return Ok(None);
    }

    let source = std::fs::read_to_string(&path)?;
    let store: FallbackStore = serde_json::from_str(&source)?;
    if store.schema != "ASD-FALLBACKS-V1" {
        return Err(format!("unsupported fallback store schema {:?}", store.schema).into());
    }
    if store.profile_id != profile_id(profile)? {
        return Err(format!(
            "fallback store profile mismatch: local={} profile={}",
            store.profile_id,
            profile_id(profile)?
        )
        .into());
    }
    if store.target_gpu_uuid != target_gpu_uuid(profile)? {
        return Err(format!(
            "fallback store GPU mismatch: local={} profile={}",
            store.target_gpu_uuid,
            target_gpu_uuid(profile)?
        )
        .into());
    }

    let mut ranks = BTreeSet::new();
    for fallback in &store.fallbacks {
        if fallback.rank == 0
            || !is_sha256(&fallback.decision_identity_sha256)
            || fallback.evidence_sha256.is_empty()
            || fallback
                .evidence_sha256
                .iter()
                .any(|evidence| !is_sha256(evidence))
        {
            return Err(format!(
                "invalid fallback store entry for decision {}",
                fallback.decision_id
            )
            .into());
        }
        if !ranks.insert((
            fallback.decision_id.as_str(),
            fallback.decision_identity_sha256.as_str(),
            fallback.rank,
        )) {
            return Err(format!(
                "duplicate fallback rank {} for decision {}",
                fallback.rank, fallback.decision_id
            )
            .into());
        }
    }

    Ok(Some(store))
}

fn bool_str(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

fn print_layout(home: &Path, architecture: &str) {
    println!("ASD_USER_LAYOUT");
    println!("  home={}", home.display());
    println!("  current_profile={}", home.join("profiles/current.asd").display());
    println!("  current_fallbacks={}", home.join("fallbacks/current.json").display());
    println!("  current_history={}", home.join("history/current.json").display());
    println!(
        "  artifacts_dir={}",
        home.join("artifacts").join(architecture).display()
    );
    println!("  artifact_resolution=external_cubin>external_ptx>qualified_provider_fallback");
    println!("  fallback_resolution=cold_control_plane>cached_decision_identity_lookup");
    println!("  execution_resolution=cold_control_plane>per_device_resolved_function_cache>launch");
    println!("  hot_path_filesystem_io=false");
}

fn print_artifact_state(home: &Path, architecture: &str, decision: &Decision) {
    if decision.provider != "raw_cuda" {
        println!("  artifact_model=provider_managed");
        return;
    }

    let external_prefix = format!("candle.{architecture}-exact-grouped.");
    if !decision.implementation.starts_with(&external_prefix) {
        println!("  artifact_model=legacy_embedded_raw");
        return;
    }

    let root = home.join("artifacts").join(architecture);
    let artifact = |extension: &str| {
        root.join(format!("{}.{}", decision.implementation, extension))
    };
    let cu = artifact("cu");
    let cubin = artifact("cubin");
    let ptx = artifact("ptx");
    let manifest = artifact("manifest");

    println!("  artifact_model=asd_v3_externalizable_raw");
    println!("  source_cu={} available={}", cu.display(), bool_str(cu.is_file()));
    println!(
        "  external_cubin={} available={}",
        cubin.display(),
        bool_str(cubin.is_file())
    );
    println!(
        "  external_ptx={} available={}",
        ptx.display(),
        bool_str(ptx.is_file())
    );
    println!(
        "  manifest={} available={}",
        manifest.display(),
        bool_str(manifest.is_file())
    );
    let selected = if cubin.is_file() {
        "external_cubin"
    } else if ptx.is_file() {
        "external_ptx"
    } else {
        "unavailable"
    };
    println!("  builtin_raw=absent_phase_b");
    println!("  phase_b_primary_resolution={selected}");
}

fn print_decision(
    home: &Path,
    architecture: &str,
    decision: &Decision,
    fallback_store: Option<&FallbackStore>,
) -> Result<(), String> {
    println!();
    println!("DECISION id={} state={}", decision.id, decision.state);
    let identity = decision_identity_sha256(decision)?;
    println!("  decision_identity_schema=ASD-DECISION-V1");
    println!("  decision_identity_sha256={identity}");
    println!("  PRIMARY provider={}", decision.provider);
    println!("  implementation={}", decision.implementation);
    println!("  evidence_sha256={}", decision.evidence);
    println!("  min_integrated_speedup_x={}", decision.min_speedup);
    println!("  exact_signature={}", decision.signature);
    print_artifact_state(home, architecture, decision);

    let mut fallbacks = fallback_store
        .map(|store| {
            store
                .fallbacks
                .iter()
                .filter(|fallback| fallback.decision_id == decision.id)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if fallbacks
        .iter()
        .any(|fallback| fallback.decision_identity_sha256 != identity)
    {
        return Err(format!(
            "fallback decision identity mismatch for {}; reinstall/sync fallbacks/current.json",
            decision.id
        ));
    }
    fallbacks.sort_by_key(|fallback| fallback.rank);

    println!(
        "  fallback_store={}",
        if fallback_store.is_some() { "local" } else { "missing" }
    );
    println!("  qualified_fallbacks={}", fallbacks.len());
    for fallback in fallbacks {
        println!(
            "  FALLBACK rank={} provider={} implementation={} protocol={} qualification={}",
            fallback.rank,
            fallback.provider,
            fallback.implementation_id,
            fallback.protocol,
            fallback.qualification,
        );
        println!(
            "    decision_identity_sha256={}",
            fallback.decision_identity_sha256
        );
        println!(
            "    required_cudnn_version_raw={}",
            fallback
                .required_cudnn_version_raw
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_owned())
        );
        println!("    evidence_sha256={}", fallback.evidence_sha256.join(","));
        println!("    note={}", fallback.note);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = asd_home().ok_or("unable to resolve ASD user data home")?;
    let path = profile_path(Some(&home)).ok_or("unable to resolve ASD profile path")?;
    let source = std::fs::read_to_string(&path).map_err(|err| {
        format!(
            "failed to read ASD Exact Profile {}: {err}; pass --profile PATH or install profiles/current.asd under the ASD home",
            path.display()
        )
    })?;
    let profile = parse_profile(&source)?;
    let profile_id = profile_id(&profile)?;
    let architecture = profile
        .fields
        .get("target.architecture")
        .map(String::as_str)
        .unwrap_or("unknown");

    if let Some(fragment_path) = arg_value("--merge-fallback-fragment") {
        merge_fallback_fragment(&home, &profile, &fragment_path)?;
        return Ok(());
    }

    if flag("--merge-fallback-seed") {
        merge_fallback_seed(&home, &profile)?;
        return Ok(());
    }

    if flag("--install-fallback-seed") {
        install_fallback_seed(&home, &profile)?;
        return Ok(());
    }

    if flag("--layout") {
        print_layout(&home, architecture);
        return Ok(());
    }

    let fallback_store = load_fallback_store(&home, &profile)?;
    let promoted = profile
        .decisions
        .iter()
        .filter(|decision| decision.state == "promoted")
        .count();

    let mut providers = BTreeMap::<&str, usize>::new();
    for decision in profile
        .decisions
        .iter()
        .filter(|decision| decision.state == "promoted")
    {
        *providers.entry(&decision.provider).or_default() += 1;
    }

    println!("=== ASD EXACT PROFILE CURRENT STATE ===");
    println!("dispatch_authority=profile");
    println!("phase=phase_a");
    println!("asd_home={}", home.display());
    println!("profile_file={}", path.display());
    println!("profile_id={profile_id}");
    println!("architecture={architecture}");
    println!("target_gpu_uuid={}", target_gpu_uuid(&profile)?);
    println!(
        "fallback_store_file={} available={}",
        fallback_path(&home).display(),
        bool_str(fallback_store.is_some())
    );
    println!("promoted={promoted}");
    for (provider, count) in providers {
        println!("provider.{provider}={count}");
    }
    print_layout(&home, architecture);

    let filter = decision_filter();
    let mut shown = 0usize;
    for decision in profile
        .decisions
        .iter()
        .filter(|decision| decision.state == "promoted")
    {
        if filter
            .as_deref()
            .is_some_and(|wanted| wanted != decision.id)
        {
            continue;
        }
        print_decision(&home, architecture, decision, fallback_store.as_ref())?;
        shown += 1;
    }
    if filter.is_some() && shown == 0 {
        return Err(format!(
            "decision {:?} not found among promoted profile rows",
            filter.unwrap()
        )
        .into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_v2_promoted_decision() {
        let source = "ASD-EXACT-POLICY-V2\n\
policy_id=test-profile\n\
target.vendor=nvidia\n\
target.architecture=sm61\n\
target.scope=device\n\
target.sm=61\n\
target.gpu_uuid=GPU-test\n\
decision|ct1d-sm61-s32-g2-raw-exact|promoted|raw_cuda|op=conv_transpose1d,dim=1,batch=1,c_in=128,c_out=128,spatial=32,weight_shape=128x64x3,groups=2,kernel=3,stride=2,padding=1,output_padding=1,dilation=1,dtype=f32,input_layout=contiguous_zero_offset,weight_layout=contiguous_zero_offset|impl=candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256|evidence=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa|min_integrated_speedup_x=none\n";
        let profile = parse_profile(source).unwrap();
        assert_eq!(profile.decisions.len(), 1);
        assert_eq!(profile.decisions[0].state, "promoted");
        assert_eq!(profile.decisions[0].provider, "raw_cuda");
        assert_eq!(
            decision_identity_sha256(&profile.decisions[0]).unwrap(),
            "c2e29bb308d621be1059ebba47d50ea574f7101a3aa2ff2498543af464ca510f"
        );
        assert_eq!(
            profile.decisions[0].implementation,
            "candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256"
        );
    }

    #[test]
    fn seed_schema_is_valid() {
        let seed: FallbackSeedStore = serde_json::from_str(FALLBACK_SEED).unwrap();
        assert_eq!(seed.schema, "ASD-FALLBACK-SEED-V1");
        assert!(!seed.fallbacks.is_empty());
    }

    #[test]
    fn merge_record_replaces_same_decision_rank() {
        let record = |evidence: &str| FallbackRecord {
            decision_id: "test".to_owned(),
            decision_identity_sha256: "a".repeat(64),
            rank: 1,
            provider: "cudnn".to_owned(),
            implementation_id: "impl".to_owned(),
            protocol: "provider-evidence-v2".to_owned(),
            evidence_sha256: vec![evidence.to_owned()],
            required_cudnn_version_raw: Some(91002),
            qualification: "exact_parity+stable_execution".to_owned(),
            note: "test".to_owned(),
        };
        let mut store = FallbackStore {
            schema: "ASD-FALLBACKS-V1".to_owned(),
            profile_id: "profile".to_owned(),
            target_gpu_uuid: "GPU-test".to_owned(),
            fallbacks: vec![record(&"b".repeat(64))],
        };
        assert!(merge_fallback_record(&mut store, record(&"c".repeat(64))));
        assert_eq!(store.fallbacks.len(), 1);
        assert_eq!(store.fallbacks[0].evidence_sha256[0], "c".repeat(64));
    }

    #[test]
    fn rejects_unknown_profile_header() {
        assert!(parse_profile("ASD-UNKNOWN\n").is_err());
    }
}
