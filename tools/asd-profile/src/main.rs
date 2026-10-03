#[path = "../../../candle-kernels/src/asd_fallback.rs"]
mod asd_fallback;
#[path = "../../../candle-kernels/src/asd_paths.rs"]
mod asd_paths;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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

fn decision_filter() -> Option<String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut skip = false;
    for arg in args {
        if skip {
            skip = false;
            continue;
        }
        if arg == "--profile" || arg == "--home" {
            skip = true;
            continue;
        }
        if arg == "--layout" {
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
    println!(
        "  artifacts_dir={}",
        home.join("artifacts").join(architecture).display()
    );
    println!("  artifact_resolution=external_cubin>external_ptx>builtin_raw>qualified_provider_fallback");
}

fn print_artifact_state(home: &Path, architecture: &str, decision: &Decision) {
    if decision.provider != "raw_cuda"
        || !decision
            .implementation
            .starts_with("candle.sm61-exact-grouped.")
    {
        println!("  artifact_model=legacy_embedded_executor");
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
    let builtin_disabled = env_truthy("CANDLE_ASD_BUILTIN_RAW_DISABLE");
    let selected = if cubin.is_file() {
        "external_cubin"
    } else if ptx.is_file() {
        "external_ptx"
    } else if !builtin_disabled {
        "builtin_raw"
    } else {
        "unavailable"
    };
    println!(
        "  builtin_raw={}",
        if builtin_disabled { "disabled" } else { "enabled" }
    );
    println!("  phase_a_primary_resolution={selected}");
}

fn print_decision(home: &Path, architecture: &str, decision: &Decision) {
    println!();
    println!(
        "DECISION id={} state={}",
        decision.id, decision.state
    );
    println!("  PRIMARY provider={}", decision.provider);
    println!("  implementation={}", decision.implementation);
    println!("  evidence_sha256={}", decision.evidence);
    println!("  min_integrated_speedup_x={}", decision.min_speedup);
    println!("  exact_signature={}", decision.signature);
    print_artifact_state(home, architecture, decision);

    let fallbacks = asd_fallback::qualified_fallbacks_for_decision(&decision.id);
    println!("  qualified_fallbacks={}", fallbacks.len());
    for fallback in fallbacks {
        println!(
            "  FALLBACK rank={} provider={} implementation={} protocol={} qualification={}",
            fallback.rank,
            fallback.provider.as_str(),
            fallback.implementation_id,
            fallback.protocol,
            fallback.qualification,
        );
        println!(
            "    required_cudnn_version_raw={}",
            fallback
                .required_cudnn_version_raw
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_owned())
        );
        println!(
            "    evidence_sha256={}",
            fallback.evidence_sha256.join(",")
        );
        println!("    note={}", fallback.note);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = asd_home().ok_or("unable to resolve ASD user data home")?;
    let path = profile_path(Some(&home)).ok_or("unable to resolve ASD profile path")?;

    if std::env::args().any(|arg| arg == "--layout") {
        print_layout(&home, "sm61");
        return Ok(());
    }

    let source = std::fs::read_to_string(&path).map_err(|err| {
        format!(
            "failed to read ASD Exact Profile {}: {err}; pass --profile PATH or install profiles/current.asd under the ASD home",
            path.display()
        )
    })?;
    let profile = parse_profile(&source)?;
    let profile_id = profile
        .fields
        .get("profile_id")
        .or_else(|| profile.fields.get("policy_id"))
        .map(String::as_str)
        .unwrap_or("unknown");
    let architecture = profile
        .fields
        .get("target.architecture")
        .map(String::as_str)
        .unwrap_or("unknown");
    let promoted = profile
        .decisions
        .iter()
        .filter(|decision| decision.state == "promoted")
        .count();

    let mut providers = BTreeMap::<&str, usize>::new();
    for decision in profile.decisions.iter().filter(|decision| decision.state == "promoted") {
        *providers.entry(&decision.provider).or_default() += 1;
    }

    println!("=== ASD EXACT PROFILE CURRENT STATE ===");
    println!("dispatch_authority=profile");
    println!("phase=phase_a");
    println!("asd_home={}", home.display());
    println!("profile_file={}", path.display());
    println!("profile_id={profile_id}");
    println!("architecture={architecture}");
    println!("target_gpu_uuid={}", profile.fields.get("target.gpu_uuid").map(String::as_str).unwrap_or("unknown"));
    println!("promoted={promoted}");
    for (provider, count) in providers {
        println!("provider.{provider}={count}");
    }
    print_layout(&home, architecture);

    let filter = decision_filter();
    let mut shown = 0usize;
    for decision in profile.decisions.iter().filter(|decision| decision.state == "promoted") {
        if filter
            .as_deref()
            .is_some_and(|wanted| wanted != decision.id)
        {
            continue;
        }
        print_decision(&home, architecture, decision);
        shown += 1;
    }
    if filter.is_some() && shown == 0 {
        return Err(format!("decision {:?} not found among promoted profile rows", filter.unwrap()).into());
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
            profile.decisions[0].implementation,
            "candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256"
        );
    }

    #[test]
    fn rejects_unknown_profile_header() {
        assert!(parse_profile("ASD-UNKNOWN\n").is_err());
    }
}
