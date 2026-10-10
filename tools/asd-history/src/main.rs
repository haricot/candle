use serde::Deserialize;
use std::path::{Path, PathBuf};

const SEED: &str = include_str!("../history.v1.json");

#[derive(Debug, Deserialize)]
struct HistoryData {
    schema: String,
    gains: Vec<HistoricalGain>,
    provider_challenges: Vec<ProviderChallenge>,
}

#[derive(Debug, Deserialize)]
struct HistoricalGain {
    transition_id: String,
    decision_id: String,
    state: String,
    generation: u32,
    predecessor_transition_id: Option<String>,
    exact_signature: String,
    from_provider: String,
    from_implementation: String,
    to_provider: String,
    to_implementation: String,
    speedup_x: f64,
    earlier_observed_speedup_x: Option<f64>,
    source_commit: String,
    source_artifact: String,
    implementation_commit: Option<String>,
    benchmark_commit: Option<String>,
    evidence_sha256: Option<String>,
    note: String,
}

#[derive(Debug, Deserialize)]
struct ProviderChallengeReplication {
    replication_id: String,
    evidence_sha256: String,
    incumbent_median_us: f64,
    challenger_median_us: f64,
    incumbent_drift_pct: f64,
    challenger_drift_pct: f64,
    incumbent_p90_us: f64,
    challenger_p90_us: f64,
    decision: String,
}

#[derive(Debug, Deserialize)]
struct ProviderChallenge {
    challenge_id: String,
    decision_id: String,
    protocol: String,
    measurement_plane: String,
    harness_revision: String,
    exact_signature: String,
    incumbent_provider: String,
    incumbent_implementation: String,
    incumbent_identity: String,
    incumbent_profile_evidence_sha256: String,
    challenger_provider: String,
    challenger_implementation: String,
    challenger_identity: String,
    replications: Vec<ProviderChallengeReplication>,
    consensus: String,
    incumbent_retained: bool,
    profile_change: bool,
    note: String,
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
        if arg == "--source" || arg == "--home" {
            skip = true;
            continue;
        }
        if arg == "--install-seed" {
            continue;
        }
        if !arg.starts_with('-') {
            return Some(arg);
        }
    }
    None
}

fn asd_home() -> Option<PathBuf> {
    if let Some(root) = arg_value("--home") {
        return Some(root);
    }
    if let Some(root) = std::env::var_os("CANDLE_ASD_HOME").filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(root));
    }
    if let Some(root) = std::env::var_os("XDG_DATA_HOME").filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(root).join("asd"));
    }
    std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(|home| PathBuf::from(home).join(".local/share/asd"))
}

fn current_history_path(home: &Path) -> PathBuf {
    home.join("history/current.json")
}

fn parse(source: &str) -> Result<HistoryData, Box<dyn std::error::Error>> {
    let history: HistoryData = serde_json::from_str(source)?;
    if history.schema != "ASD-HISTORY-V1" {
        return Err(format!("unsupported ASD history schema {:?}", history.schema).into());
    }
    Ok(history)
}

fn print_gain(gain: &HistoricalGain) {
    println!(
        "HISTORICAL_STATE transition={} decision={} state={} generation={} speedup_x={:.6}",
        gain.transition_id, gain.decision_id, gain.state, gain.generation, gain.speedup_x,
    );
    println!(
        "  predecessor_transition={}",
        gain.predecessor_transition_id.as_deref().unwrap_or("none")
    );
    println!(
        "  from provider={} implementation={}",
        gain.from_provider, gain.from_implementation
    );
    println!(
        "  to   provider={} implementation={}",
        gain.to_provider, gain.to_implementation
    );
    if let Some(value) = gain.earlier_observed_speedup_x {
        println!("  earlier_observed_speedup_x={value:.6}");
    }
    println!("  source_commit={}", gain.source_commit);
    println!("  source_artifact={}", gain.source_artifact);
    if let Some(commit) = gain.implementation_commit.as_deref() {
        println!("  implementation_commit={commit}");
    }
    if let Some(commit) = gain.benchmark_commit.as_deref() {
        println!("  benchmark_commit={commit}");
    }
    if let Some(evidence) = gain.evidence_sha256.as_deref() {
        println!("  evidence_sha256={evidence}");
    }
    println!("  exact_signature={}", gain.exact_signature);
    println!("  note={}", gain.note);
}

fn print_provider_challenge(challenge: &ProviderChallenge) {
    println!(
        "PROVIDER_CHALLENGE challenge={} decision={} protocol={} consensus={} measurement_plane={}",
        challenge.challenge_id,
        challenge.decision_id,
        challenge.protocol,
        challenge.consensus,
        challenge.measurement_plane,
    );
    println!(
        "  incumbent provider={} implementation={}",
        challenge.incumbent_provider, challenge.incumbent_implementation
    );
    println!("  incumbent_identity={}", challenge.incumbent_identity);
    println!(
        "  incumbent_profile_evidence_sha256={}",
        challenge.incumbent_profile_evidence_sha256
    );
    println!(
        "  challenger provider={} implementation={}",
        challenge.challenger_provider, challenge.challenger_implementation
    );
    println!("  challenger_identity={}", challenge.challenger_identity);
    println!("  harness_revision={}", challenge.harness_revision);
    println!("  incumbent_retained={}", challenge.incumbent_retained);
    println!("  profile_change={}", challenge.profile_change);
    println!("  replications={}", challenge.replications.len());
    for replication in &challenge.replications {
        let incumbent_advantage_x =
            replication.challenger_median_us / replication.incumbent_median_us;
        println!(
            "  REPLICATION id={} evidence_sha256={} decision={} incumbent_median_us={:.6} challenger_median_us={:.6} incumbent_advantage_x={:.6}",
            replication.replication_id,
            replication.evidence_sha256,
            replication.decision,
            replication.incumbent_median_us,
            replication.challenger_median_us,
            incumbent_advantage_x,
        );
        println!(
            "    incumbent_drift_pct={:.3} challenger_drift_pct={:.3} incumbent_p90_us={:.6} challenger_p90_us={:.6}",
            replication.incumbent_drift_pct,
            replication.challenger_drift_pct,
            replication.incumbent_p90_us,
            replication.challenger_p90_us,
        );
    }
    println!("  exact_signature={}", challenge.exact_signature);
    println!("  note={}", challenge.note);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = asd_home().ok_or("unable to resolve ASD home")?;
    let local_path = current_history_path(&home);

    if flag("--install-seed") {
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&local_path, SEED)?;
        println!("ASD_HISTORY_INSTALLED={}", local_path.display());
        println!("schema=ASD-HISTORY-V1");
        return Ok(());
    }

    let explicit = arg_value("--source");
    let (source, source_kind, source_path) = if let Some(path) = explicit {
        (std::fs::read_to_string(&path)?, "explicit", Some(path))
    } else if local_path.is_file() {
        (
            std::fs::read_to_string(&local_path)?,
            "local",
            Some(local_path.clone()),
        )
    } else {
        (SEED.to_owned(), "repo_seed", None)
    };

    let history = parse(&source)?;
    let decision = decision_filter();

    println!("=== ASD EXACT PROFILE HISTORICAL STATE ===");
    println!("dispatch_authority=false");
    println!("historical_gain_composition=forbidden");
    println!("history_schema={}", history.schema);
    println!("history_source={source_kind}");
    if let Some(path) = source_path {
        println!("history_file={}", path.display());
    } else {
        println!("history_file=repo_seed");
        println!("history_install_hint=cargo run --manifest-path tools/asd-history/Cargo.toml -- --install-seed");
    }

    match decision {
        Some(decision_id) => {
            let gains = history
                .gains
                .iter()
                .filter(|gain| gain.decision_id == decision_id)
                .collect::<Vec<_>>();
            let challenges = history
                .provider_challenges
                .iter()
                .filter(|challenge| challenge.decision_id == decision_id)
                .collect::<Vec<_>>();
            println!("decision_id={decision_id}");
            println!("transitions={}", gains.len());
            println!("provider_challenges={}", challenges.len());
            for gain in gains {
                print_gain(gain);
            }
            for challenge in challenges {
                print_provider_challenge(challenge);
            }
        }
        None => {
            println!("transitions={}", history.gains.len());
            println!("provider_challenges={}", history.provider_challenges.len());
            for gain in &history.gains {
                print_gain(gain);
            }
            for challenge in &history.provider_challenges {
                print_provider_challenge(challenge);
            }
        }
    }

    Ok(())
}
