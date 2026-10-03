#[path = "../../../candle-kernels/src/asd_history.rs"]
mod asd_history;

use asd_history::{
    history_for_decision, provider_challenges_for_decision, GAINS, PROVIDER_CHALLENGES,
};

fn print_gain(gain: &asd_history::HistoricalGain) {
    println!(
        "HISTORICAL_STATE transition={} decision={} state={} generation={} speedup_x={:.6}",
        gain.transition_id,
        gain.decision_id,
        gain.state.as_str(),
        gain.generation,
        gain.speedup_x,
    );
    println!(
        "  predecessor_transition={}",
        gain.predecessor_transition_id.unwrap_or("none")
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
    if let Some(commit) = gain.implementation_commit {
        println!("  implementation_commit={commit}");
    }
    if let Some(commit) = gain.benchmark_commit {
        println!("  benchmark_commit={commit}");
    }
    if let Some(evidence) = gain.evidence_sha256 {
        println!("  evidence_sha256={evidence}");
    }
    println!("  exact_signature={}", gain.exact_signature);
    println!("  note={}", gain.note);
}

fn print_provider_challenge(challenge: &asd_history::ProviderChallenge) {
    println!(
        "PROVIDER_CHALLENGE challenge={} decision={} protocol={} consensus={} measurement_plane={}",
        challenge.challenge_id,
        challenge.decision_id,
        challenge.protocol,
        challenge.consensus.as_str(),
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
    for replication in challenge.replications {
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

fn main() {
    let decision = std::env::args().nth(1);
    println!("=== ASD EXACT PROFILE HISTORICAL STATE ===");
    println!("dispatch_authority=false");
    println!("historical_gain_composition=forbidden");

    match decision {
        Some(decision_id) => {
            let gains = history_for_decision(&decision_id);
            let challenges = provider_challenges_for_decision(&decision_id);
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
            println!("transitions={}", GAINS.len());
            println!("provider_challenges={}", PROVIDER_CHALLENGES.len());
            for gain in &GAINS {
                print_gain(gain);
            }
            for challenge in &PROVIDER_CHALLENGES {
                print_provider_challenge(challenge);
            }
        }
    }
}
