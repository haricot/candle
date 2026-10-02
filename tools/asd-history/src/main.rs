#[path = "../../../candle-kernels/src/asd_history.rs"]
mod asd_history;

use asd_history::{history_for_decision, GAINS};

fn print_gain(gain: &asd_history::HistoricalGain) {
    println!(
        "HISTORICAL_STATE transition={} decision={} state={} speedup_x={:.6}",
        gain.transition_id,
        gain.decision_id,
        gain.state.as_str(),
        gain.speedup_x,
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

fn main() {
    let decision = std::env::args().nth(1);
    println!("=== ASD EXACT PROFILE HISTORICAL STATE ===");
    println!("dispatch_authority=false");
    println!("historical_gain_composition=forbidden");

    match decision {
        Some(decision_id) => {
            let gains = history_for_decision(&decision_id).collect::<Vec<_>>();
            println!("decision_id={decision_id}");
            println!("transitions={}", gains.len());
            for gain in gains {
                print_gain(gain);
            }
        }
        None => {
            println!("transitions={}", GAINS.len());
            for gain in &GAINS {
                print_gain(gain);
            }
        }
    }
}
