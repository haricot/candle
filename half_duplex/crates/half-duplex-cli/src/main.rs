//! HDB0 CLI: useful WAV inspection and a clearly labeled synthetic turn simulation.
//! Not a speech recognizer and not a microphone pipeline.
use half_duplex_core::{AsrBackend, AudioChunk, Event, PcmAudio, PcmRingBuffer, TurnController, UnavailableAsr, to_16khz};
use half_duplex_wav::read_wav;
use std::{env, error::Error, fs, path::{Path, PathBuf}};

const CANDLE_COMMIT: &str = "f5ce7f6f8b7a68d14a55419f9f221e1ae3537fd3";

fn help() {
    println!("half-duplex v0.1.2 | HDB0 SOFTWARE-ONLY (no ASR, VAD, microphone or Smart Turn)");
    println!("Usage:");
    println!("  half-duplex inspect-wav <file.wav> [report.json]");
    println!("  half-duplex simulate-turn <file.wav> [report.json]");
    println!("  half-duplex selftest [report.json]");
    println!("  half-duplex version");
    println!("RIFF/WAVE PCM16 or Float32, 1-2 channels; simulate auto-normalizes 8-192 kHz to 16 kHz.");
}

fn quote_json(s: &str) -> String {
    let mut output = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            c if c <= '\u{001f}' => output.push_str(&format!("\\u{:04x}", c as u32)),
            _ => output.push(c),
        }
    }
    output.push('"');
    output
}
fn maybe_write(output: Option<&String>, json: &str) -> Result<(), Box<dyn Error>> {
    if let Some(path) = output {
        let p = Path::new(path);
        if let Some(parent) = p.parent() { fs::create_dir_all(parent)?; }
        fs::write(p, json)?;
        println!("REPORT={}", p.display());
    } else { println!("{json}"); }
    Ok(())
}
fn stats_json(audio: &PcmAudio, source: &str) -> String {
    format!(
        "{{\"schema_version\":1,\"stage\":\"HDB0\",\"status\":\"WAV_INSPECTION_ONLY\",\"source\":{},\"sample_rate\":{},\"channels_normalized\":1,\"samples\":{},\"duration_seconds\":{:.6},\"peak\":{:.7},\"rms\":{:.7},\"asr_implemented\":false,\"candle_commit\":{}}}",
        quote_json(source), audio.sample_rate(), audio.sample_count(), audio.duration_seconds(), audio.peak(), audio.rms(), quote_json(CANDLE_COMMIT)
    )
}
fn event_json(e: &Event) -> String {
    match e {
        Event::SpeechStarted { turn_id, sample } | Event::PauseDetected { turn_id, sample } | Event::TurnContinued { turn_id, sample } => format!(
            "{{\"kind\":{},\"turn_id\":{},\"sample\":{}}}", quote_json(e.kind()), turn_id, sample
        ),
        Event::TurnCompleted { turn_id, start_sample, end_sample } => format!(
            "{{\"kind\":{},\"turn_id\":{},\"start_sample\":{},\"end_sample\":{}}}",
            quote_json(e.kind()), turn_id, start_sample, end_sample
        ),
        Event::TranscriptReady { turn_id, text, backend_id, language } => format!(
            "{{\"kind\":{},\"turn_id\":{},\"text\":{},\"backend\":{},\"language\":{}}}",
            quote_json(e.kind()), turn_id, quote_json(text), quote_json(backend_id), quote_json(language)
        ),
        Event::ResponseFinished { turn_id } => format!("{{\"kind\":{},\"turn_id\":{}}}", quote_json(e.kind()), turn_id),
    }
}

fn simulate(audio: PcmAudio, source: &str, out: Option<&String>) -> Result<(), Box<dyn Error>> {
    let source_rate = audio.sample_rate();
    let source_samples = audio.sample_count();
    let resampled = source_rate != 16_000;
    let audio = to_16khz(&audio)?;
    if audio.sample_count() < 8_000 { return Err("simulation requires >= 0.5s of audio".into()); }
    let mut ring = PcmRingBuffer::new(16_000, (audio.sample_count() + 1).max(16_000))?;
    let mut position = 0_u64;
    for samples in audio.samples().chunks(320) {
        let chunk = AudioChunk::new(position, 16_000, samples.to_vec())?;
        position = chunk.end_sample();
        ring.append(&chunk)?;
    }
    let mut turn = TurnController::new();
    let len = audio.sample_count() as u64;
    let first_pause = len / 4;
    let continuation = len / 3;
    let second_pause = (len * 3) / 4;
    let mut events = Vec::new();
    events.push(turn.speech_started(0)?);
    events.push(turn.pause_detected(first_pause)?);
    events.push(turn.endpoint_decision(false, continuation)?);
    events.push(turn.pause_detected(second_pause)?);
    events.push(turn.endpoint_decision(true, len)?);
    let extracted = ring.extract(0, len)?;
    let mut asr = UnavailableAsr;
    let asr_result = match asr.transcribe(&extracted) {
        Ok(_) => return Err("HDB0 invariant broken: UnavailableAsr unexpectedly succeeded".into()),
        Err(e) => e.to_string(),
    };
    let ev = events.iter().map(event_json).collect::<Vec<_>>().join(",");
    let json = format!(
        "{{\"schema_version\":1,\"stage\":\"HDB0\",\"status\":\"SYNTHETIC_SOFTWARE_ONLY_NO_ASR\",\"source\":{},\"source_sample_rate\":{},\"source_samples\":{},\"sample_rate\":16000,\"samples\":{},\"resampled\":{},\"resample_method\":{},\"duration_seconds\":{:.6},\"events\":[{}],\"final_phase\":{},\"candle_commit\":{},\"asr_backend\":{},\"asr_status\":{}}}",
        quote_json(source), source_rate, source_samples, audio.sample_count(), resampled,
        quote_json(if resampled { "windowed_sinc_blackman_r32" } else { "identity" }),
        audio.duration_seconds(), ev, quote_json(turn.phase().as_str()),
        quote_json(CANDLE_COMMIT), quote_json(asr.id()), quote_json(&asr_result)
    );
    maybe_write(out, &json)?;
    println!("HDB0=PASS_SYNTHETIC_SOFTWARE_ONLY_NO_ASR");
    Ok(())
}
fn selftest(out: Option<&String>) -> Result<(), Box<dyn Error>> {
    // Synthetic amplitude only; it does not represent speech and does not run FireRedVAD.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/hdb0-tone-16k.wav");
    let audio = read_wav(&path)?;
    simulate(audio, "SYNTHETIC_FIXTURE_NOT_SPEECH", out)
}
fn main_inner() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("help") | Some("--help") | Some("-h") | None => help(),
        Some("version") => println!("HDB0=0.1.2 CANDLE_EXTENDED_COMMIT={CANDLE_COMMIT}"),
        Some("inspect-wav") => {
            let path = args.get(1).ok_or("usage: inspect-wav <file.wav> [report.json]")?;
            if args.len() > 3 { return Err("too many arguments".into()); }
            let audio = read_wav(path)?;
            let json = stats_json(&audio, path);
            maybe_write(args.get(2), &json)?;
            println!("HDB0=WAV_INSPECTED_NO_ASR");
        }
        Some("simulate-turn") => {
            let path = args.get(1).ok_or("usage: simulate-turn <file.wav> [report.json]")?;
            if args.len() > 3 { return Err("too many arguments".into()); }
            simulate(read_wav(path)?, path, args.get(2))?;
        }
        Some("selftest") => {
            if args.len() > 2 { return Err("too many arguments".into()); }
            selftest(args.get(1))?;
        }
        Some(_) => return Err("unknown command; run --help".into()),
    }
    Ok(())
}
fn main() {
    if let Err(e) = main_inner() {
        eprintln!("HDB0=FAIL {e}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn escape_json_preserves_control_characters() {
        assert_eq!(quote_json("a\n\"\\\u{0001}"), "\"a\\n\\\"\\\\\\u0001\"");
    }
    #[test]
    fn microphone_24k_can_run_synthetic_turn_simulation() {
        let wav = PcmAudio::new(24_000, vec![0.0; 12_000]).unwrap();
        simulate(wav, "SYNTHETIC_24K_MICROPHONE", None).unwrap();
    }
    #[test]
    fn sample_fixture_loads() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/hdb0-tone-16k.wav");
        let audio = read_wav(path).unwrap();
        assert_eq!(audio.sample_rate(), 16_000);
        assert!(audio.duration_seconds() >= 0.5);
    }
}
