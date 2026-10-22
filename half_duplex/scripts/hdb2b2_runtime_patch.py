#!/usr/bin/env python3
"""Idempotent additional profiling CLI for local pinned Parakeet Candle runtime.

Does not modify normal ASR, HDB2A or HDB2B execution paths. CUDA stage
measurements use explicit device synchronization and should not be summed to
predict uninstrumented latency.
"""
import argparse
from pathlib import Path
import subprocess

PIN = '53c0d3065f9dd8de0356b25391e60ed3a471192b'
MARKER = '// HDB2B2_PROFILE_V1'
ARGS_ANCHOR = '    hdb2b_bench_warmup: usize,\n'
MODEL_READY_ANCHOR = '    let hdb2b_model_ready_ms = t0.elapsed().as_secs_f64() * 1000.0;\n'
BEFORE_LEGACY = '    // HDB2B_IN_PROCESS_BENCH_V1\n'
ARGS = '''
    /// HDB2-B2 measured full offline ASR calls, model preloaded.
    #[arg(long, default_value_t = 0)]
    hdb2b2_hot_runs: usize,

    /// Untimed full ASR warmups before HDB2-B2 measurements.
    #[arg(long, default_value_t = 3)]
    hdb2b2_warmup: usize,

    /// Separate phase-profile runs with CUDA synchronization barriers.
    #[arg(long, default_value_t = 3)]
    hdb2b2_profile_runs: usize,
'''
BENCH = '''    // HDB2B2_PROFILE_V1
    // Two separate experiment types: (1) total hot latency, (2) segmented
    // phase profiling. GPU stage barriers intentionally perturb execution.
    if args.hdb2b2_hot_runs > 0 {
        anyhow::ensure!(!args.stream, "HDB2-B2 supports only full offline ASR");
        anyhow::ensure!(args.hdb2b2_hot_runs <= 30, "HDB2-B2 max 30 hot runs");
        anyhow::ensure!(args.hdb2b2_warmup <= 10, "HDB2-B2 max 10 warmups");
        anyhow::ensure!(args.hdb2b2_profile_runs >= 1 && args.hdb2b2_profile_runs <= 10,
            "HDB2-B2 profile runs must be 1..10");

        let mut warmup_ms = Vec::<f64>::new();
        let mut hot_samples_ms = Vec::<f64>::new();
        let mut expected_text: Option<String> = None;
        let mut last_result: Option<AlignedResult> = None;
        let device_ref = model.device_ref().clone();
        for iteration in 0..(args.hdb2b2_warmup + args.hdb2b2_hot_runs) {
            device_ref.synchronize()?;
            let started = Instant::now();
            let current = model.transcribe(&args.input, &decoding_config, None, 0.0, None)?;
            device_ref.synchronize()?;
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            anyhow::ensure!(!current.text.trim().is_empty(), "HDB2-B2 empty ASR transcript");
            if let Some(ref prior) = expected_text {
                anyhow::ensure!(prior == &current.text, "HDB2-B2 inconsistent transcript");
            } else {
                expected_text = Some(current.text.clone());
            }
            if iteration < args.hdb2b2_warmup {
                warmup_ms.push(elapsed);
            } else {
                hot_samples_ms.push(elapsed);
            }
            last_result = Some(current);
        }

        // Isolate WAV decode, log-mel, encoder, TDT, token/sentence alignment.
        // This path is intentionally distinct from the ordinary total hot calls.
        let mut profile_samples = Vec::<serde_json::Value>::new();
        let pre = model.preprocessor_config().clone();
        for _ in 0..args.hdb2b2_profile_runs {
            device_ref.synchronize()?;
            let total_start = Instant::now();
            let phase = Instant::now();
            let samples = load_audio(&args.input, pre.sample_rate)?;
            let load_audio_ms = phase.elapsed().as_secs_f64() * 1000.0;

            let phase = Instant::now();
            let mel = get_logmel(&samples, &pre, &device_ref)?;
            device_ref.synchronize()?;
            let logmel_ms = phase.elapsed().as_secs_f64() * 1000.0;

            let phase = Instant::now();
            let (features, lengths) = match &mut model {
                ParakeetModel::Tdt(m) => m.encoder.forward(&mel, None)?,
                _ => anyhow::bail!("HDB2-B2 profiler requires genuine TDT model"),
            };
            device_ref.synchronize()?;
            let encoder_ms = phase.elapsed().as_secs_f64() * 1000.0;

            let phase = Instant::now();
            let (tokens, _) = match &mut model {
                ParakeetModel::Tdt(m) => m.decode(
                    &features, Some(&lengths), None, None, &decoding_config)?,
                _ => anyhow::bail!("HDB2-B2 profiler requires genuine TDT model"),
            };
            device_ref.synchronize()?;
            let tdt_ms = phase.elapsed().as_secs_f64() * 1000.0;

            let phase = Instant::now();
            let mut profiled = tokens.into_iter().map(|hyp| {
                parakeet_candle::parakeet::alignment::sentences_to_result(
                    &parakeet_candle::parakeet::alignment::tokens_to_sentences(
                        &hyp, &decoding_config.sentence))
            });
            let current = profiled.next().ok_or_else(|| anyhow::anyhow!("empty TDT batch"))?;
            let alignment_ms = phase.elapsed().as_secs_f64() * 1000.0;
            anyhow::ensure!(expected_text.as_ref() == Some(&current.text),
                "HDB2-B2 segmented ASR text differs from full inference");
            let profile_wall_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            profile_samples.push(serde_json::json!({
                "wav_decode_ms": load_audio_ms,
                "logmel_ms": logmel_ms,
                "fastconformer_ms": encoder_ms,
                "tdt_decode_ms": tdt_ms,
                "text_alignment_ms": alignment_ms,
                "profile_wall_ms": profile_wall_ms
            }));
        }
        let result = last_result.as_ref().ok_or_else(|| anyhow::anyhow!("missing ASR result"))?;
        print_output(&args, &device, result)?;
        eprintln!("HDB2B2_BENCH_JSON={}", serde_json::json!({
            "schema_version": 1,
            "model_ready_ms": hdb2b_model_ready_ms,
            "warmup_ms": warmup_ms,
            "hot_samples_ms": hot_samples_ms,
            "phase_profile_samples": profile_samples,
            "hot_scope": "wav_decode+logmel+fastconformer+tdt+alignment",
            "phase_profile_scope": "separate synchronized runs (not additive uninstrumented latency)",
            "dtype": "f32",
            "text_consistent": true,
            "measured_device": format!("{device:?}"),
            "process_elapsed_ms": t0.elapsed().as_secs_f64() * 1000.0
        }));
        return Ok(());
    }

'''

def replace_once(text, old, new):
    n = text.count(old)
    if n != 1:
        raise ValueError(f'HDB2-B2 runtime pin drift: expected 1 anchor, saw {n}: {old[:70]!r}')
    return text.replace(old, new, 1)

def patch_main(text):
    if MARKER in text:
        if text.count(MARKER) != 1 or text.count('hdb2b2_hot_runs') < 2:
            raise ValueError('HDB2-B2 marker exists but source is corrupted')
        return text
    if 'HDB2B_IN_PROCESS_BENCH_V1' not in text:
        raise ValueError('HDB2-B baseline runtime patch is required first')
    text = replace_once(text, ARGS_ANCHOR, ARGS_ANCHOR + ARGS)
    text = replace_once(text, MODEL_READY_ANCHOR, MODEL_READY_ANCHOR)
    text = replace_once(text, BEFORE_LEGACY, BENCH + BEFORE_LEGACY)
    return text

def repair(runtime):
    runtime = runtime.resolve()
    src = runtime/'src/main.rs'
    if not src.is_file():
        raise FileNotFoundError(src)
    head = subprocess.check_output(['git','-C',str(runtime),'rev-parse','HEAD'],text=True).strip()
    if head != PIN:
        raise ValueError(f'Expected pinned runtime {PIN}, got {head}')
    before = src.read_text(encoding='utf-8')
    after = patch_main(before)
    if before != after:
        src.write_text(after,encoding='utf-8')
        print('HDB2B2_RUNTIME=PATCHED_UNBUILT')
    else:
        print('HDB2B2_RUNTIME=ALREADY_PATCHED')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime-dir',type=Path,required=True)
    a=p.parse_args()
    repair(a.runtime_dir)

if __name__=='__main__':main()
