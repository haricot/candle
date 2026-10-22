#!/usr/bin/env python3
"""Add auditable in-process warm measurement to the pinned *local* runtime.

This patch works AFTER HDB2-A bootstrap/repair. It does not modify Candle
or upstream Git commits. It does not change the normal transcription path.
"""
from pathlib import Path
import argparse
import subprocess

PIN='53c0d3065f9dd8de0356b25391e60ed3a471192b'
MARKER='// HDB2B_IN_PROCESS_BENCH_V1'

ARGS_ANCHOR='    model_dir: Option<PathBuf>,\n'
ARGS_ADDITION='''
    /// Number of measured hot ASR calls in the same loaded model.
    #[arg(long, default_value_t = 0)]
    hdb2b_bench_runs: usize,

    /// Number of unmeasured warmup ASR calls.
    #[arg(long, default_value_t = 1)]
    hdb2b_bench_warmup: usize,
'''
LOAD_ANCHOR='    let mut model = from_config_value(config, vb)?;\n'
LOAD_ADDITION='''    let hdb2b_model_ready_ms = t0.elapsed().as_secs_f64() * 1000.0;
'''
BENCH_ANCHOR='    if args.debug {\n        print_debug(&mut model, &args.input)?;\n    }\n'
BENCH_CODE='''    // HDB2B_IN_PROCESS_BENCH_V1
    // No simulated ASR. All samples actually invoke Candle FastConformer/TDT.
    // Each run includes WAV decoding, log-mel, encoder and token/duration decoder.
    // Decode performs host-side token extraction; GPU work is synchronized there.
    if args.hdb2b_bench_runs > 0 {
        anyhow::ensure!(!args.stream, "HDB2B benchmark only supports offline ASR");
        anyhow::ensure!(args.hdb2b_bench_runs <= 30, "HDB2B max 30 timed samples");
        anyhow::ensure!(args.hdb2b_bench_warmup <= 10, "HDB2B max 10 warmups");
        let mut warmup_ms: Vec<f64> = Vec::new();
        let mut samples_ms: Vec<f64> = Vec::new();
        let mut expected_text: Option<String> = None;
        let mut last_result: Option<AlignedResult> = None;
        for iteration in 0..(args.hdb2b_bench_warmup + args.hdb2b_bench_runs) {
            let started = Instant::now();
            let current = model.transcribe(&args.input, &decoding_config, None, 0.0, None)?;
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            anyhow::ensure!(!current.text.trim().is_empty(), "HDB2B produced empty transcript");
            if let Some(ref expected) = expected_text {
                anyhow::ensure!(expected == &current.text, "HDB2B transcript varied across iterations");
            } else {
                expected_text = Some(current.text.clone());
            }
            if iteration < args.hdb2b_bench_warmup {
                warmup_ms.push(elapsed);
            } else {
                samples_ms.push(elapsed);
            }
            last_result = Some(current);
        }
        let result = last_result.as_ref().ok_or_else(|| anyhow::anyhow!("HDB2B no result"))?;
        print_output(&args, &device, result)?;
        eprintln!("HDB2B_BENCH_JSON={}", serde_json::json!({
            "schema_version": 1,
            "model_ready_ms": hdb2b_model_ready_ms,
            "warmup_ms": warmup_ms,
            "hot_samples_ms": samples_ms,
            "hot_scope": "wav_decode+logmel+encoder+tdt_decode",
            "dtype": "f32",
            "text_consistent": true,
            "measured_device": format!("{device:?}"),
            "process_elapsed_ms": t0.elapsed().as_secs_f64() * 1000.0
        }));
        return Ok(());
    }

'''

def exactly(text, old, new):
    n=text.count(old)
    if n != 1:
        raise ValueError(f'HDB2B pin drift: expected 1 anchor, found {n}: {old[:95]!r}')
    return text.replace(old,new,1)

def patch_main(text):
    if MARKER in text:
        if text.count(MARKER)!=1 or text.count('hdb2b_bench_runs')<2:
            raise ValueError('HDB2B marker corrupted')
        return text
    if 'HDB2A_CANDLE_LOCAL_MODEL' not in text:
        raise ValueError('HDB2A local runtime loader missing; repair/bootstrap HDB2A first')
    text=exactly(text,ARGS_ANCHOR,ARGS_ANCHOR+ARGS_ADDITION)
    text=exactly(text,LOAD_ANCHOR,LOAD_ANCHOR+LOAD_ADDITION)
    text=exactly(text,BENCH_ANCHOR,BENCH_CODE+BENCH_ANCHOR)
    return text

def repair(runtime):
    runtime=runtime.resolve()
    src=runtime/'src/main.rs'
    if not src.is_file(): raise FileNotFoundError(f'Runtime not found: {src}')
    commit=subprocess.check_output(['git','-C',str(runtime),'rev-parse','HEAD'],text=True).strip()
    if commit!=PIN: raise ValueError(f'Wrong runtime commit {commit}; expected {PIN}')
    old=src.read_text(encoding='utf-8')
    new=patch_main(old)
    if new!=old:
        src.write_text(new,encoding='utf-8')
        print('HDB2B_RUNTIME=PATCHED_REBUILD_REQUIRED')
    else: print('HDB2B_RUNTIME=ALREADY_PATCHED')
    print(f'HDB2B_RUNTIME_DIR={runtime}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime-dir',required=True,type=Path)
    args=p.parse_args()
    repair(args.runtime_dir)

if __name__=='__main__':main()
