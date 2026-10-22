# HDB1-R1 v0.5.1 — Cancellable endpoint hold and microphone restart

## Evidence, 2026-10-18

User's first natural spoken intervention was incorrectly split into three turns:
- VAD candidate at 3.4s, Smart Turn 0.982320, ASR `Muchas...`;
- VAD candidate at 8.1s, Smart Turn 0.913980, ASR `Je teste du bambini tous...`;
- VAD candidate at 11.4s, Smart Turn 0.988007, ASR `J'espère que chaque seulement`.

The next speech onsets were 3.8s and 9.0s, both within 1200ms after a
candidate. Because v0.5.0 immediately finalized at the first positive
Smart Turn output, `--max-turns 3` ended capture before the second intended
intervention.

## Changes

1. **Cancellable candidate hold**: `speech_end` sets a pending candidate.
   `speech_start` within the hold cancels it, preserving buffered PCM.
   After `--endpoint-hold-ms` uninterrupted silence (default 1200ms), run the
   real ONNX Smart Turn **once** and finalize only on completion >= threshold.
   Rejected predictions keep the pending buffer until speech resumes or
   `--max-hesitation-ms` expires, with a distinct forced reason. EOF has its
   own explicit status and is never relabelled as Smart Turn approval.
2. **Capture lifecycle**: after each accepted mic turn with ASR, stop the
   old `pw-record` and start a new capture before listening for another turn.
   Previously the loop could continue consuming queued, stale mic audio.
   The v0.5.1 transition explicitly logs `capture_restart_required_after_asr`
   and `capture_restarted_after_asr`.
3. **Audio isolation**: discard pre-roll from the previous finalized turn, so
   fragments cannot bleed into the next turn.
4. **Evidence**: `endpoint_candidates_deferred`,
   `endpoint_candidates_cancelled` counters and timestamped event records.
   Tests include an exact synthetic replay of the user's event *times*,
   not a fake test of ONNX predictions.

## Runtime limits

This is an interim endpoint controller fix, **not** a resident Candle engine.
ASR is still a blocking, fresh Rust process for each confirmed turn. PCM
arriving while the ASR process runs is not processed and is discarded when
`pw-record` restarts. This is a known half-duplex limitation. A resident
worker with controlled audio turn-taking is the follow-up optimization.

The 1200-ms hold intentionally adds turn-finalization latency on a genuine
end of speech and is a calibration parameter, not a universally optimal value.
Real mic detection, ASR quality, and multi-turn reliability are **NOT YET
QUALIFIED** for the v0.5.1 build despite deterministic software tests.

## Run

```bash
export HDB2A_RUNTIME_DIR="$HOME/tmp/haricot-half-duplex/integration/parakeet-candle/runtime"
export HDB2A_MODEL_DIR="$HOME/tmp/haricot-half-duplex/models/transcribepsy-eu-sub-candle"
export HDB2A_REPORT_ROOT="$HOME/tmp/haricot-half-duplex/reports"
./run-hdb1.sh selftest
./run-hdb1.sh mic --max-turns 3 --variant cuda-cudnn --endpoint-hold-ms 1200
```

Do not use threshold 0.99 as a default: it rejects nearly all confident
predictions on tested audio and causes forced/EOF termination.
EOF and timeout are not semantically confirmed endpoints.
