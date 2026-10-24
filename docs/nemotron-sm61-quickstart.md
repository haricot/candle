# Nemotron on the qualified Extended composition

Physical SM61 run: <https://github.com/haricot/candle/actions/runs/40029177676>, attempt 2.
Candidate: `40b5a8acc43b5f8ef925370ae12daddd929f39f2`.
Native CPU/CUDA F32 tokens, text and all frame decisions match for `fr-FR` and
`auto` on the one public French clip. This is not a broad ASR quality claim.

The canonical source configuration contains twelve standalone features and
excludes the Half Duplex application. Each canonical standalone ref receives
its own prepared tree; `extended` receives the combined GPU-tested candidate.
Temporary preparation/candidate refs may be removed after immutable archival
tags and promotion checks. Preserve `main`, `cuda_legacy`, the standalone model
refs, Half Duplex/endpointing application branches and branches used by open PRs.

## Local transcription

Fetch the promoted branch with an explicit refspec (single-branch clones may not
create `origin/extended` with a plain `git fetch origin extended`):

```bash
git -C "$HOME/tmp/candle-stt-qualification" fetch origin \
  +refs/heads/extended:refs/remotes/origin/extended
git -C "$HOME/tmp/candle-stt-qualification" worktree add --detach \
  "$HOME/tmp/candle-nemotron-promoted" refs/remotes/origin/extended
cd "$HOME/tmp/candle-nemotron-promoted"

export CUDA_COMPUTE_CAP=61
export NVCC=/opt/cuda/bin/nvcc
export NVCC_CCBIN=gcc-14
cargo build --release -p candle-examples --example nemotron_asr --features cuda,cudnn
```

Use the existing verified model directory, or download it without conversion:

```bash
export NEMOTRON_MODEL_DIR="$HOME/tmp/nemotron35-model"
bash tools/nemotron_asr_streaming/download-model.sh "$NEMOTRON_MODEL_DIR"
```

Record an utterance and stop recording with Ctrl-C:

```bash
pw-record --rate 16000 --channels 1 --format s16 "$HOME/tmp/nemotron-mic.wav"
```

Then transcribe:

```bash
./target/release/examples/nemotron_asr \
  --model-dir "$NEMOTRON_MODEL_DIR" --input "$HOME/tmp/nemotron-mic.wav" \
  --device cuda-cudnn --dtype f32 --mode streaming --language fr-FR \
  --output "$HOME/tmp/nemotron-mic-result"
jq '{device,text,load_seconds,inference_seconds,rtf}' \
  "$HOME/tmp/nemotron-mic-result/result.json"
```

The WAV example feeds PCM packets into the persistent native streaming engine,
then prints a final result. It does not yet display live partial transcripts or
detect the end of a conversational turn. Reuse the same loaded `Model` in an
application and create a fresh `StreamState` per utterance to avoid reloading
weights. The frontend runs as RustFFT on CPU; neural inference uses CUDA.

## Half Duplex and Myna

The native ASR CLI is also a generic external adapter: it accepts `--input WAV
--output DIRECTORY --model-dir DIRECTORY --device LABEL`, and writes a
`result.json` object containing string `text`. A controller supporting this
contract can invoke it unchanged with device `cuda-cudnn`; the defaults are
French (`fr-FR`), streaming and F32. Controller source, model assets and its own
pinned dependencies remain separate from the ASR binary's revision.

An external process reloads Nemotron for each invocation. For repeated turns,
use the Rust model API in a resident callback/service to retain loaded weights
and create a fresh stream per utterance. No Python/ONNX runtime is required.

Canonical Myna has Rust microphone capture, a dictation client and desktop text
injection. Its server/backends are separate. Myna's own documentation marks its
earlier Nemotron evaluation as parked and never packaged; it is not a Candle
backend. A native Rust server implementing Myna's session protocol over its Unix
socket is needed to use this Candle engine with that client without Python/ONNX.
One shared native ASR service can later serve Myna dictation and the Half Duplex
controller while keeping the model implementation in `candle-transformers`.
