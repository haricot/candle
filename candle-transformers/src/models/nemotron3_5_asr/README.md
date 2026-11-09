# Nemotron 3.5 ASR Streaming — native Candle

Branch: `nemotron_asr_streaming_standalone`, based on `haricot/candle:main`
at `c68b24997319b8d76e5a4e775dab197c7f26322a`.

## Implemented and qualified scope

The complete Rust engine implements PCM → preemphasis → Slaney log-Mel →
causal factor-8 subsampling → cache-aware FastConformer → language conditioning
→ two-layer LSTM / greedy **RNN-T**. The example uses the original tokenizer
through Rust `tokenizers`. Sessions have independent, bounded caches and a final
flush; the engine retains neither the accumulated transcript nor full audio.

No Python or ONNX is needed at runtime. The loader accepts the unchanged official
HF `model.safetensors` directly; no exported/remapped checkpoint is required.
The native NeMo `.nemo` archive has a different tensor layout and is not accepted.

**CPU F32: eight real-model comparisons pass** against the pinned official
Transformers implementation. Tokens, text, blank decisions and frame advances
match exactly within each protocol. Mel/encoder tensors pass elementwise absolute
and relative bounds. See [results](../../../../tools/nemotron_asr_streaming/RESULTS-CPU.md).
One public French utterance and its repetition do not establish general ASR
quality. CUDA and reduced precision remain to be qualified on hardware.

## Transcribe without Python

From the repository root, with Rust 1.99.0 and `curl`:

```bash
export NEMOTRON_MODEL_DIR="$HOME/tmp/nemotron-3.5-asr-model"
bash tools/nemotron_asr_streaming/download-model.sh "$NEMOTRON_MODEL_DIR"
cargo build --release -p candle-examples --example nemotron_asr
./target/release/examples/nemotron_asr \
  --model-dir "$NEMOTRON_MODEL_DIR" --input /absolute/path/input.wav \
  --device cpu --dtype f32 --mode streaming --language fr-FR \
  --output /absolute/path/transcription
```

Input is mono PCM16 WAV at 16 kHz. To resample, or to record an utterance with
PipeWire (stop recording with Ctrl-C):

```bash
ffmpeg -i original.wav -ac 1 -ar 16000 -c:a pcm_s16le input.wav
# Or:
pw-record --rate 16000 --channels 1 --format s16 input.wav
```

`--mode offline` selects whole-file inference; `--language auto` enables automatic
language detection. `--lookahead` accepts 0, 3, 6 or 13 (default 3).
`--pcm-chunk-ms` controls PCM delivery independently of encoder blocks. The WAV
example feeds packets without wall-clock pacing and prints the final transcript.
The model API returns new tokens at every push for application/microphone integration.

## CUDA, including the promoted SM61 backend

The standalone branch adds the model without changing CUDA kernels. For a GPU
supported by the upstream backend, build with `--features cuda,cudnn` and select
`--device cuda-cudnn --dtype f32`. Failed CUDA initialization is an error, with
no silent CPU fallback. The small audio frontend uses RustFFT on CPU; neural
tensors are placed on the selected device.

For the user's Pascal/SM61 machine, prepare a local integration with the backend
promoted by successful run `39846015167`, pinned at
`6b29b7f43a574fae32984b34b46de7c1d9b145eb`:

```bash
# From a clean, updated checkout of nemotron_asr_streaming_standalone:
bash tools/nemotron_asr_streaming/prepare-extended.sh \
  "$HOME/tmp/candle-nemotron-extended"
cd "$HOME/tmp/candle-nemotron-extended"
export CUDA_COMPUTE_CAP=61
export CUDAARCHS=61
# Keep the CUDA toolkit, cuDNN and host compiler of your working HDB build.
bash tools/nemotron_asr_streaming/qualify-local.sh \
  "$NEMOTRON_MODEL_DIR" /absolute/path/input.wav \
  "$HOME/tmp/nemotron-cuda-qualification-01" cuda-cudnn
```

Preparation creates a detached worktree and stages a local squash merge; it does
not move or publish Extended. Choose a new destination path. Qualification builds
once, then runs CPU and CUDA F32 using the **same binary and WAV**, for `fr-FR`
and `auto`. It checks exact tokens, text, frame advances and WAV SHA-256. Each
result records the actual device and timings. A mismatch exits unsuccessfully
and keeps the candidate `result.json`. This check uses native CPU as the baseline
for the supplied audio; it is not an independent ASR quality benchmark.

No Podman, PyTorch wheel or Python is needed for these commands. F16/BF16 flags
exist for later experiments; neither is qualified here. Start with F32 on SM61.

## Model API and streaming protocol

`Model::load(config, processor, vb)` accepts a root VarBuilder for the official
checkpoint. `transcribe_mono` provides whole-file inference. For streaming:

```rust,ignore
let mut stream = model.new_stream("fr-FR", 3)?;
for pcm_packet in packets {
    let new_tokens = model.push_pcm(&mut stream, pcm_packet, 16_000)?;
    // Append new_tokens.tokens to the application's token buffer.
}
let final_tokens = model.finish(&mut stream)?;
```

Create a new session for every utterance; weights are reusable. State belongs to
the model that created it. Arbitrary PCM packet boundaries preserve preemphasis
and STFT overlap. Finishing twice returns an empty trace; pushing after finish
is an error. Invalid PCM/rate and failed inference leave session state unchanged.

The official fixed-Mel-block protocol uses first block `1+8*r`, subsequent
blocks `8*(r+1)`, producing `r+1` encoder frames per block. Only valid centered-STFT
Mel frames are sent. A partial final block is padded with **zero Mel**, and its
entire output is decoded, as in the reference. Offline and streaming outputs can
differ, including punctuation; compare matching protocols.

Caches retain at most 56 attention frames, 8 temporal-convolution frames per layer,
and one row per subsampling stage. Global frame indices continue after cache
eviction. Frontend/pending Mel storage is bounded independently of utterance
duration. Optional observer methods expose actual encoder blocks without replay.
Batch size is one. Beam search, diarization, VAD, endpointing and microphone
capture remain outside this model; there is no HDB2 application dependency.

## Official artifacts and architecture

Model: [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
at revision `ea30d66debe3740a08b573244286791d423d6b3e`.
The downloader verifies all six files using [model.sha256](../../../../tools/nemotron_asr_streaming/model.sha256).
The checkpoint contains 655 F32 tensors, 637,997,088 parameters. The CLI field
`expected_model_revision` states the target; use the downloader to verify local files.

| Component | Configuration |
| --- | --- |
| Audio | 16 kHz, 128 Mel bins, window 400, hop 160, FFT 512, preemphasis 0.97; no Mel normalization |
| Encoder | 24 FastConformer layers, width 1024, 8 heads, FFN 4096 |
| Subsampling | Causal depthwise Conv2d, factor 8; asymmetric padding, final frequency width 17 |
| Language conditioning | 128 slots; one-hot concatenation; 1152 → 2048 → ReLU → 1024 → 640 |
| Predictor / joint | 2 LSTMs of width 640; ReLU joint; 13,088 scores, blank 13,087 |
| Context | 56 left encoder frames; right contexts 0, 3, 6, 13 (default 3) |

Contexts give nominal blocks of 80, 320, 560 and 1120 ms, not measured latency.
Context 1 is mentioned on the card but absent from the pinned configs, so it is
rejected. Prompt slots come from the published dictionary (`fr-FR`/`fr`: 8,
`fr-CA`: 100, `auto`: 101); recognition coverage follows the card's language tiers.
Fixtures contain unmodified public configurations; neural weights stay external.

Semantics follow the Apache-2.0 [official Transformers source](https://github.com/huggingface/transformers/tree/944e22eb226890dec60299386579275a66b3b44d/src/transformers/models/nemotron3_5_asr)
and its common `nemotron_asr_streaming` encoder and `parakeet` generation code.

## Tests and optional reference campaign

```bash
cargo test -p candle-transformers --lib nemotron3_5_asr
```

Ordinary CI runs ten small Rust tests and builds the CLI, without Python, models
or audio. `workflow_dispatch` can additionally enable `full_reference`; it is
disabled for ordinary pushes. Reproduce that campaign in a separate Python 3.12
environment for **qualification only**, with `ffmpeg` available:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install \
  'transformers @ https://github.com/huggingface/transformers/archive/944e22eb226890dec60299386579275a66b3b44d.tar.gz' \
  librosa soundfile numpy pyarrow
NEMOTRON_REFERENCE_PYTHON=python bash tools/nemotron_asr_streaming/qualify-reference.sh \
  "$NEMOTRON_MODEL_DIR" /absolute/new/path/cpu-reference
```

The campaign downloads pinned CAMEO/CaFE data (CC-BY-NC-SA-4.0), extracts one
public French clip and repeats it for cache eviction. Reference and candidate
run sequentially in eight cases. Optional CI uploads metrics only, excluding audio,
weights and activations. Bounds are `|candidate-reference| <= atol + rtol*|reference|`,
with `atol=rtol=1e-4` for qualified CPU F32 Mel/encoder stages. Bitwise equality
and cosine similarity are diagnostics.

## Existing implementations audited on 2026-10-23

No matching module or PR was found in `huggingface/candle`; unpublished or
unindexed implementations may still exist.

| Project | Existing target / difference |
| --- | --- |
| [nanonite/nemotron-candle-port](https://github.com/nanonite/nemotron-candle-port) | Earlier English model, Candle 0.10, remapped weights and desktop app |
| [info-wordcab/asr-rs](https://github.com/info-wordcab/asr-rs) | Earlier English Nemotron, converted NeMo weights, CPU-oriented Candle fork/server |
| [ervanalb/nemotron-asr-rs](https://github.com/ervanalb/nemotron-asr-rs) | Rust FFI to GGML/C++, not Candle |

No application from these projects is vendored or required.
