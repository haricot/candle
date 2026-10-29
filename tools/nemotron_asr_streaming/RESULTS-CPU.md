# Native Nemotron 3.5 ASR — CPU F32 qualification, 2026-10-23

**PASS within the tested scope:** eight end-to-end comparisons against the
unmodified official Transformers generator. Tokens, text and every RNN-T
token/blank/frame-advance decision match exactly within each protocol.
All dumped Mel and FastConformer activations are finite and meet the bounds below.
Machine-readable measurements: [results-cpu-f32.json](results-cpu-f32.json).

## Provenance

- Official model `nvidia/nemotron-3.5-asr-streaming-0.6b`, revision
  `ea30d66debe3740a08b573244286791d423d6b3e`.
- Unchanged `model.safetensors`, SHA-256
  `9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174`:
  655 F32 tensors / 637,997,088 parameters, loaded with native HF names.
- Reference: Transformers `944e22eb226890dec60299386579275a66b3b44d`,
  PyTorch `2.8.0+cpu`, Python 3.12.14, CPU F32, SDPA, two computation threads.
  No reference kernels were replaced. A read-only encoder hook records actual
  encoder blocks from the official `generate` calls.
- Candidate: native Rust/Candle F32, two Rayon threads, RustFFT frontend,
  Rust tokenization, no ONNX/Python runtime. Encoder observations come from
  actual streaming blocks, without accumulated-audio replay.
- Public CAMEO/CaFE revision `c7f2fc9f0538ca871d38e183ba8ad6c329120ecc`,
  CC-BY-NC-SA-4.0; `cafe_12`, neutral French, minimum `file_id`.
  PCM SHA-256 `c333bd62e3224931788c8ec8b379161d5c62220d9794d8666636aee480bb8baf`.
  Source/attribution: <https://huggingface.co/datasets/amu-cai/CAMEO>.

The public clip contains 55,649 mono PCM16 samples at 16 kHz (3.478 seconds).
It is repeated once, without a reset between repetitions, for 111,298 samples
(6.956 seconds). This exercises eviction of the 56-frame attention cache.
The repetition tests stream mechanics; it does not add corpus diversity.
Weights, audio and raw activations are not committed.

## Exact discrete results

| Input | Protocol | Prompt | Lookahead | Encoder frames | Tokens | Decisions | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| French clip | Streaming | fr-FR | 3 | 48 | 12 | 60 | Exact |
| French clip | Offline | fr-FR | 3 | 45 | 15 | 60 | Exact |
| French clip | Streaming | auto | 3 | 48 | 12 | 60 | Exact |
| French clip | Offline | auto | 3 | 45 | 15 | 60 | Exact |
| Repeated clip | Streaming | fr-FR | 0 | 88 | 25 | 113 | Exact |
| Repeated clip | Streaming | fr-FR | 3 | 88 | 25 | 113 | Exact |
| Repeated clip | Streaming | fr-FR | 6 | 91 | 25 | 116 | Exact |
| Repeated clip | Streaming | fr-FR | 13 | 98 | 25 | 123 | Exact |

The clip is transcribed as `Un cheval fou dans mon jardin` in streaming and
`Un cheval fou dans mon jardin.` offline, identically in both implementations.
The difference in punctuation and frame counts is between protocols, not engines.
Streaming decodes the full zero-Mel-padded final block, as the official generator does.

## Floating-point measurements

Elementwise acceptance: `abs(candidate-reference) <= 1e-4 + 1e-4*abs(reference)`
for Mel and encoder outputs. Bitwise differences and cosine similarity are
diagnostics, not acceptance requirements.

| Stage / protocol | Largest absolute difference |
| --- | ---: |
| Log-Mel, all cases | 1.9836426e-4 |
| Encoder, short streaming | 4.3362379e-6 |
| Encoder, short offline | 4.4256449e-6 |
| Encoder, repeated / context 0 | 5.9604645e-7 |
| Encoder, repeated / context 3 | 4.3362379e-6 |
| Encoder, repeated / context 6 | 9.8347664e-7 |
| Encoder, repeated / context 13 | 6.0796738e-6 |

Mel differences reflect F32 FFT/window/filter reduction order; logarithms amplify
relative power errors in low-energy bins. The absolute floor covers values near
zero, while the relative term scales with log-Mel magnitude. Although the largest
absolute Mel difference exceeds `1e-4`, every element passes the combined bound.
For the short clip its maximum error/bound ratio is 0.120 or less.
Encoder normalization/matmul reductions have another order in Candle and PyTorch;
the proposed CPU F32 bound leaves margin above observed differences while exact
decisions independently guard behavior. These are empirical bounds for this
qualification, not universal mathematical error guarantees or CUDA thresholds.
A new material divergence should be isolated before increasing them.

## State and integration checks

Ten Rust unit tests pass: real config/prompt contracts, projection order, missing
bias rejection, PCM packet invariance, bounded frontend storage, chunk masks after
cache eviction, greedy ties/nonfinite rejection, forced RNN-T advance, independent
sessions, invalid-state rejection, and final flush behavior.

An additional real-model repeated-clip run with 7 ms PCM packets (instead of
20 ms) matches the official context-3 result exactly using the CLI's
`--expect-result` check. The downloader verifies all six local model files.

CUDA, SM61 runtime compatibility, F16/BF16 accuracy, diverse speakers/languages,
long conversations and real-time latency are **not established by this report**.
The local native qualification script is the next hardware check.
