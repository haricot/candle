# Nemotron 3.5 ASR Streaming — standalone port

Branch: `nemotron_asr_streaming_standalone`, based directly on
`haricot/candle:main` at `c68b24997319b8d76e5a4e775dab197c7f26322a`.

## Current status: N0 components, not an ASR runtime

Implemented:

- typed parsing/validation of the published HF model and processor configs;
- language prompt lookup from the published dictionary (`fr-FR` / `fr`: 8,
  `fr-CA`: 100, automatic detection: 101);
- the native Candle language-conditioning network: broadcast one-hot prompt,
  concatenation, `1152 -> 2048 -> ReLU -> 1024`, then encoder projection to 640;
- nominal audio chunk planning from the checkpoint's supported right contexts;
- small CPU tests requiring no model download, Python, ONNX or audio files.

The nonlinear network order is `Linear(1152, 2048) -> ReLU ->
Linear(2048, 1024) -> Linear(1024, 640)`.

The PCM frontend, cache-aware FastConformer, RNN-T predictor/joint, tokenizer
integration and complete transcription are **not implemented yet**. There is no
microphone/transcribe command and no real-model, streaming or CUDA parity claim.
The tests cover these initial components only, using synthetic projection weights.

## Existing implementations checked on 2026-10-23

No matching model module or pull request was found in `huggingface/candle`.
Public searches cannot prove that no unpublished/third-party implementation exists.

| Project | What exists | Difference from this target |
| --- | --- | --- |
| [nanonite/nemotron-candle-port](https://github.com/nanonite/nemotron-candle-port) | Candle 0.10 Rust model, encoder cache API, reference tools and desktop application | Targets `nvidia/nemotron-speech-streaming-en-0.6b`; exports/remaps weights; no verified 3.5 prompt support found |
| [info-wordcab/asr-rs](https://github.com/info-wordcab/asr-rs) | Candle fork with FastConformer, cache-aware Nemotron RNN-T and HTTP server | Converter explicitly targets the earlier English model and renames NeMo tensors; CPU-focused application/fork |
| [ervanalb/nemotron-asr-rs](https://github.com/ervanalb/nemotron-asr-rs) | Streaming Rust interface with GPU backends | FFI bindings to `nemotron-asr.cpp` / GGML, not a Candle implementation |
| [huggingface/transformers](https://github.com/huggingface/transformers/tree/944e22eb226890dec60299386579275a66b3b44d/src/transformers/models/nemotron3_5_asr) | Native Nemotron 3.5 ASR and common streaming encoder | Python reference for the officially published Safetensors layout |

These projects are references for the architecture audit. No third-party
application/fork has been vendored, and their claimed runtime behavior has not
been independently qualified here.

## Pinned official artifacts

Model: [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
at revision `ea30d66debe3740a08b573244286791d423d6b3e`.

The two small JSON files in `fixtures/` are unmodified public configuration
metadata from that revision, used to prevent invented defaults and language IDs:

| File | SHA-256 |
| --- | --- |
| `model-config.json` (`config.json` upstream) | `62d186fd91f518e00e7867500f1f5819225e8ee95ea3e21b546514bf2048e845` |
| `processor-config.json` (`processor_config.json` upstream) | `ec47870f1091ea4f25539208387b45b902c92d0e3f997a30061ef88f73437ab0` |

Neural weights remain external. The target is the **official published
`model.safetensors`**, using its HF tensor names directly; there is no new
checkpoint export or conversion format. The native NeMo `.nemo` artifact has
another tensor layout and is not accepted by the current projection loader.

## Architecture and streaming constraints

| Component | Official configuration |
| --- | --- |
| Audio | mono 16 kHz, 128 Mel bins, 400-sample window, 160-sample hop, FFT 512, preemphasis 0.97 |
| Encoder | 24 FastConformer layers, width 1024, 8 attention heads, FFN width 4096 |
| Subsampling | causal depthwise Conv2d, factor 8; differs from TranscribePsy's noncausal frontend |
| Conformer temporal convolution | causal, kernel 9; retain context between blocks |
| Language conditioning | 128 prompt slots; actual recognition coverage follows the model card's language tiers |
| Predictor | two LSTM layers, width 640, persistent state per stream |
| Joint | ReLU, 13,088 scores including blank ID 13,087; **RNN-T, not TDT** |
| Context | 56 left encoder frames, right context selected from the pinned config |

The pinned model and processor configs support right contexts **0, 3, 6, 13**:
nominal chunks of **80, 320, 560, 1120 ms**. Their default is 3 (320 ms).
The model card also mentions 160 ms/right-context 1; that value is absent from
both pinned JSON lists, so `chunk_plan` rejects it. A future revision can change
that contract explicitly. Chunk duration is not measured end-to-end latency.

An actual streaming port must carry bounded attention K/V, subsampling and
temporal convolution caches, frontend boundary state, and RNN-T predictor state
across blocks. It must flush the final partial block and reset every state for
each new stream. Re-running the full accumulated utterance and slicing its text
does not meet this target. Cache shapes/flush behavior will be defined from the
reference implementation, not guessed from the nominal block sizes.

## Next implementation gates

1. Native Mel and causal FastConformer with explicit streaming state. Verify
   per-stage outputs and final-block handling against the pinned reference.
2. RNN-T decoding with predictor/joint loaded under official tensor names,
   persistent LSTM state and language conditioning; original tokenizer outside
   the tensor model layer. Retain exact token/blank/frame decisions in tests.
3. One public French utterance, forced `fr-FR` and `auto`, whole-file and chunked
   CPU F32 comparisons. Use numerical activation tolerances and exact discrete
   outputs, then separately qualify CUDA through the existing Extended backend.

The standalone branch changes model code only. The half-duplex controller,
FireRedVAD, Smart Turn, microphone acquisition and GPU runner orchestration stay
in their respective projects/branches. No HDB2 generator is a model dependency.

## Tests

```sh
cargo test -p candle-transformers --lib nemotron3_5_asr
```

`.github/workflows/nemotron-asr-streaming.yml` runs these initial contracts.
It downloads Rust dependencies only, not the multi-gigabyte checkpoint.
