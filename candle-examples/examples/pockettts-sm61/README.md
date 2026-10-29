# PocketTTS SM61 standalone — P2-R5-A1 streaming component

This is the first **independent Candle** feasibility gate for a streaming
FlowLM/Mimi inference implementation on NVIDIA Pascal sm_61. It contains no
application-specific policies, voice/emotion presets, or external baseline
references.

## Status

- **Implemented:** deterministic f32 CPU operator contract for a FlowLM-shaped
  linear projection, a continuous Mimi latent 32-to-512 Conv1d projection,
  and depthwise grouped ConvTranspose1d (512 groups, kernel 32, stride 16).
- **Implemented:** two successive one-frame ConvTranspose1d windows are
  overlap-added and compared exactly with the equivalent two-frame call.
- **Not implemented:** full model checkpoint loading, FlowLM transformer/LSD, complete
  Mimi decoder/SEANet, streaming KV caches, waveform generation, numeric
  model parity, GPU throughput or audio quality.
- **Not yet measured:** CUDA execution on sm_61 or any speedup.

## CPU gate

From the Candle repository root:

```sh
cargo test -p candle-transformers --lib pockettts::tests -- --nocapture
cargo test -p candle-examples --example pockettts-sm61 -- --nocapture
cargo run -p candle-examples --example pockettts-sm61 -- --device cpu
```

Expected final marker:

```text
POCKETTTS_SM61_A1=PASS backend=cpu dtype=f32 flow_projection=PASS mimi_projection=PASS grouped_ct1d=PASS stream_overlap_max_abs=0
```

## Extended CUDA SM61 gate (pending orchestration integration)

A standalone checkout based on main is **not** a supported Pascal CUDA
runtime: it does not include the legacy SM61 extensions. GPU compilation
and physical execution must be done against an **immutable Extended
integration candidate** after the orchestration incorporates this
standalone source. The current Extended composite has a fixed six-source
manifest, so the extra source must be explicitly registered and validated.

Do not interpret the CPU gate as GPU compatibility or throughput evidence.

## A1 streaming contract

The reusable model module at candle-transformers/src/models/pockettts.rs
contains the checkpoint-loadable FlowLM input and Mimi continuous-latent
projections and a stateful, bias-aware grouped ConvTranspose1d overlap-add
operator. Streaming commits T*stride samples per chunk and flushes the
remaining tail once. Library unit tests cover variable chunk sizes,
a nonzero convolution bias, exact offline-vs-streaming agreement in the
deterministic CPU scope, and invalid state/configuration rejection.

It is not a full FlowLM/Mimi decoder. Model weights, stateful transformer,
SEANet layers, PCM parity and physical GPU performance are still pending.

## Dependency boundary

The code is confined to the standalone Candle branch. External
applications and their test harnesses are intentionally out of scope.
