# PocketTTS SM61 standalone — P2-R5-A0 operator gate

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
- **Not implemented:** model weight loading, FlowLM transformer/LSD, complete
  Mimi decoder/SEANet, streaming KV caches, waveform generation, numeric
  model parity, GPU throughput or audio quality.
- **Not yet measured:** CUDA execution on sm_61 or any speedup.

## CPU gate

From the Candle repository root:

```sh
cargo test -p candle-examples --example pockettts-sm61 -- --nocapture
cargo run -p candle-examples --example pockettts-sm61 -- --device cpu
```

Expected final marker:

```text
POCKETTTS_SM61_A0=PASS backend=cpu dtype=f32 flow_projection=PASS mimi_projection=PASS grouped_ct1d=PASS stream_overlap_max_abs=0
```

## CUDA SM61 gate

On a machine with a compatible CUDA toolkit and a Pascal SM61 GPU:

```sh
CUDA_COMPUTE_CAP=61 NVCC_CCBIN=gcc-14 \
  cargo run --release -p candle-examples --example pockettts-sm61 \
  --features cuda -- --device cuda
```

The CUDA gate must execute successfully on the actual GPU. A successful
compilation alone is not evidence of runtime compatibility. The code uses
f32 deliberately to avoid unsupported Pascal BF16 paths.

Do not promote this initial operator gate as a full FlowLM/Mimi CUDA engine.
The subsequent gates should implement and verify inference with real,
frozen model weights, streaming state and end-to-end latency before
performance claims.

## Dependency boundary

The code is confined to the standalone Candle branch. External
applications and their test harnesses are intentionally out of scope.
