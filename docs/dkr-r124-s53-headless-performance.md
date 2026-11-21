# R1.24 S5.3 — Headless performance qualification (candidate)

Baseline: sealed S5.1 native catalogue and completed S5.2 numeric result, both unchanged.
This stage adds only test-only Rust plus Python orchestration. It must NOT rebuild
PTX/FFI, alter the catalogue, or modify production Candle code.

Protocol:
- Build a candle-nn test archive with `cuda-dkr-embedded` outside the headless run.
- Verify S5.1 seal fingerprints and S5.2 numerical evidence before building and replaying.
- In headless mode repeat three paired cold/warm runs. Per pair, create an
  empty `CUDA_CACHE_PATH`, run the archive in one new process ("cold"), and
  run a second new process with the same `CUDA_CACHE_PATH` ("warm").
  No global cache deletion; preserve other projects' cache state.
- Every run executes seven cases: Q4K MMVQ batch 1, Q8_0 MMVQ batch 8,
  Q4K MMQ batch 16, Q8_0 MMQ batch 32, GGUF MoE Q4K topk2 prefill,
  SIMT F16 MoE topk1 decode, SIMT F16 MoE topk2 prefill.
- For each case prepare weights and inputs before the timed operation;
  record first synchronized call, 24 warmups, 80 synchronized wall-clock
  samples, and per-process median, p95, min, max. The report aggregates
  medians over the three processes per phase, plus effective
  tokens/second estimators based on already-prepared tensor batches.
- Verify each process loaded the expected native SM61 modules, capture
  GPU clocks/temperature/P-state/VRAM before and after, keep immutable
  archive hashes, and retain all raw Nextest logs and private cache directories.

Measurement constraints:
- "cold" only means an empty, isolated CUDA JIT disk cache path for that
  pair. It does not clear driver-internal, OS, power-state or GPU caches.
- Host-synchronized duration includes kernel launch orchestration, output
  allocation, workspace management, and synchronization. It is NOT a pure
  CUDA-event kernel duration, and the tokens/s values are workload
  row throughput rather than language-model decode throughput.
- Nextest extraction is external to the Rust timing windows. Those
  overheads remain visible in the outer logs, not in the reported sample.
- No artificial performance threshold or claims of speedup before data.
  S5.4 will analyze VRAM and cache residence independently.
- Residual architecture-specific Rust recipe policy is still OPEN (S5.5).
