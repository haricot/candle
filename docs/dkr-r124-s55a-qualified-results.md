# R1.24 S5.5a — Qualified producer-owned MMQ launch table

Date: 2026-11-21. GPU: NVIDIA GeForce GTX 1080 Max-Q (SM61).
CUDA driver 580.196.07; clocks P0, 1227/5005 MHz; 60–61°C;
external VRAM reported at 11 MiB.

## Grounded evidence

- Archived Rust source commit: `9a9b709d2f160a4100dae7031835ae6dc884c195`.
- Python runner correction: `c29c4e82a88208cb3cc5cfe306987b3169417023` (script-only).
- Archive hash: `23800cc1081c614fee3d427f7dba7782ef798a000f02ce3740aac6cc53ab8f5c`.
- MMQ launch manifest hash: `dce36892279f1d636c63c902c99811184cb9591ebda39fd06f1cf560cbe263c9`.
- Local result: `target/dkr-r124-s55a-qualification/headless-20261121T192505618861Z/result.json`.
- headless-relogin run `20261121-202503-1000-73008`, exit code 0.
- Reuse evidence: `DKR_R124_S55A_ARCHIVE_REUSE=PASS source_unchanged=1`.
- `DKR_R124_S55A_PASS` and report `status: PASS`.

## Acceptance gates satisfied

| Gate | Evidence |
|---|---|
| Launch-plan parity | `DKR_R124_S55A_TABLE_PARITY=PASS codecs=10` — 10 codecs × 5 row counts × 8 batch sizes = 400 combinations, same matrix/quant grids, blocks, shared-memory sizing and template key as frozen S31 |
| Quantized numeric | 120/120 GGUF MMVQ/MMQ cases |
| MoE GGUF | 6/6 codecs |
| Nextest numeric suite | 14/14 PASS (9 BF16, 3 S5.2 and 2 SIMT MoE tests) |
| Embedded native images actually loaded | 25/25 CUBIN modules, PTX/CUBIN SHA provenance traced |
| Producer-owned MMQ recipe selection | All 10 codecs logged with `DKR_R124_S55A_MMQ_PLAN source=embedded` |
| Negative admission tests | Both filesystem-source and native-directory override tests PASS |
| Performance replay | Three rounds, seven workloads, 80 repeated measurements per workload and round |

The qualified path is the opt-in feature
`cuda-dkr-neutral-mmq` together with `cuda-dkr-cubin-embedded`.
The PTX-only and S5.4b CUBIN-only paths remain preserved.

## First-use latency, S5.3 cold PTX reference vs S5.5a CUBIN

These medians compare separate runs on the same GPU, not a matched
within-process A/B and not end-to-end model startup.

| Workload | PTX cold first (ms) | S5.5a CUBIN first (ms) | Ratio |
|---|---:|---:|---:|
| MMVQ Q4_K batch 1 | 7088.096 | 3.254 | 2178.272x |
| MMVQ Q8_0 batch 8 | 0.066 | 0.063 | 1.048x |
| MMQ Q4_K batch 16 | 3202.765 | 5.205 | 615.325x |
| MMQ Q8_0 batch 32 | 2962.755 | 0.897 | 3302.960x |
| MoE GGUF Q4_K tokens 8 | 289.437 | 0.216 | 1339.986x |
| MoE SIMT FP16 decode tokens 3 | 141.997 | 0.348 | 408.037x |
| MoE SIMT FP16 prefill tokens 8 | 0.055 | 0.055 | 1.000x |

Stabilized P50 is essentially unchanged: MMVQ Q4_K 27 µs,
MMQ Q4_K 46 µs, MMQ Q8_0 50 µs. Substantial ratios measure
**elimination of the JIT first-touch cost**, not better steady-state
kernel throughput; second operations that reuse a loaded module
naturally have little to gain.

## Decision

**S5.5a PASS.** The generic MMQ table interpreter in
`dkr-kernel-registry` produces the qualified S31 launch geometries
from a build-verified, producer-authored table. Existing sealed
25-family native artifacts and image transport remain intact. This
is evidence for **MMQ launch-policy extraction**, not full architectural
neutrality yet.

Pending S5.5b/c: FP8, FP4, BF16, MoE/Routed producer-owned typed
launch descriptors, removal of remaining frozen S31-specific
recipe assumptions from production Candle, followed by a broader
configuration and model-level regression gate. Keep default
promotion separate from experimental qualification.
