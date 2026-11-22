# R1.24 S5.5d — Direct PTX/CUBIN numeric equivalence and Qwen3 cold-start: qualified

**Status: PASS, experimental SM61 Cudaforge/CUBIN path.** Date: 2026-11-22;
headless-relogin run `20261122-084953-1000-11020`, finished
2026-11-22 08:52:23 Europe/Paris, `exit_code=0`.
Qualification marker: `DKR_R124_S55D_PASS`.

**Actual source HEAD under test:** `612f5d2b8f170b26c54cdd06c3dcc58cd14fdf5e`.
This source includes the S5.5d model-probe feature propagation fix,
and its PTX/CUBIN archives and separate release model executables are
independently SHA-sealed in the qualification receipt.

## Test environment and provenance

- GPU: NVIDIA GeForce GTX 1080 Max-Q, compute capability 6.1;
  NVIDIA driver 580.196.07.
- GPU P0; clocks seen before/after runs: 1227, 1354, 1493 MHz graphics;
  memory 5005 MHz. External GPU memory: 11 MiB.
  **Clocks were not fixed or continuously recorded during kernels.**
- Model: `Qwen3-0.6B-Q4_K_M.gguf`, 396,705,472 bytes;
  SHA-256 `ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a`.
- Prompt token IDs `[1,2,3,4,5,6,7,8]`; 8 decode steps;
  greedy deterministic decoding. Model has 310 tensors, vocabulary 151,936.
- Cudaforge S5.5c catalogue SHA-256:
  `91ac006b963a89362c6aaaee052252b4465880c079b0c86f399c4891a5003295`.
- PTX Nextest archive SHA-256:
  `561b101fd545730cb9f71a4e70eb4e29fb044b5547a06a20638a7eac30321313`.
- CUBIN Nextest archive SHA-256:
  `a2b5484cdcdf5c492bc8f85fcd0486c76d0a7e593bc0597b32a898acafb39ce2`.
- PTX release Qwen3 probe SHA-256:
  `41a44ff9b6712f37400026dd8bc292671d87f3cfc584aeef0af52045261f43cf`.
- CUBIN release Qwen3 probe SHA-256:
  `070e578d2629fc667b7bacb1622379556288c96f8f5fc01d72801339b588b14d`.
- Local full report:
  `target/dkr-r124-s55d-qualification-r2/headless-20261122T074955864763Z/result.json`.
- Attached evidence: `Texte collé(20261122-075620).txt`, `result(9).json`.

## Direct numeric equivalence, identical source inputs

PTX and CUBIN compiled from the same Rust source; GPU tests used
identical tensor fixtures and CPU-quantized GGUF source-weight
byte hashes.

| Family | Complete cases |
|---|---:|
| GGUF MMVQ/MMQ (10 codecs × 3 activation dtypes × 4 batches) | 120 |
| BF16 roundtrip/matmul/reduction/unary | 4 |
| FP8 E4M3FN, SIMT and DP4A across batches | 10 |
| FP4, scalar/reference/DP4A | 15 |
| MoE GGUF/SIMT | 8 |
| **Total** | **157** |

**PASS: 157/157 cases, 76,292 output float32 elements, zero mismatches,
zero maximum absolute error, and all 157 cases have exact float32
bit-pattern parity** (`bitwise_float_equal_cases=157`).
These are direct PTX-vs-CUBIN comparisons; not merely two independent
comparisons against a third-party numerical reference.

The numeric PTX test took approximately 51.47 seconds in a fresh
cache/process, whereas the CUBIN test took approximately 0.21 seconds.
These are complete test durations, **not kernel timings**.

## End-to-end Qwen3 GGUF paired comparison

Three pairs of independently launched model binaries, alternating
test order, with private fresh CUDA JIT caches. Both binaries use
the same GGUF SHA and token prefix.

- Each of the three pairs generated the same nine token IDs:
  `3764, 3, 9, 2, 9, 2, 9, 2, 9`.
- Full vocabulary logits compared for prefill and all eight decode
  steps: `151936 × 9 = 1,367,424` float32 logits per pair;
  **4,102,272 logits in total**.
- Each pair reported **zero mismatches and max absolute error 0.0**.
  The comparison checked floating-point values under declared
  tolerances, not raw serialized bytes for signed-zero equivalence;
  do not make a bitwise logits claim from that report.
- `DKR_R124_S55D_MODEL_PAIR=PASS` rounds 1, 2 and 3;
  `DKR_R124_S55D_MODEL=PASS rounds=3 full_logits_comparison=1`.

### Model cold-start performance (median of 3 independent trials)

| Metric | PTX median | CUBIN median | PTX/CUBIN |
|---|---:|---:|---:|
| CUDA device initialization | 64.398 ms | 66.369 ms | 0.970× |
| GGUF header read | 732.555 ms | 734.445 ms | 0.997× |
| Load/construct model | 10,808.216 ms | 143.974 ms | **75.071×** |
| First prefill | 16,934.677 ms | 24.290 ms | **697.187×** |
| Total in-process work | 28,612.649 ms | 1,028.631 ms | **27.816×** |
| Outside process wall time | 28,763.855 ms | 1,157.251 ms | **24.855×** |

The enormous model-build/prefill improvements arise mainly from
**avoiding the cold PTX CUDA JIT compilation cost** through embedded
precompiled SM61 CUBINs. They **do not represent comparable
improvements in warmed-up kernel throughput**. Post-first-token
decode per-token timings in these short sequences are around 7.0 ms
(PTX) and 6.0–6.5 ms (CUBIN), but GPU clocks were not controlled
and these few samples do not establish a general throughput speedup.

## Decision and limitations

**Close R1.24 S5.5d PASS** on this exact SM61 experimental path:
- S5.5a: MMQ launch recipes PASS;
- S5.5b: FP8/FP4/BF16/MoE typed recipes PASS;
- S5.5c: Cudaforge recipe producer/catalogue attestation PASS;
- S5.5d: direct PTX/CUBIN numerical equivalence and real Qwen3
  model cold-start PASS.

This is sufficient evidence to **propose** a staged, SM61-scoped
opt-in-to-default promotion review. It is **not** itself authorization
to switch all Candle CUDA users to a new backend, delete the PTX
fallback, or claim support for other GPU architectures or models.

Scope caveats:
- One Qwen3-0.6B GGUF, eight fixed input tokens and eight greedy
  decoding steps; not production prompt diversity or multi-turn KV.
- CUDA processes and JIT caches cold, but the Linux page cache is
  **not** purged; storage cold-start unmeasured.
- Clocks, power draw and thermal behavior not force-matched; ratios
  are indicative for cold-start on this hardware, not guaranteed.
- Full model logits have zero *numeric* differences in three
  comparisons, but report does not prove all logits match as raw
  IEEE-754 bit strings.
- The catalogue is produced by an autonomous executable using
  Cudaforge, not yet a generalized multi-architecture public
  Cudaforge catalogue API. Remaining S31 compatibility code
  can be retired only after a separate controlled migration.
- The embedded catalogue contains 25 native families; an individual
  model or case loads only the families it needs.

No modifications to sealed S5.1 PTX, 25 S5.4b CUBINs, original
qualified manifests, or model GGUF are part of this evidence commit.
