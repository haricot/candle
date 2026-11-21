# R1.24 S5.5b — Producer-owned FP8/FP4/BF16/Routed recipes: qualified SM61

**Status: GPU PASS, opt-in only.** Evidence supplied on 2026-11-21.
Target: NVIDIA GeForce GTX 1080 with Max-Q Design, compute capability 6.1;
driver 580.196.07. Headless run ID `20261121-205015-1000-86510`,
exit code 0, finished 20:50:51 +01:00.

## Immutable evidence and provenance

- Qualified source HEAD: `04d9cf382c363a5553c87e7d668e869a5290e111`.
- Rust Nextest archive SHA-256:
  `50b78cb997afbff1bdd35d50d7662efbf9088d5b95eafaf2b5dbbe6717ef4ac2`.
- S5.5b derived four-recipe manifest SHA-256:
  `f1042023e52e45c7aeb01b9727393e293fa62eb97031dfb31ae948c40c62363a`.
- S5.5a MMQ manifest SHA-256:
  `dce36892279f1d636c63c902c99811184cb9591ebda39fd06f1cf560cbe263c9`.
- User-local result:
  `target/dkr-r124-s55b-qualification/headless-20261121T195017827614Z/result.json`.
- User-supplied headless log:
  `Texte collé(20261121-201100).txt`, run ID as above.
- Exact report schema: `CANDLE-DKR-R124-S55B-RESULT-V1`,
  `status: PASS`.

No PTX, CUBIN, native catalogue index or historical qualification artifacts
were rebuilt or modified by this documentation commit.

## Acceptance gates

| Gate | Verified outcome |
|---|---|
| Typed producer-owned recipe regression | `DKR_R124_S55B_TYPED_PARITY=PASS fp8=1 fp4=1 bf16=1 routed=1` |
| S5.5a MMQ tile geometry regression | `DKR_R124_S55A_TABLE_PARITY=PASS codecs=10`; reference covers 400 combinations |
| Historical S5.2/BF16/SIMT numeric suite | 14/14 tests PASS |
| Quantized native MMVQ/MMQ | 120/120 configurations PASS (10 codecs × 3 dtypes × 4 batch sizes) |
| GGUF MoE | 6/6 codecs PASS |
| Added FP8 and FP4 CUDA suite | 6/6 PASS |
| CUBIN execution and provenance | 25/25 distinct native families actually loaded in `format=cubin`, with PTX and CUBIN SHA references |
| Runtime source-override rejection | 2/2 negative tests PASS |
| Cold performance replay | 7 workloads × 3 independent processes, 80 timed samples per workload, PASS |

The six additional FP8/FP4 tests are:
`f8_sm61_linear_simt_dp4a_cuda`,
`modelopt_native_modes_match_reference_with_partial_tiles`,
`dkr_fp4_auto_matches_selected_route_cuda`,
`cuda_mxfp4_dp4a_batch_1_to_8`,
`cuda_mxfp4_prefill_parity`,
`cuda_mxfp4_indexed_moe_parity`.

Runtime confirms:
`DKR_R124_S55B_RECIPES_CATALOG source=embedded
families=fp8,fp4,bf16,routed target=sm61`,
`CANDLE_DKR_CUBIN_CATALOG source=embedded images=25 target=sm61`,
`DKR_R124_S55A_MMQ_CATALOG source=embedded codecs=10 target=sm61`,
`CANDLE_DKR_NATIVE_CATALOG source=embedded manifests=25`.

## First-call performance

The PTX cold reference is the **separate S5.3 qualification**. S5.5b
CUBIN values are medians of the three headless runs. Each first-call
time includes synchronized host overhead and module load; these are not
pure kernel timings, nor full application/model startup.

| Workload | S5.3 PTX cold first (ms) | S5.5b CUBIN first (ms) | PTX / CUBIN | PTX P50 (µs) | CUBIN P50 (µs) |
|---|---:|---:|---:|---:|---:|
| MMVQ Q4_K B1 | 7088.096 | 3.145 | 2253.767x | 27 | 27 |
| MMVQ Q8_0 B8 | 0.066 | 0.063 | 1.048x | 34 | 34 |
| MMQ Q4_K B16 | 3202.765 | 5.246 | 610.516x | 46 | 45 |
| MMQ Q8_0 B32 | 2962.755 | 0.847 | 3497.940x | 50 | 50 |
| MoE GGUF Q4_K T8 | 289.437 | 0.252 | 1148.560x | 39 | 38 |
| MoE SIMT FP16 decode T3 | 141.997 | 0.381 | 372.696x | 50 | 50 |
| MoE SIMT FP16 prefill T8 | 0.055 | 0.054 | 1.019x | 50 | 50 |

The second operations touching previously loaded modules naturally show
little first-call improvement. Ratios represent eliminated PTX JIT,
**not** improved steady-state throughput. Recorded benchmark GPU clocks:
P0, 1227 MHz graphics, 5005 MHz memory, approximately 54°C, and
11 MiB external/background VRAM measured outside the test process.
No claim of peak in-process VRAM is made.

## Architectural decision

**Close S5.5b as qualified for the opt-in SM61 runtime path**
(`cuda-dkr-neutral-recipes` with `cuda-dkr-cubin-embedded`).
The runtime consumes verified generated recipe values instead of
enforcing their equality to fixed S31 values. S31 equality is retained
**as a test oracle**, not as the runtime admission policy.

S5.5b does **not** establish complete hardware-neutral architecture:
some recipe type definitions, GGUF packing/codec rules and legacy
S31 static policy remain in candle-kernels/Candle. Continue to
**S5.5c** to move these last dependencies into a versioned,
Cudaforge-backed producer/catalogue contract.

For default promotion, a later S5.5d gate must include matched-input
PTX/CUBIN deltas for FP8/FP4 edge cases, model-level startup,
controlled-clock A/B, and full application-level correctness checks.
The immutable baseline remains separately available.
