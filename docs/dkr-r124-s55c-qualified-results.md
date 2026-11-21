# R1.24 S5.5c — Cudaforge catalogue, SM61 GPU-qualified

Status: **PASS (opt-in feature only)**. Evidence provided 2026-11-21,
headless-relogin run `20261121-213445-1000-113503`;
start 21:34:45, finish 21:35:25 Europe/Paris, exit_code 0.

## Identifiers and immutable sources

- Rust source HEAD **under test**:
  `788be6f7de8cae01364535d7f3a9ef27880b4c29`
- Verified Nextest archive:
  `fb319ccddcfe57cafc04aec661cd2e1f78ae5437113cfcc0e226c2045a173822`
- Attested Cudaforge catalogue SHA-256:
  `91ac006b963a89362c6aaaee052252b4465880c079b0c86f399c4891a5003295`
- S5.5a MMQ recipe manifest SHA-256:
  `dce36892279f1d636c63c902c99811184cb9591ebda39fd06f1cf560cbe263c9`
- S5.5b typed recipes manifest SHA-256:
  `f1042023e52e45c7aeb01b9727393e293fa62eb97031dfb31ae948c40c62363a`
- Cudaforge-admitted S5.5b-format result SHA-256:
  `af623dc44059cf589b75f85353881a98e2af356c2506532a4f8470689dc0a749`
- Local S5.5c report:
  `target/dkr-r124-s55c-qualification/headless-20261121T203451750539Z/s55c-result.json`
- Local full S5.5b-gate replay:
  `target/dkr-r124-s55c-qualification/headless-20261121T203451750539Z/result.json`
- Supplied files: `Texte collé(20261121-203646).txt`,
  `result(8).json`, `s55c-result.json`.

No changes to sealed PTX packages, CUBIN images, original S5.1 seal,
or the qualified S5.5a/b manifests are required for this result.

## GPU and gates

NVIDIA GeForce GTX 1080 Max-Q, SM61, driver 580.196.07;
P0/1227 MHz graphics, 5005 MHz memory, 44–49 °C during this replay,
background GPU memory 11 MiB (not in-process peak).

| Gate | Result |
|---|---|
| Cudaforge build attestation test | PASS, 25 families, four recipes, ten codecs |
| S5.5a MMQ table geometry vs S31 oracle | PASS, 400 configurations |
| S5.5b typed FP8/FP4/BF16/Routed recipes vs S31 oracle | PASS |
| BF16/S5.2/MoE SIMT historical numerical suite | 14/14 PASS |
| MMVQ/MMQ numerical cases | 120/120 PASS |
| GGUF MoE codecs | 6/6 PASS |
| Additional FP8/FP4 tests | 6/6 PASS |
| Actual embedded native CUBIN module loads | 25/25 PASS, with PTX/CUBIN digests |
| Negative runtime filesystem overrides | 2/2 PASS |
| Seven-workload microbenchmark | 3 independent headless rounds PASS |
| Cudaforge provenance runtime trace | PASS, SHA matches attested catalogue |
| Final `DKR_R124_S55C_PASS` and result status | PASS |

The S5.5c runner executes the entire S5.5b numerical/performance
replayer **on the S5.5c-feature binary** and then verifies the new
Cudaforge provenance trace. This is not using the historical S5.5b
binary or simply declaring success from an older test report.

## First-call comparison: historical PTX S5.3 vs S5.5c CUBIN

First-use medians are synchronized host-side measurements from
different headless sessions, **not** end-to-end LLM startup
and not steady-state performance claims.

| Workload | PTX cold first (ms) | S5.5c CUBIN first (ms) | PTX/CUBIN | CUBIN P50 (µs) |
|---|---:|---:|---:|---:|
| MMVQ Q4_K B1 | 7088.096 | 3.110 | 2279.131x | 27 |
| MMVQ Q8_0 B8 | 0.066 | 0.062 | 1.065x | 34 |
| MMQ Q4_K B16 | 3202.765 | 5.221 | 613.439x | 46 |
| MMQ Q8_0 B32 | 2962.755 | 0.862 | 3437.071x | 50 |
| MoE GGUF Q4_K T8 | 289.437 | 0.220 | 1315.623x | 38 |
| MoE SIMT FP16 decode T3 | 141.997 | 0.331 | 428.994x | 50 |
| MoE SIMT FP16 prefill T8 | 0.055 | 0.054 | 1.019x | 50 |

The values near 1x are expected for another operation reusing an
already loaded module. The dramatic ratios demonstrate elimination
of the expensive initial PTX JIT, not proportional improvement of
continuous inference throughput. P50 is essentially unchanged.

## Architectural acceptance and remaining restrictions

**Close S5.5c for the experimentally qualified SM61 Cudaforge
catalogue path.** The `cuda-dkr-cudaforge-catalog` feature is
still **opt-in**. The producer owns the 10-codec MMQ tables
and four typed recipe descriptors; the Candle build script
cross-attests the 25 CUBINs and the derived manifests; the runtime
uses the embedded typed data and emits the actual producer digest.

Do not claim universal hardware neutrality, production-ready
default selection, or a general Cudaforge public catalogue API.
The Cudaforge dependency is used by a standalone `candle-dkr-s31-builder`
binary; the generic Cudaforge crate has not been extended by S5.5c.
Some S31 code and original kernel ABI rules remain in Candle/
candle-kernels for compatibility and regression tests.

**Next: S5.5d**, an independent matched-input PTX-versus-CUBIN
numerical diff on BF16/FP8/FP4 edge cases and an end-to-end model
cold-start with similar GPU clocks and launch conditions. Keep the
S5.1–S5.5c results and original byte hashes immutable.
