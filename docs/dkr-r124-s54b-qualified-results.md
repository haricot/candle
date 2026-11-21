# R1.24 S5.4b — Embedded CUBIN runtime qualification: GPU PASS

Qualification performed on 2026-11-21, NVIDIA GTX 1080 Max-Q
(SM61, driver 580.196.07), headless-relogin v0.1.4.

## Evidence

- Source HEAD under test: `65bb4672256ca4adef6c4ccb7d3cb72307adc172`.
- S5.4b archive SHA-256:
  `ff99fb29c1b2141069ad26fb58db006d09ce882c9bd0f84b76b1de4350403e67`.
- Local test report:
  `target/dkr-r124-s54b-qualification/headless-20261121T155006810716Z/result.json`.
- headless-relogin: run_id `20261121-165004-1000-177653`,
  process exit 0, marker `DKR_R124_S54B_PASS`.
- 14/14 selected numerical Nextest tests PASS (nine BF16 tests, S5.2
  core, quantized and GGUF MoE tests, two SIMT MoE tests). Two
  separate negative admission tests PASS.
- `DKR_R124_S54B_NUMERIC=PASS cases=120 gguf_moe=6`.
- `DKR_R124_S54B_CUBIN_MODULES=25/25`: each of the 25 families
  actually loaded via Candle in the `format=cubin` path, with PTX
  provenance and CUBIN SHA-256 traces. No missing family.
- `DKR_R124_S54B_PERF=PASS rounds=3 cases=7`.
- CUBIN catalog admission is `source=embedded images=25 target=sm61`.
  Strict runtime file source/directory overrides rejected.

## Measured first-call savings

The reference is a **separate** S5.3 cold-PTX run. New values are
medians over 3 S5.4b fresh-process cold CUBIN runs. These are
synchronized host wall-time calls, **not** pure GPU kernel-event
measurements and **not** end-to-end model startup times.

| Workload | S5.3 PTX cold first (ms) | S5.4b CUBIN first (ms) | PTX/CUBIN |
|---|---:|---:|---:|
| mmvq_q4k_b1 | 7088.096 | 3.235 | 2191.065x |
| mmvq_q8_0_b8 | 0.066 | 0.061 | 1.082x |
| mmq_q4k_b16 | 3202.765 | 4.775 | 670.736x |
| mmq_q8_0_b32 | 2962.755 | 0.781 | 3793.540x |
| moe_gguf_q4k_t8 | 289.437 | 0.228 | 1269.461x |
| moe_simt_f16_decode_t3 | 141.997 | 0.345 | 411.586x |
| moe_simt_f16_prefill_t8 | 0.055 | 0.051 | 1.078x |

The second workload using a previously loaded module does not
experience the original JIT latency. Ratios are only meaningful for
the first-touch workloads and must not be summed.

## Stabilized P50 per operation (microseconds)

| Workload | PTX S5.3 P50 | CUBIN S5.4b P50 | PTX/CUBIN |
|---|---:|---:|---:|
| mmvq_q4k_b1 | 27 | 26 | 1.038x |
| mmvq_q8_0_b8 | 34 | 32 | 1.062x |
| mmq_q4k_b16 | 46 | 44 | 1.045x |
| mmq_q8_0_b32 | 50 | 47 | 1.064x |
| moe_gguf_q4k_t8 | 39 | 36 | 1.083x |
| moe_simt_f16_decode_t3 | 50 | 48 | 1.042x |
| moe_simt_f16_prefill_t8 | 50 | 48 | 1.042x |

The PTX reference recorded GPU graphics clocks near 1227 MHz,
while the S5.4b CUBIN campaign recorded P0/1354 MHz, memory
5005 MHz and temperatures 47–50 degrees C. The small P50 differences
should **not** be attributed to CUBIN without matched clocks and
within-session A/B experiments. GPU memory outside the benchmark
process was 11 MiB; this is not the transient in-process peak.

## Acceptance and remaining scope

**S5.4b qualified, opt-in only.** The feature
`cuda-dkr-cubin-embedded` can serve as the SM61 experimental runtime
path for the qualified operation matrix. The PTX-only
`cuda-dkr-embedded` build is retained as an independent baseline
and rollback.

The numerics reuse S5.2 tolerances and deterministic inputs, not
bit-for-bit PTX/CUBIN equality. They do not cover every symbol,
shape, FP8/FP4 adversarial input, model-level inference or
complete end-to-end boot. No runtime source override or missing-CUBIN
fallback is permitted in strict mode.

Before wider/default promotion, perform a direct matched-input
PTX-vs-CUBIN comparison (especially FP8/FP4/MMQ), a model-level
cold-start qualification with matched clocks, and check embedded
binary size and memory tradeoffs. S5.5's remaining hard-coded
SM61 launch recipes inside production Rust are an independent,
still-open architectural gate.
