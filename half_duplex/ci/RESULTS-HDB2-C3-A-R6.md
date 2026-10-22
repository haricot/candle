# HDB2-C3-A-R6 — verified encoder padding-mask results

Date: 2026-10-20. Branch: `half_duplex`.
Tested source: `ea021b7b8df8634e6620498798baa57ffb7e5410`.
Workflow: https://github.com/haricot/candle/actions/runs/39449511991
Artifact: https://github.com/haricot/candle/actions/runs/39449511991/artifacts/12278613843

**Padding-mask attribution confirmed in the tested scope.** Applying both
native attention and convolution padding masks closes the remaining final
encoder discrepancy on the real CaFE clip with 51 valid / 52 total frames.
Neither mask alone passes the fixed numerical threshold.

The workflow passed: 133 Python contract tests, CPU Rust compilation, real
corpus checks, 3 native R6 Rust unit tests and all 8 real native forward passes
(4 modes x 2 clips). The artifact contains 16 JSON reports. Its SHA256 is
`771740cb12b9506bd5636e95ef3abc8e02f5c6beef3973afd8e8f25e81858994`.

## Clip 1 — one padded encoder frame

Same original NeMo checkpoint, exact shared Mel bytes, lengths and native R5
positional math. Values compare valid frames at the final layer (`layer_23`).

| Mode | max_abs | RMSE | cosine | Fixed numerical gate |
|---|---:|---:|---:|---|
| No masks | 0.2231819704 | 0.01195931845 | 0.9750901218 | MISMATCH |
| Attention only | 0.2014631703 | 0.01322838881 | 0.9694557302 | MISMATCH |
| Convolution only | 0.02134536952 | 0.003164997682 | 0.9982781697 | MISMATCH |
| Both | 5.103647709e-7 | 5.248183370e-8 | 0.99999999999953 | PASS |

Thresholds remain max_abs <= 1e-3 and cosine >= 0.999. With both masks, all
24 encoder blocks pass the same numerical gate; there is no differing block.
Attention alone slightly worsens cosine. Convolution masking removes most of
the error, while attention masking is also needed to meet the gate.

NeMo's actual boolean masks match the native masks exactly, including blocked
keys AND padded queries. The convolution mask is applied at the genuine NeMo
boundary: after GLU, before depthwise convolution. On block 0, the padded
pre-depthwise value reaches max_abs=40.03883362 with attention enabled; the
convolution mask makes it exactly zero. Its complete depthwise input then
passes the numerical comparison with the actual observed NeMo input.

## Clip 0 — no padding (50 / 50 frames)

All four modes reproduce the R5 baseline **bit-for-bit on all frames of all
26 encoder stages**. Their final max_abs is unchanged at 3.87430191e-7,
cosine=0.99999999999977. All 24 encoder blocks pass numerically.

## Controls

- Mode OFF reproduces all 26 complete R5 stage tensors bitwise on both clips.
- Pre-encoder, post-positional output and positional table remain bitwise
  identical to R5 across every mode on both clips.
- All 8 runs compute native masks exactly matching the actual NeMo masks.
- Recursively eval-mode NeMo references retain the exact C1/C3 self-check.
- R6 uses a separate source tree and Cargo target; the R5 executable SHA
  remains unchanged across the R6 build.
- Native unit tests verify padded keys/queries, fully masked softmax rows,
  replacement of huge blocked scores, head broadcasting and GLU-boundary
  zeroing. No internal encoder activation or positional table is injected.

These controls isolate the mask treatment as the cause of the improvement
in this two-clip CPU experiment.

## Qualification boundary

**Final encoder output and all 24 encoder blocks are numerically aligned on
both clips with both masks enabled.** This does not establish bitwise equality
of the whole pipeline. The earlier pre_encode and post-scaling absolute
threshold failures remain unchanged, as do R5's non-bitwise positional table
and long-sequence positional threshold failures.

This run measures an isolated native encoder candidate. TDT numerical parity,
normal end-to-end ASR quality, streaming caches and GPU are not qualified by R6.
