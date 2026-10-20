# HDB2-C3 — FastConformer & TDT Exact Parity

**Branch:** `haricot/candle:half_duplex` — CPU diagnostic, no production promotion.

## Verified C2 baseline (real CaFE/CAMEO CPU, 2026-10-19)

- Same-checkpoint real ASR NeMo and Candle C2 run successfully on two French CaFE utterances.
- Log-Mel parity PASS, RMSE ≈ 2.45e−6 / 3.81e−6, cosine ≈ 1.
- FastConformer final cosine ≈ 0.9574 / 0.9391, **numeric parity not established**.
- Candle C2 text == NeMo text on both excerpts (one reference transcript was itself imperfect).
- TDT logits/tokens/duration numeric parity remains **NOT MEASURED**.

## HDB2-C3-A — first numerical divergence

From a *real* C1 `--probes` report, with an already-built isolated C2 candidate:

```bash
python3 -m hdb2c.c3 /path/to/genuine-C1-probes-report
```

The probe compares NeMo and Candle CPU, using **exactly the same raw NeMo log-Mel tensor injected into Candle**; this isolates the encoder from frontend differences. It captures 26 time-major tensors for each engine: `pre_encode`, `post_pos_enc`, and each of 24 `layer_00` through `layer_23` blocks. The NeMo hook trace must reproduce the existing C1 NeMo encoder output; Candle's consumed Mel must be **bit identical** to NeMo input. Both are checked.

Output: `c3_layers_cpu.json`, with valid-frame-only RMSE, cosine, max absolute error, first mismatching stage, shapes, fingerprint of source PCM and NeMo checkpoint. An execution reporting `FIRST_DIVERGENCE_LOCALIZED` is a *successful diagnostic*, NOT a parity PASS.

Raw trace `.f32le` tensors remain in the runner's temporary report directory. **Only the JSON report is uploaded as a GitHub Actions artifact.** All sources and conversions are run in separate pinned runtime clones; no changes are made to the developer's working CUDA application.

### Fail-closed requirements

- Original checkpoint SHA256 matches converted weights metadata and the real NeMo report.
- Both C1 reports have the same PCM SHA256 and probe tensors exist.
- Exactly 24 NeMo encoder layer hooks execute, exactly once each.
- NeMo trace reproduces the C1 encoder output and valid lengths.
- Complete C3 Rust trace produced: no missing stages, no non-finite values.
- Same original NeMo Mel is fed to the Candle encoder and verified **bit-for-bit**.
- Stages compare **valid subsampled frames only**. A wrong feature dimension or insufficient frame count is a mismatch, not silently cropped.
- Quantitative goal (initial, conservative): `max_abs ≤ 0.001` and `cosine ≥ 0.999` per stage.

## HDB2-C3-B — TDT (not yet qualified)

After attribution of the **first** encoder divergence, feed the same NeMo encoder activations to both TDT decoders (not merely the same waveform) and compare first-step token/duration logits, blank handling, recurrent state, and greedy duration progression. Until independent token/duration trace comparison passes, `TDT_NUMERIC_PARITY=NOT_ESTABLISHED`, even if text is identical. No TDT model implementation has been changed by C3-A.

## Audio/data privacy

GitHub Actions uses CaFE/Oréau via CAMEO. Personal microphone WAVs stay in local `tests/local/` ignored by Git. Do not publish NeMo weights, personal audio, or raw activations.
## C3-A-R1: NeMo self-reference preflight

The first C3 diagnostic stopped before comparing NeMo/Candle because the fresh NeMo encoder output did not match its own C1 reference at an all-frame absolute tolerance of 1e-6. This release replays the C1 transcribe-then-encoder sequence and compares only the valid [B,D,T] frames, explicitly recording overall padding drift. Limits are max_abs <= 1e-4 and cosine >= 0.999999; any larger discrepancy still fails closed and prints actual metrics. CPU C2 and CUDA are unchanged.

## C3-A-R2 — Explicit valid-frame propagation (CPU/CaFE)

The initial C3-A diagnostics on two genuine CaFE excerpts found `pre_encode` as the first divergent stage (cosine approx 0.913 / 0.949), while the NeMo self-reference reproduced exactly. C3-A-R2 specifically rules out **missing propagation of NeMo valid Mel length**: the Rust C3 probe receives the already-verified bit-identical NeMo `[B,T,128]` Mel tensor plus `HDB2C3_NEMO_MEL_VALID_FRAMES`, validates `0 < valid <= total`, constructs a real Candle `Tensor<i64>` of shape `[1]`, and invokes `m.encoder.forward(&mel, Some(&lengths))`. It checks returned encoder valid frames against NeMo and publishes the recorded input and output lengths in the JSON stage report. The normal C2 probe and ASR remain untouched. R2 uses a separate immutable candidate `hdb2c-c3-r2-v0.7.1`.

**Important limitation:** passing explicit lengths does not by itself prove equivalence of NeMo's internal padding or attention masks. The report marks `padding_mask_parity=NOT_ESTABLISHED_LENGTH_PROPAGATION_ONLY`. If `pre_encode` still differs, proceed to R3 convolution-level instrumentation; do not label TDT or full encoder parity PASS. The real CPU job runs on public CaFE neutral + sadness only, never personal WAVs.

### R2-R1: adapt to the **actual** C2 probe source

C2 already supplies a sample-derived `valid_t` to `m.encoder.forward(&mel,Some(&valid_t))`. The first R2 real-model run failed because its source transformer expected a stale `None` call and correctly refused unknown source. R2-R1 now anchors the real post-C2 callsite, passes the **NeMo preprocessor's actual** valid frames when Mel injection is active, and preserves C2's `valid_t` as the fallback in the isolated C3 probe only. CPU/NeMo length equality is checked in the JSON report. This can confirm length equivalence; it **cannot** establish convolution padding-mask parity.

## HDB2-C3-A-R3 — DW-Striding weight and per-operation parity (2026-10-20)

The R2 CPU/CaFE run verified matching NeMo/Candle Mel lengths (394/403) and encoder lengths (50/51) but preserved first divergence at `pre_encode`. R3 captures source NeMo and loaded Candle **actual** tensors for `conv.0/.2/.3/.5/.6`, their weight+bias, ReLU activations, and projection `out` (12 weight tensors, nine operation outputs). All shapes and valid-frame counts must be comparable, with per-layer valid time windows: ceil(T/2), ceil(T/4), ceil(T/8). ReLU is instrumented by invocation count because NeMo may reuse a single module instance. The exact same NeMo log-Mel is injected in Candle. It reports weight equality separately from activation equality, with first operation divergence in `c3_dw_cpu.json` plus the original 26-stage `c3_layers_cpu.json`.

**No mathematical modifications in R3 A/B.** The suspected difference is NeMo `MaskedConvSequential` masking invalid temporal frames before each layer vs Candle's unmasked loop. Do not change the computations until the weight audit and first-operation diagnostics support the hypothesis. Numerical diagnostic success does not mean full encoder or TDT parity. Only JSON metrics are uploaded; tensors/weights, private recordings and model files are never published.

## HDB2-C3-A-R3-C — Exact Mask Propagation A/B

Two **isolated** C3 candidates run on each public CaFE sample: immutable R3 traces without temporal masks, and R3-C with the NeMo `MaskedConvSequential` policy: mask before **every** convolution and every ReLU, update valid lengths after `conv.0`, depthwise `conv.2` and `conv.5`, and apply final mask before flatten/project. Keep actual NeMo Mel and NeMo valid length identical for both. C3-R3 probes record the same loaded weights and stage activations. Report `c3_mask_ab_cpu.json` includes per-operation before/after cosine/RMSE/max_abs, pre_encode and final encoder statuses, and explicitly states TDT NOT_TESTED. JSON metrics only are uploaded; raw activations, weight dumps and WAVs remain temporary. **No changes to C2 ASR, TDT, GPU, or production code.** Parity success requires the measured thresholds; workflow success alone is not qualification.

### R3-C-R1 — Correct original Mel length, not downsampled length

The green initial run #39407278819 masked input Mel using Rust `lengths` after it had been overwritten by three stride operations, incorrectly using 50/51 instead of 394/403 frames. The resulting early `conv.0` divergence is an instrumentation artifact and cannot validate or invalidate NeMo-style masks. R1 captures original Mel lengths **before** the shadowed Rust variable is downsampled, propagates them after each stride, and unit-tests the exact source anchor. C2 ASR, the R3 unmasked control, TDT and CUDA are unchanged.
