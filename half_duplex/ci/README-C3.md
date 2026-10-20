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
