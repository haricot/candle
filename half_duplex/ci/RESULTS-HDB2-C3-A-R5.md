# HDB2-C3-A-R5 — verified real CPU results

Date: 2026-10-20. Branch: `half_duplex`.

Tested source commit: `d15fc800a68355cca0e8089fc3b1e5343c956dcb`.
Workflow: https://github.com/haricot/candle/actions/runs/39440057205
Artifact: https://github.com/haricot/candle/actions/runs/39440057205/artifacts/12273838314

The workflow completed successfully: 125 Python contract tests, CPU Rust
compilation, real corpus checks and both real NeMo/Candle CaFE comparisons.
Both R5 Rust binaries compiled and executed. The artifact contains 14 JSON
reports, including `c3_positional_0.json` and `c3_positional_1.json`.

**Overall strict positional parity: NOT_ESTABLISHED.** A successful diagnostic
workflow is not a numerical or bitwise qualification of the whole encoder.

## Reference correction confirmed on both clips

After NeMo `transcribe()`, the parent model was in eval mode but **700 child
modules were in training mode**, including both positional dropouts at p=0.1.
R5 recursively restores eval and verifies zero training modules before fresh
C1 and C3 probes. Their valid-frame NeMo self-comparison is exact: max_abs=0,
RMSE=0. The original R4 positional max_abs=1 therefore used an invalid
inference-state reference; its values must not be treated as a native-math
A/B against the newly corrected reference.

## Native positional operator

Per clip: 17 common-input cases (actual encoder input plus short/long operator
fixtures, both scaling modes and offsets). Across both clips:

- Scaling output: **34/34 bitwise identical**, max_abs=0.
- Positional table: 13/17 cases per clip pass the fixed 1e-5 numerical gate.
- Four T=513 cases per clip reach max_abs=3.0517578125e-5 and fail that gate.
- Positional byte equality is not established. No tolerances were relaxed.
- The live positional table passes the numerical gate on both real clips:
  max_abs=3.807246685028076e-6; RMSE is 1.17588e-7 / 1.23188e-7.

The native f32 frequency-order treatment reduces differing f32 table words
on clip 0 from 25,354 (R4) to 3,418 (R5), but this is not exact parity.
The remaining error has not been independently attributed between exp and
sin/cos implementations.

## Uninjected encoder measurements

Both engines consume byte-identical NeMo Mel and exact valid lengths. No
positional tables or internal activations are injected into the full encoder.

| Quantity | Clip 0 | Clip 1 |
|---|---:|---:|
| Valid / total encoder frames | 50 / 50 | 51 / 52 |
| pre_encode max_abs | 0.00146484375 | 0.001220703125 |
| post-scaling max_abs | 0.03125 | 0.03125 |
| layer_23 max_abs | 3.87430191e-7 | 0.2231819704 |
| layer_23 cosine | 0.99999999999977 | 0.9750901218 |
| layer_23 numerical gate | PASS | MISMATCH |

The fixed pre-encoder 1e-3 absolute threshold is exceeded on both clips,
although its cosine is ~1. Thus **all-stage encoder parity is not qualified**.
The strong remaining final-layer discrepancy occurs on the clip with one
padded encoder frame. Attention/convolution padding-mask parity is the next
hypothesis to test; R5 alone does not prove that attribution.

## Instrumentation and build control

- R3C vs R4: **exact on all 26 valid-frame encoder stages, both clips**.
- R4 vs R5 pre_encode: bitwise identical on both clips.
- R4 executable SHA remains unchanged across the independent R5 build.
- The first attempt (39436600013) used a shared Cargo target and stopped at
  the second clip's exact instrumentation gate. R5 now has its own complete
  Cargo target; the corrected run passes that unchanged gate on both clips.

TDT numeric parity, GPU parity and live microphone quality remain outside
this CPU positional experiment.
