# HDB2-C3-A-R7 — strict global bytes and actual production TDT

Continue on `half_duplex` from the verified R6 both-mask encoder. The user
requests global bitwise equality and TDT parity, with the existing GitHub
workflow. This revision measures those gates independently; a successful
diagnostic workflow does not itself establish either gate.

## Executions

1. Restore the original checkpoint in the pinned CPU NeMo environment,
   recursively verify eval, verify every decoder/joint exported weight byte,
   and observe the checkpoint's actual `GreedyBatchedTDTInfer` calls.
   The decoder is not replaced. Include-duration only adds output metadata.
2. Run an isolated native R7 candidate with byte-identical C1 Mel to retain
   the R6 encoder control, then decode its actual encoder output with TDT.
3. Run PCM -> native frontend -> R6 both-mask encoder -> native TDT, with
   no intermediate tensor injection. Compare all valid and padded frames,
   all decoder decisions, emitted tokens, timestamps, durations and text.
4. Separately replay actual NeMo predictor token/state inputs and joint
   encoder/predictor inputs through native operators. Verify consumed bytes
   explicitly. These operator checks cannot establish global pipeline parity.

## Source correction

Align the native batched-compatible transition with NeMo: a blank predicted
with duration zero advances one frame and preserves the committed label/state.
Keep the existing same-frame nonblank symbol cap. Reject duration head/config
size mismatches rather than clamping an invalid argmax index.

## Gates

- Strict f32: compare complete little-endian words including signed zero;
  report changed words, actual changed bits, maximum ULP and first counterexample.
- Predictor/state: max_abs <= 1e-4 and cosine >= .999999.
- Joint logits: max_abs <= 1e-3 and cosine >= .999999.
- Encoder: retain R6 numerical criteria; report exact bytes separately.
- Discrete TDT: every blank/nonblank decision, duration head argmax, frame,
  effective advance, forced advance, token sequence, token durations and
  decoded text must agree exactly.
- Global exact equality requires frontend, complete encoder and corresponding
  autonomous TDT activations to agree bitwise. Close numbers or identical
  transcripts do not satisfy it.

Use a separate source tree and Cargo target; verify the R6 binary and its
26-stage shared-Mel outputs remain unchanged. Native tests cover blank+zero,
symbol limits and positive-duration boundary cases. Only numerical JSON
reports are uploaded; raw audio, weights and activations stay in runner temp.
