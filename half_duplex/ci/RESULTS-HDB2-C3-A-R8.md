# R8 CPU — measured 14-clip results, 2026-10-21

Source: [`ce3461ff`](https://github.com/haricot/candle/commit/ce3461ffac311c91346801b5a7b5bd811c9cac60).
[Run 39566816910](https://github.com/haricot/candle/actions/runs/39566816910)
completed all 14 real CaFE/Oréau clips without a runtime error. The real ASR job
failed its qualification gate as intended: `NUMERIC_PARITY_NOT_ESTABLISHED`.
A completed campaign does not establish all-stage numerical or bitwise parity.

Artifact `half-duplex-real-cpu-cafe-reports`, ID `12331416073`, ZIP SHA256
`f509bb181eaf39efb457804b75f44e75b314bd139bd422ea73f482dd9d113fed`.
Original checkpoint SHA256
`0acdaf1e1ebf09094400263006d134d33b0ebb871ebb064253cfd3a7c528903c`.

## Controls and passing results

- Independent R7/R8 native observation controls pass bitwise on all 14 clips,
  including observed encoder stages, predictor states, logits and TDT results.
- Actual NeMo frontend and encoder instrumentation is byte-neutral on all clips.
  Derived common-input frames reproduce the actual NeMo STFT bytes.
- All 11 frontend stages pass unchanged numerical gates on all 14 clips.
  The first bit difference is the symmetric Hann window, max abs 3.7252903e-7.
  The worst final log-Mel absolute error is 2.5367737e-4.
- Echo-verified identical inputs and weights: all five subsampling convolutions
  and all three ReLUs pass numerically on all clips (112/112 comparisons).
- Valid final encoder output passes on all 14 clips; worst max abs 9.2387199e-7.
- All 283 native TDT decisions match NeMo: tokens, frames, duration, advance and
  forced-advance flags. Shared-Mel decisions also match. Common-input TDT
  operators pass numerically with exact argmax on all 14 clips.

## Remaining differences

The common-input final subsampling linear projection fails the unchanged
`max_abs <= 1e-3` gate on six clips (worst 2.197265625e-3). Its arithmetic is
therefore a local source of error, independently of frontend drift. Autonomous
pre-encode errors reach 9.7045898e-3; the following sqrt(640) scaling amplifies
them. Earlier stages and block-0 internals still exceed absolute gates; layer 12
also fails on Oréau surprise (1.3122559e-3). All 24 block outputs pass on 13/14
clips. Final output passes on every clip. No thresholds were relaxed.

Final text is exact on 12/14 clips. Oréau neutral and happiness have identical
IDs, frames, durations and all TDT decisions, but different decoded text.
This localizes the remaining text discrepancy after token selection; its exact
cause is not established by the R8 public metrics, which omit raw text and IDs.

| Clip | Common linear max abs | Final encoder max abs | Text exact |
|---|---:|---:|:---:|
| cafe-anger-6794242ad35a | 0.0009765625 | 7.3807314e-07 | yes |
| cafe-disgust-ae49ff259f21 | 0.00219726562 | 9.23871994e-07 | yes |
| cafe-fear-ee405e4aa7b7 | 0.0009765625 | 6.78002834e-07 | yes |
| cafe-happiness-7cf7df31b5c7 | 0.00122070312 | 3.7997961e-07 | yes |
| cafe-neutral-3863734a5ee4 | 0.000732421875 | 4.87081707e-07 | yes |
| cafe-sadness-343548448bb1 | 0.00122070312 | 4.54485416e-07 | yes |
| cafe-surprise-e70c8fa23bb9 | 0.001953125 | 3.60305421e-07 | yes |
| oreau-neutral-bd77387a7b93 | 0.000732421875 | 2.68220901e-07 | no |
| oreau-surprise-17c071bdf88e | 0.00122070312 | 3.12924385e-07 | yes |
| oreau-sadness-51a5eef70768 | 0.0009765625 | 3.57627869e-07 | yes |
| oreau-happiness-40d7959b6532 | 0.0009765625 | 3.12924385e-07 | no |
| oreau-disgust-bd4f77f5c27b | 0.000732421875 | 3.27825546e-07 | yes |
| oreau-anger-aa4ce25e2ed2 | 0.0009765625 | 2.91503966e-07 | yes |
| oreau-fear-264f4f372712 | 0.00146484375 | 6.1839819e-07 | yes |

Strict global FP32 bit equality fails on all 14 clips, separately from numerical
and discrete-decision qualification.

## Attribution limit and follow-up

R8 reconstructs each native operand from the preceding output, omitting the
production temporal mask applied between operations. Consequently its full
`input_difference` can include padded positions and cannot establish that an
error is exclusively inherited from the frontend. The common-input replay
itself consumes and echoes the actual NeMo operands, so its local operator
comparison remains valid. R8B records native operands after masks and replaces
that attribution with an observation of the actual inputs. It also compares
both FP32 linear outputs against independent FP64 dot products rounded to FP32,
and publishes tokenizer-category/codepoint diagnostics without transcripts.
These supplemental checks do not change autonomous arithmetic or existing gates.

CUDA/cuDNN sm61 and local microphone are prepared, **not yet measured**.
Instructions: [README-R8.md](README-R8.md). No streaming, production promotion,
GPU parity or microphone qualification follows from these CPU results.
