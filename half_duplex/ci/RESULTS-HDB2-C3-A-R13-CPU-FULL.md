# R13 — full CPU qualification on 14 real clips

Candidate `11a1422803e925197959e3ff228c819f90080063` passes **14/14 complete CPU numerical qualifications**
in independent [GitHub Actions run 39668765979](https://github.com/haricot/candle/actions/runs/39668765979), completed on 2026-10-21.
The actual `real-cpu-asr` job succeeded, alongside Python contracts, Rust CPU
and corpus preparation. The isolated R12 job was skipped as expected.

All original R7/R8B numerical thresholds remain unchanged. This result covers
the integrated R13 native CPU candidate, from real PCM through the original
checkpoint frontend, all 24 encoder blocks, production TDT and rendered text.
NeMo references and independent baseline controls were freshly executed on the
CI worker. Shared-Mel and common-input replays are separate diagnostics; the
native pipeline receives no internal reference activation.

| Measurement | Independent CPU CI result |
| --- | --- |
| Complete numerical qualification | 14/14 PASS |
| Independent R7/R8B byte controls | 14/14 PASS |
| Frontend stages bit-identical | 154/154 |
| Encoder numerical stage gates, native + shared-Mel | 1400/1400 PASS |
| Native layer outputs bit-identical | 267/336 |
| Shared-Mel layer outputs bit-identical | 267/336 |
| Final valid encoder bit-identical | 11/14 clips |
| Final valid encoder maximum absolute difference | 2.384185791015625e-7 |
| Declared encoder-stage maximum absolute difference | 6.103515625e-5 |
| Production TDT decisions exact | 283/283 on each path |
| Token IDs, frames, durations and text | Exact on all 14 clips, both paths |
| TDT activation maximum absolute difference | 4.57763671875e-5 |
| Global bitwise qualification | FAIL_DIFFERENT_F32_BITS |
| GPU / microphone | NOT_RUN / NOT_RUN |

## Per-clip native results

| Corpus | Emotion | Numerical gates | Exact layer outputs | Final encoder max abs | Exact TDT decisions |
| --- | --- | --- | --- | --- | --- |
| cafe | anger | PASS | 24/24 | 0 | 22 |
| cafe | disgust | PASS | 24/24 | 0 | 34 |
| cafe | fear | PASS | 24/24 | 0 | 28 |
| cafe | happiness | PASS | 24/24 | 0 | 18 |
| cafe | neutral | PASS | 24/24 | 0 | 16 |
| cafe | sadness | PASS | 24/24 | 0 | 24 |
| cafe | surprise | PASS | 24/24 | 0 | 20 |
| oreau | neutral | PASS | 24/24 | 0 | 15 |
| oreau | surprise | PASS | 24/24 | 0 | 17 |
| oreau | sadness | PASS | 24/24 | 0 | 16 |
| oreau | happiness | PASS | 1/24 | 1.266598701e-07 | 15 |
| oreau | disgust | PASS | 1/24 | 2.384185791e-07 | 11 |
| oreau | anger | PASS | 24/24 | 0 | 25 |
| oreau | fear | PASS | 1/24 | 1.788139343e-07 | 22 |

## Numerical parity and remaining bit differences

The three Oréau clips happiness, disgust and fear retain small encoder
differences starting at layer_01 on this CI worker. Their block0 stages and
layer_00 are bit-identical. All declared encoder gates still pass. The cause
of the residual difference is not established by this report; do not attribute
it to a specific CPU dispatch without new measurements. Earlier local
measurements reported 336/336 exact layer outputs; that count must not be
transferred to the independent CI result, which is 267/336.

Subsampling ReLU signed-zero differences and TDT floating-point differences
also remain. None changes the measured production decisions or final text.
Do not describe the complete CPU pipeline as globally bit-identical.

## Reproducible evidence and scope

The [persistent JSON](results/r13-ci-full-cpu-20261021.json) includes the
unmodified campaign summary, all 14 PCM identities, original numerical limits,
per-stage encoder metrics, per-clip TDT/control summaries, source/binary/SDK
hashes and checksums of all 29 original JSON files. The original
[artifact](https://github.com/haricot/candle/actions/runs/39668765979/artifacts/12379343896) is SHA256
`86e4539b15f9bd3608c7af387d50c8dd289c533a78462b63fdc8e409cf1066c3`; it expires on 2026-11-04.

Original checkpoint SHA256:
`0acdaf1e1ebf09094400263006d134d33b0ebb871ebb064253cfd3a7c528903c`.
CI native executable SHA256:
`60cb56c94f3a955a451071955cccf422d364ecab9bc7bbe38fdbb52fef79f6e3`.

The CI candidate uses pinned MKL 2024.2.2, oneDNN 3.11.2 and standalone
SLEEF 3.8, with four threads, eight-lane ordered LayerNorm and 16-lane vector
activations on this worker. It does not link ATen/libtorch. The native Rust
candidate passed 22 tests in this same job. Source/SDK revisions and compiler
identities are recorded in the JSON. Instructions are in
[README-R13-CPU-FULL.md](README-R13-CPU-FULL.md).

The initial `c80fc34` full campaign failed five short Oréau clips; it is
superseded by this `11a1422` result after correcting small 1x1 convolution
dispatch and SiLU scalar-tail boundaries. No threshold was relaxed.

This is an isolated CPU FP32 offline batch1 candidate qualified on these 14
real public clips. Default Candle CPU, the user's physical laptop, other
shapes/precisions, microphone behavior and CUDA are separate qualifications.
GPU workflow and local runner entry points remain paused; this evidence
commit does not enable or launch a CUDA campaign.
