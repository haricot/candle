# R11 — local oneDNN convolution experiment on all 14 clips

This continues the R10 CPU-only experiment on the same AMD EPYC9V74/9-thread
host, original checkpoint, actual independently captured NeMo references and
14 pinned CaFE/Oreau PCM files. It is not a default-runtime promotion, a
new hosted-CI result or a GPU qualification. Historical R7/R8B executable
controls are not rerun in this local extension.

R11 retains R10's native checkpoint frontend, fused projection and corrected
text. The five subsampling convolutions use standalone oneDNN3.11.2 at the exact
Torch-reported revision `03c022d3ffdcee958cfacbe720048e725fdf644c`. The SDK
build enables CPU/GNU OpenMP, convolution and reorder; GPU runtime is NONE.
The bridge reads native model weights and production-masked operands. It
never reads a NeMo activation during the autonomous PCM pipeline.

Redacted evidence and original native source/SDK/binary hashes:
[`results/r11-local-cpu-20261021.json`](results/r11-local-cpu-20261021.json).
File SHA256: `963affff05ba6f0cb69ccac8ade52c8b269fb18fe329991737e4a7292229ed0d`.
Reproducible experimental components: [`experimental/cpu_parity`](experimental/cpu_parity/README.md).

| Check | Observed result |
| --- | --- |
| Native frontend, all 11 stages | bit-identical, 154/154 |
| Five convolutions on identical actual NeMo inputs | bit-identical, 70/70 |
| Common-input projection | bit-identical, 14/14 |
| Autonomous pre_encode | bit-identical, 14/14 |
| Autonomous post_pos_enc / scaling | bit-identical, 14/14 |
| Final text / discrete TDT / TDT numerical gates | pass, both paths, 14/14; 283 exact decisions |
| Valid final encoder | numerical pass, 14/14; worst max_abs 7.897615432739258e-7 |
| Block-0 FF1 numerical gate | fail, 14/14; worst 0.013671875 |
| Block-0 attention numerical gate | fail, 14/14; worst 0.16263961791992188 |
| Layer-12 output numerical gate | fail, 1/14; worst 0.0018310546875 |
| Complete internal-stage CPU numerical qualification | not established |

Some ReLU traces differ only by the sign of zero. Their numerical difference
is zero; this is retained as a bitwise difference rather than normalized away.
The following convolutions and final subsampling projection still match
actual NeMo bits. Native and shared-Mel paths have the same remaining errors.

The remaining first chronological arithmetic difference after exact block-0
input is normalization. On CaFE fear, `block0_norm_ff1` differs by
1.33514404296875e-5. Candle's CPU kernel uses sequential FP32 sum/sum-of-squares
variance; actual Torch uses Welford/cascade moments with vector/FMA ordering.
Feed-forward projections and activations also retain their original backend;
the whole FF1 error must not be attributed to normalization without further
common-input decomposition.

A separate, standalone ordered-normalization experiment matches that actual
norm output bit-for-bit on all 14 reference inputs. Correct ordering is
8-lane AVX2 with FP32 FMA contraction. The pinned Torch dispatch header does
not register this ordinary LayerNorm kernel for AVX512, despite the global
AVX512 capability report. Sixteen-lane and contraction-off variants fail.
The successful standalone source is archived; it has not been integrated
into the native encoder or qualified in a full PCM campaign.

The portable R11 preparation script was also executed against the real R10
candidate and SDK. Its freshly rebuilt probe reproduces the measured R11
prototype on one real CaFE fear clip: all 75 observed encoder/operand stages,
all 28 TDT decisions/activations, returned result and all 11 frontend traces
are byte-identical. This is a rebuild control, not a fresh R7/R8B control.

The early pipeline now has exact evidence in this local SDK profile. Block-0
and layer-12 failures still prevent CPU qualification and any GPU resumption.
The R10 hosted workflow is prepared; no new R10 run was visible in the Actions
API at the last check, so no hosted pass is claimed.
