# R8B CPU — actual operands and residual attribution

[Run 39597750497](https://github.com/haricot/candle/actions/runs/39597750497)
completed all **14/14** real clips, zero execution errors, on source
[`b1d6a027`](https://github.com/haricot/candle/commit/b1d6a0272f98a1e2fdb563ccd99a38f0d479827f).
The real ASR job fails the unchanged qualification gates as intended:
`NUMERIC_PARITY_NOT_ESTABLISHED`. All 153 Python tests and 9 native Rust tests
pass. No autonomous arithmetic or threshold was changed by these diagnostics.

Artifact `half-duplex-real-cpu-cafe-reports`, ID `12344787867`, ZIP SHA256
`5c6ccdde32444d7e58e835f419dbf95c36ca38c0c287aec7347c51993b553226`.

All original R7/R8B observed tensors, TDT activations, decisions and returned
results match bitwise on all 14 native observation controls. Actual NeMo
frontend and encoder observation controls also pass. All 283 autonomous TDT
decisions remain exact; the final encoder remains numerically qualified on
14/14 clips. Full initial/internal-stage gates and global FP32 bits still fail.

## Text: the two discrepancies contain the unknown token

The actual SentencePiece model has SHA256
`eacec2b0a77f336d4a2ca4a25a7047575d3c2b74de47e997f4c205126ed3135e`.
Its unknown token (ID 0) renders as **space + U+2047 + space**, whereas the
native vocabulary concatenation emits the literal five-character `<unk>`.
It has no control/byte pieces or denormalizer; the actual NeMo tokenizer uses
`legacy=false`. Converted vocabulary entries exactly match the original model.

| Clip | Unknown tokens emitted | Native text exact | Reference/native character count |
|---|---:|:---:|---:|
| Oréau neutral | 1 | no | 32 / 34 |
| Oréau happiness | 1 | no | 19 / 21 |
| Other 12 clips | 0 each | yes | equal |

On both mismatches the first differing native character is `<` (U+003C) while
the reference has a space (U+0020). IDs, frames, durations and all TDT decisions
are identical. The exact unknown-surface handling is the concrete text fix to
implement; this diagnostic revision **does not yet modify the native text**.

A separate tokenizer-only experiment loaded this exact public model and
compared all 8192 singleton IDs, 5000 random sequences (seed 20261021, uniform
lengths 1..50 and vocabulary IDs), and three
unknown-token edge cases against SentencePiece. Using its original unknown
surface yields 0/13195 mismatches after the existing final trim; the old join
fails 15 cases. This is operator evidence, not another ASR corpus run or a
production integration claim.

## Linear: distinguish FP32 accumulation from higher precision

Common inputs and weights remain byte-verified. FP64 dot products are computed
independently by Candle and Torch, rounded to FP32 for comparison, and never
injected into autonomous inference.

| Comparison on identical operands | Worst max abs | Original numerical gate passed |
|---|---:|---:|
| Torch FP32 vs Torch FP64 rounded | 0.001220703125 | 13/14 |
| Candle FP32 vs Torch FP64 rounded | 0.001220703125 | 10/14 |
| Candle FP64 rounded vs Torch FP64 rounded | 0 | 14/14 |
| Candle FP64 rounded vs Torch FP32 | 0.001220703125 | 13/14 |

The two independent FP64 results rounded to FP32 are **bitwise identical on
all 14 clips**. This isolates an FP32 arithmetic-order/accumulation residual in
the common-input projection, rather than a layout or weight conversion error.
It does not identify a particular BLAS kernel as the sole cause. Increasing
precision alone would still fail the original Torch-FP32 comparison on CaFE
fear (1.220703125e-3), and does not remove inherited autonomous input errors.
No gate is relaxed or replaced by the supplemental FP64 diagnostic.

## Observe the actual masked native operands

R8B records the native input immediately after every production temporal mask.
It compares this operand against the actual NeMo pre-hook input. This removes
the large padding artefacts in R8's reconstructed operands. All original
production stage bytes remain covered by the independent baseline control.

| Operator | Worst absolute input difference, after masks |
|---|---:|
| conv_00 | 0.0002536773682 |
| relu_00 | 0.0003936290741 |
| conv_02 | 0.0003930032253 |
| conv_03 | 0.0003563761711 |
| relu_03 | 0.003093719482 |
| conv_05 | 0.0009560585022 |
| conv_06 | 0.0003471374512 |
| relu_06 | 0.001791954041 |
| out | 0.001751422882 |

## Window arithmetic experiment (isolated primitive only)

A separate local 400-sample symmetric Hann comparison against Torch2.12.1+cpu
finds 215 differing coefficients with the existing Rust phase expression
`2*f32_PI*i/(N-1)`, max abs3.8743019e-7. Computing the scale
`(2*f64_PI/(N-1)) as f32` once, then multiplying by the FP32 index, reduces this
to 14 coefficients and max abs5.9604645e-8. Using an FP32 scale instead does not
produce the same improvement. Residual scalar/vector cosine differences remain.
These are host-specific primitive measurements, separate from the R8/R8B CI
window values and any full-model claim; the autonomous frontend was not changed.

## Physical follow-up

The physical GTX 1080 Max-Q preflight reported capability **6.1**, driver
**580.178.04**, CUDA toolkit **12.9.2** and cuDNN **9.10.2.21**, all passing.
This is hardware/toolchain availability evidence, not inference qualification.

CUDA run [39598582574](https://github.com/haricot/candle/actions/runs/39598582574)
is queued for exact candidate `8e827fadddfb6b5e555289836728aee52c6b9bb1`.
The local launcher fix in `d0a322d` addresses Python's non-seekable `/dev/tty`
read/write failure. A real controlling-pseudo-terminal regression verifies
hidden token entry. Updating the local launcher does not change the queued
candidate. Usage: [README-R8.md](README-R8.md).

GPU ASR, microphone and streaming remain **not measured** at this checkpoint.
