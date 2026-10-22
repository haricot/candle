# R10 — local native CPU measurement on all 14 clips

The original `.nemo` SHA256 remains
`0acdaf1e1ebf09094400263006d134d33b0ebb871ebb064253cfd3a7c528903c`.
These measurements use an AMD EPYC 9V74, 9 CPU threads, Python 3.13.16,
NeMo 2.7.3, Torch 2.12.1+cpu and standalone MKL 2024.2.2. Torch reports MKL
2024.2 build20240605. The standalone native binary does not link libtorch.
Other CPUs and the default Candle backend are not qualified by this result.

All 14 WAVs and PCM hashes match the existing pinned CaFE/Oreau selection.
Actual NeMo is restored independently from the original checkpoint for each
clip; its frontend/encoder observation controls pass. The local new-profile
measurement reuses those actual reference tensors. It does not rerun the
independent R7/R8B executable controls locally; the new CI entry point retains
and reruns them rather than inferring them from earlier campaigns.

Redacted per-clip metrics, source/SDK hashes and binary identity:
[`results/r10-local-cpu-20261021.json`](results/r10-local-cpu-20261021.json).
File SHA256: `2330599f553ea8369c902a54078e29c1b9d0ff0951a360a0e494169a588231b4`.
No WAV, transcript, emitted token sequence or raw activation is included.

| Check | Observed result |
| --- | --- |
| Native frontend, all 11 primitive tensors | bit-identical, 154/154 |
| Frontend common-input experiments, 7 stages | bit-identical, 98/98 |
| Common-input final projection | bit-identical, 14/14 |
| Common-input 5 convolutions / 3 ReLUs | numerical gates pass, 112/112 |
| Final text / discrete TDT / TDT numerical gates | pass, both paths, 14/14; 283 exact decisions |
| Valid final encoder | numerical pass, 14/14; worst max_abs 4.805624485015869e-7 |
| Autonomous pre_encode | numerical pass 13/14; worst 0.001220703125 |
| Scaling / block 0 internal gates | fail, 14/14; post_pos_enc worst 0.03125 |
| All internal-stage numerical gates | not established |
| Global FP32 equality against NeMo | not established |

The window and Mel coefficients now come from fixed model buffers, independently
verified against actual NeMo bytes. Native MKL supplies FFT, vector sqrt/log
and SGEMM; power 2 uses an explicit FP32 multiplication. The final projection
initializes its output with checkpoint bias before SGEMM, reproducing NeMo's
fused addmm behavior. These are native parameter/operator changes; the PCM
pipeline reads no reference activation.

Both native and shared-Mel paths now receive identical Mel bytes, and have
identical remaining error metrics. The remaining subsampling discrepancy is
upstream of a common-input projection that is now exact. NeMo selects oneDNN
for all five convolutions on the captured shapes, whereas this profile retains
Candle's original convolution backend. Matching the oneDNN version is the
next controlled experiment, not a validated convolution correction yet.

The absolute gate is demanding at this checkpoint's internal scale: one
example's post_pos_enc reaches 90822.09375, where adjacent FP32 values differ
by 0.0078125, exceeding the unchanged 0.001 limit. Relative error and excellent
final encoder/TDT behavior do not override that gate. One layer-12 output also
fails (0.00146484375).

Local verification: all 164 Python tests and 3 new native Rust tests pass;
the actual profile binary compiles and completes all 14 clips without errors.
The independent hosted CPU campaign remains necessary. No GPU or microphone
test is launched from these results.
