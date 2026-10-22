# R13 — full CPU qualification before CUDA

The R12 CPU CI qualified one LayerNorm operator on 14 actual common inputs.
It did not execute or qualify the full CPU pipeline. All GPU entry points
remain paused; a successful CPU campaign does not automatically re-enable them.

R13 integrates that unchanged ordered LayerNorm implementation into all 24
Conformer blocks and their convolution norms. It also includes the original
checkpoint frontend, standalone MKL 2024.2.2 (including feed-forward and
attention linears/matrix products), oneDNN 3.11.2 subsampling and temporal
convolutions, and R9
SentencePiece rendering. This is an isolated native CPU candidate, with no
ATen/libtorch dependency in the inference executable.

Relative attention follows NeMo's FP32 operation order: add the two score
matrices before dividing by the FP32 square root of the head dimension.
Position encoding uses native MKL vector exp/sin/cos. SiLU, GLU and softmax use
standalone SLEEF 3.8 at PyTorch 2.12.1's pinned source revision, with native
AVX2/AVX512 dispatch and the same CPU grain boundaries, scalar tails and
reduction order. Small 1x1 temporal convolutions use MKL addmm at PyTorch's
20480-element dispatch boundary; larger inputs use oneDNN. The ordered
LayerNorm retains the separately verified eight-lane algorithm unchanged.

The complete test uses all 14 real CaFE/Oréau clips. It reruns the independent
R7/R8B baseline controls and the actual NeMo reference, then evaluates native
PCM inference, frontend stages, subsampling, all 24 layer outputs, block 0
internal stages, production TDT decisions/activations and final text. Common
input replays and shared-Mel diagnostics remain separate from native inference.
No internal reference activation is injected into the native pipeline.

Every original numerical threshold is retained. A missing clip, execution
error or failed numerical/text/decision gate prevents the full pass. Execution,
numerical parity and global bitwise parity have separate verdicts. A numerical
pass is not a bitwise claim, a production promotion or a microphone test.

CI input `r13_cpu_full=true`, or commit marker `[r13-cpu-full]`, selects the
complete CPU campaign. It publishes `r13-cpu-summary.json` and per-clip metrics
with the source, executable, checkpoint and native-library identities. CUDA is
not run by this workflow. A failed numerical qualification exits 2.

In an already prepared CPU environment, from `half_duplex`:

```bash
uv pip install --no-config --python "$HDB2C_PYTHON" \
  --default-index https://pypi.org/simple \
  'mkl==2024.2.2' 'mkl-include==2024.2.2'
bash ci/experimental/cpu_parity/build_onednn.sh \
  /absolute/path/to/onednn-source /absolute/path/to/onednn-build
bash ci/experimental/cpu_parity/build_sleef.sh \
  /absolute/path/to/sleef-source /absolute/path/to/sleef-build
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=4
"$HDB2C_PYTHON" -m hdb2c.c3_r13_cpu \
  --corpus /absolute/path/to/cameo/manifest.json \
  --onednn-source /absolute/path/to/onednn-source \
  --onednn-build /absolute/path/to/onednn-build \
  --sleef-source /absolute/path/to/sleef-source \
  --sleef-build /absolute/path/to/sleef-build --require-numeric
```

Linux x86-64 with AVX2/FMA, GCC/G++, ar, GNU OpenMP, CMake, Ninja and the
pinned Rust/Python/NeMo environment are required. The helper builds pinned
CPU-only oneDNN. Baseline sources and executables remain independently hashed.



The completed independent CPU campaign on `11a1422` passed all14 full numerical
qualifications with the original gates. TDT283/283 decisions and all texts
match. Global bitwise parity remains incomplete: CI has267/336 exact layer
outputs per path and11/14 exact final encoder outputs, with max final encoder
abs2.384185791015625e-7. See [the measured results](RESULTS-HDB2-C3-A-R13-CPU-FULL.md)
and [persistent CI evidence](results/r13-ci-full-cpu-20261021.json) before
interpreting the numerical pass. GPU remains paused.
