# Local CPU experiments after R10

These experimental native components do not change the default runtime or
the R10 hosted campaign. GPU and CUDA microphone tests remain deferred until
all CPU numerical gates pass. [Measured results](../../RESULTS-HDB2-C3-A-R11-CPU.md)
include all 14 real corpus clips and identify the remaining failures.

The oneDNN bridge uses the exact source revision reported by Torch 2.12.1:
3.11.2, commit `03c022d3ffdcee958cfacbe720048e725fdf644c`. Both convolution and
reorder primitives are required. The configuration builds CPU/GNU OpenMP
only. GCC/G++, CMake >=3.22, Ninja and the existing standalone MKL SDK are
required. The bridge contains no ATen or libtorch dependency.

From the repository root, with an existing R10 candidate built by
`hdb2c.c3_r10_cpu`:

```bash
bash half_duplex/ci/experimental/cpu_parity/build_onednn.sh \
  /absolute/path/to/onednn-source /absolute/path/to/onednn-build
"$HDB2C_PYTHON" half_duplex/ci/experimental/cpu_parity/prepare_probe.py \
  --parent /absolute/path/to/r10-candidate \
  --target /absolute/path/to/new-r11-candidate \
  --onednn-source /absolute/path/to/onednn-source \
  --onednn-build /absolute/path/to/onednn-build
```

The preparation rejects modified R10 source, a different oneDNN revision and
a non-CPU SDK. It copies into a new directory and changes only the five
subsampling convolutions and their separate replay. The new native probe
retains R10's CLI: `--device cpu`, `--model-dir`, `--input`, `--output`, plus
separate `--operator-cases`, `--shared-mel` or `--tdt-cases` diagnostics. Use
the same actual NeMo fixtures/PCM hashes. Its build and source manifest remain
local. The benchmark is unqualified while any original gate fails.

`ordered_layernorm.cpp` is an independent scalar-lane implementation of the
FP32 Welford/cascade order in the pinned Torch CPU sources:

- [moments_utils.h](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/native/cpu/moments_utils.h)
- [layer_norm_kernel.cpp](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/native/cpu/layer_norm_kernel.cpp)
- [DispatchStub.h](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/native/DispatchStub.h)

The latter disables ordinary AVX512 registration for this kernel, so the
tested Torch uses **8-lane AVX2 moment ordering**, even though global CPU
capability reports AVX512. Compiling this standalone source with
`-O2 -mavx2 -mfma -fno-math-errno -ffp-contract=fast` and calling
`hdb2c12_layernorm(..., lanes=8, ...)` matches the first norm on all 14 real
common inputs bit-for-bit. The 16-lane and contraction-off variants do not.
This operator has **not been integrated or qualified in the full pipeline**.
It is retained to investigate the block-0 normalization/feed-forward residual,
without implying that the whole CPU pipeline has passed.

The standalone check can reuse the actual `clip-XX/nemo` folders from a local
R10 campaign run with `--keep-raw`:

```bash
g++ -shared -fPIC -O2 -mavx2 -mfma -fno-math-errno -ffp-contract=fast \
  half_duplex/ci/experimental/cpu_parity/ordered_layernorm.cpp -o /tmp/ordered-layernorm.so
"$HDB2C_PYTHON" half_duplex/ci/experimental/cpu_parity/check_layernorm.py \
  --model "$HDB2A_MODEL_DIR" --corpus /absolute/path/to/cameo/manifest.json \
  --references /absolute/path/to/r10-report --library /tmp/ordered-layernorm.so \
  --output /absolute/path/to/ordered-layernorm-metrics.json
```
