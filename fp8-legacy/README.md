# FP8 legacy standalone — W8ABF16 recovery

This branch extends `fp8_candle` at `fa922939e36cdd3f84cfdbbd235f8ce2bcc8cc4b`
on Candle main 0.11.1. It restores exactly the two W8ABF16 provider/kernel files
from `9cba2e898af3d25a89257f1f2ec5acece0404dcb`, the Candle used by Mistral
Legacy V3's successful historical SM61 campaign (run 42380889678). That historical
result is a source reference, not execution proof for this new branch.

The packed E4M3FN weight path now accepts F16 or BF16 rank-two activations.
BF16 bits widen directly to F32; GEMV and prefill4 accumulate and output F32.
Weights stay compact, including for BF16 inputs. No BF16-to-F16 narrowing,
native BF16 arithmetic, native FP8 hardware or new common dispatch layer is added.
Existing F16, explicit DP4A and F16-expansion APIs remain compatible. BF16 input
is rejected for DP4A and expanded-F16 residency.

For BF16 callers, prepare explicitly with `Fp8Sm61Mode::SimtW8A16`. The historical
residency/profile planner in this standalone still describes F16 workloads;
it must not authorize F16 expansion for a BF16 request. Model-wide policy belongs
in Mistral's subsequent integration. This change does not move or expand that
planner, and does not alter `cuda_legacy` or its orchestration/promotion refs.

## Validation

Use Rust 1.99.0 and CUDA 12.9. The source receipt and reviewed lockfile are in
this directory. CPU tests, CUDA compilation and physical execution are distinct:

```sh
python3 fp8-legacy/validate.py cpu
CUDA_COMPUTE_CAP=61 NVCC_CCBIN=gcc-14 python3 fp8-legacy/validate.py cuda-compile
CUDA_COMPUTE_CAP=61 NVCC_CCBIN=gcc-14 python3 fp8-legacy/validate.py gpu
```

Hosted CI tests CPU contracts and compiles all selected tests plus real PTX for
SM61, SM70, SM75 and SM80. Each target must expose all four F16/BF16 packed
GEMV/prefill4 symbols. The manually run physical test checks an independent
CPU oracle, F16 and BF16 range, block scales, N/tokens tails, constant compact
residency, incompatible modes and invalid layouts. It reports the actual device
CC; no GPU is initialized by hosted checks and no runner is registered here.

Compilation on several architectures is not physical correctness, performance
qualification or full-model validation on those architectures. Provider names
retain their historical `sm61` suffix; workload profiles and architecture-specific
optimizations must be qualified on their actual target. Formats, scale conventions,
K divisibility and layouts remain explicit constraints; `CC < 80` is not itself
an execution contract. A later `cuda_legacy` assembly must consume this exact
source and pass fresh integrated tests, including its cuDNN/MoE providers.
