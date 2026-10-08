# Standalone FP4 on main: software NVFP4 opt-in for legacy CUDA

- Source of this feature branch: `fp4_candle@9e92a11d99c451bbf7bc1a2a7b32ff8f04aa4179`, which descends from the then-current `main@66a8cf184a5a519671454066b1b9efd446ec9f5c` without the other five CUDA-ASD features.
- The six-feature integrated, physically validated release `cuda_asd_runner_v2@ac983d16750a136a8563def01926ee83a7831d22` and the originally promoted `fp4_candle` ref are unchanged.
- This new branch keeps the original MXFP4 GGUF support, reference quantizer, dequantizer, SM61 DP4A decode/GEMV, prefill, indexed MoE and tests.

## What `cuda-legacy-fp4` means

This is a real **opt-in for the experimental software NVFP4 path**, not a claim of hardware FP4 Tensor Core capability and not a gate on normal MXFP4. The optional NVFP4 code handles packed E2M1 values, E4M3FN block scales and a global F32 scale through integer LUT/DP4A SIMT kernels. It is experimental and is deliberately not a `GgmlDType` in Candle.

The default `cuda` build omits NVFP4 experimental kernels from the PTX and does not register GPU tests that need those kernels. MXFP4 (including the usual SM61 DP4A paths) continues to compile and work by default, preserving the original feature branch's existing GGUF behavior.

With `--features 'cuda cuda-legacy-fp4'`, `candle-core` forwards the feature to `candle-kernels`, whose build script adds `-DCANDLE_CUDA_LEGACY_FP4=1` to PTX compilation. CUDA experimental NVFP4 symbols and their opt-in tests become available. This does not change the native Blackwell FP4 instruction support of any GPU.

## Independent validation required

Hosted checks (no physical GPU):

```bash
export CUDA_COMPUTE_CAP=61
cargo fmt --all -- --check
cargo test -p candle-core --no-default-features --test mxfp4_tests
cargo check -p candle-core --features 'cuda' --tests
cargo check -p candle-core --features 'cuda cuda-legacy-fp4' --tests
```

GPU tests on one physical SM61 card (only after hosted CUDA checks pass):

```bash
cargo test -p candle-core --features 'cuda cuda-legacy-fp4' --lib cuda_nvfp4_legacy_lut_probe -- --nocapture
cargo test -p candle-core --features 'cuda cuda-legacy-fp4' --lib cuda_nvfp4_sm61_compute_parity -- --nocapture
cargo test -p candle-core --features cuda --test mxfp4_tests -- --nocapture
```

Verify that the normal `cuda` PTX has **no** `nvfp4_*` entries and that opt-in PTX **does** contain the expected NVFP4 symbols. Record exact CUDA version, SM, policy, lockfile, compiler result, numerical tolerances and kernel timing separately. These checks have NOT been executed merely by publishing this isolated branch. No auto-promotion is enabled.
