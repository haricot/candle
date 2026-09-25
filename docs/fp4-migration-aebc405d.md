# FP4 migration on pinned upstream

- Original source: `fp4_candle` at `78c44dac3c2a12e70ba0b4e99d3d11f240800f14`.
- Original common ancestor: `8e686231a49a6456b812dc4af1c8084dcb1c006b`.
- Pinned upstream: `aebc405d2b4bf42808387e0ca597bf7dad9b565f`.
- Shared Pascal-compatible baseline: `23cd66e38694c54c2184a57ba9f2c25a7ba1c682` (original SM61 guards at `db6bfa7d`, plus `vec![` syntax repair).
- The original `fp4_candle` branch has **not** been rewritten.

## Migration strategy

The 28 original FP4 feature commits are consolidated into three subsystem-oriented
commits, rather than replayed byte-for-byte. Their final runtime, reference,
test and benchmark code is retained. The new history is not a literal
28-commit rebase; provenance remains in the source branch and commit messages.

1. MXFP4 reference type and GGUF/Metal integration: merge the FP4 additions
   with current upstream changes in `k_quants.rs` and `quantized/mod.rs`.
2. MXFP4/NVFP4 CUDA: final FP4 quantized dispatch and `quantized.cu`
   (DP4A, prefill, indexed MoE and experimental NVFP4 variants).
3. Tests: MXFP4 llama.cpp goldens, round-trip, DP4A/prefill/MoE parity,
   benchmark gates and NVFP4 reference/quality probes. Merge
   `quantized_tests.rs` without reverting independent upstream tests.

## Shared SM61 compatibility and deliberate exclusions

This candidate inherits two *generic* fixes from the shared SM61 baseline:
omit the native FP16 atomic reduction and static WMMA MoE objects below SM70.
It does **not** duplicate these fixes in the FP4 feature.

Three changed files in the old FP4 branch are intentionally not copied:
- `candle-kernels/build.rs`: the old `with_compute_override` path at
  `cudaforge 0.1.6` is superseded by the generic SM61 baseline, which
  excludes static WMMA MoE objects instead of cross-compiling them.
- `candle-kernels/Cargo.toml`: retain the shared baseline's cudaforge version
  and upstream `cutile` feature instead of the old version change.
- `candle-kernels/src/compatibility.cuh`: retain upstream's current
  toolkit-version half-NaN guard. The old `atomicAdd(__half*)` CAS emulation
  is not referenced in FP4 `quantized.cu`; generic pre-SM70 native FP16
  reduction remains excluded by the common baseline.

## Acceptance checks

Run Candle Candidate CI on the **exact candidate commit SHA**:
- Rustfmt, CPU compilation and MXFP4/NVFP4 reference tests.
- Hosted CUDA 12.9 SM61 compile of `candle-core` and `candle-nn`
  including FP4 test targets, without executing kernels.
- Save exact `Cargo.lock`, SHA, logs and exit codes as evidence.

GPU parity, memory safety and performance remain pending until the
integrated `cuda_asd_runner_v2` SHA is fixed.

## RC2 formatting correction

RC1 CPU compilation and all eight non-ignored FP4 reference tests passed;
its format check failed on three FP4 Rust files and the shared baseline's
`build.rs` syntax error. RC2 merges the corrected SM61 baseline and uses
the GitHub Actions stable rustfmt toolchain on the three affected files.
The one-time formatter workflow is absent from the final candidate tree.
