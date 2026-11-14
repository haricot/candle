# DKR-R1.22/S3.1 — Candle native module bridge

Branch base: d99f6d77e507688bd9e4b654fbe20ac9e18238fc. This branch is standalone and does not depend on Mistral.rs.

The six original Candle translation units are compiled unchanged using cudaforge
0.1.6, the original legacy BF16/FP8/FP4 flags and SM61. The standalone builder is
an independent Cargo workspace; its output is not a dependency of Candle's
normal build. The new candle-core feature cuda-dkr-s31 opts into the bridge.
Set CANDLE_DKR_S31_DIR to the absolute output directory before creating a CUDA
device. With no variable, built-in modules are used. With the variable, all six
manifests must pass and missing/corrupt modules are errors, not silent fallbacks.
Other Candle modules, memcpy/alloc/BLAS/cuDNN/RNG paths stay unchanged.

This is native module ABI externalization, not yet the Mistral.rs dkr.typed.v1
catalog or a prevalidated launch-plan implementation. No new fusion or numeric
algorithm is introduced. The original embedded modules still build as fallback;
removing them is deliberately deferred. The package digest verifies integrity,
not authenticity or full source provenance; only load trusted local artifacts.

## Deferred GPU qualification

Use a separate target directory for the builder and retain the normal Candle
build/cache. Do not change CUDA_COMPUTE_CAP or run cargo clean.

1. Build packages once using tools/dkr-s31-builder/Cargo.toml and --output.
2. Compile candle-core once with cuda-dkr-s31.
3. Run identical tensor tests with CANDLE_DKR_S31_DIR unset and set, including
   all supported dtypes, non-contiguous and offset layouts, broadcast, where,
   cast, const_set, copy2d, strided copy, empty tensors and device streams.
4. Verify actual external dispatch (not yet instrumented), parity and latency.
   No performance claims or merge before the qualification.

S3.2 should reuse the same loading boundary for reduction/indexing; a neutral
shared runtime/typed catalog and launch plans require separate design work.
