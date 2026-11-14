# DKR-R1.22/S3.1–S3.2 — Candle native module bridge

Branch base: d99f6d77e507688bd9e4b654fbe20ac9e18238fc. This branch is standalone and does not depend on Mistral.rs.

The eight original Candle translation units are compiled unchanged using cudaforge
0.1.6, the original legacy BF16/FP8/FP4 flags and SM61. The standalone builder is
an independent Cargo workspace; its output is not a dependency of Candle's
normal build. The new candle-core feature cuda-dkr-s31 opts into the bridge.
Set CANDLE_DKR_S31_DIR to the absolute output directory before creating a CUDA
device. With no variable, built-in modules are used. With the variable, all eight
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
4. Verify module-load evidence with CANDLE_DKR_TRACE=1, parity and latency.
   A module-load event is not a per-operation launch counter.
   No performance claims or merge before the qualification.

S3.2 adds reduce.cu and indexing.cu unchanged, including Candle-NN softmax,
RMSNorm and LayerNorm through the same loader. Device construction snapshots
all eight validated PTX texts; subsequent cold loads perform no file reads.
The original feature and environment variable names remain for compatibility.
A neutral
shared runtime/typed catalog and launch plans require separate design work.

## S3.3 — GGUF and fast MMVQ

The same provider now requires ten modules, adding `quantized.cu` and `mmvq_gguf.cu`.
The latter reuses 240 existing specialized matvec entries and three Q8_1 quantizers.
The Rust path preserves batch-specific launch geometry and the resident Q8_1 scratch.
The builder and loader reject an MMVQ PTX that lacks any of the 243 expected entries.
The legacy static MMVQ path remains available with external activation unset.
Fast MMQ remains static pending a separate launch-plan port; its template symbol,
shared-memory, tile selection and stream-K launch contracts must not be replaced
with generic GEMM. No CUDA source algorithm has been changed.

## S3.3.1 — Fast MMQ

Adds the eleven original MMQ compilation units without changing CUDA source algorithms.
Each format requires 16 native C++ entries (eight widths, two boundary variants),
and the quantizer requires all three Q8_1 layouts. The builder discovers the exact
Itanium-mangled symbol names from PTX and fails if absent or ambiguous; the loader
independently reconstructs and compares this inventory, including exact PTX scalar
parameter widths. This is a compiler-ABI gate, not an
assumption that all NVCC versions will emit the same names.
SM61 uses tile_y=64, 32x8 threads, conventional tiling, exact DP4A shared bytes.
A bounded 64-entry LRU per thread retains resolved CUDA functions and geometry;
no Stream-K fixup allocation is needed in the selected external path.
Original static routes remain available when the provider is unset. S3.3.1 is
not compiled or GPU qualified yet; the previous 'MMQ pending' note above is
superseded at source level only. Total native packages: 21.
