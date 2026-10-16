# SIMT F16 MoE — pinned-upstream migration

- Source branch: moe_simt_f16_candle, commit 9e332eab583dc6801d4524963e1cb15ce85b922b (unchanged).
- Source commits: 088d26773c75bef16bd4b80bbac8129b2c30254d and ABI fix 9e332eab583dc6801d4524963e1cb15ce85b922b.
- Pinned upstream: aebc405d2b4bf42808387e0ca597bf7dad9b565f.
- Corrected SM61 baseline: 23cd66e38694c54c2184a57ba9f2c25a7ba1c682.
- Integration target: separate cuda_asd_runner_v2. Source feature branch is not rewritten.

## Three-way resolution

The common SM61 baseline excludes WMMA and WMMA-GGUF static translation units
below sm70. The old build.rs must not be copied wholesale.

1. Retain the current cudaforge source list, cutile optional source, upstream
   rerun-if-changed rules and SM61 exclusion of static WMMA objects.
2. Compile the independent moe_simt_f16.cu translation unit and export
   CANDLE_CUDA_COMPUTE_CAP for Rust's compile-time backend selector.
3. Below sm70, pass -DNO_WMMA_KERNEL so the SIMT unit supplies the original
   WMMA FFI stub while the unsupported WMMA translation units stay excluded.
4. Merge the Rust dispatch with upstream's cutile module declaration and
   use the final original FFI signatures.

The SIMT FP16 path uses FP16 storage with FP32 FMA arithmetic. BF16 on Pascal
is deliberately rejected: select_moe_backend(61, 1) returns None. SM70+
continues to use WMMA where supported.

## Validation

The Rust selector tests cover FP16 SM53/SM61/SM70 and BF16 SM80 architecture
thresholds, plus unsupported dtypes. Candle Sync auto-rustfmt formats the
temporary candidate before its SHA is frozen. Candidate CI verifies CPU
and compiles CUDA 12.9 with CUDA_COMPUTE_CAP=61 and matching Cargo.lock.

Compile-only success is not GPU runtime proof. After integration, run GTX 1080
FFI linking, SIMT numerical parity, top-k routing, prefill/decode, masking
and performance checks on the final integrated SHA.
