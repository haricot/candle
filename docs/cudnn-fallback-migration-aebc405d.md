# cuDNN fallback — one-time migration

**Pin**: upstream `aebc405d2b4bf42808387e0ca597bf7dad9b565f`.
**Shared sm61 baseline**: `db6bfa7d02e8ff8419e1ba2f7baabc739fc752bc`.
**Original branch**: `cudnn_fallback_candle`, `0ccb2719d41a029e403be3b68c141ee168126ae6` (preserved).
**Candidate**: `rebase/20260925-aebc405d/cudnn_fallback_candle-work`.

## Scope and three-way resolution

The original seven feature commits are grouped into independently inspectable
functional, dispatch, and test/provenance commits. This is **not** a literal
seven-commit `git rebase`; the historical branch remains the audit trail.

1. Typed `CudnnError` conversion plus thread-local quarantine for
   convolution on devices with compute capability below sm70 after a
   cuDNN `EXECUTION_FAILED` status. The disabled state is keyed by CUDA
   device ID, not a global all-GPU fallback.
2. Explicit generic CUDA Conv1D and Conv2D paths and fallback routing
   for `NOT_SUPPORTED`, `NOT_SUPPORTED_ARCH_MISMATCH`, and (on
   pre-Volta only) `EXECUTION_FAILED`. Other errors propagate as errors.
3. The upstream `cuda_backend/mod.rs` changes (cutile registration,
   small-reduction kernel routing, copy bounds and noncontiguous Conv2D
   kernel materialization fix) are retained. The original Conv2D body
   was extracted to `conv2d_cuda`; therefore the upstream copy fix was
   **manually relocated** to that new helper during three-way resolution.

## Intentionally not copied from the historical branch

`.github/workflows/cuda-sm61-validation.yml` is a legacy manual
validation workflow. The consolidated `Candle Candidate CI` on the
`mirror_orch` default branch now supplies SHA-pinned CPU and CUDA
12.9 SM61 compilation checks, without running GPU kernels. Reintroducing
the old workflow would cause overlapping maintenance and CI runs.

## Gates

Before integration, run `Candle Candidate CI` with the final candidate
SHA and `validation=cpu_cuda`. Preserve the SHA-256 of its exact
`Cargo.lock`, CPU/CUDA logs and result JSON.

The unit tests for classification are compiled with `--features cudnn`;
they are not executed by the hosted CUDA compile-only job. Runtime parity,
pre-Volta failure injection, thread-local quarantine behavior and
fallback correctness require the later physical GTX 1080 GPU validation.
