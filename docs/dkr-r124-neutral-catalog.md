# DKR R1.24 — Architecture-neutral native catalogue

## S1: verified artifact admission (initial implementation)

Baseline: `95f9087c5210908c9ff7356bf35e914cddac178a` (R1.23 embedded).
S1 preserves the qualified S3.1 native package ABI and its exact-target semantics.
The runtime no longer chooses a fixed GPU target: the verified artifact identity
provides the target for the registry and exact-target admission. Both filesystem
and embedded paths use the same package validator at their trust boundary.
The legacy S3.1 validation rules still intentionally accept SM61 only; a new
schema must carry independently qualified ABI and launch metadata, not relax
the old schema. Synthetic registry tests are not GPU qualifications.

## Migration gates (not yet completed)

- S2: manifest-driven native family inventory, stable operation/entry ABI, explicit
  target and compatibility bounds; separate legacy S3.1 builder/validator and
  MMQ launch recipes into package-specific adapters. Reject unsupported recipes.
- S3: migrate FP8 residence/dispatch, FP4/BF16 and MMQ/MMVQ/MoE host launch
  specialization behind typed contracts. Keep tensor checks and memory safety
  in Candle. Preserve static compatibility mode explicitly.
- S4: catalogue-only feature, with no mandatory static PTX or libmoe.a for
  migrated families; audit all FFI consumers before removing linkage. Use
  cudaforge for package production, not per-build compilation of migrated kernels.
- S5: qualify only SM61 on physical GPU: identical PTX hashes, filesystem versus
  embedded parity with files hidden, exact route provenance, no implicit static
  fallback, numeric/performance comparison to sealed R1.23 baseline.

No changes to CUDA kernels, legacy flags, launch geometry or Mistral pins in S1.
No claim of multi-architecture GPU qualification or catalogue-only compilation.
