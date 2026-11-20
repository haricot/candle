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

## S2A: explicit native inventory (candidate)

The optional root-level `native-catalog.json` declares schema, native ABI,
launch recipe, exact architecture bounds and the list of required families.
The build script, filesystem provider and standalone cudaforge producer share
the same inventory parser and package validation. Absence of an index invokes
the strict legacy S3.1 adapter; an invalid or incomplete index never falls back.
Both transport sources now converge on one registration path. No CUDA kernel
or launch geometry is changed.

The only admitted recipe is currently `s31-native-v1` (SM61 exact target).
A new recipe must supply separately qualified argument ABI and launch geometry;
the generic index does not confer compatibility with an unqualified device.

To index an existing qualified package directory without recompiling PTX:

    cargo run --manifest-path tools/dkr-s31-builder/Cargo.toml --release -- --index-existing /absolute/native/package/directory

New packages built with `--output` also emit the index. The index does not
modify per-family manifests or PTX, and old R1.23 readers ignore it.

S2A CPU tests exercise complete inventory, reordered inventory, duplicates,
missing families, traversal, unknown ABI/recipe, mismatched target/range,
absent package and invalid index. These are not GPU qualification.

## Migration gates (not yet completed)

- S2B: extract the S3.1 MMQ launch geometry into an explicitly selected recipe
  adapter; extend typed operation/entry ABI metadata without weakening S3.1.
  Reject unsupported recipes. S2A only establishes the inventory/transport layer.
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
