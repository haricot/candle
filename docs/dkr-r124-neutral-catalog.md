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

## S2B: catalogue-selected MMQ recipe (candidate)

The selected recipe ID is embedded with the verified table or parsed from the
filesystem inventory. Native MMQ plan preparation delegates tile selection,
shared-memory caps and launch dimensions to the admitted recipe. The direct
SM61 check in fast_mmq is removed; device admission occurs at catalogue load.
The S3.1 recipe still deliberately uses the same DP4A formulas, entry keys and
launch geometry. Argument packing remains the existing native ABI.
The plan cache and CUDA launch path are unchanged. Known native modules must
not silently fall back to static PTX if a validated catalogue is selected.
CPU tests compare 200 recipe configurations against the historical selector
and reject invalid device limits, shapes and codecs.

## S3.1: FP8 native dispatch extraction (candidate)

The generic F8E linear, residency planner, format and loader now live under
`fp8_linear`, `fp8_dispatch`, `fp8_format` and `fp8_loader`. The former public
SM61 module and type names remain compatibility reexports; no Mistral pin
change is needed for these names. The native recipe supplies the five entry
symbols, token tile, rows per block, block width, K alignment and DP4A groups.
The producer writes explicit FP8 metadata in the optional native index field.
Existing manifests/indexes without it resolve the exact frozen S3.1 recipe;
edited descriptors or missing required PTX entries fail closed. The build
embeds a verified typed FP8 descriptor and the filesystem path uses the same
recipe and package checks. The selected descriptor is pinned per linear.
Historical static execution stays explicit when no native catalogue is selected.
No automatic DP4A or FP16 admission changes, no PTX edits, no launch ABI changes.
Submitted telemetry is now recorded explicitly for FP8 launches; this adds
observability, not a new execution route. The embedded table test checks
its five entry symbols when built with DKR_EMBED_NATIVE_DIR.
S3.1 retains F8E-v1 tensor/packing ABI and old custom-op trace names; it does
not add native FP8 Tensor Core support or remove the static feature yet.
The FP8 recipe is frozen in the package-side source, not arbitrary executable
metadata. New architectures require separately qualified descriptors and ABI.

## S3.2: FP4 / NVFP4 dispatch extraction (candidate)

ModelOpt NVFP4 native, linear, FP4 typed facade and format modules use generic
names; old APIs remain reexports. The catalogue selects a frozen FP4 descriptor
for NVFP4 scalar W4A16, FP16 cast, Q8_1 quantization, reference and tiled W4A8,
plus GGUF MXFP4 GEMV, prefill and indexed MoE. Auto thresholds, DP4A admission,
entry symbols and launch geometry are in the package-side recipe. Static
compatibility remains available without a selected catalogue. The index
optionally records the FP4 descriptor and validates it against its qualified
ABI; all fifteen required entries must exist in the quantized PTX.
Embedded and filesystem metadata use the same descriptor. No PTX, packing,
scale convention, argument ABI or auto thresholds changed. NVFP4 and MXFP4
remain distinct formats. Launch bounds reject invalid dimensions.
Generic GGUF Q8_1 staging, MXFP4 dequantization/embedding, experimental NVFP4 probes and the other dtype paths are not migrated here.
The native instance caches device capability once; a GPU test compares Auto with the expected explicit route at the batch 8/9 boundary.
This does not remove the legacy compilation feature or qualify another GPU.

## S3.3: BF16 hardware dispatch extraction (candidate)

The native inventory supplies a typed BF16 GEMM route and the two required
conversion entry symbols. The S3.1 recipe selects exact BF16 to F32 widening,
cuBLAS SGEMM and rounding once to BF16. The build verifies both cast entries
and embeds the recipe; filesystem and embedded runtime use the same rules.
Earlier indexes lacking the BF16 descriptor use the frozen S3.1 adapter.
An edited descriptor cannot admit a different matmul/convert ABI.
CudaDevice resolves and caches the selected route at device construction.
The legacy static path (when no DKR catalogue is selected) retains its
pre-Ampere boundary, in the package-side adapter. Native BF16 on SM80+
remains the old static route; no new GPU architecture is qualified.
Tensor layout materialization, strides, broadcast, exponent range, output
rounding and ordinary F32 GEMM behavior are unchanged. The cast entry names
are fixed by the generic Candle cast ABI, not arbitrary manifest aliases.
Other BF16 kernels are already routed through native module lookup; no CUDA
source or PTX changes. cuDNN failure policy, MoE and legacy build flags are
outside this stage; catalogue-only compilation remains S4.

## Migration gates (not yet completed)

- Extend stable typed operation/argument ABI and recipes for additional
  architectures only after independent qualification. S2 does not authorize
  arbitrary manifest-driven launch geometry or change the S3.1 argument ABI.
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
