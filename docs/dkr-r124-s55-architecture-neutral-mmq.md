# R1.24 S5.5 — Architecture-neutral catalogue (S5.5a: MMQ extraction)

## Architecture audit

Prior to S5.5a, the native runtime read the frozen
`s31-native-v1` recipe, then called
`NativeRecipe::mmq_geometry`, which delegates to
`contract::mmq_tile` and `contract::mmq_smem_bytes`.
The tile widths (8–64), fixed row tile (64), warp/block layout,
and codec-dependent shared-memory formula were consequently
compiled into Candle/candle-kernels. The FP8/FP4/BF16/Routed
recipes remain represented by frozen S31 types/constants.

## S5.5a — New experimental opt-in feature

- `cuda-dkr-neutral-mmq` implies `cuda-dkr-embedded` and
  can be combined with `cuda-dkr-cubin-embedded`.
- A dedicated offline recipe producer,
  `tools/dkr-r124-s55a-prepare.py`, emits a manifest of the
  10 GGUF MMQ codec families and 80 tile/shared-memory choices.
  The SM61-specific memory equation lives in this **offline**
  producer, not in the generic runtime interpreter.
- The producer verifies S5.1's sealed native index hash and exact
  PTX/package hashes, and verifies all 16 MMQ entry-map keys per
  codec. It writes a new derived manifest; existing PTX, CUBIN,
  S5.1–S5.4b reports and the original index remain unchanged.
- `candle-core/build.rs` checks schema, exact target, source-index
  SHA, individual original PTX digests, package symbol-table keys,
  bounded row/grid/shared-memory parameters and complete tile
  inventory before generating a typed, static embedded table.
  The runtime does not read a recipe file from the filesystem.
- `dkr-kernel-registry::mmq_recipe` now owns the generic table
  interpreter and MMQ launch-plan selection. It has **no** SM61,
  NVCC, CUDA driver, PTX, GGUF tensor representation or CUBIN
  policy; it chooses the first feasible tile minimizing grid Y.
- `ExternalModules::mmq_configs` calls that generic interpreter
  under the feature, leaving the verified function-cache, stream,
  launch arguments and CUBIN selection unchanged.
- Baseline without `cuda-dkr-neutral-mmq` continues using
  the previously qualified frozen S31 recipe.
- The test `dkr_s55a_neutral_mmq_matches_frozen_s31_qualified_shapes`
  compares 10 codecs across 5 row counts × 8 batch sizes = 400
  expected launch configurations, including tile choice,
  workspace grids, blocks and shared memory, with rejection
  checks. An additional headless pass reruns S5.2 numerics,
  all 25 CUBIN module provenance records and S5.3 workloads.

This is a **staged migration** and does not claim the whole
architecture-neutral goal is complete. S31 constants used by
legacy builds, FP8/FP4/BF16/Routed recipe implementations, and
the old S31-only catalogue verifier have not been removed.

## How to qualify on SM61

From /home/np/tmp/candle_dkr/candle:

    git pull --ff-only origin feature/dkr-r124-neutral-catalog
    python3 -m py_compile tools/dkr-r124-s55a-prepare.py tools/dkr-r124-s55a-qualify.py
    python3 tools/dkr-r124-s55a-prepare.py --self-test
    python3 tools/dkr-r124-feature-audit.py
    python3 tools/dkr-r124-s55a-prepare.py \
      --seal "$PWD/target/dkr-r124-s51/seal.json" \
      --out "$PWD/target/dkr-r124-s55a"

Build Rust tests with the S5.4b CUBIN inventory (no new NVCC
or ptxas, no cargo clean):

    python3 tools/dkr-r124-s55a-qualify.py build \
      --seal "$PWD/target/dkr-r124-s51/seal.json" \
      --mmq-dir "$PWD/target/dkr-r124-s55a" \
      --cubin-dir "$PWD/target/dkr-r124-s54b" \
      --s52-result "$PWD/target/dkr-r124-s52/headless-20261121T124142625489Z/result.json" \
      --s53-result "$PWD/target/dkr-r124-s53/headless-20261121T130221097950Z/result.json" \
      --out "$PWD/target/dkr-r124-s55a-qualification"

Only after the archive build passes, save graphical work and run:

    headless-relogin --with-after-run-openlog \
      /usr/bin/env -u DISPLAY -u WAYLAND_DISPLAY \
      python3 "$PWD/tools/dkr-r124-s55a-qualify.py" headless \
        --receipt "$PWD/target/dkr-r124-s55a-qualification/receipt.json"

Expected PASS markers (do not claim them before GPU testing):

    DKR_R124_S55A_TABLE_PARITY=PASS codecs=10
    DKR_R124_S55A_NUMERIC=PASS cases=120 gguf_moe=6
    DKR_R124_S55A_NEUTRAL_MMQ=PASS codecs=10
    DKR_R124_S55A_CUBIN_MODULES=25/25
    DKR_R124_S55A_PERF=PASS rounds=3 cases=7
    DKR_R124_S55A_PASS

## Subsequent S5.5 workstreams

S5.5b: producer-owned typed FP8/FP4/BF16/Routed descriptors
with bounded verifier and explicit ABI schema (no S31
constant equality required in the runtime).

S5.5c: relocate package/entry-name knowledge and remaining
S31 recipes from generic Candle crates to a reusable
Cudaforge-backed producer and versioned catalogue transport.

S5.5d: matched-input PTX-vs-CUBIN numeric delta diagnostics,
FP8/FP4 edge cases, model-level cold-start and controlled-clock
regression qualification before default promotion.

Every stage must preserve the original PTX-only baseline,
S5.4b validated binary behaviour, fail-closed admission and
no silent static/legacy fallback.
