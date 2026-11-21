# R1.24 S5.5b — Producer-owned FP8, FP4, BF16 and Routed launch recipes

## Design and scope

This is the next incremental extraction after the S5.5a neutral MMQ
PASS. The original S5.1 sealed PTX, S5.4b 25 embedded CUBIN,
and S5.5a MMQ table remain immutable.

**Experimental Cargo feature:** `cuda-dkr-neutral-recipes`
(implies `cuda-dkr-neutral-mmq` and `cuda-dkr-embedded`).
Use alongside `cuda-dkr-cubin-embedded` to retain the qualified
SM61 CUBIN startup benefit.

The offline producer `tools/dkr-r124-s55b-prepare.py` supplies
four strict typed descriptors:

- FP8: SIMT / DP4A GEMV and prefill symbols, quantizer, row/grid
  and activation group parameters;
- FP4: scalar, cast, reference, tiled and MXFP4 kernel symbols,
  switching thresholds, threads/warps and tile geometry;
- BF16: native vs F32-widened GEMM route and conversion symbols;
- Routed/MMVQ/MoE: quantization, SIMD warp, padding, shared memory,
  MoE routing and GGUF tile parameters.

The producer also records the S5.1 seal and optional legacy index
fingerprints, and the PTX+manifest hashes for five affected native
families (`quantized`, `cast`, `mmvq_gguf`,
`moe_simt_f16`, `moe_gguf`).

`candle-core/src/cuda_backend/dkr_s55b_build.rs` admits that
manifest only at build time after checking schema, source S5.5a
fingerprints, exact target CC, family hashes, real verified PTX
entrypoint names, integer bounds, block sizes and native input/ABI
constraints. It generates typed Rust constants under OUT_DIR.
The runtime has **no recipe file access** and no silent PTX/static
fallback in the CUBIN-enabled experimental mode.

Under `cuda-dkr-neutral-recipes`, Candle reads these build-verified
`Fp8Recipe`, `Fp4Recipe`, `Bf16Recipe` and
`RoutedRecipe` values. It does **not** require equality to the
frozen S31 constants for runtime admission. A feature-gated test
uses those frozen values solely as a regression oracle. The generic
CUDA device interface, context, streams, kernel arguments, MMQ
table interpreter and binary loader remain unchanged.

**Limitation:** these four descriptor _values_ are now producer
owned; the existing recipe type definitions and some GGUF codec
packing/dispatch rules still reside in Candle/candle-kernels. Full
producer/runtime separation and removal of the remaining S31
special cases are S5.5c, not yet closed.

## Step 1: update and prepare the four recipes, no CUDA compilation

From `/home/np/tmp/candle_dkr/candle`:

```sh
git pull --ff-only origin feature/dkr-r124-neutral-catalog
python3 -m py_compile tools/dkr-r124-s55b-prepare.py tools/dkr-r124-s55b-qualify.py
python3 tools/dkr-r124-s55b-prepare.py --self-test
python3 tools/dkr-r124-feature-audit.py

python3 tools/dkr-r124-s55b-prepare.py \
  --seal "$PWD/target/dkr-r124-s51/seal.json" \
  --out "$PWD/target/dkr-r124-s55b"
```

The producer verifies the existing five source packages and writes
only `target/dkr-r124-s55b/manifest.json`. It does not run ptxas
or nvcc, and does not touch global CUDA caches.

## Step 2: build exactly one Rust Nextest archive

```sh
python3 tools/dkr-r124-s55b-qualify.py build \
  --seal "$PWD/target/dkr-r124-s51/seal.json" \
  --mmq-dir "$PWD/target/dkr-r124-s55a" \
  --recipes-dir "$PWD/target/dkr-r124-s55b" \
  --cubin-dir "$PWD/target/dkr-r124-s54b" \
  --s52-result "$PWD/target/dkr-r124-s52/headless-20261121T124142625489Z/result.json" \
  --s53-result "$PWD/target/dkr-r124-s53/headless-20261121T130221097950Z/result.json" \
  --out "$PWD/target/dkr-r124-s55b-qualification"
```

This builds Rust under the combined experimental
`cuda-dkr-cubin-embedded,cuda-dkr-neutral-recipes` features,
including FP8/FP4 numerical CUDA test binaries. No NVCC, ptxas,
static FFI recompilation, or `cargo clean` is needed.

A failing archive build is a source issue: inspect
`target/dkr-r124-s55b-qualification/archive-build.log` before
considering headless; never delete qualified assets.

## Step 3: preflight non-GPU typed recipe parity from the archive

```sh
cargo nextest run \
  --archive-file "$PWD/target/dkr-r124-s55b-qualification/s55b-tests.tar.zst" \
  --workspace-remap "$PWD" \
  --run-ignored all --no-capture \
  -E 'test(dkr_s55b_typed_recipes_match_qualified_s31) | test(dkr_s55a_neutral_mmq_matches_frozen_s31_qualified_shapes)'
```

Expected: 2/2 PASS; in particular
`DKR_R124_S55B_TYPED_PARITY=PASS fp8=1 fp4=1 bf16=1 routed=1`
and S5.5a MMQ geometry parity. This preflight is important before
any graphical logout.

## Step 4: headless GPU gates

```sh
headless-relogin --with-after-run-openlog \
  /usr/bin/env -u DISPLAY -u WAYLAND_DISPLAY \
  python3 "$PWD/tools/dkr-r124-s55b-qualify.py" headless \
  --receipt "$PWD/target/dkr-r124-s55b-qualification/receipt.json"
```

The headless runner verifies receipt/archive and all source hashes,
then runs:
1. Typed S5.5b and MMQ S5.5a parity tests;
2. the previous S5.2 numeric matrix (120 MMVQ/MMQ cases,
   6 GGUF MoE codecs, BF16 and SIMT MoE) and all 25 native
   CUBIN provenance traces;
3. six further FP8/FP4 CUDA tests (SIMT/DP4A FP8,
   NVFP4 ModelOpt and MXFP4 vector/prefill/indexed MoE);
4. both negative filesystem override admission tests;
5. seven S5.3 microbenchmarks across three independent cold
   processes with private CUDA caches and GPU clock logs.

Expected markers **only after running the tests on SM61**:

```
DKR_R124_S55B_TYPED_PARITY=PASS fp8=1 fp4=1 bf16=1 routed=1
DKR_R124_S55B_NUMERIC=PASS cases=120 gguf_moe=6
DKR_R124_S55B_CUBIN_MODULES=25/25
DKR_R124_S55B_NEUTRAL_RECIPES=PASS fp8=1 fp4=1 bf16=1 routed=1
DKR_R124_S55B_FP8_FP4_GPU=PASS tests=6
DKR_R124_S55B_PERF=PASS rounds=3 cases=7
DKR_R124_S55B_PASS
```

No speedup claims are made until the GPU reports are available.
Passing numerical tests certifies the existing S5.2 tolerances,
not bitwise-identical outputs for every possible CUDA entrypoint.
S5.5b remains opt-in; default promotion requires S5.5c and
model-level comparison with unchanged effective clocks.
