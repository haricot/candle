# R1.24 S5.5c — Cudaforge produces a sealed, typed native catalogue

**Status:** experimental source published, local compilation and GPU
qualification still required. This stage does **not** promote the CUBIN
runtime as the default Candle backend.

## Producer and consumer boundaries

The independent Cargo package
\`tools/dkr-s31-builder\` already depends on \`cudaforge = "=0.1.6"\`.
S5.5c adds a **second, offline-only binary**
\`dkr-cudaforge-catalog\` to that package. It uses
\`cudaforge::KernelBuilder\` solely for explicit compute-capability
selection; it **does not run CUDA compilation**, NVCC, ptxas or static
linking.

The producer-owned, versioned recipe profile
\`tools/dkr-s31-builder/recipes/sm61.json\` is the input:
10 GGUF MMQ codecs × eight tiles, plus FP8, FP4, BF16 and
Routed/MoE launch policy. It is **outside candle-core and
candle-kernels**. The historical S31 code and S5.5a/b Python
producers remain as independent regression oracles, not the new
runtime's source of policy.

Before producing output the binary verifies:

- the original S5.1 25-family seal, all 25 PTX hashes and
  per-family manifest hashes, and optional legacy index state;
- all 25 prebuilt CUBIN files (SHA-256, size, ELF magic, target,
  provenance to the corresponding sealed PTX);
- 16 MMQ template keys per codec, each resolved to a real PTX
  entry; FP8/FP4/BF16 symbol names in the qualified packages;
- the exact target SM61 and the producer recipe schema;
- **semantic equality** of generated S5.5a and S5.5b manifests
  to the already GPU-qualified historical references, when the
  recommended \`--compare-s55a\` and \`--compare-s55b\` flags are used.

It publishes only a **new directory** (never modifies the sealed
catalogue):

    target/dkr-r124-s55c/
      mmq/manifest.json          # S5.5a schema-compatible
      recipes/manifest.json      # S5.5b schema-compatible
      catalogue.json             # S5.5c provenance envelope

The envelope binds the two recipe manifests, the CUBIN inventory,
the S5.1 seal and the producer-owned recipe spec through SHA-256.
No images are copied or regenerated.

The new Cargo feature \`cuda-dkr-cudaforge-catalog\` implies both
\`cuda-dkr-neutral-recipes\` and \`cuda-dkr-cubin-embedded\`.
Candle's build script checks the producer, target, schema,
recipe-spec digest and all manifest/provenance links. It requires
the MMQ/typed-recipe paths to refer **exactly** to the Cudaforge
output directories. The S5.5a/b typed and PTX entrypoint validation
still runs. It emits an embedded attestation for runtime traces:
\`DKR_R124_S55C_CATALOGUE_ADMITTED producer=cudaforge ...\`.

The runtime never reads a producer manifest from disk. The feature
is opt-in and fails closed on any missing/mismatched CUBIN or recipe.

## 1. Build the Cudaforge producer (once)

From \`/home/np/tmp/candle_dkr/candle\`:

~~~sh
cd /home/np/tmp/candle_dkr/candle
git pull --ff-only origin feature/dkr-r124-neutral-catalog
python3 -m py_compile tools/dkr-r124-s55c-qualify.py
python3 tools/dkr-r124-feature-audit.py

cargo run --release \
  --manifest-path tools/dkr-s31-builder/Cargo.toml \
  --bin dkr-cudaforge-catalog -- --self-test
~~~

No \`cargo clean\`, \`nvcc\`, \`ptxas\` or CUDA build. The standalone
builder's Cargo dependencies must be available (once on this host).

## 2. Produce and compare manifests against the GPU-qualified baseline

~~~sh
cargo run --release \
  --manifest-path tools/dkr-s31-builder/Cargo.toml \
  --bin dkr-cudaforge-catalog -- \
  --seal "$PWD/target/dkr-r124-s51/seal.json" \
  --cubins "$PWD/target/dkr-r124-s54b" \
  --compare-s55a "$PWD/target/dkr-r124-s55a/manifest.json" \
  --compare-s55b "$PWD/target/dkr-r124-s55b/manifest.json" \
  --out "$PWD/target/dkr-r124-s55c"
~~~

Expected (only after execution):

    DKR_R124_S55C_SEMANTIC_PARITY=PASS manifest=mmq
    DKR_R124_S55C_SEMANTIC_PARITY=PASS manifest=recipes
    DKR_R124_S55C_PRODUCER=PASS backend=cudaforge target=sm61
    DKR_R124_S55C_CATALOGUE=PASS families=25 codecs=10 recipes=4

Any semantic discrepancy stops production *before* writing the
new directory. Keep the original S5.5a/b manifests intact.
\`target/dkr-r124-s55c\` must not exist before the first run.

## 3. Build only the Rust Nextest test archive

~~~sh
python3 tools/dkr-r124-s55c-qualify.py build \
  --seal "$PWD/target/dkr-r124-s51/seal.json" \
  --catalogue-dir "$PWD/target/dkr-r124-s55c" \
  --cubin-dir "$PWD/target/dkr-r124-s54b" \
  --s52-result "$PWD/target/dkr-r124-s52/headless-20261121T124142625489Z/result.json" \
  --s53-result "$PWD/target/dkr-r124-s53/headless-20261121T130221097950Z/result.json" \
  --out "$PWD/target/dkr-r124-s55c-qualification"
~~~

The archive compiles under only \`cuda-dkr-cudaforge-catalog\`,
which transitively enables all required neutral and CUBIN
features. It includes the S5.2 tests, S5.5a/b regression tests,
BF16, SIMT MoE, six additional FP8/FP4 tests, and S5.3 benchmarks.
No CUDA recompilation.

Expected marker:

    DKR_R124_S55C_BUILD=PASS backend=cudaforge features=neutral,cubin

## 4. Preflight without logout

~~~sh
cargo nextest run \
  --archive-file "$PWD/target/dkr-r124-s55c-qualification/s55c-tests.tar.zst" \
  --workspace-remap "$PWD" --run-ignored all --no-capture \
  -E 'test(dkr_s55c_cudaforge_catalogue_build_attestation) | test(dkr_s55b_typed_recipes_match_qualified_s31) | test(dkr_s55a_neutral_mmq_matches_frozen_s31_qualified_shapes)'
~~~

Expect 3/3 PASS including:

    DKR_R124_S55C_ATTESTATION_TEST=PASS families=25 recipes=4 codecs=10
    DKR_R124_S55B_TYPED_PARITY=PASS fp8=1 fp4=1 bf16=1 routed=1
    DKR_R124_S55A_TABLE_PARITY=PASS codecs=10

This catches build/ABI errors before a graphical session is
interrupted. Never start headless if any preflight fails.

## 5. Reuse S5.5b's complete headless GPU matrix

After saving desktop work:

~~~sh
headless-relogin --with-after-run-openlog \
  /usr/bin/env -u DISPLAY -u WAYLAND_DISPLAY \
  python3 "$PWD/tools/dkr-r124-s55c-qualify.py" headless \
  --receipt "$PWD/target/dkr-r124-s55c-qualification/receipt.json"
~~~

S5.5c verifies its own attestation, then invokes the **same
immutable-archive Nextest gates** as S5.5b. The result references
the full numerical/performance report and adds a Cudaforge
receipt with its own schema.

Expected only on a successful real SM61 run:

    DKR_R124_S55C_CATALOGUE=PASS producer=cudaforge families=25
    DKR_R124_S55C_NUMERIC=PASS cases=120 gguf_moe=6 fp8_fp4_tests=6
    DKR_R124_S55C_CUBIN=PASS families=25
    DKR_R124_S55C_PERF=PASS rounds=3 cases=7
    DKR_R124_S55C_PASS

## Remaining scope

- This is a **producer-source migration** for the currently
  qualified target, not proof that other GPU compute capabilities
  support this recipe.
- Historical S31 recipe implementations remain for regression and
  PTX-only rollback. The S5.5c runtime selects produced values,
  not those S31 constants, but some general Candle CUDA/quantization
  types and legacy code still exist.
- S5.5d must include matched-input PTX/CUBIN numeric deltas and
  model-level cold-start, controlled-clocks regression before any
  default promotion.
- SHA-256 attestation proves *internal consistency*, not an
  external cryptographic signature or a trusted release.
