# R1.25 / P1 — Progressive qualified Cudaforge/CUBIN default on SM61

**Status: source implemented; local Rust build and GPU canary pending.**
Baseline R1.24 S5.5d GPU-qualified on the GTX 1080 Max-Q:
157/157 full PTX/CUBIN numerical cases (76,292 bitwise-equal F32s),
three Qwen3-0.6B GGUF pairs (zero numerical logit difference),
25 sealed PTX/CUBIN family manifests.

## What "default" means at this stage

Candle has **not** changed its generic \`cuda\` or Cargo \`default\`
feature for all users. This is intentionally **Phase P1, an opt-in
deployment profile**. Only an explicitly built
\`cuda-dkr-qualified-default\` binary defaults to the verified
precompiled image. That profile implies
\`cuda-dkr-cudaforge-catalog\`, all verified neutral recipes,
and 25 CUBINs at compile time.

The catalogue's attested target supplies the compute capability.
The generic runtime does **not** embed an SM61-specific
dispatch table. The currently qualified producer profile targets
SM61 only, and device admission rejects different compute
capabilities. Missing/modified CUBINs or an inconsistent recipe
manifest fail **closed** at the build/catalogue admission gate.

One environment variable controls deployment image selection,
read **once per CudaDevice** on initialization:

| \`CANDLE_DKR_IMAGE_MODE\` | Behavior |
|---|---|
| Unset or \`auto\` | Default: verified CUBIN; retry verified embedded PTX **only** if loading its CUDA module/function fails, with explicit stderr fallback trace |
| \`cubin\` | Verified CUBIN exclusively; any load error propagates, no fallback |
| \`ptx\` | Explicit rollback to the corresponding embedded, verified PTX; still validates the sealed CUBIN catalogue and target at admission |
| Any other value | Reject during CUDA device creation before executing inference |

The runtime **never** retries a failing kernel computation in a
different backend, preventing duplicate side effects.
The mode is fixed for the CUDA context lifetime: changing the
environment without constructing a new device does not invalidate
its content-addressed module cache. The PTX and CUBIN are the same
ABI identity and the same 25-family producer catalogue, and share
the same per-context cache (one selected module per artifact).

With \`CANDLE_DKR_TRACE=1\`:
- \`DKR_R125_P1_POLICY default=cubin requested=auto rollback=verified_ptx source=embedded\`
- \`CANDLE_DKR_IMAGE_LOADED ... format=cubin ...\`
- explicit PTX mode prints \`format=ptx\`;
- only a real pre-launch admission failure in \`auto\` prints
  \`DKR_R125_P1_FALLBACK ... selected=ptx ...\` **unconditionally**.

The previous \`cuda-dkr-cudaforge-catalog\` feature retains its
original **strict-CUBIN** behavior and the prior S5.5a–d
test/archive artifacts remain unchanged.

## Before starting: qualified local inputs

Use the local \`target\` directory already populated by S5.5a–d
and the same Qwen3-0.6B GGUF. No CUDA kernel build, ptxas, NVCC,
\`cargo clean\`, re-quantization or new model download.

~~~
cd /home/np/tmp/candle_dkr/candle
git pull --ff-only origin feature/dkr-r124-neutral-catalog

export MODEL_GGUF="$PWD/target/dkr-r124-s55d-models/Qwen3-0.6B-Q4_K_M.gguf"
export P1_OUT="$PWD/target/dkr-r125-p1-qualification"

python3 -m py_compile tools/dkr-r125-p1-qualify.py
python3 tools/dkr-r124-feature-audit.py
~~~

Feature audit expected:
\`DKR_R125_P1_FEATURE_GRAPH=PASS\`.
It also checks the feature propagation into
\`candle-nn\` and the standalone Qwen3 probe; static
CUDA FFI flags remain absent.

## Build one promoted numeric archive and one model executable

~~~
python3 tools/dkr-r125-p1-qualify.py build \
  --seal "$PWD/target/dkr-r124-s51/seal.json" \
  --cubin-dir "$PWD/target/dkr-r124-s54b" \
  --catalogue-dir "$PWD/target/dkr-r124-s55c" \
  --s55d-result "$PWD/target/dkr-r124-s55d-qualification-r2/headless-20261122T074955864763Z/result.json" \
  --model "$MODEL_GGUF" \
  --out "$P1_OUT"

echo "DKR_R125_P1_BUILD_EXIT=$?"
~~~

This compiles Rust via \`cargo nextest archive\` and one
release Qwen3 GGUF probe. It snapshots a SHA-256-bound receipt
with the exact model, binaries, source HEAD and S5.5c producer
catalogue. It only writes to a **fresh** qualification directory.
Previous archives and 25 CUBINs are not overwritten.

The CPU-only unit preflight is archived with the numeric test:

~~~
cargo nextest run \
  --archive-file "$P1_OUT/promotion-tests.tar.zst" \
  --workspace-remap "$PWD" \
  --run-ignored all --no-capture \
  -E 'test(qualified_default_policy_has_explicit_rollback_only) | test(dkr_s55c_cudaforge_catalogue_build_attestation) | test(dkr_s55b_typed_recipes_match_qualified_s31)'
~~~

Expected:
\`DKR_R125_P1_POLICY=PASS auto=cubin rollback=ptx invalid=reject\`
plus S5.5b/Cudaforge attestation test results. If the
build or preflight fails, **do not** log out into headless.

## Headless promotion canary — same binary, three policies

After saving your desktop work and successful build/preflight:

~~~
headless-relogin --with-after-run-openlog \
  /usr/bin/env -u DISPLAY -u WAYLAND_DISPLAY \
  python3 "$PWD/tools/dkr-r125-p1-qualify.py" headless \
    --receipt "$P1_OUT/receipt.json"
~~~

The canary performs, in this order:

1. Verify unchanged S5.5d PASS, model SHA, S5.5c
   Cudaforge receipt, 25 CUBIN manifests, and both new binaries.
2. Run the feature-gated unit tests and reject an invalid
   \`CANDLE_DKR_IMAGE_MODE\`.
3. Run **one identical Nextest numeric binary** three times
   with unset (\`auto\`), forced PTX and forced CUBIN. Each
   performs 157 tests; compare full F32 outputs and recorded
   quantized-weight byte hashes, requiring zero violations.
4. Inspect **actual image traces** for each mode, rejecting
   unexpected PTX inside the auto/CUBIN launch or unexpected
   CUBIN inside the forced PTX mode. Any CUBIN admission
   fallback is a failure for the promotion canary even if
   normal runtime auto would permit it.
5. Run **one identical Qwen3 model binary** under all three
   modes, with empty private CUDA JIT caches, a fresh process,
   identical GGUF and token IDs. Require identical greedy
   token IDs and a full 9×151,936-logit comparison with zero
   tolerance violations.
6. Store the timing differences and GPU pre/post context
   without treating unconstrained clocks as a throughput
   qualification.

Expected only after actual SM61 execution:

~~~
DKR_R125_P1_NUMERIC=PASS policies=auto,ptx,cubin cases=157
DKR_R125_P1_DEFAULT=PASS policy=auto selected=cubin
DKR_R125_P1_ROLLBACK=PASS policy=ptx selected=ptx
DKR_R125_P1_INVALID=PASS policy=invalid rejected=1
DKR_R125_P1_MODEL=PASS policies=3 logits_full=1
DKR_R125_P1_PASS
~~~

Failure leaves its logs and outputs intact; only a complete
PASS writes a final \`result.json\`.

## Phased promotion decision

- **P1 (this patch):** application/deployment opt-in
  to qualified-default, rollback and canary tests.
- **P2 (after P1 GPU PASS):** make this feature the
  **application-specific default on qualified SM61 builds**,
  preserve \`CANDLE_DKR_IMAGE_MODE=ptx\`, and collect a
  broader model/prompt/regression matrix.
- **P3 (later):** evaluate cross-architecture catalogue
  distribution and default selection in generic packages.
  Do **not** enable in generic \`cuda\` without a separately
  attested catalogue for each target or without a clean
  way to build CUDA for other GPUs.

### ASD V3 boundary

This patch changes the DKR native module image policy only.
It does **not** merge or change ASD V3, its
\`ResolvedAsdPlan\` selection, its user-local profiles, its
fallback-store authority or its content-addressed external
artefacts. The future DKR+ASD integration must preserve
the **ASD Exact Profile selection priority** for an eligible
specialized convolution and use DKR only for its own
general kernel families. Never interpret the 25 DKR families
as the seven qualified ASD exact kernels.
