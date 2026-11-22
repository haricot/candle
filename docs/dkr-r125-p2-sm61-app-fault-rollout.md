# R1.25 P2 — SM61 app-scoped default and genuine CUDA admission-fault fallback

**State:** P2 implementation published; build and SM61 headless qualification **pending**.
Predecessor P1 was GPU-qualified on GTX 1080 Max-Q (SM61), source
`8c7e6610599f12caaa810e580258bf7e18b863e0`,
P1 Cudaforge/25-family catalogue SHA-256:
`91ac006b963a89362c6aaaee052252b4465880c079b0c86f399c4891a5003295`.

## Scope: application-level promotion, NOT generic CUDA default

P2 adds two independent Cargo profiles to the standalone offline Qwen3
GGUF application `dkr-s55d-model-probe`:

| Feature on app package | Purpose | Contains malformed-CUBIN fault code? |
|---|---|---|
| `sm61-app` | **Shipping candidate** for SM61. Enables `promoted`/`cuda-dkr-qualified-default`. Unset `CANDLE_DKR_IMAGE_MODE` defaults to verified CUBIN; `ptx` is explicit rollback | **No** |
| `p2-fault-canary` | **Qualification binary only**. Enables `promoted` plus `cuda-dkr-p2-test-fault` in Candle core/nn. | **Yes**, behind optional isolated feature |

No default features of `candle-core`, `candle-nn`, `candle-transformers`
or the general `cuda` feature have been changed. This is an **explicitly
built SM61 application profile**, not a universal backend override.
Do not deploy or distribute the fault-enabled executable.

## Real CUDA driver fault, with immutable verified artefacts

The test-only feature recognises exactly one optional knob:

`CANDLE_DKR_P2_TEST_FAULT_CAST=1`

It is **read once during CUDA device creation** and only exists in the
separate `p2-fault-canary` binary. When the runtime first tries to
load the native CUBIN for `cast`, it calls
`CudaContext::load_module(Ptx::from_binary(vec![0u8;64]))`.
This is an **actual CUDA driver API call with an intentionally invalid
temporary image**. It does not touch the original sealed image, PTX,
manifest, SHA-256 identity or cache entry.

The expected driver error is emitted as
`DKR_R125_P2_DRIVER_REJECTED family=cast error=...`.
Under `auto`, P1's existing module-admission-only fallback must then
load the **verified embedded PTX** for `cast`; other native families
remain on the attested CUBIN. The same error under strict `cubin`
must abort without generating a successful inference report. Under
forced `ptx`, the malformed CUBIN must never be submitted to the driver.

An invalid `CANDLE_DKR_P2_TEST_FAULT_CAST` value fails closed. The
shipping application excludes the hook entirely; even if the test
variable is mistakenly inherited, it has no effect on that binary.
The environment-based fault hook is deliberately **not** exposed in
a production feature. No kernel launch failures are retried.

## Model/prompt matrix: same GGUF, full logits

Instead of P1's single eight-token input, P2 runs **the same
shipping Rust release executable** in `auto`, `ptx` and `cubin`
for these three deterministic Qwen3 input/decode lengths:

| Scenario | Initial prompt tokens | Decode steps | Full logits per backend |
|---|---:|---:|---:|
| short | 8 | 8 | 9 × 151,936 |
| medium | 24 | 16 | 17 × 151,936 |
| long | 48 | 24 | 25 × 151,936 |

Each mode and scenario runs in a separate fresh CUDA process and
isolated JIT cache. The runner checks that all loaded kernel families belong to the
25-family sealed catalogue and that the same per-scenario family set
is selected in PTX, auto and CUBIN. Longer prefill may legitimately
select additional MMQ families. It requires the expected image format,
and compares generated greedy tokens and **every F32 logit**
against the PTX run (absolute tolerance 0.01, relative tolerance
0.005). This is numerical equivalence, not a claim that all raw
logit bit patterns are identical.

P2 also reruns all **157 S5.5d fixed-input numerical cases** in
`ptx` and fault-enabled `auto`, using the same
`p2-fault-canary` Nextest archive. The latter must show a single
`cast` PTX fallback with other modules remaining CUBIN, zero
numerical mismatches and the same 16 quantized-weight SHA-256s.
This establishes that a *mixed but individually verified* fallback
configuration is numerically correct.

Finally the P2 canary runs Qwen3 `short` under fault-enabled
`auto` and strict `cubin`; a real driver rejection must be visible
in both, but only `auto` may succeed. This also tests full logits
of the fallback model against the shipping application's
regular `auto` output.

## Reuse all S5.5/P1 artifacts, do not rebuild CUDA

On the GTX 1080 Max-Q host:

~~~sh
cd /home/np/tmp/candle_dkr/candle
git pull --ff-only origin feature/dkr-r124-neutral-catalog

export MODEL_GGUF="$PWD/target/dkr-r124-s55d-models/Qwen3-0.6B-Q4_K_M.gguf"
export P2_OUT="$PWD/target/dkr-r125-p2-qualification"

python3 -m py_compile tools/dkr-r125-p2-qualify.py
python3 tools/dkr-r125-p2-qualify.py self-test
python3 tools/dkr-r124-feature-audit.py
~~~

Expected CPU-only markers:

    DKR_R125_P2_SELF_TEST=PASS scenarios=3 prompt_lengths=8,24,48
    DKR_R125_P2_FEATURE_AUDIT_PASS mode=sm61-app fault=0
    DKR_R125_P2_FEATURE_AUDIT_PASS mode=p2-fault-canary fault=1
    DKR_R125_P2_FEATURE_GRAPH=PASS

Compile the two profile-specific Rust Qwen3 executables and
one Nextest archive. This will not run NVCC/ptxas, rebuild native
CUDA kernels, or overwrite earlier qualification results:

~~~sh
python3 tools/dkr-r125-p2-qualify.py build \
  --seal "$PWD/target/dkr-r124-s51/seal.json" \
  --cubin-dir "$PWD/target/dkr-r124-s54b" \
  --catalogue-dir "$PWD/target/dkr-r124-s55c" \
  --p1-result "$PWD/target/dkr-r125-p1-qualification/headless-20261122T084545343274Z/result.json" \
  --model "$MODEL_GGUF" \
  --out "$P2_OUT"

echo "DKR_R125_P2_BUILD_EXIT=$?"
~~~

Expected only upon successful local compilation:

    DKR_R125_P2_BUILD=PASS shipping_profile=sm61-app fault_profile=isolated scenarios=3
    DKR_R125_P2_BUILD_EXIT=0

The release artifacts, SHA-256s and exact compiled source HEAD are
sealed in `$P2_OUT/receipt.json`. The shipping candidate is
`$P2_OUT/bin/sm61-app-qwen3`. The fault-enabled binary is a separate
file `$P2_OUT/bin/p2-fault-canary-qwen3`: **never deploy it**.

### CPU-only test archive check before headless

~~~sh
cargo nextest list \
  --archive-file "$P2_OUT/p2-fault-tests.tar.zst" \
  --workspace-remap "$PWD" \
  -E 'test(qualified_default_policy_has_explicit_rollback_only) | test(dkr_s55c_cudaforge_catalogue_build_attestation) | test(dkr_s55d_capture_outputs)'
~~~

The listing must contain both unit tests and the ignored 157-case
capture test. If the build/list check fails, **do not** log out or
change the P1-qualified default.

### SM61 GPU canary, after saving the graphical session

~~~sh
headless-relogin --with-after-run-openlog \
  /usr/bin/env -u DISPLAY -u WAYLAND_DISPLAY \
  python3 "$PWD/tools/dkr-r125-p2-qualify.py" headless \
  --receipt "$P2_OUT/receipt.json"
~~~

Expected only after a real successful GPU run:

    DKR_R125_P2_FAULT_NUMERIC=PASS cases=157 elements=76292
    DKR_R125_P2_MODEL=PASS scenario=short modes=3 logits=1367424
    DKR_R125_P2_MODEL=PASS scenario=medium modes=3 logits=2582912
    DKR_R125_P2_MODEL=PASS scenario=long modes=3 logits=3798400
    DKR_R125_P2_FAULT=PASS actual_driver_rejection=1 auto_fallback=ptx strict_cubin=rejected
    DKR_R125_P2_ROLLOUT=PASS sm61_app_cubin_default=1
    DKR_R125_P2_PASS

The local `headless-*/result.json` is written only on full PASS;
partial logs remain available on any failure.

## Go/no-go

P2 can be closed only if the final PASS report includes all
three model scenarios, unchanged PTX/CUBIN numerics, the real CUDA
driver error and successful fallback, strict-CUBIN fail-closed,
unmodified artefacts, and no test-hook contamination in the shipping
SM61 application. **Do not claim P2 PASS from code publication alone.**

In a later, explicitly scoped `mistral.rs` application rollout,
its Cargo features must propagate this same qualified profile
through `candle-core` and `candle-nn`; it must not trigger the
old `cuda-dkr-s31` static FFI path. That integration is outside
this Candle-only canary and requires its own targeted test.

## ASD V3

ASD V3's `ResolvedAsdPlan`, exact convolution specialization,
external artifact/manifest verification and hot-swap/fallback
authority are unchanged. This P2 deployment profile only controls
the native DKR family image. Do not merge ASD's seven qualified
kernels into the 25-family DKR catalogue and do not lower ASD
profile priority for matching convolution signatures.
