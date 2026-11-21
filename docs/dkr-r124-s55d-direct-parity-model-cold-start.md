# R1.24 S5.5d — Direct PTX/CUBIN numeric + Qwen3 model cold-start

**Status:** experimental qualification implemented, **not yet GPU-tested**.
Qualified baseline: S5.5c (SM61) with Cudaforge catalogue digest
\`91ac006b963a89362c6aaaee052252b4465880c079b0c86f399c4891a5003295\`.
S5.1 native packages, 25 S5.4b CUBINs and S5.5a/b/c report files
are never altered or recompiled.

## Rationale

S5.5a–c use the same S5.2 numerical tolerances, but run only the
CUBIN path in those campaigns. That **does not** constitute a direct
PTX vs CUBIN comparison. S5.5d adds two independently built
binaries from the **same source HEAD**, one with
\`cuda-dkr-embedded\` (PTX) and one with
\`cuda-dkr-cudaforge-catalog\` (precompiled CUBIN).

The test \`candle-nn/tests/dkr_s55d_parity.rs\` serializes **every**
FP32-converted output element for 157 deterministic cases:

| Family | Cases |
|---|---:|
| 10 GGUF codecs × 3 activation dtypes × 4 batch sizes | 120 |
| BF16 rounding, matmul, reduction and unary edge values | 4 |
| FP8 E4M3FN SIMT and DP4A, five batch sizes each | 10 |
| NVFP4 reference and DP4A, five batch sizes each | 15 |
| Six GGUF MoE codec routes + two SIMT MoE routes | 8 |
| **Total** | **157** |

GGUF quantized weights are created on **CPU**, hashed and uploaded
bit-identically to each mode. The Python comparator requires the
same 16 quantized weight SHA-256s, tensor shape, dtype, case names,
and actual float arrays. NaNs/Infs, absent cases, excessive
deviations or invalid provenance are **failures**. It reports
per-case max absolute/relative errors and number of elements
outside the declared tolerance.

Tolerances for **PTX-vs-CUBIN output differences**, not to be
confused with S5.2's independent dequantization-reference contract:

| Family | atol | rtol |
|---|---:|---:|
| GGUF MMVQ/MMQ | 0.001 | 0.001 |
| BF16 | 0.005 | 0.002 |
| FP8 | 0.010 | 0.005 |
| FP4 | 0.005 | 0.002 |
| MoE | 0.005 | 0.002 |

S5.5d also uses a new standalone Cargo workspace binary,
\`tools/dkr-s55d-model-probe\`, rather than \`candle-examples\`.
This matters because \`candle-examples/build.rs\` would compile CUDA
custom-op PTX. The standalone binary links only the existing
Candle and candle-transformers crates. It runs **Qwen3 GGUF**
through the actual \`quantized_qwen3::ModelWeights::from_gguf\`
path, **without tokenizer, network/model downloads, NVCC, ptxas
or static CUDA FFI**. Both model binaries are compiled under
separate \`ptx\` and \`cubin\` features, copied to uniquely named
files, tested with a GPU-free \`--self-test\`, then SHA-256-sealed.

It uses identical user-supplied local Qwen3 GGUF weights and
initial token IDs, deterministic greedy next-token selection,
same number of decode steps and full F32 logits for **every**
step. The qualification fails if either the greedy token sequence
or any logit exceeds the explicit model tolerance
(atol 0.010, rtol 0.005). All complete logits are recorded as
little-endian float32 files and compared, not just top-1 scores.

Time measurements are split into CUDA context creation, GGUF
header read, GGUF model construction, first prefill, each
decode step, and total work in the model process. These are
synchronized GPU-side operations measured with host wall clocks
where relevant. Three alternating-order pairs use new processes
and separate initially empty **CUDA_CACHE_PATH** directories.
No forced GPU clocks, no \`drop_caches\`, and no claim of a
cold OS filesystem/page-cache read. The report records pre/post
GPU power state, clocks, memory and temperature; compare
conditions before interpreting small timing differences.

## Prerequisite: choose one local Qwen3 GGUF

The model is a **required explicit input**; the runner intentionally
does not download it, or infer the model family from a filename.
It must be compatible with Candle's \`quantized_qwen3\` loader,
not Qwen3-MoE or LFM.

Possible local discovery:

~~~sh
find -L "$HOME/.cache/huggingface/hub" -type f \
  -iname '*Qwen3*.gguf' ! -iname '*moe*' -print
~~~

Choose the **same** local GGUF for both modes and set
\`MODEL_GGUF=/absolute/path/to/your/Qwen3-*.gguf\`.
A small 0.6B Q4_K model is recommended for a first run on 8 GiB;
a Qwen3-4B Q4_K can also be tried if GPU memory permits.

## Step 1 — pull and static preflight

~~~sh
cd /home/np/tmp/candle_dkr/candle
git pull --ff-only origin feature/dkr-r124-neutral-catalog
python3 -m py_compile tools/dkr-r124-s55d-qualify.py
python3 tools/dkr-r124-s55d-qualify.py self-test
python3 tools/dkr-r124-feature-audit.py
test -f "$MODEL_GGUF" || { echo 'set MODEL_GGUF to a real local Qwen3 GGUF'; exit 1; }
~~~

Run these in the existing shell, and do **not** run a headless
logout unless all previous gates succeed.

## Step 2 — build two Nextest archives and two standalone model binaries

~~~sh
python3 tools/dkr-r124-s55d-qualify.py build \
  --seal "$PWD/target/dkr-r124-s51/seal.json" \
  --cubin-dir "$PWD/target/dkr-r124-s54b" \
  --catalogue-dir "$PWD/target/dkr-r124-s55c" \
  --s55c-result "$PWD/target/dkr-r124-s55c-qualification/headless-20261121T203451750539Z/s55c-result.json" \
  --s52-result "$PWD/target/dkr-r124-s52/headless-20261121T124142625489Z/result.json" \
  --s53-result "$PWD/target/dkr-r124-s53/headless-20261121T130221097950Z/result.json" \
  --model "$MODEL_GGUF" \
  --tokens 1,2,3,4,5,6,7,8 \
  --decode 8 \
  --out "$PWD/target/dkr-r124-s55d-qualification"
~~~

This needs no new CUDA kernels; it may build additional **Rust**
code in both feature configurations. It does **not** invoke
\`cargo clean\`. If any compilation fails, read
\`archive-ptx.log\`, \`archive-cubin.log\`,
\`model-build-ptx.log\` or \`model-build-cubin.log\`.
Do not launch headless on a build failure.

Expected only on successful compilation:

    DKR_R124_S55D_MODEL_PROBE=PASS mode=ptx no_cuda_build=1
    DKR_R124_S55D_MODEL_PROBE=PASS mode=cubin no_cuda_build=1
    DKR_R124_S55D_BUILD=PASS modes=ptx,cubin numeric=157 model=qwen3-gguf

A receipt containing the model's SHA-256, the two archives,
the two executables, Cudaforge provenance and fixed input token
IDs is written to \`target/dkr-r124-s55d-qualification/receipt.json\`.

## Step 3 — verify both archived tests are present without running GPU work

~~~sh
cargo nextest list \
  --archive-file "$PWD/target/dkr-r124-s55d-qualification/s55d-ptx-tests.tar.zst" \
  --workspace-remap "$PWD" \
  -E 'test(dkr_s55d_capture_outputs)'

cargo nextest list \
  --archive-file "$PWD/target/dkr-r124-s55d-qualification/s55d-cubin-tests.tar.zst" \
  --workspace-remap "$PWD" \
  -E 'test(dkr_s55d_capture_outputs)'
~~~

Each archive must list one ignored \`dkr_s55d_capture_outputs\`
test. Listing does not create a CUDA context.

## Step 4 — headless PTX/CUBIN numeric and true-model tests

Only after saving any graphical work:

~~~sh
headless-relogin --with-after-run-openlog \
  /usr/bin/env -u DISPLAY -u WAYLAND_DISPLAY \
  python3 "$PWD/tools/dkr-r124-s55d-qualify.py" headless \
    --receipt "$PWD/target/dkr-r124-s55d-qualification/receipt.json"
~~~

Headless requires that the S5.5c PASS report and all input hashes
remain unchanged. It first evaluates both numeric archives in
new processes, using private CUDA caches, checks image provenance
(PTX baseline must **not** load CUBIN; Cudaforge path must load
actual CUBIN) and compares >70,000 tensor elements.

It then runs the **same real GGUF** with the two binaries in three
fresh-process pairs, alternating which mode runs first, compares
all greedy token IDs and **all logits at every generation step**,
and reports medians for each performance metric. No threshold is
set for speedup: a slower CUBIN model trial is recorded honestly,
not hidden behind a passing numeric test.

Expected only after all GPU gates genuinely succeed:

    DKR_R124_S55D_NUMERIC=PASS cases=157 elements=...
    DKR_R124_S55D_MODEL_PAIR=PASS round=1 logits=... tokens=9
    DKR_R124_S55D_MODEL_PAIR=PASS round=2 logits=... tokens=9
    DKR_R124_S55D_MODEL_PAIR=PASS round=3 logits=... tokens=9
    DKR_R124_S55D_MODEL=PASS rounds=3 full_logits_comparison=1
    DKR_R124_S55D_PERF=PASS rounds=3 modes=ptx,cubin
    DKR_R124_S55D_PASS

The receipt is written only after **all** tests pass.
Logs and partial numeric data remain available for diagnosis
if a test fails. No source commit or CUBIN file is modified
by the headless runner.

## Promotion decision remains separate

S5.5d PASS would verify the 157-case tensor contract and one
specific Qwen3 GGUF model. It would not prove all models, tensors,
FP8/FP4 corner cases, other GPU compute capabilities, production
latencies under load, or bitwise identical outputs. The user must
review end-to-end differences and operational stability before
allowing a new default backend.

The Cudaforge feature is still **opt-in**.
