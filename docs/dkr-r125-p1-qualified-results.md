# R1.25 P1 — Qualified CUBIN default and explicit PTX rollback (SM61)

**Status: PASS for phase P1 canary.** This qualification does not alter the
generic Candle `cuda` default and does not qualify ASD V3 or other GPU
architectures.

## Provenance

- GPU: NVIDIA GeForce GTX 1080 with Max-Q Design, compute capability 6.1,
  NVIDIA driver 580.196.07, reported P0, memory clock 5005 MHz.
- Headless-relogin: `20261122-094541-1000-29285`, 2026-11-22
  09:45:41–09:47:16 Europe/Paris, **exit_code=0**.
- **Exact source HEAD used to build and run the canary**:
  `8c7e6610599f12caaa810e580258bf7e18b863e0`.
- Catalogue Cudaforge SHA-256:
  `91ac006b963a89362c6aaaee052252b4465880c079b0c86f399c4891a5003295`.
- Model: `Qwen3-0.6B-Q4_K_M.gguf`, SHA-256
  `ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a`.
- Prompt IDs: `1,2,3,4,5,6,7,8`; eight decode steps, greedy decoding;
  vocabulary 151,936, model tensors 310.
- Local report:
  `target/dkr-r125-p1-qualification/headless-20261122T084545343274Z/result.json`.
- Uploaded verification evidence:
  `Texte collé(20261122-102615).txt`,
  `result(10).json`, `numeric-comparison.json`, all three
  `numeric-{auto,ptx,cubin}.json` and all three
  `model-{auto,ptx,cubin}.json`.

## Policy and provenance gates

- `DKR_R125_P1_POLICY=PASS auto=cubin rollback=ptx invalid=reject`
- 3/3 policy/typed-recipes/Cudaforge-attestation Nextest tests PASS.
- Invalid `CANDLE_DKR_IMAGE_MODE` rejected before any numerical output.
- The `auto` mode (environment unset) loads **CUBIN**. The forced
  `cubin` mode loads **CUBIN**. The forced `ptx` mode loads **PTX**.
- During real Qwen3 inference, **nine actual native families** were
  loaded in each mode: `quantized`, `cast`, `unary`, `indexing`,
  `reduce`, `mmvq_gguf`, `fill`, `affine`, `binary`.
  All nine image traces agree with the configured mode; there are
  no surprise mixed images, and no `DKR_R125_P1_FALLBACK` marker.
- The catalog inventory is 25 families, **not** a claim that all
  25 are loaded by this particular Qwen3 model.

### Direct numeric comparison

Each of the three modes ran the exact same 157 deterministic cases,
with **16 identical quantized-source-weight SHA-256s** across modes.
The complete F32 values were compared using the S5.5d comparator.

| Comparison | Cases | F32 elements | Mismatches | Float32-bit-equal cases | Max abs error |
|---|---:|---:|---:|---:|---:|
| PTX vs auto | 157/157 | 76,292 | 0 | 157 | 0.0 |
| PTX vs cubin | 157/157 | 76,292 | 0 | 157 | 0.0 |

Tests include GGUF MMVQ/MMQ (120 combinations), BF16, FP8,
FP4 and MoE. **Exact-bit parity applies to these captured
test tensors**, not automatically to any future kernel or model.

### Qwen3-0.6B full logits

The same compiled `promoted-qwen3` executable was launched in
three fresh CUDA processes with isolated JIT caches and the
environment policy varied. All returned the same nine greedy
token IDs:

`[3764, 3, 9, 2, 9, 2, 9, 2, 9]`.

Comparing full prefill+decode logits to PTX:

| Comparison | Values | Numerical mismatches |
|---|---:|---:|
| PTX vs auto | 1,367,424 | 0 |
| PTX vs cubin | 1,367,424 | 0 |
| **Total** | **2,734,848** | **0** |

Model logits passed the declared numerical tolerances.
The comparison does **not** establish exact IEEE-754 bitwise
identity for all logits (signed zeros, etc.).

### Model timing (one new-process run per policy, GPU-synchronized)

| Metric | PTX | auto (CUBIN) | strict CUBIN |
|---|---:|---:|---:|
| CUDA device initialization | 64.961 ms | 66.420 ms | 64.880 ms |
| GGUF header read | 734.694 ms | 734.666 ms | 736.910 ms |
| Model construction | 10,802.156 ms | 153.252 ms | 148.322 ms |
| First prefill | 16,936.983 ms | 26.194 ms | 24.209 ms |
| Total in-process work | 28,603.748 ms | 1,046.213 ms | 1,035.656 ms |

- PTX / `auto` in-process cold-start ratio: **27.34×**.
- PTX / strict CUBIN ratio: **27.62×**.
- Model initialization and prefill benefits primarily reflect
  avoiding cold PTX driver JIT; they do not establish a 27×
  improvement in steady-state token throughput.
- Each policy was measured once in P1; for three-round medians
  see the earlier S5.5d campaign. GPU clocks and OS page cache
  were not strictly controlled.

## P1 decision: PASS, promote only the opt-in deployment profile

The canary confirms the new feature `cuda-dkr-qualified-default`
chooses CUBIN when unset, that a forced PTX rollback functions on
the same binary, and that output equivalence is preserved on the
documented model/test matrix. Existing feature
`cuda-dkr-cudaforge-catalog` remains strict-CUBIN. Other
architectures and generic Candle `cuda` remain unaffected.

**Outstanding evidence before P2/application rollout:**
1. The `auto` path's *real* module-admission failure followed
   by PTX retry has **not** been fault-injected; the P1 run
   verified the normal default and explicit PTX rollback only.
2. Wider model/prompt lengths, launch failures, resource
   pressure, process restarts, and negative-path telemetry.
3. Application-specific integration; do not enable universal
   Candle CUDA defaults from this P1 report alone.
4. ASD V3's `ResolvedAsdPlan`, exact convolution profiles,
   fallback-store priorities and hot profile updates are out
   of scope; preserve them in any future DKR/ASD convergence.

No sealed S5.1 PTX, S5.4b CUBIN, S5.5c catalogue or GGUF
input was changed by this qualification or by this document.
