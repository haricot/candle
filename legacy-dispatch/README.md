# Candle Legacy Dispatch — format integration

This is a source integration candidate on pinned Candle main 0.11.1, not a
published GPU/model/performance qualification. The exact scope, sources and
payload hashes are in `sources.json`. The three format branches retain separate
providers; `candle-legacy-dispatch` assembles their reviewed deltas.

| Format | Integrated path | Numerical boundary |
| --- | --- | --- |
| BF16 | Ordinary tensor fallback plus prepared BF16 linear, sharing one strict F32 GEMM helper | BF16 range preserved; prepared GEMM rounds before bias; private resident weights |
| FP8 | E4M3FN packed SIMT W8A16 / W8ABF16, existing GEMV and prefill4 kernels | F16/BF16 activation contract explicit; F32 accumulation/output; BF16 cannot select F16 expansion |
| FP4 | Existing typed E7C NVFP4 modes and GGUF MXFP4 path | The two encodings remain distinct; W4A8/DP4A is an explicitly selected approximate contract |

Only the formats listed in the current manifest are present in a format branch.
The BF16 pilot is not a mandatory dependency of the FP8/FP4 branches. All use
the same reviewed pre-Volta build policy, which omits native-only WMMA objects
below SM70 and retains fast F16 reductions. No new MoE SIMT algorithm is added.

## FP8 recovery correction

`fp8_candle` alone was behind the working Mistral path. The W8ABF16 API and CUDA
kernels are extracted from Candle `9cba2e8`, used by Mistral FP8 standalone v2
and the successful Legacy V3 reference. Current main's manifests and the newer
FP8 validation fixes are retained. No whole old Candle or Mistral tree is merged.
The new activation-aware planner filters out F16 expansion for BF16 projections
before considering performance profiles. Typed construction checks that contract
before allocating weights and enforces the activation dtype during execution.
Historical untyped APIs remain compatibility entry points, not an ASD authority.

Existing donor audit documents describe their original revisions and may refer
to historical dependency versions. This directory's source receipt and reviewed
lockfile describe the actual integrated candidate.

## Validation

Use Rust 1.99.0. Portable CPU checks and static CUDA checks are independent:

```sh
RUSTFLAGS='-C target-cpu=x86-64' python3 legacy-dispatch/validate.py cpu
CUDA_COMPUTE_CAP=61 NVCC_CCBIN=gcc-14 RUSTFLAGS='-C target-cpu=x86-64' \
  python3 legacy-dispatch/validate.py cuda-compile
```

Hosted CI runs CPU contracts, compiles actual SM61 PTX and all selected hardware
tests, and verifies required PTX entry points. The combined branch also compiles
SM80. It never initializes a GPU, starts/registers a runner or accesses a model
cache. Each source change requires fresh evidence for the complete candidate.

The physical step is explicitly run by the operator on the SM61 machine:

```sh
CUDA_COMPUTE_CAP=61 NVCC_CCBIN=gcc-14 RUSTFLAGS='-C target-cpu=x86-64' \
  python3 legacy-dispatch/validate.py gpu
```

This executes the selected self-contained tests plus the explicitly named
ignored physical contracts. External-checkpoint and benchmark-only ignored tests
stay separate. Compilation, PTX presence and historical Mistral results are not
physical qualification of the new binaries. Full-model and timing validation
remain distinct subsequent gates.

## ASD and remaining scope

This initial integration connects the BF16 provider/pilot and preserves FP8 R4D
residence dispatch plus FP4 E7C typed selection. It does not pretend that the
BF16 `SelectionPolicy` is already a format-neutral ASD adapter. Shared
cross-format admission and the final generic policy contract remain explicit
follow-on work; callers must partition memory between the current per-format
planners. The built-in paths work without ASD. Flow remains the producer of
measured/qualified profiles and external kernels; adding a linear external ABI
requires an explicit new runtime contract. No live ASD store is modified here.
