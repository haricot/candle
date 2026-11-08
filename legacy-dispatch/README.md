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

This integration connects the BF16 provider/pilot and preserves FP8 R4D
residence dispatch plus FP4 E7C typed selection. S1 adds the format-neutral
planning/preference contract and execution facade described below. The BF16
`SelectionPolicy` remains specific to BF16. Shared cross-format admission and
the connection to an actual ASD policy remain follow-on work; callers must
partition memory between layers and the current per-format planners. The built-in paths work without ASD. Flow remains the producer of
measured/qualified profiles and external kernels; adding a linear external ABI
requires an explicit new runtime contract. No live ASD store is modified here.

## Common linear contract (S1)

`dispatch::Bf16LinearBackend` is the explicit name of the BF16 pilot's execution
choices. The old `LinearBackend` is a deprecated compatibility alias. Other S0
planning types and `SelectionPolicy` still describe BF16; they are not the
format-neutral extension interface.

`dispatch::contract` separates weight encoding, activation dtype, accumulation,
output dtype, bias rounding, approximation permission and input layout. A
`LinearProvider` reports its implementation ID, device kind, constraints and
logical memory needs. `select_linear` rejects incompatible candidates and
resource excesses before calling `LinearPreference`. A preference cannot add a
candidate, cross from CUDA to CPU, change a numerical contract or bypass a budget.
IDs are extensible: a future F16 provider does not require a new BF16 enum variant.
These are planning declarations, not hardware execution evidence.

`dispatch::providers::PreparedLinear` rebuilds availability from the compiled
providers, inspects the actual device, validates the payload through the existing
constructors, and checks input/output contracts at execution. It offers rank-two
linear execution through the existing providers:

| Common adapter | Contract | Current architecture scope |
| --- | --- | --- |
| BF16 | BF16 weights/activations/output, F32 accumulation; optional BF16 GEMM rounding before bias | Existing BF16 selector's CPU/CUDA rules |
| Packed FP8 | E4M3FN weights, F16 or BF16 activations, F32 accumulation/output | SM61 only in this adapter |
| ModelOpt NVFP4 | E2M1 weights + E4M3FN block-16 scales and global multiplier; F16 activations, F32 accumulation/output | SM61 only in this adapter |

The common contract can express F16 weights and GGUF MXFP4, but currently reports
no eligible built-in adapter for those requests. Existing MXFP4, FP8 measured
F16-expansion and explicitly approximate DP4A APIs remain available separately.
They need adapters with their own numeric/resource contracts before a common
preference can select them. Broadening the packed-provider architecture scope
requires corresponding provider validation; `< SM80` alone is not eligibility.

Each `PreparedLinear::prepare` receives a dedicated per-layer memory partition.
There is no common resident ledger yet: callers must partition budgets across
layers/formats and reserve source tensors, retained outputs, preparation peaks
and backend/allocator overhead separately. The existing BF16 `DispatchContext`
continues to offer shared admission for its own directly prepared BF16 layers.
No ASD store, external kernel ABI or tuning producer is connected by S1.

The `dispatch_linear_contract` tests exercise extension with an independent F16
planning fixture, eligibility before preference, dtype/format/layout separation,
resource rejection, CPU execution parity and the manually run SM61 facade test.
The fixture does not advertise an executable F16 provider.
