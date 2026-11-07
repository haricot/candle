# Candle Dispatch S0: Device Capabilities and BF16 Linear

Branch: `capability-dispatch-standalone`.

Base: `haricot/candle:main` at
`cfcbd7186d13271843ce7eb41d25659bd0835850` (Candle 0.11.1).

Status: S0 A+B is implemented and CPU-checked. CUDA compilation and physical
qualification are separate gates; no GPU or performance result is claimed here.
The adaptive-dispatch audit is complete at
`0203aa56fd689b01838e19bf9f7e12887f78b17a`. The initial 404 was the GitHub
installation's selected-repository scope, subsequently corrected by the user.
See `adaptive-dispatch-audit.md` for the recovery decision.

## Objective

An application describes an operation and its numerical and resource contract.
Candle determines which available implementations can satisfy that contract,
prepares a device-bound execution plan and executes it. The first concrete
consumer is a dense BF16 linear layer, with SM61 as the physical reference.
Support for additional architectures requires its own evidence.

The standalone starts from pinned main. Source recovery is selective; existing
Legacy/Extended/ASD branches are donors, not wholesale merge prerequisites.
Their complete Cargo manifests, compiled profiles, applications and experiments
must not replace current main. Sources and dispositions are in `sources.json`
and `recovery.md`.

## A: common capability and selection foundation

- Separate immutable per-device/context facts from available implementation
  artifacts and variable workload resources. Free memory is not a static GPU
  capability.
- Express operation-specific eligibility: storage, conversion, arithmetic and
  accumulation capabilities are distinct. A device-wide `supports_bf16` flag
  cannot establish that a particular GEMM provider accepts BF16 operands.
- Keep mathematical operation/format/layout identity separate from run IDs,
  timestamps and application layer names. Device/context identity and relevant
  policy/artifact generations belong in resolved-plan cache identity.
- Separate legal execution, numerical qualification and measured preference.
  A performance profile never enables an incompatible implementation.
- Provide deterministic selection, explicit reasons for rejected candidates,
  and a policy extension point for ASD. An explicit provider request must still
  satisfy the same operation, device and resource contracts.
- Prepare module/function handles and execution metadata outside the hot path.
  No filesystem polling, hashing, candidate search or NVCC invocation per
  forward. Keep CUDA submission and observed completion distinct.
- Treat physical architecture, compiled kernel targets and external artifact
  compatibility as separate facts. Do not replace every SM61 equality guard
  with a blanket pre-SM80 check.

## B: prepared dense BF16 linear pilot

The pilot contract is `y = x @ w.T + bias`, with BF16 inputs/weights/output,
F32 accumulation and an explicit bias/rounding order matching the reference.
The implementation must preserve BF16 exponent range; a hidden F16 intermediate
or additional activation quantization is not an equivalent backend.

Preparation validates weight, bias and execution-device compatibility and
describes persistent and transient bytes. Resource admission accounts for
existing weights, KV reservations, workspace and peak preparation memory.
Related layers must not each claim the same model-wide free-memory snapshot.

Runtime activation shape/layout validation covers leading batch dimensions,
nonzero offsets, prefix views with larger backing allocations, transposes and
broadcast strides. Required materialization is part of the plan and its memory
cost. A contiguous view is not necessarily compact or zero-offset.

Resident weight representation is chosen at preparation. Shape-dependent
prefill/decode kernel selection may vary within the prepared contract without
repacking all weights on every call. Weight mutation must either be excluded
by an enforceable inference-only ownership contract or invalidate prepared
state through explicit generation tracking; a shallow Tensor clone is not a
proof of immutable storage. The exact ownership mechanism remains an API audit
item rather than an assumed capability of existing Candle tensors.

CPU execution supplies numerical references and software tests. CUDA execution
uses actually available compatible providers. Existing CPU, Metal and newer
CUDA behavior outside the opt-in pilot remains under the upstream contracts.
The initial standalone is not a global replacement of `candle_nn::Linear`.

## ASD and tuner boundary

The shared foundation must work without a profile, user artifact store or tuner.
ASD V3 supplies reusable plan/cache/identity mechanisms and a future policy
adapter. Existing convolution signatures and `asd.xwo.f32.v1` cannot be silently
relabeled as a BF16 linear ABI. The full ASD module/profile integration is a
separate milestone, with an explicit dependency on this foundation's contract.

Flow-Adaptive Tuner retains candidate generation, architecture-specific source,
compilation, evidence feedback, search budgets and artifact publication. The
later `asd_tuned` integration consumes qualified decisions. S0 does not embed
the tuner, import D6 predictions as execution authority or restore the retired
embedded NVRTC tuning runtime.

FP8/FP4 execution, MoE routing, attention/KV implementation, stateful ShortConv,
full CPU/GPU trace collection and new kernel search are outside A+B. Their
existing contracts inform extension design without activating those features.

## Acceptance sequence

1. Preserve the completed adaptive-dispatch audit and exact source pins as the
   basis for the selector implementation and its explicit exclusions.
2. Implement and check A against pinned main, with pure selection tests for
   incompatible providers, unavailable artifacts, resource admission, explicit
   provider requests and device/context/generation isolation. Simulated device
   descriptions are software evidence, not physical GPU qualification.
3. Implement B and a small public consumer with no application-side SM61 branch.
   Validate BF16 values outside the F16 range, scale-sensitive values, bias
   ordering, layout normalization and reported memory requirements.
4. Check CPU and CUDA compilation separately. Test actual SM61 execution and
   compare prepared versus reference paths with fixed workload and declared
   tolerances. Reuse the tuner's measurement discipline, while adapting its
   convolution-specific probes explicitly. Diagnostics stay out of timing runs.
5. Adapt a separate Mistral consumer against an exact Candle revision and run a
   fresh complete integration campaign. Historical Legacy V3 results remain
   reference evidence for their original bytes, not proof for a new binary.

The assistant does not register/start the user's GPU runner or install/update
the live ASD store. Physical execution remains an explicit user-run step.


## Public S0 consumer

```rust,ignore
use candle_core::dispatch::{DispatchContext, ResourceBudget};
let budget = ResourceBudget::from_available(available, reserved, headroom, workspace)?;
let session = DispatchContext::new(&device, budget, None)?;
let linear = session.prepare_bf16_linear(&weight, bias.as_ref(), max_rows)?;
let output = linear.forward(&input)?;
```

`PreparedBf16Linear` implements Candle's `Module` and is also re-exported by
`candle_nn`. Existing `Linear` construction and global precision settings retain
upstream behavior. A prepared layer always takes a private compact weight/bias
snapshot, so subsequent source-variable mutation does not update that layer.
Reprepare to use new weights. Layer clones share the snapshot and admission;
last-clone drop refunds its resident reservation.

S0 supports eager CPU/CUDA inference. Each session binds an actual Candle
Device (including its context/stream identity) and inspects it once. A pure
`select_bf16_linear` function accepts declared facts for testing/planning, but
those declarations cannot be injected into an executing session. The CUDA
session resolves the needed conversion/copy/add modules during construction;
incompatible compiled PTX is rejected by the actual CUDA loader. There is no
process-global dispatch cache or profile authority in this milestone.

`SelectionPolicy` is a cold-path preference hook. It receives only eligible
providers, and even an explicit `RequireBackend` cannot bypass eligibility.
The default order is native CUDA BF16, CUDA F32 accumulation, or CPU F32 on a
CPU session; it never moves CUDA work to CPU. Unknown CUDA major versions use
no assumed native-BF16 provider. This is an architectural planning rule, not
physical qualification of every listed architecture.

The numerical contract rounds GEMM to BF16 before adding BF16 bias, then
rounds the final result to BF16. CUDA F32/BF16 GEMMs use explicit F32 compute
without changing process-wide reduced-precision flags. All input activation
views are logically normalized before GEMM. `max_rows` is an enforced bound.

Budget admission shares permanent snapshot reservations across session/layer
clones. Source tensors and retained outputs are external to that reservation.
Workspace figures are conservative logical tensor-buffer estimates per call,
not measured allocator peaks or aggregate concurrent-inference guarantees.
The caller reserves backend-private memory and determines concurrency. CUDA
graph capture, asynchronous workspace-pool admission and runtime ASD refresh
are later contracts, not implied by this eager pilot.

Use the reviewed dependency resolution and portable CPU flags:

```sh
cp dispatch-standalone/Cargo.lock Cargo.lock
export RUSTFLAGS='-C target-cpu=x86-64'
cargo check --locked -p candle-core -p candle-nn --no-default-features
cargo test --locked -p candle-core --no-default-features --test dispatch_bf16
cargo run --locked -p candle-core --no-default-features --example dispatch_bf16_linear
```

The branch CI checks CPU software contracts and compiles the CUDA consumer and
hardware test for SM61 and SM80 without executing a GPU. Its pinned SDK is the
already-published CUDA 12.9.2 / Rust 1.99.0 environment. Hosted compilation is
not physical validation. The hardware contract remains explicitly ignored
until the user runs it on an available physical GPU:

```sh
CUDA_COMPUTE_CAP=61 NVCC_CCBIN=gcc-14 RUSTFLAGS='-C target-cpu=x86-64' \
  cargo test --locked -p candle-core --no-default-features --features cuda \
  --test dispatch_bf16 cuda_physical_bf16_linear_contracts -- --ignored --nocapture
```

This command is a standalone correctness contract, not a complete model or
performance campaign. No live ASD store or runner registration is involved.
