# Candle Dispatch S0: Device Capabilities and BF16 Linear

Branch: `capability-dispatch-standalone`.

Base: `haricot/candle:main` at
`cfcbd7186d13271843ce7eb41d25659bd0835850` (Candle 0.11.1).

Status: scope and source-recovery preparation only. No Dispatch runtime, BF16
pilot, hardware qualification or performance result is implemented by this
initial documentation commit. `haricot/adaptive-dispatch` must be inspected
before fixing the selector API and choosing the final reusable implementation.
The connected GitHub reader returned 404 for its repository, README and Cargo
manifest on 2026-11-07; this does not distinguish a missing repository from
insufficient access. No conclusion about its contents or license is possible.

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

1. Complete the adaptive-dispatch source/license/API audit and record exact
   revisions, recovered functions and disposition before selector implementation.
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
