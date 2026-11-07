# Source recovery inventory

This inventory records read-only source review and proposed extraction targets.
It does not certify fresh GPU behavior, benchmark results or completed recovery.
The machine-readable revision pins and access state are in `sources.json`.

| Source | Reuse target | Adaptation / limit |
| --- | --- | --- |
| Candle ASD V3, `candle-core/src/conv/asd_exact_cuda.rs` | Resolved plan, prepared launch, device/generation cache identity, cached misses | Extract operation-neutral lifecycle; keep convolution ABI in its adapter |
| Candle ASD V3, `candle-core/src/asd_runtime_extensions.rs` | Decision identity, profile generations, qualification metadata | Replace compile-target assumptions in new generic contracts; include device/context in every relevant cache key |
| Candle ASD V3, `candle-core/src/cuda_backend/device/asd_modules.rs` | Artifact identity, verification/loading and content-addressed module reuse | Future external-provider adapter; current raw ABI is convolution F32, not linear BF16 |
| Candle ASD V3, `candle-core/src/asd_fallback_store.rs` | Qualified fallback identity and explicit refresh | Preserve decision/evidence binding; no unqualified fallback or live-store write in S0 |
| Candle BF16 donor, `candle-core/src/cuda_backend/mod.rs` | BF16 to F32 conversion, F32 accumulation, BF16 output | Port the reviewed calculation to main without importing FP8/FP4 or donor Cargo files |
| Candle FP8 donor, `candle-core/src/quantized/fp8_sm61_dispatch.rs` | Global residency budget, construction cost and single-representation invariant | Design reference for resource contracts only; no FP8 provider in S0 |
| Candle D7, `candle-core/src/diagnostics.rs` | Semantic operation identity, optional diagnostics, missing-capability reporting | Minimal decision/plan diagnostics; no full collector dependency |
| Flow main, `src/selection.rs` and `docs/ARCHITECTURE.md` | Counterbalanced external measurement, raw-output retention, plan hot-path comparison | Remains external; BF16 probes need their own declared numerical/execution contract |
| Flow D5, `tools/asd-artifacts.py` and `docs/ARTIFACTS_D5.md` | Source/artifact receipts, architecture catalogue and verified publication | Stays in Flow; existing publisher supports the existing raw ABI only |
| Flow D6-B2, `docs/D6_B2_EVIDENCE_OOD.md` | Bound raw evidence and abstention outside measured support | Offline search guidance; hashes are integrity checks, not runner attestation or runtime eligibility |
| Flow D6-B3, `docs/D6_B3_EQUAL_BUDGET_REPLAY.md` | Blinded replay and declared comparison budgets | Retrospective/synthetic tests are not new kernel or engine performance evidence |
| Flow D7, `docs/D7_ENGINE.md` and `docs/D7_GENERIC_TRACE_CONTRACT.md` | Coverage, untraced measurement, independent numerical qualification and semantic trace schema | Keep coverage/submission/completion distinct; no whole-engine gain inferred from a microbenchmark |
| `haricot/adaptive-dispatch` | Capability/observation/model separation and numerical workload descriptors | Audit complete at the pinned revision; design recovery only, no copied code or dependency |

## Concrete extraction hazards found in ASD V3

At the pinned ASD V3 revision, `lookup_exact` caches by operation key and global
generation before checking the actual GPU UUID. Its cache-hit condition is not
itself device-scoped. The prepared-plan cache, by contrast, includes DeviceId.
The new foundation requires context/device isolation throughout both layers;
the existing code is not a blanket multi-GPU guarantee.

The profile target reader compares profile architecture with
`CUDA_BUILD_COMPUTE_CAP`. An external module is separately checked against the
runtime device. S0 must model compile inventory and runtime capabilities
explicitly rather than assuming one build architecture is the active device.

The existing `ExactOperationCall` signature and external kernel ABI encode
convolution semantics and F32 buffers. Preserve these persisted contracts;
define a distinct versioned linear contract if/when an external BF16 provider
is added. Reusing the loader does not establish buffer/scale/layout equivalence.

## adaptive-dispatch audit completed

See `adaptive-dispatch-audit.md`. The repository is distinct from the historical
adaptive-static-dispatch crate and the current ASD V3 convolution runtime.
Its capability/model separation informs S0; its heuristic predictor and legacy
capability mutation are deliberately outside the execution authority.

The initial source-access gate is closed. No Cargo dependency, capability table,
probabilistic model, test macro or CUDA source is copied from that repository.
The S0 implementation is new Candle code using existing Candle tensor/back-end
operations, with the reviewed BF16 conversion/accumulation semantics as reference.


## Minimum build guards recovered after hosted compilation

The SM61 build at Dispatch commit df2e2d520f5ffe0d3e059166e0e903fd014c06fc
failed on main's unguarded `SUM_OP(__half, sum_f16)` (scalar-half atomicAdd).
Recover the existing donor's SM70 guard, retaining `fast_sum_f16`. Also recover
the donor build policy that omits WMMA-only static objects below SM70.
These changes involve `candle-kernels/src/reduce.cu` and `candle-kernels/build.rs`;
they introduce no BF16 general arithmetic or MoE algorithm. New source bytes
receive a fresh complete CPU/CUDA CI run.
