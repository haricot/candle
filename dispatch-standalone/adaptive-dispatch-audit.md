# adaptive-dispatch recovery audit

Repository: `haricot/adaptive-dispatch`, `master` at
`0203aa56fd689b01838e19bf9f7e12887f78b17a`.
Tree: `aed6386567384355293925814e30056023a74615`.
Review date: 2026-11-07. Source review only; no donor GPU/performance validation
was executed in this audit.

## Findings and disposition

| Source | Existing responsibility | S0 disposition |
| --- | --- | --- |
| `ir_caps/src/caps/hw.rs` | Named capabilities and `StaticCaps` architecture descriptions | Recover the declarative-capability idea; implement only the operation facts required by the BF16 pilot |
| `ir_caps/src/arch_metrics.rs` | Explicit origins (`NvidiaSpec`, runtime, microbenchmark, user) and separation of observables/models/derived values | Preserve the authority separation: actual-device inspection drives execution; declared facts are pure planner inputs |
| `ir_caps/src/caps/library.rs` | Library-oriented heuristics from architecture/metrics | Do not treat a predicted affinity as proof that a concrete cuBLAS operation/artifact is available |
| `ir_caps/src/caps/table.rs` | Architecture table and legacy overrides | No table import; legacy permission must not mutate physical facts |
| `ir_caps/src/compute_cap.rs` | Build configuration, environment/detection helpers and architecture suffix handling | Keep build-target selection distinct from the executing Candle Device/context; no environment/nvidia-smi authority in the hot path |
| `adapt_dsp/src/predictor.rs` | Hardware-conditioned priors, workload shape, layout/numerics/statefulness/provenance, heuristic rankings and safety assessments | Numerical/layout descriptors inform the explicit BF16 contract; prediction, confidence and search pruning remain tuner concerns |
| `adapt_dsp/src/calibration.rs`, `synperf/`, `telemetry.rs` | Calibration, simulation and observation | Outside A+B; no microbenchmark or predictor is run while preparing/forwarding a layer |
| `candle_utils_tests/` | Test macros and tensor comparison helpers | Existing Candle tests already supply the needed primitives; no parallel test dependency |

## Specific compatibility observations

`legacy_override` sets `has_bf16` or `has_fp8` on pre-native architectures when
permission is granted. That merges permission to use software support with a
physical property. S0 keeps architecture facts unchanged and registers a
separate software-capable provider. A policy can choose only a provider that
also satisfies device, implementation and resource eligibility.

Some library capability entries are explicitly derived from roofline,
tensor-core dominance or cache/register estimates. These are useful search
priors, not executable library-call contracts or numerical qualification.
Their confidence/throughput values do not enter S0 selection.

The build helpers may choose specialized architecture suffixes automatically.
Such a compilation target is not a universal compatibility guarantee for all
GPUs sharing a major architecture. S0 resolves built-in conversion modules on
the actual runtime device; later external providers must retain their own
precise artifact compatibility contract.

## Dependency and license surface

The root workspace patches cudaforge to git revision
`64a7f7a7926cc0c1a9b896ab853dedbcb74617a6`. S0 retains pinned Candle main's
own dependencies and does not import that patch or either crate.

The README displays an MIT/Apache-2.0 badge. At the reviewed tree there are no
LICENSE files and neither reviewed crate manifest declares a license field.
This records the available metadata, not a license determination. This recovery
uses design observations only: no donor source text or dependency is copied.

## Result

The source-access audit is complete. The repository contributes useful concepts
but is not a drop-in Candle dispatch runtime. The S0 code remains a small
deterministic operation-specific executor. ASD V3 is the later profile/module
adapter; Flow-Adaptive Tuner remains the search/measurement producer.
