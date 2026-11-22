# R1.25/P2 — SM61 app-scoped Cudaforge default and driver-fault rollback: reported PASS

**Status: PASS in the supplied P2 qualification report; raw headless CUDA
log was not supplied for independent trace-by-trace inspection.**
Evidence date: 2026-11-22. This is a **documentation-only** follow-up
to the tested source commit; no CUDA artifact, source or binary changes.

## Immutable provenance

- **Exact source HEAD under test:** `60fb2f3276f897d463e318e070be8a70c9a735c5`.
- JSON schema: `CANDLE-DKR-R125-P2-RESULT-V1`, `status=PASS`,
  `target_cc=61`.
- Submitted evidence: `result(20261122-174629).json`.
- Catalogue Cudaforge SHA-256:
  `91ac006b963a89362c6aaaee052252b4465880c079b0c86f399c4891a5003295`.
- Model GGUF SHA-256:
  `ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a`.
  The model used for the qualification is Qwen3-0.6B Q4_K_M.
- Shipping candidate feature: `sm61-app`.
- Separate qualification-only fault feature: `p2-fault-canary`.
- `test_fault_compiled_into_shipping_app=false` in the report.
- The **generic Candle `cuda` feature remains unchanged**.

## Actual driver-admission failure and PTX fallback

The report asserts all of these gates:

| Gate | Reported evidence |
|---|---|
| Deliberately malformed image refused by real CUDA driver | `fault_driver_rejected=true` |
| Target family | `cast` |
| Qualified `auto` fallback | `auto_fallback_family=cast` |
| Strict `cubin` fails closed | `strict_cubin_rejected=true` |
| Test hook absent from shipping binary | `test_fault_compiled_into_shipping_app=false` |
| Mixed-image fallback numeric parity | 157 cases, 76,292 elements, **0 mismatches** |
| Fault-enabled Qwen3 model logits parity | 1,367,424 elements, **0 mismatches** |

The submitted JSON summarizes these results but does **not** contain
the original `cuModuleLoad` rejection log, image-format traces or
the negative-test stderr. These can be examined separately if a
stronger independently audited provenance chain is required.

The test-only hook attempts to load an invalid in-memory module
without altering the attested CUBIN image or package SHA.
The application binary does **not** contain this test hook.

## Three Qwen3 prompt/decode scenarios: full logits

The P2 shipping-profile model was invoked in `ptx`, unset `auto`
and strict `cubin` modes, with the same GGUF and token fixtures.
All compared the **complete** greedy-prefill/decode float32 logits
against PTX; no mismatches were reported under the declared tolerances.

| Scenario | Prompt IDs | Decode steps | Logits per comparison | `auto` mismatches | strict CUBIN mismatches |
|---|---:|---:|---:|---:|---:|
| Short | 8 | 8 | 1,367,424 | 0 | 0 |
| Medium | 24 | 16 | 2,582,912 | 0 | 0 |
| Long | 48 | 24 | 3,798,400 | 0 | 0 |
| **Total** | | | **7,748,736 per alternative** | **0** | **0** |

**15,497,472 logit values** were compared in aggregate
(two alternatives against PTX over three scenarios);
there were no reported numerical mismatches or maximum absolute
differences. This is numerical equality from the report; unlike
the 157-case P1 tensor comparison, the model logit comparison is
tolerance-based and does **not** prove all raw bit patterns match.

The Qwen3 native family set changes appropriately with longer
prefill, including `mmq_instance_q4_k`,
`mmq_instance_q6_k` and `mmq_quantize` in the medium and
long scenarios. All modes in each scenario passed the family-set
and numerical gates.

## Cold-start model timings: one run per policy/scenario

In-process total processing times, **not steady-state tokens/s**:

| Scenario | PTX cold (s) | auto/CUBIN (s) | strict CUBIN (s) | PTX/auto |
|---|---:|---:|---:|---:|
| Short | 28.573483 | 1.034686 | 1.041644 | **27.62x** |
| Medium | 34.941315 | 1.091977 | 1.101012 | **32.00x** |
| Long | 34.992996 | 1.148198 | 1.155988 | **30.48x** |

These gains primarily avoid cold PTX JIT initialization.
Each reported row comprises one run per policy; GPU clock
histories and Linux OS page caches were not controlled. Do not
describe these ratios as warmed steady-state decode throughput
improvement or as production latency guarantees.

## Decision and scope

**Close R1.25/P2 as PASS for this SM61 standalone Qwen3 application
canary, based on the reported evidence.**

The P2 campaign shows that a scoped `sm61-app` feature can use
the qualified Cudaforge/CUBIN default, retain explicit PTX rollback,
and recover from a CUDA module-admission error without touching
the sealed native artefacts.

**Not yet demonstrated or changed:**
- No generic/default `cuda` Cargo feature promotion for all architectures.
- No integration in the separate `mistral.rs` application.
- No ASD V3 `ResolvedAsdPlan`, dynamic profile or exact convolution
  routing changes; its priority over general DKR paths must remain.
- No production fault test hook; never distribute the
  `p2-fault-canary` binary.
- No evidence for arbitrary GGUF models, other Pascal SKUs or
  GPUs with different compute capabilities.

## Next appropriate step

Review the runtime integration points in `mistral.rs` before
allowing `cuda-dkr-qualified-default` in that application's
SM61-targeted build. Pass the feature coherently through Candle
core and nn, preserve deterministic rollback and explicit image
traces, and run the same model/numeric gates there before default
promotion. Keep generic CUDA and ASD routing separate.
