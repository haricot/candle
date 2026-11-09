# Semantic Trace & Stateful — experimental Extended integration

This standalone keeps the useful D7 work from Candle PRs #27 and #28 while
discarding their old, overlapping ASD history. The initial integration branch
is based on the current `extended` tree because the stateful F16 candidate uses
the ASD V3 runtime profile and external CUDA manifest APIs already present there.

## What is integrated

- Optional `d7-trace` semantic capture with bounded events and explicit loss and
  incomplete-drain reporting.
- Application-owned `ScopeObserver` hooks, Linux monotonic/thread-CPU clocks,
  and optional NVTX attribution.
- Immutable `causal_depthwise_conv1d_step` tensor reference for F16/F32.
- The external-ABI F16 ShortConv candidate, guarded by exact device, artifact,
  layout, dtype and runtime-profile checks.

The candidate is **not selected by default**. Experimental candidate decisions
require a matching runtime profile plus the explicit
`CANDLE_ASD_ALLOW_STATEFUL_CANDIDATE=1` opt-in. Unsupported calls retain the
reference operation. Trace collection is opt-in through the `d7-trace` feature.

## Qualification status

- CPU compilation and semantic/stateful tests: required for this integration.
- Historical physical SM61 exactness evidence: available in D7, tied to older
  source and model pins; it is **not** evidence for this rebased integration.
- Physical SM61 on this integration commit: **NOT RUN / UNQUALIFIED**.
- End-to-end untraced model performance or speedup: **NOT ESTABLISHED**.
- Complete CPU/GPU activity correlation: **PENDING**; the generic observer is
  only the application-facing bridge, not an embedded CUPTI collector.

This is an experimental, disabled-by-default integration into `extended`, not
an authorization to bypass the existing Unified physical-SM61 gate for other
standalones. Do not mark this revision as hardware-qualified or silently promote
the F16 candidate to a production ASD profile.

## Provenance

Cherry-picked functional commits from `d7-trace-t123` and `d7-stateful-f16`:

1. `443e9b11b3854251774a3b09f5b755cd5c09b7a2` — bounded semantic capture
   and immutable ShortConv reference.
2. `84a0b24c030d3af912dcab9a3939f8b0de1dbd32` — stateful F16 candidate ABI.
3. `8a0e6d099e680496c623c116e0ebc1fd516ec3ce` — capture drain correctness.
4. `5d166459144ed2b830f1d237286e851e654efb7b` — CPU clock and observer.

Keep the original PRs and evidence archives until this branch is independently
verified and all downstream Mistral/tuner API contracts are rechecked.
