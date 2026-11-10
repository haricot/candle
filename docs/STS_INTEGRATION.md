# STS-S0 — main-based Semantic Trace & Stateful Observer standalone

This source branch is based **exactly** on `haricot/candle:main` at
`cfcbd7186d13271843ce7eb41d25659bd0835850`. Its source delta does not
include the existing 13 Extended standalone contributions.

## Standalone contract

- `d7-trace` provides bounded semantic events, drain/loss metadata, CPU clocks,
  optional NVTX attribution, and an application-owned `ScopeObserver`.
- `stateful::causal_depthwise_conv1d_step` implements immutable input/next-state
  with F16/F32 support through Candle tensor operations (CPU or CUDA).
- On the standalone, the optional ASD F16 executor is **not compiled or
  selected**. Its original source is retained under `stateful/cuda.rs` and
  `stateful/validation.rs`, gated off in `stateful.rs` until integration.
- The exact, reviewed ASD V3 bridge changes are kept as
  `.github/sts-asd-v3-exact.patch`, with a separate
  `.github/sts-asd-v3-activate.patch`. Both are inert on the standalone.
- No part of the historical `extended` branch is imported into this source.

## Unified integration (14th source)

After the existing 13 sources, Unified composes this standalone's exact
`main`-based delta and verifies the pinned bridge preimages. It then applies
the two source-owned patches to the already assembled ASD V3 implementation,
checks the expected postimages and commits the audited integration. In this
order, the F16 candidate remains dependent on ASD V3's exact-match manifest,
device identity, artifact and runtime policy. The normal failure/fallback path
uses the immutable tensor reference.

Do not apply the ASD bridge to the source branch directly: the ASD V3 APIs are
not available in `main` and would implicitly import other standalone history.
Do not use the old `extended`-based branch as an exact-delta source.

## Qualification status

- Main-based CPU tests: required.
- Main-based CUDA source build: a reference-path build, **not** the F16 executor.
- Unified integration tests: must verify bridge provenance and original STS
  file identities before a candidate can be published.
- Physical SM61 on this new branch/candidate: **NOT RUN / UNQUALIFIED**.
- Model-level speedup, CUDA execution parity on this SHA, and complete CPU/GPU
  activity correlation: **NOT ESTABLISHED**.
- Experimental F16 candidate remains disabled unless an exact runtime candidate
  is present and `CANDLE_ASD_ALLOW_STATEFUL_CANDIDATE=1` is explicitly set.

This does not waive Unified's existing physical SM61 promotion gates for its
other sources. Historical evidence does not qualify this new candidate.

## Provenance

Functional recovery from old PR #27 and #28:

1. `443e9b11b3854251774a3b09f5b755cd5c09b7a2` — trace and reference.
2. `84a0b24c030d3af912dcab9a3939f8b0de1dbd32` — candidate ABI.
3. `8a0e6d099e680496c623c116e0ebc1fd516ec3ce` — capture drain.
4. `5d166459144ed2b830f1d237286e851e654efb7b` — CPU observer clocks.

The old Extended-based STS integration is preserved as historical evidence,
not used as a source for new Unified candidate composition.
