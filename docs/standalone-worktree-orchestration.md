# Standalone source → detached preparation → disposable integration

The old six-feature `candle-integration.yml` and physical SM61 promotion are frozen; this v1 supports the two independently reconstructed source branches first. It does **not** rewrite or re-promote the previously GPU-validated six-feature release.

## Ref ownership

- **Source-of-truth (immutable to automation):** `fp8_candle_standalone` and `fp4_candle_standalone`, both descended from fork `main`.
- **Orchestrator:** `mirror_orch` contains the workflow and script, not a second code baseline. The hosted job fetches **fork main** at an exact full SHA, pins each standalone source SHA, and uses `git worktree add --detach` for per-feature preparation.
- **Preparation strategy:** `merge` by default for traceability; optional `rebase --rebase-merges --onto <pinned-main> <merge-base>` acts **only** on a detached worktree.
- **Integration:** a separate detached worktree starts at the pinned `main` commit and merges exact FP8 then FP4 prepared SHAs. Every input and SHA is recorded in `candle-integration/standalone-sources.json`.
- **Output:** at most **one** new remote integration ref per successful campaign: `integration/standalone-<campaign>-fp8-fp4`. No temporary per-feature branches need to be pushed by this end-to-end path. Feature diffs, SHA manifest and conflicts are retained as a workflow artifact.
- **Preview:** the first run should use `publish=false` and `validation=prepare_only`. A conflict fails closed with an artifact, and no integrated ref is created. To publish an accepted merge, rerun with a **new campaign identifier** and `publish=true`.
- **CI:** only after publication, the optional handoff explicitly dispatches `Candle Candidate CI` with `branch=standalone_fp48_integration`, the exact aggregate SHA and both `candle-core/cuda-legacy-fp8` and `candle-core/cuda-legacy-fp4`. No physical GPU runner is requested, and no promotion happens.

The old `Candle Sync` workflow now maps the legacy **test suite names** `fp8_candle` and `fp4_candle` to their **source refs** `fp8_candle_standalone` and `fp4_candle_standalone`. It can emit optional `sync/<campaign>/<feature>` throwaway feature candidates for independent CI, or reuse a current standalone SHA directly. This mapping does **not** move the historical integrated aliases `fp8_candle` and `fp4_candle`.

## How this differs from a GitHub merge queue

A GitHub merge queue creates temporary refs to validate the combined result without modifying pull request heads. This workflow follows the same temporary-candidate principle, but has a deterministic *two-feature* merge order, exact source provenance and separate CPU/CUDA requirements. It is **not** a substitute for full merge-queue policy, and a successful compile is not physical GPU parity.

## Before expanding to all six features

Move the other source branches to true standalone-main histories first, then extend this orchestrator's **explicit allowlist**, merge order and source manifest schema. Do not point it at the old integrated aliases. Do not alter the old six-source manifest, review-gate rerere cache or GPU promotion tags. New conflicts require one reviewable resolution proposal; never run `-X ours` or blindly use `rerere.autoupdate=true`.

After any future promotion, cleanup must target only temporary `sync/`, `rebase/` and obsolete `integration/standalone-...` refs with exact SHA leases and preservation tags. Never delete or advance any `*_standalone` head.
