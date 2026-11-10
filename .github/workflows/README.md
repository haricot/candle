# Haricot Candle — active GitHub Actions workflows

This directory contains orchestration workflows on `orch/unified-candle-v1`.
This is the repository's default branch. Legacy and Extended retain independent
qualification paths while Unified validates the shared preparations once.

## Current CUDA/SM61 qualification

| Workflow file | Scope | Entry point |
| --- | --- | --- |
| `candle-extended-composite.yml` | **Current Extended**: dynamic standalone sources; hosted `cpu_cuda_compile`; binary-only `cpu_execute`; physical `SM61 Extended runtime`; validated promotion | Push to `orch/extended-v1` (selected paths), or `workflow_dispatch` |
| `candle-legacy-cuda.yml` | Separate CUDA legacy qualification | `orch/legacy-cuda-v1` |
| `candle-unified-orchestrator.yml` | **Unified R3**: one Extended-qualified Legacy/Extended pair and automatic SHA-guarded post-promotion retention | `orch/unified-candle-v1`, explicit dispatch with `promote: true` |
| `candle-cuda-ci-image.yml` | Cached, fingerprinted CUDA 12.9.2/SM61 build environment | Reusable workflow and explicit image dispatch |

**Unified campaign map:** open a Unified run, then its **Summary** and the
**Unified CI map and campaign history** job. The diagram and source inventory
are generated from the checked-out workflow and source contracts; the job
includes per-gate results, current workflow states and recent run history.
It runs even if preparation fails. GitHub's workflow sidebar also retains
historical entries that are not present in the current branch.

The Extended workflow does **not** require the older six-feature integration
workflow. It is driven by `.github/extended-sources.json`, not a fixed number
of source branches. The `lean4_standalone` source can remain in the composition,
but its formal proof runs separately in `haricot/haricot-verify`.

## Additional workflows retained

- `candle-sync.yml`: **manual preparation only** of detached legacy candidates,
  without dispatching CPU, CUDA, SM61 GPU or promotion. These candidates are
  unqualified until they pass a separately reviewed qualification route.
- `candle-cleanup.yml`: manual, SHA-leased cleanup of legacy `sync/` and
  `rebase/` refs, dry-run by default and explicit confirmation for deletion.
  It is **not** an automatic campaign and cannot be replaced by Unified R3
  retention of `prepare/` / `integration/` refs.
  Its `unified_historical` mode runs the R4.3 historical reachability audit
  with an opt-in atomic archive-and-prune operation; default remains dry-run.
  Apply requires the SHA-bound `PRUNE_HISTORICAL_<digest>` token printed by
  a preceding dry-run; a changed inventory invalidates the token.
- `candle-cuda1292-container-smoke.yml`: CUDA container smoke tests on PRs.
- `maturin.yml`, `python.yml`, `rust-ci.yml`:
  upstream-related build and package CI.
- `trufflehog.yml`: security scanning.

The old `ci_cuda.yaml` definition targeted upstream SM86/AWS runners and CUDA 13. It has been disabled since January 2026 and is removed from the three maintained orchestration branches. The digest-pinned CUDA 12.9 SM61 producer and its current consumers are retained.

## Unified R3 retention

On a successful promoted Unified run, `candle-unified-retention.py` maintains
one rolling `archive/unified/previous/<canonical-ref>` backup per promoted
target, then atomically removes current-campaign preparation, integration and
archive refs, along with older preparation/integration refs at provably
identical SHAs. The helper defaults to dry-run and fails closed if a ref moved,
proof is absent or an open PR protects the ref. Legacy archives with distinct
history are **not** deleted automatically. See
`docs/CANDLE-UNIFIED-R3-RETENTION.md` before modifying maintenance workflows.

## Retired orchestration files

These workflows were superseded or bound to one-time historical evidence.
These earlier files were removed from `orch/extended-v1` without deleting Git
branches, artifacts, prior runs, or the historical revisions that contain them:

- `candle-six-gpu-drift-resume.yml`
- `candle-six-temporary-cleanup.yml`
- `candle-standalone-integration.yml`
- `extended-dynamic-config-ci.yml`
- `fp8-rebase-bootstrap.yml`
- `candle-cuda1292-candidate-request.yml`
- `candle-integration.yml`
- `candle-integration-gpu.yml`
- `candle-integration-gpu-handoff.yml`
- `candle-integration-promote.yml`

**Unified R3.2** additionally removes the following seven redundant workflows
from the default branch (without deleting source branches or historical runs):

- `candle-build-once-contracts.yml`
- `candle-hosted-disk-contracts.yml`
- `candle-standalone-publication.yml`
- `candle-candidate-ci.yml`
- `candle-gpu-ci.yml`
- `candle-promote.yml`
- `candle-asd-extended.yml`

The old publishing and validation workflow files may still exist on other
long-lived refs, notably `orch/extended-v1` and `ci/*`. Removing them from the
default branch does **not** remove those copies or deactivate their push events.
Review those refs separately before any cross-branch retirement.

GitHub Actions may still display **historical** workflow runs after the YAML
files are removed. Their history is preserved intentionally.

For the local Pascal runner, start a `candle/extended` one-shot session only
when the physical `SM61 Extended runtime` job is queued.

R3.1 migrated workflow-dispatch authority to `orch/unified-candle-v1` and
retired the completed R2-only workflow. R3.2 narrows the default-branch workflow
inventory to **11** files while preserving the manual legacy cleanup.
See `docs/CANDLE-UNIFIED-R3-2-WORKFLOW-RETIREMENT.md` for safety boundaries.
