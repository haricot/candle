# Candle Unified — R3.1: Default Branch Migration & Orchestration Consolidation

## Target topology

- **Default / workflow-dispatch authority:** `orch/unified-candle-v1`.
- **Upstream/source baseline:** `main` (kept, *not* made the default).
- **Stable products:** 13 standalone heads, `extended` and `cuda_legacy` (kept).
- **Independent qualification:** `orch/legacy-cuda-v1` and `orch/extended-v1` (kept).
- **Historical orchestrator:** `mirror_orch` (remove only after final gates).

This is a **staged** migration. Committing R3.1 does not change the GitHub
repository default or delete any remote ref.

## What migrates

The seven live workflows that used to dispatch on / verify `mirror_orch` now
pin `orch/unified-candle-v1`: Candle Sync, Candle Candidate CI, Candle GPU CI,
Candle Promote, Candle Cleanup, Candle ASD Extended, and the CUDA 12.9.2 PR
container smoke. Candidate CI -> GPU CI -> Promote still checks exact immutable
candidate SHAs, successful hosted CPU/CUDA evidence and physical SM61 proof;
the new branch is required for **new** evidence. A run from the retired mirror
is not silently accepted as a new Unified result.

The generic legacy `candle-cleanup.yml` remains manually dispatched for
promotion-tagged `sync/` / `rebase/` refs, with dry-run and explicit confirmation.
Its *completed*, pinned historical `.github/cleanup-request.json` push/finalize
path is retired. **R3** owns the separate automatic post-promotion retention for
current Unified campaign `prepare/` and `integration/` refs, with one rolling
previous state per canonical target. Do not remove the manual legacy cleanup
without auditing whether any unique old `sync/` / `rebase/` ref needs it.

The R2-only `candle-unified-promote-r2.yml` is removed from the active workflow
directory: campaign 42518965385 has already been promoted by run 42587341887.
Its historical workflow revisions, runs, and tags remain readable. The
`candle-sm61-hygiene.yml` request was old and remains absent from Unified.

## Release gates (in this order)

1. Review and merge [R3 PR #31](https://github.com/haricot/candle/pull/31),
   including its SHA-leased post-promotion cleanup selftest. Review/merge the
   R3.1 migration PR without overwriting R3 changes. Check the final Unified
   branch includes **both** test scripts and their steps in
   `candle-unified-orchestrator.yml`.
2. Pass `python3 -B .github/scripts/candle-default-branch-migration-selftest.py`,
   the R3 19-case retention selftest, YAML parse and `git diff --check`.
   Verify the Unified static-contracts workflow succeeds on its final commit;
   a normal push does **not** request Cargo, CUDA or the physical GPU runner.
3. Audit open PR heads **and bases**. In particular PR #26 currently targets
   `mirror_orch`: close it if superseded or explicitly migrate/review its base.
   PR #31 must no longer depend on an unmerged base. Do not delete a branch
   still serving an open PR, a live run or an unpublished unique commit.
4. Audit GitHub branch rules / required check names, environments, webhook and
   external automation targeting `mirror_orch`. Update them for Unified. Make
   an immutable SHA record of both the old default and new Unified branch;
   keep normal commit history and previous promotion tags.
5. As a repository administrator, open **Settings > General > Default branch**
   (GitHub may group this under Settings > Branches), select
   `orch/unified-candle-v1` and confirm. This repository setting cannot be
   changed by merging a YAML file.
6. Read back the repository's `default_branch` metadata and the final workflow
   inventory. Confirm that the Actions UI exposes **Run workflow** for
   `candle-sync.yml`, `candle-unified-orchestrator.yml` and qualification jobs.
   Use an existing immutable candidate in a dry-run or static check if needed;
   **do not** select a real promotion or rebuild CUDA just to prove the UI.
7. Only when there is no open PR, pending job, ruleset or external caller that
   needs `mirror_orch`, archive its final SHA in the migration audit and delete
   that *specific branch* with an exact-ref check. No other source, standalone,
   historical archive or product branch should be removed in R3.1.

## Rollback / stop conditions

If a dependency still points at `mirror_orch`, stop before switching default.
If branch default has changed but workflow_dispatch is missing or static
contracts fail, restore the previous default using the recorded SHA and repair
the workflow/branch rules before any ref deletion. Do not run maintenance jobs
with old October manifests or delete `mirror_orch` while PR #26 targets it.

The historical shell helper `candle-six-temporary-cleanup.sh` still contains
deliberate mirror-only guards and is **not called** by any active workflow;
do not retarget/revive it without a separate security review.
