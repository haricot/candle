# Post-SM61 Candle cleanup — 2026-10-16

## Verified evidence

- Legacy: [run 38898658386](https://github.com/haricot/candle/actions/runs/38898658386), physical SM61 job `116973510347` PASS, “Promote validated Legacy refs” PASS. Canonical `cuda_legacy`: `bfce966ad4d61a48801eb6af4db8b8eabebc01a8`.
- Extended: [run 38898680839](https://github.com/haricot/candle/actions/runs/38898680839), physical SM61 job `117002054129` PASS, “Promote validated Extended refs” PASS. Canonical `extended`: `f5ce7f6f8b7a68d14a55419f9f221e1ae3537fd3`.
- No open PRs at audit time. This must be rechecked on execution.

## Cleanup scope

The pinned manifest `.github/post-validation-prune-20261016.json` contains **99 exact heads**:

- 44 `archive/*` branch refs;
- 47 `prepare/{legacy,extended}/*` branch refs;
- 8 `integration/{legacy-cuda,extended}/candidate/*` branch refs.

Every ref receives a distinct **annotated tag** under
`archive/candle-pruned-20261016/<original-branch-name>` before any deletion.
The script verifies peeled tag SHA, checks promotion evidence, verifies all 13 protected
heads, checks open PRs twice, and applies all branch deletions in one atomic
force-with-lease push. Any unexpected SHA or tag collision **fails closed**.

The workflow `Candle Post-Validation Branch Prune` is **manual only**.
First run on `mirror_orch` with `dry_run=true` (default). After reviewing the
preview, use `dry_run=false` and `confirmation=ARCHIVE_AND_DELETE_99`.
No destructive action is triggered by merging the PR.

## Workflows removed in this PR

- `fp8-rebase-bootstrap.yml`: September one-shot with hard-coded prior candidate.
- `candle-six-gpu-drift-resume.yml`: old one-shot DW5x5 resume with pinned request.
- `candle-six-temporary-cleanup.yml`: completed historical 23-ref cleanup pinned to
  old run `36571847341`.

These are **not** the successful October Legacy or Extended workflows.

## Workflows / branches NOT retired

**Do not delete** `orch/legacy-cuda-v1` or `orch/extended-v1` yet.
The exact workflows that passed on 2026-10-16 currently live on those branches:
`candle-legacy-cuda.yml` and the newer `candle-extended-composite.yml`.
The latter differs from `mirror_orch`; the Legacy workflow is absent on
`mirror_orch`. First migrate and revalidate their scripts and manifests in a
separate reviewed PR; avoid reverting to the older default-branch implementation.

Keep `main`, `mirror_orch`, `cuda_legacy`, `extended`, the 8
feature / proof source branches, and other active experimental refs.
Upstream Rust/Python/PyO3/security workflows and the opt-in ASD Extended
workflow remain unchanged. The old general-purpose candidate/sync/integration
pipeline remains intact until its callers and promotion paths are checked
separately, instead of deleting shared workflows simply because the
October campaigns passed.

## Mistral.rs follow-up — not started

Mistral's `allow_old_card` currently points to
`00d0a5e6989a42d1f3d903e72c427439a6098717`.
Its Legacy CUDA orchestration workflow lives on **`haricot/mistral.rs@mirror_orch`**,
not on `allow_old_card`, and supports `expected_pr_sha`,
`expected_upstream_sha`, and `expected_candle_sha`.

After Candle pruning and workflow consolidation, start the
`Mistral CUDA Legacy (rebase PR then validate SM61)` workflow on
`mirror_orch` with `validation_scope=full`, pinning the new
`cuda_legacy` SHA `bfce966ad4d61a48801eb6af4db8b8eabebc01a8`,
and the current actual PR/upstream SHAs. Do not rerun the already
cancelled/failed October Mistral runs: start a fresh, immutable campaign.
