# Candle Unified R3 — One previous state & verified cleanup

The Unified promotion retains **one rolling pre-promotion branch per canonical
standalone and integration target**, only after the Extended and Legacy pair has
passed the hosted and physical SM61 gates. This change does not add a CUDA build.

## Retention rule

The verified promotion continues to use the existing immutable, campaign-scoped
preimage branches while it atomically advances the canonical refs. After it
reports `PROMOTION=PASS`, `.github/scripts/candle-unified-retention.py`:

1. Verifies the campaign, main, both candidate SHAs, both candidate manifests,
   promotion success, and **every** current canonical target SHA.
2. Plans the fixed rollback refs `archive/unified/previous/<target>`, pointed at
   the **immediately preceding** commit for each canonical target (13
   standalones, `extended`, `cuda_legacy` for the current Extended manifest).
3. Checks live open PR heads **and bases**, with a fatal stop on any GitHub API
   error. No open PR ref is deleted.
4. Atomically publishes/rotates the rolling rollback refs and deletes
   campaign-scoped preparations, candidate refs, per-campaign backups and
   duplicate SM61-validated branch refs, using exact-SHA leases.
5. Also removes older preparation/integration refs **only when their
   SHA is identical** to a current canonical head or the newly retained
   previous head. Old archive names remain as audit evidence until separately
   reviewed, even if their commit is reachable elsewhere.

The job-level `candle-unified-stable-ref-writer` concurrency group serializes
Unified promotions across different campaign names. Other orchestration branches
still use their own promotion contracts and are not silently modified.

The audit contains a `retention.json` file (uploaded with the promotion
artifact) and the Action step summary. The CLI defaults to **dry-run**; the
workflow explicitly supplies `--apply` only after promotion passes. An unexpected
remote change, missing proof, failed PR lookup, or failed atomic push turns the
cleanup step red, leaving the already-promoted stable refs untouched.

## Limits and follow-up

R3 prevents further growth of Unified's campaign-scoped temporary branch refs.
It is intentionally **not** a blind one-shot deletion of the 207-branch legacy
inventory: historical candidates with unique commits, old `archive/promotion`
states, `integration/candle-legacy-v3-dispatch-*` refs, and open PR branches are
left alone until their individual evidence is reviewed. A separate pinned
historical sweep may retire them later; they must not become an implicit
retention rule in the Unified promoter.

`candle-cleanup.yml` and `candle-sm61-hygiene.yml` remain independent tools for
their other historical/explicit maintenance scopes. Do not reuse their old
October request manifests or delete either workflow solely because R3 now owns
Unified's post-promotion cleanup. They can be consolidated in a separate PR
after their remaining consumers and permissions have been audited.

### Offline checks

```sh
python3 -B .github/scripts/candle-unified-retention-selftest.py
bash -n .github/scripts/candle-promote-validated.sh
```

The selftests exercise stale canonical SHA refusal, changed temporary refs,
open-PR preservation, retention-idempotency and an actual atomic push/deletion
against a disposable local bare Git repository. No GPU and no external GitHub
writes are involved.
