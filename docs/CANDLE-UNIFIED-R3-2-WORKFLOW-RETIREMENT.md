# Candle Unified — R3.2: Default-branch workflow retirement

Scope: `haricot/candle`, default branch `orch/unified-candle-v1` only.
No feature, standalone, product, archive, `sync/` or `rebase/` ref is deleted.
No CUDA image is rebuilt and no GPU campaign is launched by this migration.

## Retired on the Unified default branch

| Workflow | Reason | Continued owner |
| --- | --- | --- |
| `candle-build-once-contracts.yml` | Duplicates Unified/Extended build-once contract checks | Unified `validate_orchestrator` and `prepare` |
| `candle-hosted-disk-contracts.yml` | One-off hosted disk testing on an old CI branch | CUDA image producer and Extended checks |
| `candle-standalone-publication.yml` | Old direct publication channel with a second stable-ref writer | Unified gated `promote_validated` |
| `candle-candidate-ci.yml` | Old six-feature candidate qualification | Manifest-based Unified qualification |
| `candle-gpu-ci.yml` | Old independently dispatched six-feature SM61 campaign | Unified physical SM61 gate |
| `candle-promote.yml` | Old promotion route | Unified SHA-guarded promotion |
| `candle-asd-extended.yml` | Isolated compilation-only check, not an SM61 acceptance gate | Manifest-defined Extended/Unified qualification after source integration |

The four remaining Candle workflows (`candle-unified-orchestrator.yml`,
`candle-extended-composite.yml`, `candle-legacy-cuda.yml` and
`candle-cuda-ci-image.yml`), container smoke, legacy sync, legacy cleanup and
four general CI/security workflows form an **11-file** inventory.

## Synchronization dependency removed

`candle-sync.yml` formerly triggered `candle-candidate-ci.yml` after detached
merge/rebase preparation, with `validation=cpu_cuda` as the default. Deleting
Candidate CI without changing Sync would leave a broken default path.

R3.2 changes Sync to **prepare-only**. It keeps exact-source/upstream-SHA
provenance, conflict artifacts, optional rustfmt and immutable temporary ref
publication, but removes the `validation` input and all candidate-CI dispatch.
New temporary candidates are reported `PREPARED_UNQUALIFIED`. They are not
automatically eligible for promotion: use the independently reviewed
manifest-driven Unified/Extended qualification pipeline.

## Why `candle-cleanup.yml` stays

R3 `candle-unified-retention.py` only governs the current Unified campaign's
`prepare/`, `integration/` and related rolling archive refs. The legacy
cleanup is still the audited manual route for historical `sync/` and `rebase/`
refs: it checks proof tags, protects open PR heads, defaults to `dry_run: true`,
requires `DELETE_TEMP_REFS` before deletion, and uses SHA leases. The old
promotion workflow is removed but the existing tags remain useful evidence.

**Do not** automatically trigger Cleanup from Unified. Removing Cleanup
requires a separate inventory of uniquely preserved historical refs and a
safe replacement procedure.

## Checks and cross-branch caveat

- Run `.github/scripts/candle-default-branch-migration-selftest.py`: retired
  files must be absent, active workflows cannot reference them, Sync must not
  dispatch qualification, cleanup guards and Unified physical SM61 gate stay.
- Run existing Unified retention/pair/source selftests, YAML parsing and
  `git diff --check` without invoking CUDA or SM61 hardware.
- Removing YAML from the default branch does **not** remove copies on other
  branches. In particular, `candle-standalone-publication.yml` can still be
  triggered on `orch/extended-v1` if its own branch copy remains; any cross-ref
  retirement needs its own audit of trigger rules and reference writers.
- GitHub Actions may retain names and runs of retired workflows as history.

The repository's branch/default setting is unchanged in R3.2. Review and merge
this change through an explicit PR; do not delete additional branches here.

## R3.2 follow-up: secret scan on first branch push

PR #33's initial branch-creation push had a zero `before` SHA. The upstream
TruffleHog action interpreted that as a full-history scan (22 MB), reported
one **unverified** GitHub finding, no verified secrets, and exited 183. The
earlier default-branch push checks scanned commit deltas and passed.

The unchanged TruffleHog engine still fails on findings. Its workflow now
passes an explicit Git range: `before -> github.sha` for ordinary pushes, and
`origin/<default branch> -> github.sha` for first pushes. It does not add a
detector exclusion or change `--fail`. The one historical unverified finding
is **not** thereby declared harmless: triage a dedicated historical scan
separately before treating it as a false positive or a credential exposure.
