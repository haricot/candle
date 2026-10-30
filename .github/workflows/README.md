# Candle workflow inventory

The default branch is `mirror_orch`. Current SM61 campaigns live on
`orch/legacy-cuda-v1` and `orch/extended-v1`.

| Entry point | Maintained branch | Purpose |
| --- | --- | --- |
| `candle-legacy-cuda.yml` | `orch/legacy-cuda-v1` | Immutable Legacy preparation, hosted CPU/CUDA, physical SM61 and guarded promotion |
| `candle-extended-composite.yml` | `orch/extended-v1` | Standalone composition, reusable hosted archives, physical SM61 and guarded promotion |
| `candle-cuda-ci-image.yml` | Both orchestration branches | Reuse CUDA 12.9 SM61 environments by input fingerprint and immutable digest |
| `candle-sync.yml`, `candle-candidate-ci.yml`, `candle-gpu-ci.yml`, `candle-promote.yml` | `mirror_orch` | Explicit independent preparation, validation and promotion |
| `candle-asd-extended.yml` | `mirror_orch` | Isolated ASD qualification |
| `candle-cleanup.yml`, `candle-sm61-hygiene.yml` | `mirror_orch` | Checked, explicit maintenance requests |
| `trufflehog.yml`, `rust-ci.yml`, `python.yml`, `maturin.yml` | Maintained branches | Security, upstream CPU and Python/package CI |

Standalone and D7 workflows remain on their respective source/development
branches. Their absence from the default branch does not make them obsolete.

The former `ci_cuda.yaml` uses upstream SM86/AWS runners with CUDA 13 and has
been disabled since January 2026. Its file is removed from the three maintained
orchestration branches. `fp4-format-once.yml` is a historical one-shot formatter
absent from every current head and is retired. Previously retired integration,
bootstrap and finalization workflows stay retired.

See [the cleanup record](../SM61_CLEANUP_20261030.md) for archival safeguards.
Historical workflow runs, artifacts and logs remain available after retirement.
