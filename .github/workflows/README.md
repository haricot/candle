# Haricot Candle — active GitHub Actions workflows

This directory contains orchestration workflows on `orch/extended-v1`. The
repository's default branch is `mirror_orch`; the two branches intentionally
expose different workflow inventories.

## Current CUDA/SM61 qualification

| Workflow file | Scope | Entry point |
| --- | --- | --- |
| `candle-extended-composite.yml` | **Current Extended**: dynamic standalone sources; hosted `cpu_cuda_compile`; binary-only `cpu_execute`; physical `SM61 Extended runtime`; validated promotion | Push to `orch/extended-v1` (selected paths), or `workflow_dispatch` |
| `candle-legacy-cuda.yml` | Separate CUDA legacy qualification | `orch/legacy-cuda-v1` |
| `candle-asd-extended.yml` | Isolated ASD candidates and CPU/CUDA compilation | Explicit dispatch |

The Extended workflow does **not** require the older six-feature integration
workflow. It is driven by `.github/extended-sources.json`, not a fixed number
of source branches. The `lean4_standalone` source can remain in the composition,
but its formal proof runs separately in `haricot/haricot-verify`.

## Additional workflows retained

- `candle-sync.yml`: synchronize and prepare legacy candidates.
- `candle-candidate-ci.yml`: independent candidate CPU/CUDA qualification.
- `candle-gpu-ci.yml`: independent GPU qualification.
- `candle-promote.yml`: explicitly requested promotions outside Extended.
- `candle-cleanup.yml`: auditable, opt-in branch cleanup; **not** an automatic
  campaign and not a replacement for Extended promotion.
- `candle-cuda1292-container-smoke.yml`: CUDA container smoke tests on PRs.
- `ci_cuda.yaml`, `maturin.yml`, `python.yml`, `rust-ci.yml`:
  upstream-related build and package CI.
- `trufflehog.yml`: security scanning.

## Retired orchestration files

These workflows were superseded or bound to one-time historical evidence.
Their files were removed from `orch/extended-v1` without deleting Git branches,
artifacts, prior runs, or the historical revisions that contain them:

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

GitHub Actions may still display **historical** workflow runs after the YAML
files are removed. Their history is preserved intentionally.

For the local Pascal runner, start a `candle/extended` one-shot session only
when the physical `SM61 Extended runtime` job is queued.
