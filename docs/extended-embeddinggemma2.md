# Extended seven-source composition: EmbeddingGemma 2

This campaign extends `Candle Extended` with exactly one additional source:
`embeddinggemma2`, pinned to `f4ea3ad08aa9492db7efabcdeb521cc55608ad1f`.

The source branch is based on `main`. It is **not** a descendant of
`mirror_orch` and must not be rebased on the workflow-orchestration branch.

The first six Extended inputs, historical six-source receipts, and stable
promotion refs remain unchanged. This seven-source campaign uses
`kind=extended-seven` and only publishes immutable temporary candidate
refs. It is **not** automatically eligible for an old six-source promotion.

Checks: independent CPU compilation and model-configuration test;
seven-source integrated CPU compilation and test; hosted CUDA SM61 compile;
physical SM61 runtime gates (including native EmbeddingGemma 2 CUDA compile).
None of those gates by itself proves numerical embedding parity or quality.

The push on `orch/extended-v1` is intentionally an opt-in launch mechanism;
it auto-creates a unique `extended-<run-id>-<attempt>` campaign. Subsequent
updates must pin any new source SHA explicitly; never rely on a moving head.
