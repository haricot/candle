# Candle Unified R2 — promotion of existing qualification

Source: [run 42518965385](https://github.com/haricot/candle/actions/runs/42518965385), attempt 1, on `orch/unified-candle-v1` at `d744bb0c544b2a0e5e270bc456a45ae2904c68ba`.

- Qualified base `main`: `cfcbd7186d13271843ce7eb41d25659bd0835850`.
- Qualified Extended: `integration/extended/candidate/extended-42518965385-1` at `a0d83786cb9448dc6a36942f35a3e616c06c97d9`.
- Qualified Legacy: `integration/legacy-cuda/candidate/extended-42518965385-1` at `417fce15f880868e9e6fc069a4fc0bdae514e59d`.
- Qualified standalones: the 13 `features[].prepared_sha` entries in `candle-integration/extended.json` at the Extended candidate SHA.
- Legacy's five prepared SHAs must exactly equal the first five Extended prepared SHAs.

The R2 workflow checks the exact source run, seven successful gate jobs, eight unexpired run-bound artifacts, all source/preparation refs and both unchanged integration refs before invoking the existing archival and atomic `git push --atomic --force-with-lease` promotion.

It must never execute Cargo, compile CUDA, use the physical SM61 runner, regenerate candidates, or move `main`. It fails closed when evidence, refs or GitHub permissions are missing. Report: the `candle-unified-r2-42518965385-promotion` artifact and the workflow summary.

R2 trigger token: `[promote-validated-r2:42518965385]` (only on `orch/unified-candle-v1`).
