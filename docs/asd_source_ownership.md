# ASD source ownership

Phase D4 makes the ASD runtime boundary explicit.

## Candle owns runtime consumption

Candle consumes already-qualified ASD runtime data:

- the authoritative runtime Exact Profile (`current.asd` or an explicit override);
- `ASD-CUDA-MODULE-V1` manifests;
- CUBIN/PTX artifacts;
- qualified provider fallbacks;
- runtime device identity and architecture checks;
- cold-path manifest/artifact verification;
- generic CUDA launch through `asd_exact_cuda`.

Candle does **not** own specialized ASD CUDA source generation, architecture-specific
candidate catalogues, NVCC command lines, or installation recipes.

No new compatible raw CUDA implementation should require a Candle source change or
Candle rebuild. Its execution contract is:

```text
runtime Exact Profile
  -> implementation_id + decision identity
  -> manifest
  -> runtime-device architecture check
  -> verified CUBIN/PTX
  -> generic ASD CUDA executor
```

## Flow-Adaptive Tuner owns artifact production

The Flow-Adaptive Tuner side owns:

- search and candidate generation;
- specialized CUDA source;
- architecture-specific compilation;
- candidate provenance;
- parity/performance qualification orchestration;
- publication of the CUBIN/PTX plus manifest/profile records consumed by Candle.

The seven historical Pascal exact-grouped CUDA sources and the former
`tools/asd-sm61-install.sh` installer were removed from Candle in Phase D4.
Their previous revisions remain available in Git history, but they are no longer
artifact-production inputs of this repository.

Historical implementation ids such as
`candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256` remain valid runtime data.
Their names are not an executor contract: D3 demonstrated that an arbitrary
compatible implementation id can be loaded and executed.

## Repository invariant

A Candle change is required only when the generic runtime contract itself changes,
for example a new kernel ABI or a new execution-provider capability.

Adding another implementation that already satisfies a supported ABI must remain a
data/control-plane operation outside Candle:

```text
source/tuning outside Candle
  -> artifact + manifest + profile
  -> install/sync ASD user store
  -> refresh_asd() or process restart
  -> execute
```
