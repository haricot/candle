# ASD V3 external CUDA module manifest

Stage 1a/1b resolves promoted ASD implementation ids through a CUDA module registry.
External modules are opt-in through `CANDLE_ASD_MODULE_DIR` and may be CUBIN or PTX.

For an implementation id such as:

```text
candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256
```

the external provider checks, in order:

```text
<implementation_id>.cubin
<implementation_id>.ptx
```

A sibling manifest is optional for tuner experiments:

```text
<implementation_id>.manifest
```

When present, it is strict and must have this format:

```text
ASD-CUDA-MODULE-V1
abi_version=1
implementation_id=candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256
architecture=sm61
artifact_kind=cubin
entry=flow_v0322_ct1d_s32_g2_u1_b256
artifact_sha256=<64 lowercase hex characters>
```

The runtime hashes the actual selected artifact and rejects the module before launch if
the manifest does not match the artifact, implementation id, architecture, artifact
kind, entry symbol, or ABI.

Trace states:

- `builtin_ptx / historical_evidence_bound`: existing embedded fallback.
- `external_cubin / external_artifact_unverified`: external module without a manifest.
- `external_cubin / external_artifact_verified`: manifest and actual artifact agree.

Artifact verification is intentionally distinct from ASD promotion evidence. V1 module
manifests do not claim that a newly-built CUBIN inherits historical benchmark evidence.
A later promotion record must bind the exact artifact SHA-256 to fresh evidence.


## Flow-Adaptive Tuner handoff

The Flow-Adaptive Tuner treats `asd_core_v3` as the Candle execution and
qualification target. The tuner owns search and candidate generation; Candle
owns artifact binding, runtime loading, parity validation, performance
qualification, and promotion evidence.

The handoff unit is one SM61 CUBIN for the exact implementation id. Stage it
with:

```text
asd_v3_stage_tuner_candidate
  --candidate-cubin <tuner-output.cubin>
  --out-dir <candidate-module-dir>
  --receipt-out <candidate-receipt.txt>
```

The staging tool hashes the exact CUBIN, copies it under the required
`implementation_id`, writes the strict `ASD-CUDA-MODULE-V1` manifest, and
optionally emits an `ASD-CUDA-TUNER-CANDIDATE-V1` receipt. It refuses to
overwrite an existing staged artifact.

The candidate directory is then used as `CANDLE_ASD_MODULE_DIR_B`; the known
reference artifact directory remains `CANDLE_ASD_MODULE_DIR_A`. Candidate
generation never changes the Candle source tree and never inherits historical
performance evidence merely by reusing the same implementation id.

The qualification path is:

```text
Flow-Adaptive search
  -> candidate CUBIN
  -> stage/hash/manifest
  -> external A/B/A parity evidence
  -> settled time-equivalent performance evidence x2
  -> ASD-CUDA-PERFORMANCE-QUALIFICATION-V1
  -> PROMOTE_CANDIDATE or REJECT_CANDIDATE
```
