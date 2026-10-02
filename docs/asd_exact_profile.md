# ASD Exact Profile

The **ASD Exact Profile** is the production decision authority for exact operation
signatures. It replaces the architectural term "Exact Policy"; the legacy V2
wire format remains accepted so `stage2f-production.v2.asd` stays authoritative.

```text
                 Flow-Adaptive
                  explore/measure
                       |
                       v
+---------------------------------------+
|          ASD EXACT PROFILE            |
| exact shape -> promoted decision      |
| provider + implementation + evidence  |
+---------------------------------------+
                       |
                       v
              Execution Provider
               /       |       \
          raw CUDA    cuDNN    native
              |
       builtin/external artifact
```

## Exact decision remains the source of truth

The Stage2F profile still binds, for every promoted row:

- exact operation and geometry,
- dtype and layout constraints,
- device scope and GPU UUID,
- execution provider,
- implementation identity,
- evidence identity,
- optional integrated speedup threshold.

The twelve current Stage2F production decisions are unchanged semantically.
They still select `raw_cuda` and retain their existing evidence hashes and
implementation ids.

The file header `ASD-EXACT-POLICY-V2` and `policy_id` field are legacy wire
names. Runtime code exposes `PROFILE_ID` and `profile_id` while retaining
`POLICY_ID` / `policy_id` compatibility aliases.

## Execution providers

The generated lookup exposes:

```rust
ExactExecutionProvider::RawCuda
ExactExecutionProvider::Cudnn
ExactExecutionProvider::Native
```

Current production support is intentionally asymmetric:

| provider | Exact Profile promotion | execution status |
| --- | --- | --- |
| `raw_cuda` | yes | existing Stage2F paths |
| `cudnn` | yes, currently only grouped ConvTranspose 1D/2D | delegated backend, no Candle CUBIN required |
| `native` | contract-reserved | not promotable until Candle has an unambiguous native executor identity |

This prevents a future provider value from silently falling through to cuDNN.

## Artifact identity is provider-specific

A raw CUDA implementation may resolve to:

```text
raw_cuda
  -> builtin PTX
  -> external CUBIN/PTX
```

For this provider, artifact SHA-256, ABI and entry symbol are part of execution
identity.

A cuDNN decision has no Candle-generated CUBIN requirement. Its future evidence
must bind a provider fingerprint (cuDNN version plus the selected execution
configuration/plan identity and relevant runtime context) rather than inventing
an artifact SHA.

NVIDIA cuDNN's backend API models convolution execution through engine
configurations and execution plans; the ASD provider identity should therefore
be based on the selected cuDNN execution configuration, not on a Candle module.

## Flow-Adaptive boundary

Flow-Adaptive proposes and measures challengers for one exact signature. It does
not replace the Exact Profile.

```text
exact signature
   |
   +-- incumbent provider/implementation
   |
   +-- challenger provider/implementation
              |
              v
      parity + performance evidence
              |
              v
      promotion proposal
              |
              v
       ASD Exact Profile
```

The current Candle A/B V1/V2 evidence harness remains artifact-vs-artifact.
Provider-aware evidence for cuDNN is a subsequent step; the profile/provider
contract is introduced first without weakening the existing twelve decisions.

Thermal, clock and throttle observations remain evidence metadata only and do
not change production dispatch.


## Historical provider state

The Exact Profile keeps current dispatch authority separate from historical
decision memory. Historical lineage is exposed from:

```text
candle-kernels/src/asd_history.rs
```

and can be inspected with:

```bash
cargo run --manifest-path candle-kernels/Cargo.toml --example asd_history

cargo run --manifest-path candle-kernels/Cargo.toml --example asd_history -- \
  conv2d-dw5x5-f32-b1-c384-h8-w6-g384-s1-p2-d1-raw
```

Historical entries are tagged as either:

- `historical_only`: useful prior provider/implementation measurements which
  explain why a later decision exists but are not current promotion evidence;
- `promotion_lineage`: the integrated measurement that led to the currently
  promoted implementation, retained for decision context.

For the C384 DW5x5 decision the retained lineage is:

```text
legacy chunk-per-group + cat
  -> native cuDNN grouped Conv2D
     historical speedup = 402.470x
     earlier isolated observation = 445.265x

native cuDNN grouped Conv2D
  -> candle.depthwise-conv2d-5x5.raw.v1
     integrated speedup = 38.313419x
```

These gains describe different generations and MUST NOT be multiplied.
Historical state is never consulted by production dispatch, profile lookup,
promotion gates, or Flow-Adaptive scheduling. It is retained as human/tooling
memory so future work can see the reason and magnitude behind an exact decision.
