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

and can be inspected without building or linking CUDA through the standalone
tool:

```bash
cargo run --manifest-path tools/asd-history/Cargo.toml

cargo run --manifest-path tools/asd-history/Cargo.toml -- \
  conv2d-dw5x5-f32-b1-c384-h8-w6-g384-s1-p2-d1-raw
```

The inspector directly reuses `candle-kernels/src/asd_history.rs` as its
single source of truth. It has no CUDA dependency and therefore does not compile
the kernel catalogue or link `cudart`.

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


### Historical lineage ordering

Historical transitions carry an explicit zero-based `generation` and an
optional `predecessor_transition_id`. `history_for_decision(...)` sorts by
generation before returning entries, so chronology does not depend on the
physical order of the static table.

For the C384 DW5x5 decision:

```text
generation=0
legacy chunk-per-group + cat
  -> cuDNN grouped Conv2D
  speedup=402.470x
  earlier_observed=445.265x

generation=1
predecessor=dw5x5-c384-legacy-chunked-to-cudnn-grouped
cuDNN grouped Conv2D
  -> raw CUDA exact DW5x5
  speedup=38.313419x
```

This is lineage only; the production runtime selects the currently promoted
provider directly.

## Provider-neutral CT1D Exact Profile lookup

Grouped ConvTranspose1D no longer depends on the legacy
`CANDLE_ASD_V2_CT1D_ENABLE` opt-in to read the runtime GPU identity. When a
device-scoped Exact Profile with a target GPU UUID is embedded, CT1D obtains the
UUID from the real CUDA context and passes it to the exact lookup regardless of
whether the promoted provider is raw CUDA or cuDNN.

An unreadable UUID or a runtime/build compute-capability mismatch fails closed.
Explicit backend requests and `CANDLE_ASD_EXACT_DISABLE` retain their existing
precedence.


## Provider Evidence V1: CT1D G2 raw CUDA vs cuDNN

The first provider-aware benchmark is:

\`\`\`text
exact decision:
  ct1d-sm61-s32-g2-raw-exact

A / incumbent:
  execution_provider=raw_cuda
  implementation=candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256

B / challenger:
  execution_provider=cudnn
  implementation=candle.cudnn.grouped-transpose.v1
\`\`\`

The harness is:

\`\`\`text
candle-core/examples/asd_provider_evidence_v1.rs
\`\`\`

It refuses to benchmark if the embedded Exact Profile does not currently bind
the CT1D G2 exact signature to the promoted raw CUDA incumbent.

Provider identities are provider-specific:

\`\`\`text
raw CUDA:
  implementation id
  + module ABI
  + entry symbol
  + SHA-256 of builtin PTX

cuDNN:
  runtime cuDNN version
  + convolution-backward-data algorithm id
  + workspace size
  + exact operation geometry
\`\`\`

No artificial CUDA artifact SHA is created for cuDNN.

The protocol is unchanged from the stabilized artifact benchmark:

\`\`\`text
parity
settle A 500 ms
settle B 500 ms
A1 B1 A2 B2 A3 B3
40 timed samples per phase
32 launches per sample
drift <= 5 %
challenger speedup >= 1.01x
challenger p90 <= incumbent p90
thermal/clock/throttle telemetry = observe-only
\`\`\`

A passing challenger produces \`PROMOTE_CHALLENGER_SIGNAL\`, not an automatic
Exact Profile mutation. A second independent authoritative replication remains
required before any provider promotion.

Build check:

\`\`\`bash
export CANDLE_ASD_EXACT_POLICY=/home/np/tmp/asd-stage2g-r3-postcommit.osrFpVFJ/release/profile/stage2f-production.v2.asd
export CANDLE_ASD_TARGET_GPU_UUID=GPU-0259509a-8026-4c2b-477f-0b13c4e2117d

cargo check -p candle-core \
  --features cuda,cudnn \
  --example asd_provider_evidence_v1
\`\`\`

Authoritative run should be performed under the same headless/quiet GPU
conditions used for previous performance qualification:

\`\`\`bash
cargo run --release -p candle-core \
  --features cuda,cudnn \
  --example asd_provider_evidence_v1 -- \
  --evidence-out /home/np/tmp/candle_asd/provider-evidence-ct1d-g2-r1.txt
\`\`\`

The resulting file uses:

\`\`\`text
ASD-PROVIDER-PERFORMANCE-EVIDENCE-V1
\`\`\`

and records the incumbent/challenger identities, parity, all six authoritative
phases, aggregate medians, drift, p90, threshold result, telemetry and final
signal.
