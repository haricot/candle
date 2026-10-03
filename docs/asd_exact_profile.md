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

Provider-aware evidence is now a first-class benchmark contract. Raw CUDA
identity is artifact/module based, while cuDNN identity is provider/runtime
configuration based. Provider Evidence V1 established the first exact
raw-vs-cuDNN production-path comparison; Provider Evidence V2 hardens the
experimental discipline for all subsequent provider duels.

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

### Provider challenge history

Rejected provider alternatives are recorded separately from promotion lineage.
A stable reject is **not** represented as a `HistoricalGain`, because no
provider transition occurred.

The separate registry is:

```text
PROVIDER_CHALLENGES
```

and its entries are descriptive memory only. They do not participate in
production dispatch, Exact Profile lookup, promotion gates, or Flow-Adaptive
scheduling.

The first recorded challenge is the CT1D G2 raw-CUDA incumbent versus cuDNN
production-path challenger:

```text
challenge_id=ct1d-s32-g2-raw-vs-cudnn-v1
decision_id=ct1d-sm61-s32-g2-raw-exact
protocol=provider-evidence-v1
measurement_plane=production_path

incumbent:
  provider=raw_cuda
  implementation=candle.sm61-exact-grouped.ct1d-s32-g2-u1-b256

challenger:
  provider=cudnn
  implementation=candle.cudnn.grouped-transpose.v1

R1:
  evidence_sha256=1d86ff6da849ffc29dc30ec4eee69f9ab91330314d74221db5fd82bed11dc07b
  decision=REJECT_CHALLENGER

R2:
  evidence_sha256=7e2d4d08f2210b0418261df07921c2ef33b0738d157cfbc834f7c7746f20e367
  decision=REJECT_CHALLENGER

consensus=stable_reject
incumbent_retained=true
profile_change=false
```

The V1 protocol identifier is preserved exactly for this historical result.
Provider Evidence V2 applies only to subsequent provider challenges.

The standalone inspector prints both lineage transitions and provider
challenges. For this exact decision:

```bash
cargo run --manifest-path tools/asd-history/Cargo.toml -- \
  ct1d-sm61-s32-g2-raw-exact
```

The expected semantic shape is:

```text
transitions=0
provider_challenges=1
PROVIDER_CHALLENGE ... protocol=provider-evidence-v1 consensus=stable_reject
  incumbent_retained=true
  profile_change=false
```

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

Smoke/runtime probe (optional, non-authoritative):

```bash
cargo run --release -p candle-core \
  --features cuda,cudnn \
  --example asd_provider_evidence_v1 -- \
  --warmup-ms 20 --iters 3 --inner 1
```

Any deviation from the default 500 ms / 40 / 32 / 5% / 1.01x protocol is
reported as `authoritative_protocol=false` and can only produce
`MEASUREMENT_ONLY`.

Authoritative run should be performed under the same headless/quiet GPU
conditions used for previous performance qualification:

\`\`\`bash
cargo run --release -p candle-core \
  --features cuda,cudnn \
  --example asd_provider_evidence_v1 -- \
  --evidence-out /home/np/tmp/candle_asd/provider-evidence-ct1d-g2-r1.txt
\`\`\`

A trace-clean run must print, before the parity line:

```text
harness_revision=provider-evidence-v1-no-hot-trace-r2
hot_path_trace_expected=false
```

and must not emit repeated
`CANDLE_GROUPED_TRANSPOSE_BACKEND=cudnn ...` lines during warmup or timing.
If either condition is violated, the measurement is invalid regardless of the
reported gates.

The resulting file uses:

\`\`\`text
ASD-PROVIDER-PERFORMANCE-EVIDENCE-V1
\`\`\`

and records the incumbent/challenger identities, parity, all six authoritative
phases, aggregate medians, drift, p90, threshold result, telemetry and final
signal.


### Preserved V1 consensus: CT1D G2

The two trace-clean independent V1 replications remain valid evidence and are
not rewritten into the V2 schema:

```text
protocol=provider-evidence-v1
decision_id=ct1d-sm61-s32-g2-raw-exact

R1:
  evidence_sha256=1d86ff6da849ffc29dc30ec4eee69f9ab91330314d74221db5fd82bed11dc07b
  incumbent_raw_cuda ~= 10.507 us
  challenger_cudnn  ~= 113.393 us
  decision=REJECT_CHALLENGER

R2:
  evidence_sha256=7e2d4d08f2210b0418261df07921c2ef33b0738d157cfbc834f7c7746f20e367
  incumbent_raw_cuda ~= 10.342 us
  challenger_cudnn  ~= 115.327 us
  decision=REJECT_CHALLENGER

consensus=STABLE_REJECT
incumbent_retained=true
profile_change=false
```

These hashes will later be referenced by Provider Challenge historical state as
V1 evidence. They are intentionally not regenerated under V2.

## Provider Evidence V2 benchmark discipline

All new provider duels use:

```text
candle-core/examples/asd_provider_evidence_v2.rs
ASD-PROVIDER-PERFORMANCE-EVIDENCE-V2
protocol=provider-evidence-v2
measurement_plane=production_path
```

V2 keeps the V1 parity, drift, p90 and minimum-speedup gates but adds explicit
experimental-state controls inspired by the stricter benchmark discipline used
for reproducible model comparisons.

The model-specific notions of checkpoint, quantization, context and sampling
map to the exact-operator experiment as follows:

```text
source checkpoint     -> exact git commit + clean tracked tree
quantization/dtype    -> exact dtype in the operation signature
context/input         -> deterministic tensors + SHA-256 identities
sampling protocol     -> fixed warmup/sample/inner/gate configuration
residency             -> explicit CUDA/module/cuDNN state declaration
GPU isolation         -> headless environment + preflight idle check
alternation           -> deterministic balanced A/B ordering
best-run selection    -> forbidden
```

### Authoritative protocol gate

A V2 result is authoritative only when all of the following are true:

```text
default protocol:
  warmup=500 ms/provider
  samples=40
  launches/sample=32
  max drift=5 %
  challenger minimum speedup=1.01x

replication:
  --replication r1
  or
  --replication r2

source:
  git snapshot available
  tracked source tree clean

environment:
  DISPLAY unset
  WAYLAND_DISPLAY unset
  zero pre-existing compute processes on the target GPU
  preflight GPU utilization <= 1 %
```

Failure of any protocol/environment item does not prevent exploratory
measurement, but forces:

```text
authoritative_protocol=false
DECISION=MEASUREMENT_ONLY
```

### Balanced deterministic replication order

V2 deliberately avoids free randomization. The two independent replications
use complementary deterministic orders:

```text
R1 settling: A B
R1 timing:   A1 B1 | B2 A2 | A3 B3

R2 settling: B A
R2 timing:   B1 A1 | A2 B2 | B3 A3
```

This preserves reproducibility while reducing sensitivity to monotonic clock,
temperature or residency drift.

### Residency declaration

For the current CT1D G2 production-path comparison V2 records:

```text
cache_state=warm_after_settling
cold_cache_measured=false

CUDA context:
  warm_after_settling

raw CUDA:
  module warm_after_settling
  source=builtin_ptx

cuDNN:
  handle=thread_local_cached
  descriptors=recreated_per_call
  algorithm=repicked_per_call
  workspace=allocated_per_call
```

Those fields describe the implementation actually traversed by
`Tensor::conv_transpose1d()`; they are not claims about intrinsic cuDNN kernel
latency.

### Production path versus intrinsic provider execution

V2 makes the measurement plane explicit:

```text
measurement_plane=production_path
provider_hot_execution_measured=false
```

A future cached/prepared cuDNN implementation is a new challenger identity and
must receive new evidence. Its intrinsic or prepared hot-path measurements must
not be substituted for production-path evidence.

### Latency and throughput reporting

V2 retains per-launch latency obtained from batched wall-clock timing and also
reports derived serial exact-op throughput:

```text
latency_kind=per_launch_from_batched_wall_clock
throughput_kind=serial_repeated_exact_op
aggregation=median_of_three_phase_medians
best_run_selection=forbidden
```

This is an exact-operator microbenchmark, so model-level distinctions such as
prefill, continued prefill and generation are recorded as not applicable rather
than being imitated artificially.

### Running V2

Compile:

```bash
cargo check -p candle-core \
  --features cuda,cudnn \
  --example asd_provider_evidence_v2
```

Run R1 in a fresh headless process:

```bash
/usr/bin/env -u DISPLAY -u WAYLAND_DISPLAY \
  cargo run --release -p candle-core \
  --features cuda,cudnn \
  --example asd_provider_evidence_v2 -- \
  --replication r1 \
  --evidence-out /home/np/tmp/candle_asd/provider-evidence-v2-ct1d-g2-r1.txt
```

Run R2 independently with `--replication r2`.

Before interpreting performance, an authoritative run must show:

```text
protocol=provider-evidence-v2
harness_revision=provider-evidence-v2-discipline-r1
measurement_plane=production_path
source_tree_clean=true
headless_display_env=true
gpu_workload_preflight_status=ok
gpu_workload_preflight_active_compute_processes=0
gpu_workload_preflight_clean=true
authoritative_protocol=true
```

Provider Evidence V1 remains frozen for the already-established CT1D G2
consensus; it should not be retroactively regenerated as V2.
