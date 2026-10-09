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
decision memory. Historical state is no longer compiled into
`candle-kernels`; `asd_history.rs` has been removed from the runtime.

The standalone inspector reads data from the ASD user store:

```text
~/.local/share/asd/history/current.json
```

Initialize it from the versioned repository seed when needed:

```bash
cargo run --manifest-path tools/asd-history/Cargo.toml -- --install-seed
```

Then inspect all history or one decision without building/linking CUDA:

```bash
cargo run --manifest-path tools/asd-history/Cargo.toml

cargo run --manifest-path tools/asd-history/Cargo.toml -- \
  conv2d-dw5x5-f32-b1-c384-h8-w6-g384-s1-p2-d1-raw
```

If the local history file is absent, the tool can still display its versioned
repository seed, but production dispatch never consults historical state.

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

The CT1D harness now covers the four promoted sm61 exact decisions:

```text
ct1d-sm61-s32-g2-raw-exact
ct1d-sm61-s32-g4-raw-exact
ct1d-sm61-s32-g8-raw-exact
ct1d-sm61-s32-g16-raw-exact
```

`--decision <id>` selects the exact geometry. Omitting it preserves the
historical G2 default.

The harness also separates two purposes:

```text
--purpose performance-challenge
--purpose fallback-qualification
```

`performance-challenge` preserves the promotion gates: parity + drift +
challenger p90 non-regression + minimum speedup.

`fallback-qualification` still measures the same production paths and records
all performance fields, but resilience qualification is based on the
authoritative protocol plus exact parity and stable drift. A slow cuDNN
challenger can therefore be rejected for promotion while still producing a
valid fallback-qualification replication.

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

For the selected CT1D G2/G4/G8/G16 production-path comparison V2 records:

```text
cache_state=warm_after_settling
cold_cache_measured=false

CUDA context:
  warm_after_settling

raw CUDA:
  module warm_after_settling
  source=external_cubin | external_ptx | builtin_ptx

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

Example fallback-qualification R1 for CT1D G4:

```bash
/usr/bin/env -u DISPLAY -u WAYLAND_DISPLAY \
  cargo run --release -p candle-core \
  --features cuda,cudnn \
  --example asd_provider_evidence_v2 -- \
  --decision ct1d-sm61-s32-g4-raw-exact \
  --purpose fallback-qualification \
  --replication r1 \
  --evidence-out /home/np/tmp/candle_asd/provider-evidence-v2-ct1d-g4-fallback-r1.txt
```

Run R2 independently with the same decision and
`--replication r2`. Repeat for G8 and G16.

Before interpreting a replication as qualification evidence, an authoritative
run must show:

```text
protocol=provider-evidence-v2
harness_revision=provider-evidence-v2-ct1d-matrix-r3
purpose=fallback-qualification
measurement_plane=production_path
source_tree_clean=true
headless_display_env=true
gpu_workload_preflight_status=ok
gpu_workload_preflight_active_compute_processes=0
gpu_workload_preflight_clean=true
authoritative_protocol=true
PARITY ... pass=true
STATUS=PASS
FALLBACK_QUALIFICATION=QUALIFIED_REPLICATION
DECISION=QUALIFIED_REPLICATION
```

A single `QUALIFIED_REPLICATION` is not sufficient to publish a fallback.
Two independent authoritative replications (R1 and R2) must both qualify for
the same exact decision and challenger identity. Their two evidence SHA-256
values are then published into the user fallback store, bound to the canonical
`ASD-DECISION-V1` identity of that exact decision.

Promotion remains independent. A typical slow-but-stable resilience result can
therefore be:

```text
PROMOTION_RESULT=REJECT_CHALLENGER
FALLBACK_QUALIFICATION=QUALIFIED_REPLICATION
```

Provider Evidence V1 remains frozen for the already-established CT1D G2
consensus and current G2 qualified fallback. Running G2 through the generalized
V2 harness is useful as a control, but does not rewrite that historical V1
authority.

### CT1D G4 qualified fallback consensus

CT1D G4 completed the generalized V2 fallback-qualification protocol against
the canonical external CUBIN incumbent:

```text
decision_id=ct1d-sm61-s32-g4-raw-exact
incumbent_implementation=candle.sm61-exact-grouped.ct1d-s32-g4-u1-b256
incumbent_source=external_cubin
incumbent_cubin_sha256=72c8aeaacc322596973768f2f8bb61483ab9d05c158fa86e51841f11f746107e

challenger_provider=cudnn
challenger_implementation=candle.cudnn.grouped-transpose.v1
cudnn_runtime_version=91002
cudnn_algorithm_id=0
cudnn_workspace_bytes=0

R1:
  evidence_sha256=8258710d0fbae354bd4b95505e0caf03d13cdb73c7016c82c5bc53228c2c69c2
  authoritative_protocol=true
  parity=pass
  drift=pass
  promotion_result=REJECT_CHALLENGER
  fallback_qualification=QUALIFIED_REPLICATION

R2:
  evidence_sha256=f190a815d690c2b9e665edf985c067dac0d97419c3d237558f87bf5581e9dd8f
  authoritative_protocol=true
  parity=pass
  drift=pass
  promotion_result=REJECT_CHALLENGER
  fallback_qualification=QUALIFIED_REPLICATION

fallback_consensus=QUALIFIED
profile_change=false
```

The two hashes are represented in the fallback seed and, after installation,
in `$CANDLE_ASD_HOME/fallbacks/current.json`. The record is additionally
bound to the canonical decision SHA-256, so it cannot be reused with a changed
G4 decision. It does not promote cuDNN; it authorizes it only as the rank-1
resilience fallback when the G4 raw implementation is unavailable.

### CT1D G8 qualified fallback consensus

CT1D G8 completed the same V2 fallback-qualification protocol against its
canonical external CUBIN incumbent:

```text
decision_id=ct1d-sm61-s32-g8-raw-exact
incumbent_implementation=candle.sm61-exact-grouped.ct1d-s32-g8-u1-b256
incumbent_source=external_cubin
incumbent_cubin_sha256=3358e366a5a3baf94bb8eba3345797dbba2f5e9eb5f0f9357f13f464a7681120

challenger_provider=cudnn
challenger_implementation=candle.cudnn.grouped-transpose.v1
cudnn_runtime_version=91002
cudnn_algorithm_id=0
cudnn_workspace_bytes=0

R1:
  evidence_sha256=222dadb694d57ec7b8804bc8d473e31e3221de506a0ab3bb0286fc0a6d3406ac
  authoritative_protocol=true
  parity=pass
  drift=pass
  promotion_result=REJECT_CHALLENGER
  fallback_qualification=QUALIFIED_REPLICATION

R2:
  evidence_sha256=321a768144fcb4fac18594027406faf21228f4b82363f2a3a0f5eae4ead509e8
  authoritative_protocol=true
  parity=pass
  drift=pass
  promotion_result=REJECT_CHALLENGER
  fallback_qualification=QUALIFIED_REPLICATION

fallback_consensus=QUALIFIED
profile_change=false
```

The two hashes are represented in the fallback seed and, after installation,
in `$CANDLE_ASD_HOME/fallbacks/current.json`, bound to the canonical G8
decision identity. As for G4, cuDNN remains rejected as a promoted provider
and is authorized only as the rank-1 resilience fallback when the G8 raw
implementation is unavailable.



### Canonical decision identity and repository sync

Every Exact Profile decision now has a canonical identity:

```text
decision_identity_schema=ASD-DECISION-V1
decision_identity_sha256=<64 hex>
```

The hash is computed from a normalized ordered representation containing the
decision id, state, provider, every exact geometry/layout field, implementation
id, promotion evidence and speedup threshold. Reformatting the profile does not
change the identity; changing the semantics does.

The build-time Exact Profile adapter embeds this identity in
`ExactAsdMatch`, and `tools/asd-profile` computes the same value from the
local profile. Fallback stores use it as a foreign key. This permits a future
repository sync to accept a downloaded fallback/artifact only when it targets
the exact decision already authenticated by the local build/profile.

## ASD V3 user store and resilient provider resolution

Phase A introduces a user-scoped ASD store:

```text
~/.local/share/asd/
├── profiles/
│   └── current.asd
├── fallbacks/
│   └── current.json
├── history/
│   └── current.json
└── artifacts/
    └── sm61/
        ├── <implementation-id>.cu
        ├── <implementation-id>.cubin
        ├── <implementation-id>.ptx
        └── <implementation-id>.manifest
```

`CANDLE_ASD_HOME` overrides the ASD root and `XDG_DATA_HOME` is honored.
`CANDLE_ASD_MODULE_DIR` remains a direct artifact-directory override.

The `.cu` file is source/provenance only. Runtime raw resolution is:

```text
external CUBIN
  -> external PTX
  -> builtin raw PTX
  -> evidence-qualified provider fallback
```

Artifact discovery is control-plane work. On the first resolution of a known
implementation (or after an explicit refresh), Candle performs filesystem
discovery, artifact hashing, manifest verification, CUDA module loading and
function resolution. The result is cached per `CudaDevice` and per
implementation.

The steady-state execution path is therefore:

```text
Exact Profile decision
  -> per-device resolved implementation slot
  -> cached CudaFunction
  -> launch
```

In particular, the steady-state hot path performs no artifact `stat()`, no
artifact/manifest read, no SHA-256 computation, no manifest parsing, and no
filesystem-cache lookup. An unavailable primary is negatively cached as well,
so repeated qualified-fallback execution does not poll the filesystem.

Artifact replacement is explicit control-plane work:

```rust
device.as_cuda_device()?.refresh_asd_module(implementation_id)?;
```

or, for all ASD implementations on the device:

```rust
device.as_cuda_device()?.refresh_asd_modules();
```

After refresh, the next exact invocation resolves and validates the current
artifact once, then repopulates the execution-plane cache. Changing
`CANDLE_ASD_MODULE_DIR` or files under the ASD user store without a refresh
does not implicitly alter an already-resolved implementation.


### Hot-swappable additive decisions and kernels

ASD preserves a strict no-rebuild extension path for compatible raw CUDA
kernels. The embedded Exact Profile remains the fastest path for existing
decisions. Only when the embedded lookup misses does Candle consult:

```text
$CANDLE_ASD_HOME/profiles/extensions.asd
```

The extension file uses:

```text
ASD-EXACT-EXTENSIONS-V1
base_profile_id=<embedded profile id>
target.architecture=sm61
target.gpu_uuid=<device UUID>
decision|...same canonical decision row format...
```

It is additive-only: an extension cannot shadow an embedded promoted decision.
The extension parser compiles rows into an in-memory exact-key index on first
use. If no additive rows exist, the runtime records an atomic fast-negative;
subsequent embedded misses do not take an RwLock or HashMap lookup.

A new implementation id that is not present in the compiled seven-entry Phase A
catalogue can be loaded without rebuilding Candle when its external manifest
contains the complete launch contract:

```text
ASD-CUDA-MODULE-V1
abi_version=1
implementation_id=<new id>
architecture=sm61
artifact_kind=cubin
entry=<CUDA entry symbol>
artifact_sha256=<64 hex>
candidate_id=<diagnostic id>
kernel_abi=asd.xwo.f32.v1
output_count=<positive integer>
grid_x=<positive integer>
block_x=<positive integer>
shared_mem_bytes=<integer>
decision_identity_sha256=<ASD-DECISION-V1 hash>
```

`asd.xwo.f32.v1` means the existing host ABI of three F32 CUDA pointers:
input, weight and output. New kernels using that ABI can therefore be synced as
data/artifacts and launched by an already-built Candle binary. A genuinely new
host ABI still requires Candle support.

Performance invariant:

```text
embedded decision:
  embedded lookup -> existing static implementation slot -> CudaFunc launch

additive decision:
  first use: extension parse + manifest/SHA/module resolution
  steady state: in-memory exact-key lookup -> cached implementation -> CudaFunc launch

filesystem/json/sha in steady-state hot path = 0
```

`CudaDevice::refresh_asd()` invalidates modules, fallbacks and additive
profile state after a sync. Reload remains lazy/cold.


### No-rebuild dynamic decision/kernel proof

The architectural invariant is that a promoted ASD decision and a compatible
CUDA kernel can be added after Candle was built. The runtime additive profile
and module manifest are cold-control-plane inputs; repeated selection and launch
must use resolved in-memory state.

The validation example intentionally reuses the already-qualified CT1D G8
CUBIN under a completely new decision id and implementation id that are absent
from the embedded implementation catalogue:

```bash
cargo run --release -p candle-core \
  --features cuda,cudnn \
  --example asd_v3_dynamic_no_rebuild_validate
```

The example disables the embedded G8 selector, installs a temporary
`ASD-EXACT-EXTENSIONS-V1` decision plus `ASD-CUDA-MODULE-V1` manifest,
and resolves the renamed CUBIN dynamically. After the cold launch it deletes
the extension profile, manifest and CUBIN before the warm launch. A successful
warm launch with exact parity therefore proves that steady-state execution no
longer depends on those filesystem inputs.

Finally it calls `CudaDevice::refresh_asd()`. Because the temporary files
have already been removed, the deleted dynamic implementation must not be
resurrected. The expected terminal markers are:

```text
implementation_known_to_embedded_catalogue=false
cold_source=external_cubin
control_plane_files_removed_before_warm=true
warm_source=external_cubin
warm_launch_after_files_removed=success
refresh_asd_invalidated_dynamic_cache=true
deleted_dynamic_implementation_resurrected=false
hot_path_filesystem_dependency=false
candle_rebuild_required_for_new_compatible_kernel=false
STATUS=PASS
```

The dynamic path uses generation-invalidated thread-local last-hit caches for
both additive decision lookup and dynamic module lookup. Global maps, profile
parsing, manifest parsing, artifact SHA verification and CUDA module loading
remain cold-path work only.

### Phase D4 source ownership

The runtime is now catalogue-driven and the specialized Pascal artifact-production
sources no longer belong to Candle.

Phase D4 removed:

- `candle-kernels/src/sm61_exact_grouped/sm61_exact_grouped_k00.cu` through
  `sm61_exact_grouped_k06.cu`;
- `tools/asd-sm61-install.sh`;
- the build-generated `sm61_exact_grouped_scope.rs` compatibility surface;
- build-system exclusions and environment hooks that existed only to preserve those
  in-tree sources.

The historical implementation ids remain valid runtime data. They are resolved entirely
through the runtime Exact Profile, `ASD-CUDA-MODULE-V1` manifests and external CUBIN/PTX
artifacts.

Specialized source generation, architecture-specific NVCC compilation, candidate
provenance and artifact publication are responsibilities of the Flow-Adaptive Tuner side.
Candle owns the generic consumer:

```text
current.asd
  -> implementation id + decision identity
  -> manifest
  -> runtime-device architecture validation
  -> verified CUBIN/PTX
  -> generic ASD CUDA executor
```

No Candle rebuild is required to add a new implementation that uses an already-supported
kernel ABI.

The seven historical SM61 decisions are still useful regression fixtures. Their canonical
runtime artifacts live in the ASD user store (for example under
`$CANDLE_ASD_HOME/artifacts/sm61`) rather than in this source tree.

Fallback authority remains in `$CANDLE_ASD_HOME/fallbacks/current.json`. It is parsed and
validated only on the cold control plane. The store is authenticated against the runtime
profile id, canonical decision identity and the actual CUDA device UUID; no embedded Exact
Profile identity is consulted.

Runtime raw resolution is:

```text
external CUBIN/PTX
  -> qualified provider fallback when the promoted raw primary is unavailable
```

See `docs/asd_source_ownership.md` for the repository ownership invariant.

### Inspecting current primary/fallback state

```bash
cargo run --manifest-path tools/asd-profile/Cargo.toml -- --layout

cargo run --manifest-path tools/asd-profile/Cargo.toml -- \
  --profile /path/to/stage2f-production.v2.asd

cargo run --manifest-path tools/asd-profile/Cargo.toml -- \
  --profile /path/to/stage2f-production.v2.asd \
  ct1d-sm61-s32-g2-raw-exact
```

The tool reports promoted provider counts, exact primary identity, V3 artifact
availability, Phase A raw resolution, and ranked qualified fallbacks.

Install the current profile into the user store with:

```bash
mkdir -p ~/.local/share/asd/profiles ~/.local/share/asd/artifacts/sm61
cp /path/to/stage2f-production.v2.asd ~/.local/share/asd/profiles/current.asd
```

The build-time `CANDLE_ASD_EXACT_POLICY` mechanism remains explicit in Phase B
for reproducibility. Merely placing `current.asd` in the user store does not
silently change the Exact Profile embedded in an existing Candle binary.


### Validating the Phase B qualified fallback

The dedicated validator deliberately hides external artifacts. Because Phase B
has no embedded raw PTX, the promoted raw primary is therefore unavailable. A
strict gate forbids falling through to an unqualified generic path:

```bash
cargo run --release -p candle-core \
  --features cuda,cudnn \
  --example asd_v3_phase_b_fallback_validate
```

The validator requires the device-scoped Exact Profile to be embedded at
build time. Set the same profile and target UUID used by Provider Evidence
before running it:

```bash
export CANDLE_ASD_HOME="${CANDLE_ASD_HOME:-$HOME/.local/share/asd}"
export CANDLE_ASD_EXACT_POLICY="$CANDLE_ASD_HOME/profiles/current.asd"
export CANDLE_ASD_TARGET_GPU_UUID=GPU-0259509a-8026-4c2b-477f-0b13c4e2117d
```

The validator accepts either qualified CT1D decision:

```bash
# historical G2 fallback
cargo run --release -p candle-core \
  --features cuda,cudnn \
  --example asd_v3_phase_b_fallback_validate -- \
  --decision ct1d-sm61-s32-g2-raw-exact

# V2-qualified G4 fallback
cargo run --release -p candle-core \
  --features cuda,cudnn \
  --example asd_v3_phase_b_fallback_validate -- \
  --decision ct1d-sm61-s32-g4-raw-exact

# V2-qualified G8 fallback
cargo run --release -p candle-core \
  --features cuda,cudnn \
  --example asd_v3_phase_b_fallback_validate -- \
  --decision ct1d-sm61-s32-g8-raw-exact
```

Before validating, install/update the local fallback store from the current
profile and the versioned seed:

```bash
cargo run --manifest-path tools/asd-profile/Cargo.toml -- \
  --install-fallback-seed
```

With `CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED=1`, generic auto dispatch is
forbidden. The validation therefore fails if the exact decision cannot be
authenticated; a successful cuDNN execution must come from the
decision-identity-bound `fallbacks/current.json` record, not from the normal
`auto_rule`.

A valid run must emit the ASD V3 fallback trace with:

```text
runtime_role=fallback
reason=primary_artifact_unavailable
provider=cudnn
implementation=candle.cudnn.grouped-transpose.v1
```

and finish with:

```text
primary_artifact_available=false
qualified_fallback_required=true
STATUS=PASS
```

The validation switch `CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED=1` exists only
to prove this chain. If the qualified fallback cannot execute, the validator
fails instead of silently trying an unqualified raw/generic fallback.

### Provider Evidence V2 and external raw identity

Provider Evidence V2 revision `provider-evidence-v2-ct1d-matrix-r3` binds the
selected CT1D decision and incumbent evidence to the raw source actually
resolved at runtime:

```text
incumbent_raw_source=builtin_ptx | external_cubin | external_ptx
incumbent_artifact_sha256=<actual SHA-256>
incumbent_raw_proof_status=...
incumbent_raw_identity_verified=true|false
```

An external CUBIN/PTX without a valid matching manifest can still be useful for
non-authoritative experimentation, but cannot produce authoritative V2 evidence.


## Hot-path runtime refresh contract

ASD V3 keeps runtime extensibility and steady-state execution separate.

Compatible decisions and CUDA artifacts can still be installed without rebuilding Candle. A running process intentionally does not poll the filesystem or process environment from the exact execution hot path. After installing or synchronizing a new compatible profile decision, manifest, CUBIN or PTX, call:

```rust
cuda_device.refresh_asd();
```

The next matching call performs cold control-plane resolution and then caches the resolved execution plan. A process restart has the same practical effect when explicit refresh integration is not available.

Runtime policy environment changes alone can use:

```rust
cuda_device.refresh_asd_runtime_policy();
```

That operation snapshots grouped-transpose / ASD policy flags and invalidates execution plans without discarding already verified CUDA modules or additive profile data.

After resolution, the successful exact hot path does not parse manifests, hash artifacts, access the filesystem, or read policy environment variables. Policy changes are therefore observed only at an explicit refresh boundary (or process restart), by design.
