# ASD V3 external CUDA modules

ASD raw CUDA implementations are runtime artifacts. Candle does not compile or own
specialized ASD CUDA sources.

The authoritative runtime Exact Profile selects an `implementation_id`. The generic
CUDA registry then resolves a sibling `ASD-CUDA-MODULE-V1` manifest and its CUBIN/PTX
artifact from either `CANDLE_ASD_MODULE_DIR` or the architecture-scoped ASD user store.

For example, an implementation may be named:

```text
thirdparty.generic-exact.ct1d-s33-g2-u1-b256
```

The name is opaque to the executor. Architecture compatibility comes from the manifest
and the CUDA device that is executing the kernel.

## Manifest contract

Executable raw CUDA implementations require a manifest:

```text
ASD-CUDA-MODULE-V1
abi_version=1
implementation_id=thirdparty.generic-exact.ct1d-s33-g2-u1-b256
architecture=sm61
artifact_kind=cubin
entry=flow_phase_c_ct1d_s33_g2_u1_b256
artifact_sha256=<64 lowercase hex characters>
candidate_id=<candidate identity>
kernel_abi=asd.xwo.f32.v1
output_count=8448
grid_x=33
block_x=256
shared_mem_bytes=0
decision_identity_sha256=<64 lowercase hex characters>
```

The runtime rejects the implementation before launch if any bound identity is wrong,
including implementation id, runtime-device architecture, artifact kind/hash, kernel ABI,
launch metadata or decision identity.

The artifact source is reported as `external_cubin` or `external_ptx`. Filesystem
discovery, manifest parsing, hashing and CUDA module loading are cold-path work and are
cached until an explicit ASD refresh.

## Source ownership

The Flow-Adaptive Tuner side owns specialized CUDA source generation, compilation,
candidate provenance and artifact publication. Candle owns only the generic runtime
consumer and validation surface.

The handoff unit is:

```text
runtime profile decision
+ ASD-CUDA-MODULE-V1 manifest
+ verified CUBIN/PTX
```

Candidate generation must never modify the Candle source tree and never inherits
historical performance evidence merely by reusing an implementation id.

Generic Candle validators may still be used by tuner workflows for parity, lifecycle or
performance evidence, but architecture-specific source/installer tables do not belong in
this repository.

See `docs/asd_source_ownership.md` for the repository invariant.
