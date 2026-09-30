# ASD Conv2D DW5x5 Exact Integrated Validation V0

Status: **PASS**

This document records the integrated validation of the four exact Conv2D
DW5x5 F32 candidate points. It does not activate production dispatch and does
not introduce any generalized `depthwise && kernel == 5 => raw` rule.

## Candidate policy

- policy: `asd-exact-v2-candidate-84f3dc50433e225b`
- wire: `ASD-EXACT-POLICY-V2`
- authoritative candidate wire SHA-256:
  `2e68fc01699d9c727bc7c8493032339fd7019d52648bfa750d093c5cdce6b106`
- target: NVIDIA sm61, device scoped
- selected implementation: `candle.depthwise-conv2d-5x5.raw.v1`
- production activation: false

## Protocol

The final integrated gate uses time-equivalent preconditioning per backend:

```text
warmup_policy=time_equivalent_per_backend
warmup_ms=500
timed_samples=40
launches_per_sample=32
max_drift_pct=5.0
minimum_integrated_speedup_x=1.10
sequence=current_a,asd_b,current_a2,asd_a,current_b,asd_a2
```

This supersedes count-equivalent warmup for integrated evidence. Count-based
warmup strongly over-conditioned the much slower CURRENT/cuDNN path and was
therefore an observer-effect source for the very short raw kernels.

## Integrated results

| Exact point | CURRENT median us | ASD raw median us | Speedup | CURRENT drift | ASD drift | Parity | p90 gate | Result |
|---|---:|---:|---:|---:|---:|---|---|---|
| C48 64x48 | 190.708781 | 13.154156 | 14.497987x | 0.618% | 0.439% | exact | PASS | PASS |
| C96 32x24 | 190.300969 | 8.598406 | 22.132121x | 0.072% | 0.552% | exact | PASS | PASS |
| C192 16x12 | 189.203531 | 5.064813 | 37.356473x | 0.309% | 0.221% | exact | PASS | PASS |
| C384 8x6 | 189.641844 | 4.949750 | 38.313419x | 0.018% | 0.046% | exact | PASS | PASS |

For every exact point:

```text
selected=raw reason=exact_asd
submitted_backend=raw
synchronized=true
max_abs=0
max_rel=0
parity=true
drift=true
integrated_speedup=true
p90_non_regression=true
```

Negative exact-domain lookup:

```text
c=64 h=32 w=24
lookup=miss
selected=current
submitted_backend=current
max_abs=0
max_rel=0
pass=true
```

Final gate:

```text
MULTI_POINT_GATE cases=4 domain_miss=true production_activation=false warmup_policy=time_equivalent_per_backend pass=true
STATUS=PASS
```

## Weighted DW5x5 diagnostic

Using the workload repetition counts 2 / 4 / 4 / 2 for C48 / C96 / C192 /
C384 respectively:

```text
CURRENT weighted total = 2278.719250 us
ASD raw weighted total =   90.860688 us
reduction              = 2187.858562 us
weighted speedup        = 25.079265x
weighted reduction      = 96.012642%
```

This weighted value is diagnostic only. End-to-end application impact must be
measured after a later production promotion.

## Lifecycle

All four exact points have now passed the integrated candidate gate:

```text
MEASURED
  -> TUNER_CANDIDATE
  -> INTEGRATED VALIDATION PASS   [this document]
```

The next artifact is the Flow Adaptive Tuner V2 multi-point integrated
validation export. Production dispatch remains unchanged until a separate
promotion closure is completed.
