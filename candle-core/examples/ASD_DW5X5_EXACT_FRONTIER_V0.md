# ASD DW5x5 Exact Frontier V0

Status: measurement evidence only. No production dispatch or ASD policy activation.

## Exact domain

All cases are exact F32 NCHW Conv2D depthwise signatures with contiguous zero-offset input and weight:

- batch = 1
- weight = `[C, 1, 5, 5]`
- groups = C
- kernel = 5x5
- stride = 1
- padding = 2
- dilation = 1

Measured signatures:

1. `conv2d:f32:b1:c48:h64:w48:weight48x1x5x5:g48:k5:s1:p2:d1:contiguous_zero_offset`
2. `conv2d:f32:b1:c96:h32:w24:weight96x1x5x5:g96:k5:s1:p2:d1:contiguous_zero_offset`
3. `conv2d:f32:b1:c192:h16:w12:weight192x1x5x5:g192:k5:s1:p2:d1:contiguous_zero_offset`
4. `conv2d:f32:b1:c384:h8:w6:weight384x1x5x5:g384:k5:s1:p2:d1:contiguous_zero_offset`

Benchmark implementation revision:

`fdf973b9279783fd90e84722ce7ac87ba3ad81cf`

Measurement protocol:

- `raw_a,cudnn_b,raw_a2,cudnn_a,raw_b,cudnn_a2`
- warmup samples = 16
- timed samples = 40
- launches per sample = 32
- maximum drift = 5%
- frontier margin = 10%
- exact numerical parity required
- measurement-only custom backends; no production fallback between candidates

## Campaign results

Three campaigns were observed: two normal desktop runs and one headless run.

### C=48, H=64, W=48

Raw-vs-cuDNN median speedups:

- run 1: 11.926675x, raw drift 7.936% -> HOLD
- run 2: 13.155575x, raw drift 7.725% -> HOLD
- headless: 11.012954x, raw drift 10.853% -> HOLD

Numerical parity was exact in all three runs.

State: `measured/HOLD`.

Reason: performance direction is unambiguous, but the raw timing drift exceeds the 5% stability contract. Do not relax the drift threshold.

### C=96, H=32, W=24

Raw-vs-cuDNN median speedups:

- run 1: 18.788969x, raw drift 0.303%
- run 2: 20.621539x, raw drift 0.429%
- headless: 17.717102x, raw drift 0.207%

Numerical parity was exact and the raw direction was consistent in all three runs.

State: `tuner_candidate-ready`.

Conservative candidate speedup floor: `17.717102x`.

### C=192, H=16, W=12

Raw-vs-cuDNN median speedups:

- run 1: 31.818139x, raw drift 0.505%
- run 2: 35.115503x, raw drift 1.898%
- headless: 30.088013x, raw drift 0.093%

Numerical parity was exact and the raw direction was consistent in all three runs.

State: `tuner_candidate-ready`.

Conservative candidate speedup floor: `30.088013x`.

### C=384, H=8, W=6

Raw-vs-cuDNN median speedups:

- run 1: 32.739511x, raw drift 1.415% -> measured winner
- run 2: 33.618746x, raw drift 8.786% -> HOLD
- headless: 27.726018x, raw drift 10.048% -> HOLD

Numerical parity was exact in all three runs.

State: `measured/HOLD`.

Reason: performance direction is unambiguous, but repeatability does not yet satisfy the 5% drift contract. Do not relax the drift threshold.

## Headless weighted cost estimate

For the observed repetition counts `2 / 4 / 4 / 2`, the headless medians imply approximately:

- cuDNN aggregate DW5x5 time: 2.298750 ms
- raw aggregate DW5x5 time: 0.117424 ms
- potential aggregate reduction: 2.181327 ms
- weighted raw-vs-cuDNN speedup: 19.5766x

This estimate is diagnostic only. It is not an end-to-end promotion claim.

## Promotion state

- C48: MEASURED / HOLD
- C96: TUNER_CANDIDATE-READY
- C192: TUNER_CANDIDATE-READY
- C384: MEASURED / HOLD
- production dispatch: UNMODIFIED
- ASD production policy: UNMODIFIED

No generalized `depthwise 5x5 => raw` rule is justified by this artifact. Promotion remains exact-signature scoped.

## Next gate

1. Re-run C48 and C384 under stronger, time-equivalent GPU preconditioning while preserving the 5% drift limit.
2. Keep C96 and C192 candidate evidence conservative at 17.717102x and 30.088013x respectively.
3. Introduce a generic exact Conv2D ASD contract before integrated validation; preserve the existing ConvTranspose V1 wire unchanged.
4. Only after integrated selected/submitted identity, synchronized completion, parity, domain-miss fallback, drift, p90, and integrated speedup gates may any exact Conv2D decision enter `promoted` state.
