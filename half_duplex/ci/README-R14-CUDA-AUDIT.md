# R14 — CUDA FP64 operator audit and 42-clip corpus

The native CUDA/cuDNN R9 path produced identical text on all 14 excerpts and
identical 283 decoding decisions in Extended run 39781921938. Its final encoder
output met the existing numerical limits on all 14, but intermediate stages
exceeded the original absolute limit of 0.001. The common-input projection had
max_abs 0.0068359375 and relative_L2 at most 6.973534552982341e-7. These aggregate
numbers do not establish an elementwise absolute-plus-relative criterion.

On 2026-10-22 the user also reported successful native microphone inference
outside Podman/CI with candidate 711776ce60b3558b3f338f90ae7687552f0a0c84:
`NATIVE_PCM_FRONTEND_ENCODER_TDT`, `Cuda(CudaDevice(DeviceId(1)))`, intelligible
French text with some technical-term errors. This is user-reported functional
evidence, not an independent WER, bitwise or numerical qualification. Neither
the private recording nor its transcription is committed/published.

## What is measured

R14 leaves generated R9 Rust arithmetic unchanged. It executes the autonomous
PCM path and the separate shared-Mel diagnostic, preserves the R8B/R9 exact
instrumentation controls, and requires exact token IDs, emission frames,
durations, text and every TDT decision. CPU/GPU floating-point bit differences
are diagnostics. Exact input/weight identity and same-backend instrumentation
controls remain required.

The corpus is the pinned public CAMEO revision
`c7f2fc9f0538ca871d38e183ba8ad6c329120ecc`. Three distinct PCM excerpts are
selected per CaFE/Oréau emotion stratum: the first 14 retain their previously
recorded SHA256 identities; 28 additional utterances form a separate cohort.
Duplicate PCM, missing strata, changed anchors or changed PCM fail closed.
The additional cohort is utterance-disjoint, not guaranteed speaker-disjoint.
Its audit metrics are visible; a policy chosen after examining them requires
fresh unseen utterances before claiming independent policy validation. It does
not establish open-domain accuracy or WER against human transcriptions.

Copies of the actual NeMo checkpoint modules execute in CPU FP64 on the exact
observed FP32 operands. The original reference stays in FP32. Replays cover:

- All 5 subsampling convolutions, 3 ReLUs and the projection, including common
  NeMo operands and the autonomous candidate's actual post-mask operands.
- Block0 normalizations, both feed-forward modules, attention with the observed
  relative-position tensor/mask, and convolution with the observed pad mask.
- The 4 block0 residual operations and the positional input scaling.

An FP32 output cannot be labelled as an FP64 reference. FP64 results remain
FP64 during comparison. These are local operator replays, not a full FP64 ASR
reference. No reference activation or FP64 result is injected into inference.

For linear/convolution replays, a conservative roundoff diagnostic uses
`gamma_n = n*u/(1-n*u)`, with `u=epsilon/2` and `n=2*k+1`, and the actual
FP64 sum of absolute products. It reports a relative term and an elementwise
absolute cancellation allowance. This bound assumes normal IEEE arithmetic
without overflow/underflow and a direct dot reduction; FFT/Winograd transforms
are not certified by this bound. It can be loose and is explicitly not an
acceptance gate for a composed neural network.

## Tolerances and promotion

Every encoder stage and FP32/FP64 replay is surveyed element by element with:

`abs(candidate-reference) <= atol + rtol*abs(reference)`

The original absolute floor is 0.001. The prospective relative survey is
`[0, 1e-7, 1e-6, 1e-5, 1e-4]`; it records violation counts, first violating
indices, maximum excess and the minimum required relative tolerance as a
diagnostic. In particular, a near-zero failure cannot hide behind a large
global tensor amplitude or cosine. The results are separated by operation
family, execution mode, original 14 and additional 28 clips.

No grid point is selected automatically and no new relative threshold is
approved from the candidate outputs. The next numerical policy must be
justified from these FP64/local-operand results and validated independently.
Existing numerical gates remain mandatory during collection.
Thus this audit may still exit 2 with `NUMERIC_PARITY_NOT_ESTABLISHED`, even
when it completes all 42 clips and all decoding decisions match. Such an audit
does not promote `extended` or pretend that a proposed threshold was measured.

The integrated CPU job retains the full R13 qualification on the original 14
and additionally exercises the FP64 operator oracle on their real reference
operands. A small actual-NeMo fixture checks 23 reference replays and 41
candidate-layout replays, including an intentionally changed local operand,
to detect dtype/API mistakes before downloading the real checkpoint. A completed fixture
is not evidence of real ASR or GPU execution.

## Running

The physical Extended job selects R14 automatically through
`bash half_duplex/ci/extended_gate.sh gpu`. After the hosted CPU/CUDA stages
succeed, use the existing V6.3 runner with the new run ID:

```bash
./run-once.sh check candle/extended --run-id RUN_ID
./run-once.sh start candle/extended --run-id RUN_ID
```

JSON metrics are uploaded even on failure. Raw tensors and recordings are
excluded from uploads and per-clip temporary traces are removed by default.
Per-operator FP64 dtype/API failures make the audit incomplete and block
qualification. Original CPU and CUDA qualification paths remain available.

## Native microphone smoke test

The already-built R9 consumer can run directly on the laptop. Its diagnostic
mask selector must match CI; both attention and convolution masks are enabled:

```bash
HDB2C6_MASK_MODE=both \
"$HDB_NATIVE_BUILD/cargo-target/release/hdb2c_r9_probe" \
  --model-dir "$HDB_MODEL_DIR" --input "$HDB_MIC_RUN/input.wav" \
  --output "$HDB_MIC_RUN/output" --device cuda-cudnn &&
jq '{device, text}' "$HDB_MIC_RUN/output/result.json"
```

This is an ASR smoke test on a recorded microphone intervention. HDB1's
automatic end-of-turn controller is a separate integration.

Primary numerical background:
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html),
[NVIDIA floating-point comparison guidance](https://docs.nvidia.com/cuda/floating-point/index.html#verifying-gpu-results).
