# Stateful F16 ShortConv candidate ABI

`asd.xswyn.f16.k3.v1` extends the external manifest executor with three read-only
inputs and two fresh outputs. It is an unqualified candidate for the observed
LFM decode signature, not a promoted performance result. Production Extended
and the previously validated diagnostic collector pins are unchanged.

| Position | CUDA argument | Contract |
| --- | --- | --- |
| 0 | `const half *x` | contiguous F16 `[B,C,1]`, zero offset |
| 1 | `const half *state` | contiguous F16 `[B,C,3]`, zero offset |
| 2 | `const half *weight` | contiguous F16 `[C,1,3]`, zero offset |
| 3 | `half *y` | fresh F16 `[B,C,1]` |
| 4 | `half *next_state` | fresh F16 `[B,C,3]` |
| 5 | `uint32_t batch` | positive batch count |
| 6 | `uint32_t channels` | positive channel count |

Each row shifts `state[1], state[2], x` into a separate next-state buffer and
computes the weighted sum. Prior state, input, weight and retained next states
remain immutable, including when the caller forks from an earlier checkpoint.
Output writes use mutable cudarc kernel arguments for stream dependency tracking.
No device synchronization is inserted per operator call.

The candidate rounds each product to F16, converts it back to F32, sums in the
same sequential order as Candle's `fast_sum_small_f16`, then rounds the output
to F16. It does not replace three rounded products with a single FMA sequence.
F32 multiply/conversion avoids relying on Pascal's low-throughput native F16
arithmetic. Bias and LFM projections/gating remain outside this ABI.

The runtime profile operation is `causal_depthwise_conv1d_step`, with dtype
`f16`, spatial=1, kernel=3, groups=C, stride/dilation=1 and no padding. Its state
shape, zero-offset contiguous layout and immutable contract are implicit in
this operation/version and checked before profile lookup. This leaves existing
F32 decision signatures and identities unchanged. Wrong kernel ABI, output
count, launch coverage, artifact hash or decision binding fails closed.

Without a matching profile, unsupported dtype/kernel/layout/device and autograd
use the tensor reference. A `candidate` decision is accepted only when
`CANDLE_ASD_ALLOW_STATEFUL_CANDIDATE=1`; that opt-in applies only to this stateful
F16 raw-CUDA operation. It does not turn existing F32 candidates into promoted
decisions. The coordinated tuner stages an isolated profile/artifact directory
and leaves the user's production ASD store untouched.

`candle-core/examples/d7-shortconv-f16.cu` is compiled to a hashed sm_61 CUBIN.
The manifest supplies entry, geometry and row count; the kernel receives B/C
as scalar arguments and supports arbitrary positive B/C when an exact profile
matches. Initial staging covers B=1/2 and C=7/2048, K=3, with 128 threads/block.
The recorded real-model case is B=1, C=2048.

`stateful::validate_f16_candidate` (CUDA + `d7-trace`) runs 48 sequences across
trace-off/on, B=1/2, C=7/2048 and six signed/zero/subnormal/nonuniform patterns:
768 recurrent steps, 48 forks from retained states, and 10 layout/autograd
fallback cases. Outputs and next states must match the same-device reference
bit for bit. Every eligible traced call must have an actual `asd.submitted`
record for the verified artifact; a reference fallback cannot masquerade as
candidate validation. The validation report carries a fresh nonce and artifact
hash. Hardware parity remains pending until this program passes on CUDA.

The Mistral CLI exposes `d7-shortconv-f16-validate --out DIRECTORY` when built
with CUDA + `d7-trace`, so the coordinated launcher reuses the same engine
binary and Cargo target for validation and model capture. A standalone example
`d7_shortconv_f16_validate` invokes the same implementation.

CPU verification runs the actual profile parser in
`tools/d7-stateful-profile-check` and the existing F16/F32 immutable reference
tests. CUDA Rust typechecks and sm_61 CUBIN compilation do not replace GPU
execution, output/state parity or untraced complete-engine A/B measurements.
