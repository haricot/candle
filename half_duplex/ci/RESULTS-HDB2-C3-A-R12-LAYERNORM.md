# R12 — first LayerNorm operator, local CPU evidence

The new standalone harness was executed against the original pinned checkpoint
(`0acdaf1e1ebf09094400263006d134d33b0ebb871ebb064253cfd3a7c528903c`),
Torch 2.12.1+cpu, NeMo 2.7.3 and all 14 actual pinned public CaFE/Oréau clips.
Fresh references were captured on this Linux x86-64/AVX2 host with four Torch
threads. The original CPU C++ source is unchanged from the R11 experiment.

| Measurement | Result |
| --- | --- |
| First `norm_feed_forward1` output, standalone CPU vs original NeMo | **14/14 bit-identical**, zero differing FP32 words |
| Reference encoder output, observation hook vs no hook | **14/14 bit-identical** |
| Offline CUDA compilation | NVRTC 12.9.86, compute_61, 33,201-byte PTX emitted |
| Physical CUDA kernel execution | **Not run here**, requires local sm61 invocation |
| Integration into the Candle encoder | Not performed |
| Full STT / TDT / microphone / streaming qualification | Not evaluated by R12 |

The CPU report records exact PCM, input, weights, output, native library and
harness source hashes. Its observed normalized feature dimension is 640.
[Full JSON metrics](results/r12-local-cpu-20261021.json) contain no raw operands,
WAVs, transcripts or local paths. These are local measurements; no new hosted
CPU result or physical GPU pass is claimed.

The generated CUDA code reuses the CPU scalar-lane algorithm and has a real
device kernel entry. NVRTC compilation succeeds without CUDA hardware; that
alone establishes neither GPU execution nor numerical/bitwise GPU parity.
The driver harness checks physical compute capability 6.1 and echoes all
copied common operands to verify their bytes. It has no CPU fallback.

Validation: all 170 Python contract tests pass, including the ordered
arithmetic, invalid operand rejection, local container command boundary and
manual CPU workflow checks. The pinned uv bootstrap was also downloaded,
checksum-verified and executed. The local shell harness successfully prepared
its environment, downloaded the public checkpoint/corpus and completed the
real CPU operator campaign. Local Podman/GPU execution remains to be measured
on the user's existing unified image and physical laptop.

[Commands and exact scope](README-R12-LAYERNORM.md). The CPU CI input is ready
for manual dispatch after the requested local GPU result. No GPU workflow has
been dispatched, and the full-model CPU-first gate remains unchanged.
