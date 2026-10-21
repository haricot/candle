# R12: local CUDA LayerNorm, then independent CPU CI

This tests **only `encoder.layers.0.norm_feed_forward1`** on the actual common
inputs of the 14 pinned public CaFE/Oréau clips. It does not replace the
operator in the Candle model and does not qualify full GPU/CPU STT, TDT,
microphone, streaming or performance. The full R11 CPU pipeline remains
unqualified until its remaining declared gates pass.

The archived CPU scalar-lane Welford/cascade source is compiled unchanged with
its AVX2/FMA flags. A diagnostic CUDA port is generated from that same source
and compiled with NVRTC 12.9 for compute_61. The original source and emitted
CUDA/PTX hashes are recorded. NeMo remains the pinned **CPU-only** reference;
no `torch.cuda` is required on Pascal. GPU equality is an observed verdict,
never inferred from the existing CPU result.

## On the physical sm61 laptop, outside CI

Use the same rootless Podman account and existing unified image as the previous
GPU campaign. From the existing clone:

```bash
cd ~/tmp/candle-stt-qualification
git pull --ff-only
python3 half_duplex/ci/local_layernorm.py --device cuda
```

The launcher holds the unified runner's `gpu.lock`, checks the existing image
and GPU, and overrides its runner entrypoint with this local test. It makes no
GitHub Actions request and requires no registration token. It installs/reuses
the isolated CPU NeMo environment, fetches the public model and the 14 public
clips, verifies the model's qualified SHA256, captures the real normalization
inputs/weights/outputs, checks CPU arithmetic, then runs the CUDA kernel.
Each GPU input transfer is copied back and checked byte-for-byte.

Downloads and reports persist under `~/.cache/haricot-stt-layernorm`, with one
new `runs/<timestamp>-<pid>` folder per invocation. The last output prints the
exact host paths of `result/report.json` and `console.log`. Rerunning reuses
the model, Python environment and package caches; previous reports are kept.
The source checkout is mounted read-only. A missing/busy GPU is an error.
There is no CPU fallback for the CUDA candidate.

The report separates `cpu_vs_nemo`, `cuda_vs_nemo` and `cuda_vs_cpu`. Exit 0
requires bitwise equality on all 14 clips for the requested backend, including
the CPU reproduction control. Exit 2 means an executed operator did not meet
that strict criterion; exit 1 means preparation/execution was incomplete.
Signed zeros are compared as bits. No tolerance is relaxed.

## CPU CI after reviewing the local GPU result

The same workflow has a new manual input, `r12_layernorm=true`. It selects an
independent hosted **CPU operator** job, with new actual NeMo references and
only a JSON metrics artifact. Do not confuse this with full STT qualification.

```bash
gh workflow run half-duplex-cpu.yml --repo haricot/candle --ref half_duplex \
  -f r12_layernorm=true
```

This input does not enqueue a GPU job. The new heavy CPU operator job does not
run on push. The existing CUDA workflow remains manual-only.

Without Podman, on a prepared Linux x86-64 CPU environment with AVX2/FMA, g++,
Python and uv, the underlying harness also runs directly:

```bash
bash half_duplex/ci/r12_layernorm.sh cpu
```

The harness uses the existing `install-nemo` pins (Torch 2.12.1+cpu and NeMo
2.7.3). For local CUDA the unified CUDA 12.9.2 image supplies NVRTC and driver
access. Kernel execution on the physical sm61 GPU still needs that local run;
offline PTX compilation is not a GPU qualification.
