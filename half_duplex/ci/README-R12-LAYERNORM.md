# R12: isolated LayerNorm, local CUDA and manual GPU/CPU CI

**CPU-first pause:** the user requires complete CPU parity before any CUDA
test. The GPU workflow and launcher are disabled while the full CPU candidate
is qualified. The GPU commands below describe retained tooling; do not use
them before complete CPU qualification and explicit re-enablement.

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

## Independent CPU operator CI

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
access. Kernel execution on the physical sm61 GPU requires the local run or
the GPU CI profile below; offline PTX compilation is not a GPU qualification.

## GPU CI after the validated CPU operator run

The hosted [CPU run 39649356383](https://github.com/haricot/candle/actions/runs/39649356383)
completed the isolated LayerNorm job with bit-identical outputs on **14/14**
clips. Its `real-cpu-asr` job was skipped; it is not a full-model CPU pass.
The GPU workflow now has a separate `r12_layernorm=true` mode. It downloads
that run's real CPU artifact and verifies its 14-clip bitwise controls,
checkpoint/reference versions and unchanged core operator sources before
executing CUDA. Its artifact dependency retains the CPU run's 14-day expiry;
after expiry, obtain a fresh validated CPU run and update the pinned run ID.

On the laptop, using the existing rootless unified image and an authenticated
GitHub CLI (`gh`), run:

```bash
cd ~/tmp/candle-stt-qualification
git pull --ff-only
python3 half_duplex/ci/stt_sm61_runner.py layernorm-ci
```

This explicitly dispatches **only R12**, obtains the exact new run ID by a
unique request identifier, prints its Actions URL and starts the unified
one-job runner with the same physical GPU lock. It asks for the temporary
runner registration token in the terminal, as with the earlier CI campaign.
Get it from [New self-hosted runner](https://github.com/haricot/candle/settings/actions/runners/new).
The image and actual sm61 GPU still run on this laptop; GitHub Actions
orchestrates the job and stores the JSON result. No full STT pipeline runs
in this mode, even if the legacy `r8_corpus` input was left at its default.

The Actions artifact is `half-duplex-r12-layernorm-cuda-sm61-report`, containing
`r12-layernorm-cuda.json`. Exact CPU/NeMo, CUDA/NeMo and CUDA/CPU comparisons
and CUDA operand-byte controls remain separate. No push can enqueue a GPU
job: the workflow remains dispatch-only.

To dispatch through the GitHub UI or directly with `gh` instead, select the
`half_duplex` branch, set `run_cuda=true`, `r12_layernorm=true`, `r8_corpus=false`,
then start the existing runner with the resulting **new GPU run ID**:

```bash
python3 half_duplex/ci/stt_sm61_runner.py start --run-id NEW_GPU_RUN_ID
```

Do not use the completed CPU run ID for this runner command. The launcher
rejects a stale checkout or a run/job/commit mismatch.

The earlier local `/etc/hosts: permission denied` failure was in `diagnose`,
before any CUDA kernel execution: its Podman command omitted `--no-hosts`.
Both local preflight and local execution now pass that option. The existing
CI runner already passed it to both of its Podman invocations.
