# HDB2-C3-A-R8 — frontend, common-input subsampling and full corpus

User-authorized scope: decompose window/FFT/power/Mel/logarithm; isolate each
subsampling operator on exactly the same real input; cover all 14 existing
CaFE/Oreau clips, then CUDA/cuDNN sm61 and actual laptop microphone audio.

R8 preserves the six qualified R7 source hashes before creating a separate
candidate. It adds observation-only frontend buffers and separate operator
replays. An independently compiled R7 baseline must reproduce R8's PCM, Mel,
all observed encoder stages, every TDT activation and every discrete decision
byte-for-byte on the same backend. No output rounding or threshold change.

The actual installed NeMo `FilterbankFeatures.forward` is instrumented only
with observers. Its result must match a fresh uninstrumented forward on all
frames. Derived framing inputs must independently reproduce actual STFT bytes
before being accepted as FFT fixtures. NeMo subsampling module hooks capture
the real input/output and weights for all five convolutions, three ReLUs and
the final linear projection. The native replay echoes every consumed input;
the Python comparator rejects even a one-bit input or weight difference.

The independent PCM pipeline never receives any internal reference activation.
Shared-Mel, primitive and predictor/joint replays have separate folders/scopes.
Reports retain the old absolute/cosine gates and add relative errors and ULPs
as supplementary diagnostics. No relative metric can turn an absolute failure
into a pass. Global bit equality has a distinct verdict.

## GitHub CPU

Push `[real-asr] [r8-corpus]` on `half_duplex`, or dispatch
`Half Duplex CPU and CaFE qualification` with `r8_corpus=true`.
Two isolated Rust targets compile once. All 14 split/emotion pairs must be
unique and their PCM hashes must match the pinned CAMEO manifest. Every clip
gets a complete real reference and native run. Errors are recorded and the
remaining clips still execute. `--require-numeric` makes the job fail if any
declared numerical gate fails, even when diagnostic execution completed.

Only JSON metrics/hashes are uploaded, including per-clip and corpus summaries.
Temporary raw model tensors are deleted after each public-corpus clip to limit
disk usage. No WAV, transcription or token sequence is uploaded by R8.

## Actual CUDA/cuDNN sm61 host

Dispatch `Half Duplex CUDA sm61 candidate` with `run_cuda=true` and
`r8_corpus=true`, or push `[r8-cuda]`. Existing runner environment variables
`HDB2A_RUNTIME_DIR`, `HDB2A_MODEL_DIR`, `HDB2C_NEMO`, `HDB2C_PYTHON` must identify
the bootstrapped pinned runtime, R7B converted model and existing CPU NeMo
environment (including datasets/soundfile). No weights or environments are
reinstalled by the GPU script. It checks real compute capability 6.1, builds
with `cuda cudnn`, and rejects CPU fallback. The NeMo reference stays CPU.

## Actual microphone (local only)

After updating the checkout and exporting those existing environment paths:

```bash
cd /path/to/candle/half_duplex
bash ci/r8_microphone.sh cuda-cudnn
```

This records one 12-second intervention via PipeWire, then runs the same actual
reference/native comparisons. Raw audio and diagnostic evidence stay in the
local report directory. To compare a preexisting recording instead:

```bash
"$HDB2C_PYTHON" -m hdb2c.c3_r8 --wav /path/to/microphone.wav \
  --variant cuda-cudnn --keep-raw --require-numeric
```

The corpus campaign cannot qualify streaming, endpoint detection or the
microphone. Those remain explicitly NOT_RUN until their actual execution.


## R8B follow-up

[R8 CPU measurements](RESULTS-HDB2-C3-A-R8.md) cover all 14 clips and identify
six common-input linear failures and two post-token text mismatches. The current
R8 entry point generates a separate `r8b-v0.7.8b` candidate. It observes actual
operands after production masks, verifies all original R7 trace bytes, adds
FP64 linear diagnostics and identifies tokenizer categories/codepoints. FP64
outputs are diagnostics only. Historical R8 results remain tied to `ce3461ff`;
new reports identify revision `HDB2-C3-A-R8B`. Existing commands and gates apply.

## Physical sm61 using the unified V6.3 runner image

The archive's four existing profiles select other workflows. STT now uses a
small launcher from this repository with the **same existing unified image and
GPU lock**, without changing that archive, its image or its profiles. Dedicated
`half-duplex-stt-RUN_ID` labels bind the one-job runner to the exact campaign.
The workflow verifies the expected run and checked-out commit again.

After a `[r8-cuda]` push, or a fresh workflow dispatch with `run_cuda=true` and
`r8_corpus=true`, use the rootless GPU user on the physical laptop:

```bash
python3 half_duplex/ci/stt_sm61_runner.py check --run-id RUN_ID
python3 half_duplex/ci/stt_sm61_runner.py start --run-id RUN_ID
```

The launcher checks the selected commit, queued job, existing V6.3 image and
hardware before requesting the usual ephemeral registration token locally.
`HARICOT_RUNNER_IMAGE` overrides the same default image name as V6.3. No token
is passed in an argument or stored in a source file. The workflow bootstraps
its own temporary runtime, public checkpoint, corpus and isolated CPU NeMo
environment, then builds the real CUDA/cuDNN candidate for capability 6.1.
It uploads metrics even if unchanged numerical gates fail. Existing prepared
host environments can still use `bash half_duplex/ci/r8_sm61.sh` directly.

This session cannot start a process on the laptop or capture its microphone.
The local microphone command in this document remains a separate measurement,
with recordings and raw evidence retained locally.


R8B CPU results: [actual operands, FP64 and tokenizer attribution](RESULTS-HDB2-C3-A-R8B.md).
If an older launcher fails with `/dev/tty` being non-seekable, update the clone
with `git pull --ff-only`, then repeat `start` for the same queued run. The fix
is in commit `d0a322d`; hardware/image preparation does not need to be repeated.
