# R10 — isolated CPU MKL qualification

No GPU campaign resumes until every original CPU numerical gate passes.
R9 corrects final text on all 14 clips. R10 addresses CPU primitive arithmetic:
the original checkpoint's window/Mel buffers, batched native MKL DFTI, vector
sqrt/log, explicit FP32 squaring, native MKL Mel SGEMM and the subsampling
projection with bias included in SGEMM. It links standalone MKL, not libtorch.
The five subsampling convolutions and the remaining encoder/TDT arithmetic
retain their R8B source. R9 text rendering is included without changing token
selection. This is a separate qualification binary, not a production promotion
or a qualification of the default Candle CPU backend.

The original R7/R8B baseline, its byte controls and actual unmodified NeMo
reference rerun on each of the 14 real CaFE/Oreau WAVs. The new profile runs
native PCM, separate shared-Mel and exact-input frontend/subsampling/TDT
experiments. All fixture inputs and weights are echoed and verified. No NeMo
activation enters native PCM inference. All prior numerical thresholds remain
unchanged, including large internal activations; higher-precision diagnostic
results cannot turn their FP32 failures into a pass.

The standalone SDK is pinned to `mkl==2024.2.2` / `mkl-include==2024.2.2` in
the existing NeMo Python environment. Linux x86-64, GCC, ar and its GNU OpenMP
runtime are required. Sources, checkpoint identity and all MKL shared-library
hashes are recorded. The Cargo target and baseline executables are isolated;
altered cached source or SDK identities are rejected.

With the existing prepared CPU environment, from `half_duplex`:

```bash
uv pip install --python "$HDB2C_PYTHON" \
  --default-index https://pypi.org/simple \
  'mkl==2024.2.2' 'mkl-include==2024.2.2'
"$HDB2C_PYTHON" -m hdb2c.c3_r10_cpu \
  --corpus /absolute/path/to/cameo/manifest.json --require-numeric
```

CI dispatch input `r10_cpu=true`, or the explicit `[r10-cpu]` commit marker,
selects this CPU-only profile. There is no GPU device option. JSON reports
retain separate execution, numerical and bitwise verdicts. A complete run
with a failed gate exits 2. Public-corpus raw data are deleted per clip unless
`--keep-raw` is supplied locally. No microphone or streaming claim follows.

The initial local AMD EPYC experiment gives bit-identical frontend output
and common-input projection on all 14 clips. Full internal numerical gates
still fail after convolutions/scaling. These measurements are platform/profile
specific; the independent CI measurement is required before any wider claim.
Full local evidence: [R10 CPU results](RESULTS-HDB2-C3-A-R10-CPU.md).
