# R1.24 S5.4 — Cache JIT, VRAM et CUBIN SM61 (pilot)

## Scope and invariants

This qualification **does not change Candle's PTX runtime or the sealed
25-package inventory**. It produces derivative SM61 CUBIN objects under
`target/dkr-r124-s54/cubins/`, each paired to the S5.1 manifest/PTX SHA-256
and the exact `ptxas --version`. It verifies the S5.3 PASS report and
archive hash before proceeding. The default pilot covers:

* `fill` — simple `fill_f32` kernel for actual PTX/CUBIN output parity.
* `mmvq_gguf` — 243-entry family that dominated S5.3 first touch (~7.1s).
* `mmq_instance_q4_k` — Q4_K MMQ instance for large first-touch comparison.

Passing the pilot **does not mean Candle can yet load CUBIN** or that the
MMVQ/MMQ CUBIN symbol has been numerically exercised. The direct-driver
probe validates module loading + function resolution for those modules, and
executes a concrete 4,096-element `fill_f32` kernel for both PTX and CUBIN,
checking identical outputs. Promotion to Candle runtime must await full
numerical parity on the actual MMQ/MMVQ entrypoints and fallback semantics.

## Toolchain gate

CUDA Toolkit **13.x dropped offline compilation for SM61**. Use a separate
CUDA 12.x `ptxas` (typically 12.8 or 12.9), even when driver version is 580+.
The prepare phase fails before work if it detects 13.x or cannot locate ptxas.
Do not change the CUDA driver or system toolkit to run this experiment.
`--ptxas /path/to/cuda-12.x/bin/ptxas` selects an explicit offline assembler.

```sh
cd /home/np/tmp/candle_dkr/candle
python3 tools/dkr-r124-s54-driver-probe.py --self-test
python3 tools/dkr-r124-s54-qualify.py self-test

python3 tools/dkr-r124-s54-qualify.py prepare \
  --seal "$PWD/target/dkr-r124-s51/seal.json" \
  --s53-result "$PWD/target/dkr-r124-s53/headless-20261121T130221097950Z/result.json" \
  --ptxas /path/to/cuda-12.x/bin/ptxas \
  --out "$PWD/target/dkr-r124-s54"
```

If `ptxas` from CUDA 12.x is already on PATH, omit the explicit flag.
The preparation can resume after a partial error: successfully compiled
families keep per-CUBIN receipts and are reused if identities match.
Once the top-level receipt is created, it is deliberately immutable.
Never delete/reset the qualified PTX catalogue or global CUDA caches.

After preparation, save graphical work before headless-relogin:

```sh
headless-relogin --with-after-run-openlog \
  /usr/bin/env -u DISPLAY -u WAYLAND_DISPLAY \
  python3 "$PWD/tools/dkr-r124-s54-qualify.py" headless \
    --receipt "$PWD/target/dkr-r124-s54/receipt.json"
```

## Measurement details

Each selected family and each of three independent rounds receives:

1. PTX **cold**: driver module loaded in a fresh context and process with an
   isolated empty `CUDA_CACHE_PATH`;
2. PTX **warm**: same cache path, new process and context;
3. **CUBIN**: same cache path, new process and context.

All measurements use `CUDA_MODULE_LOADING=EAGER` to make module-load
measurements more comparable, including `cuModuleLoadData` timing,
`cuModuleGetFunction` timing and in-context `cuMemGetInfo_v2` deltas.
A driver context is created **outside** the module timer. This does not
measure NVCC build time and does not represent end-to-end model inference.

A separate replay of the already qualified S5.3 Nextest binary samples
**system-wide GPU VRAM used** through NVML at a 25ms interval while the
process is alive. This is an observational lower bound on peak memory,
not exact allocator watermarks. Sampling could perturb fine-grained
timings; S5.3 remains the primary timing baseline. No NVML polling is
performed inside the direct driver probe's timed load/lookup calls.

## Acceptance and limitations

* All sealed PTX and manifest digests unchanged.
* Offline-generated CUBIN SHA-256 and ptxas version recorded.
* PTX cold/warm and CUBIN loaded for the same selected symbol on SM61.
* `fill_f32` result is exactly 1.25 on all 4,096 elements for both formats.
* Per-process VRAM deltas, cache file counts, clocks/P-state and sampled
  live S5.3 memory recorded; no hard-coded performance ratio threshold.
* Tooling must not rebuild PTX, invoke NVCC, alter Candle's production
  Rust source, or clear shared CUDA cache.
* The conclusion is a **CUBIN feasibility pilot**, not the final 25-family
  runtime migration. Further work must include CUBIN-aware verified
  catalogue records and actual MMQ/MMVQ runtime parity, ideally with a
  driver-load-only and kernel-first-launch split.
