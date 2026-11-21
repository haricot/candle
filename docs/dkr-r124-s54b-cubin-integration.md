# R1.24 S5.4b — Experimental embedded CUBIN transport

This is an opt-in integration, not a release promotion. The S5.1 PTX
packages, original S5.2 numerics, S5.3 performance baseline, and S5.4 pilot
remain unchanged. Feature: cuda-dkr-cubin-embedded (implies
cuda-dkr-embedded). Build-only variable: DKR_EMBED_CUBIN_DIR.

The build verifies 25 exact-target CUBIN image digests, bytes, target,
producer toolchain and *corresponding sealed PTX SHA-256* before embedding
binary images using include_bytes! from OUT_DIR. The normal verified PTX
catalogue and launch ABI remain unchanged, but the experimental transport
loads the verified binary via cudarc Ptx::from_binary. No runtime filesystem
override or silent PTX fallback; no NVCC and no static FFI.

## Prepare CUBIN (before graphical logout)

Run from /home/np/tmp/candle_dkr/candle:

    git pull --ff-only origin feature/dkr-r124-neutral-catalog
    python3 -m py_compile tools/dkr-r124-s54b-prepare.py tools/dkr-r124-s54b-qualify.py
    python3 tools/dkr-r124-s54b-prepare.py \
      --seal "$PWD/target/dkr-r124-s51/seal.json" \
      --pilot-receipt "$PWD/target/dkr-r124-s54/receipt.json" \
      --out "$PWD/target/dkr-r124-s54b"

Pass --ptxas /path/to/CUDA-12.x/bin/ptxas if PATH selects CUDA 13.
Three existing S5.4 pilot images are reused when their provenance matches;
the other 22 are compiled offline from the *already sealed PTX*. Completed
per-family CUBIN files can be reused on retry. The final manifest is
immutable and all 25 entries are mandatory.

## Build an immutable Nextest archive (before graphical logout)

    python3 tools/dkr-r124-s54b-qualify.py build \
      --seal "$PWD/target/dkr-r124-s51/seal.json" \
      --cubin-dir "$PWD/target/dkr-r124-s54b" \
      --s52-result "$PWD/target/dkr-r124-s52/headless-20261121T124142625489Z/result.json" \
      --s53-result "$PWD/target/dkr-r124-s53/headless-20261121T130221097950Z/result.json" \
      --out "$PWD/target/dkr-r124-s54b-qualification"

This builds Rust test code under the new feature, without compiling CUDA.
If Cargo fails, inspect archive-build.log; do not start headless.

## Execute numerical qualification and benchmarks headlessly

    headless-relogin --with-after-run-openlog \
      /usr/bin/env -u DISPLAY -u WAYLAND_DISPLAY \
      python3 "$PWD/tools/dkr-r124-s54b-qualify.py" headless \
        --receipt "$PWD/target/dkr-r124-s54b-qualification/receipt.json"

The headless gate:
- Replays all existing S5.2 numeric cases: 120 GGUF MMVQ/MMQ configurations,
  six GGUF MoE codecs, CPU/GPU tensor family parity, BF16 and SIMT MoE.
- Requires all 25 native families loaded with format=cubin and matching
  CUBIN/PTX digests, not merely present as an embedded catalogue.
- Runs the two S4.4 anti-override tests.
- Benchmarks seven S5.3 workloads across three isolated new processes.
- Reports first touch and steady-state vs the original PTX reference.
- Captures GPU state, no Cargo rebuild or ptxas under headless.

Expected markers only after successful runtime qualification:

    DKR_R124_S54B_NUMERIC=PASS cases=120 gguf_moe=6
    DKR_R124_S54B_CUBIN_MODULES=25/25
    DKR_R124_S54B_PERF=PASS rounds=3 cases=7
    DKR_R124_S54B_PASS

Same S5.2 tolerance contract is not PTX/CUBIN bit-exact equivalence.
New CUBIN objects must qualify at runtime before promotion. No guarantee
of application-wide 1254x speedup; S5.4's ratios were module-load only.
The PTX-only build remains separately available to roll back.
Residual S31-specific launch recipes remain an S5.5 architecture gate.
