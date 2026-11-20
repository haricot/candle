R1.24 S5.2 — Extended Numerical Qualification (candidate)

Strict SM61 only. Two test-only Rust changes and a Python archive/replay harness; production sources and 25 PTX packages remain identical to S5.1.

Numerical gates: 10 GGUF codecs x 3 activation dtypes x 4 batch sizes = 120 cases (batch 1,8 MMVQ; 9,17 MMQ); effective dequantized weights times FP32 input are an independent numerical reference. Additional CPU-versus-GPU checks cover affine, binary, cast, conv, fill, indexing, reduce, sort, ternary and unary. Six GGUF MoE codecs check grouped routing and tail width with a scalar reference from effective weights. Existing BF16 and SIMT MoE tests are replayed.

Build an immutable Nextest archive with cuda-dkr-embedded before shutting down the graphical session. Run the archive under headless-relogin; no Cargo compilation or NVCC under headless. Verify unchanged sealed manifest/PTX hashes and S4.4 archive. Require selected numerical test success, exact case inventory, and runtime first-load provenance for all 25 families. Report unseen families explicitly and fail qualification if any are missing.

First-module-load traces do not establish all entrypoints or exotic shapes and dtypes. This is not a kernel throughput benchmark. Further S5.3 tests should isolate cold CUDA JIT and steady-state GPU timings. Rust recipe neutralization remains an open separate architecture gate.
