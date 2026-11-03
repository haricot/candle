# Haricot Candle — Lean 4 standalone (v0.1)

Independent of ASD-Core. Based on haricot/candle main; opt-in to the Extended
composition. Candle's normal workspace does not build or need Lean 4.

## Purpose

The Lean proof in `lean/HaricotLeanContracts/Index.lean` establishes
bounds for row-major indexing and its transposed view, over unbounded natural
numbers. The Rust contract checks shape multiplication and indices against
machine-word overflow, and can call the regular Candle CPU or CUDA
`Tensor::transpose` with `--features candle` / `--features cuda`.

**Proof boundary:** This is not a formal proof of the Candle implementation,
CUDA/PTX code, numerical precision, absence of data races, or GPU performance.
The Rust and Lean implementations are independently checked; relating the Rust
implementation and CUDA backend to the Lean model is future work.

## Tests

```bash
cargo test --manifest-path tools/lean4-standalone/Cargo.toml
cd tools/lean4-standalone/lean
lake build                    # elan resolves pinned Lean v4.34.1
```

Or, with Lean installed, `bash tools/lean4-standalone/scripts/check.sh`.

Optional Cargo features:
- `candle`: a CPU Tensor::transpose2d contract adapter.
- `cuda`: same adapter using Candle's existing GPU backend dispatch (no new kernel).
- `lean-ffi`: opt-in dependency on argumentcomputer/lean-ffi at pinned commit
  b4beefd55352ad1d864562b1f15cb52db6186b2d. This exercises the Rust-side
  ownership API; a callable Lean export and ABI-safe linkage remain a separate
  explicit qualification milestone.
- digama0/lean-sys (0.0.9 / Lean 4.23) is a **comparison candidate**, not mixed
  into this FFI build due to toolchain/ABI version differences.

No modification to CUDA kernels, ASD runtime, or main branch is included.
