# Experimental dual FP4 support: MXFP4 GGUF and ModelOpt NVFP4 (E7C)

This branch preserves existing `GgmlDType::Mxfp4` GGUF storage (E2M1 + E8M0 block32) and adds an independent ModelOpt NVFP4 (E2M1 + E4M3FN block16 + F32 global scale) runtime interface. Both are deliberately distinct.

The existing `cuda-legacy-fp4` feature opt-in gates ModelOpt software NVFP4 CUDA kernels and all E7C runtime modules. It does **not** enable native hardware FP4 Tensor Cores on Pascal sm_61. Normal GGUF MXFP4 remains available without this feature.

## Rust API

* `candle_core::quantized::fp4_e7c_runtime::{Fp4E7CLinear,Fp4Format,Fp4Execution}`
* `Fp4E7CLinear::from_mxfp4(qtensor)` requires GGUF MXFP4 and uses existing `QMatMul`.
* `Fp4E7CLinear::from_modelopt_bytes(cuda, n, k, packed, scales_e4m3, global)` uses GPU-resident E6/E7 CUDA operators; `from_modelopt_shards(...)` reads exact Safetensors tensor trio from user-specified shards.
* The ModelOpt path accepts rank-2 FP16 activations, `[tokens,k]` contiguous with zero offset. It rejects silent BF16 conversion and format aliasing. W4A8 reference and W4A8 DP4A remain selectable explicitly.

## Tests

On non-CUDA hosts:

```bash
cargo test -p candle-core --no-default-features --test mxfp4_tests
cargo test -p candle-core --no-default-features --test fp4_e7c_contract
```

On physical sm_61 with `CUDA_COMPUTE_CAP=61` and `NVCC_CCBIN=gcc-14`:

```bash
cargo test --release -p candle-core --features cuda,cuda-legacy-fp4 --test fp4_e7c_contract -- --nocapture
```

The fixture-dependent GPU test is explicitly ignored by default. To run it, first create a Safetensors fixture with `[2,32]` packed `U8` weights, `[2,4]` `F8_E4M3` block scales, and F32 scalar `.weight_scale_2`, all under `model.layers.0.mlp.experts.0.w1`; export its absolute path as `ARCH_E7C_FIXTURE`:

```bash
cargo test --release -p candle-core --features cuda,cuda-legacy-fp4 \
  --test fp4_e7c_contract -- --ignored --exact modelopt_packed_loader_and_dp4a_gpu_sm61 --nocapture
```

The E7B full-forward checkpoint test also remains ignored by default, and requires `ARCH_E7B_ASSETS`, `ARCH_E7B_DUMP`, `ARCH_E7B_IDS`, and an external checkpoint-prepared weights manifest:

```bash
cargo test --release -p candle-core --features cuda,cuda-legacy-fp4 \
  --test nvfp4_arch_e7b_glm_fullforward -- --ignored --exact arch_e7b_glm_fullforward_logits --nocapture
```

## Validation status

E7B v0.3 independently demonstrated `PASS_FULL_FORWARD_SHORT_CONTEXT` with original ModelOpt checkpoint bytes on sm_61 at 9 tokens, 1024 vocab, W4A8/DP4A minimum logit cosine 0.9999963826289395, zero argmax flips, zero W4A8/DP4A MoE route differences across 27 decisions, and 87 DP4A calls. That evidence **does not** prove E7C has passed CUDA compilation. E7C's Safetensors loader and dual-dispatch runtime require local compiler/GPU qualification before production claims.

Limitations: no model-wide automatic projection registration, no KV-cache, no autoregressive decode, no generic rank-3 expert layout support, no compressed-tensors inverse scale guessing. No auto-promotion to `extended`.
