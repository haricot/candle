# Kimodo native F32 — S51-A/B initial standalone

Reference source: https://github.com/nv-tlabs/kimodo/tree/e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954
Target checkpoint: nvidia/Kimodo-SOMA-RP-v1.2. This is an initial source port, not a qualified v1.2 model release.

Implemented: F32 two-stage transformer, PyTorch packed QKV names, sinusoidal time/position embeddings, noncausal key-padding attention, concatenated observation mask, global-to-local root bridge, regular/separated CFG, cosine DDIM x0/eta=0, explicit noise input. No new CUDA kernels or dependencies. Fixed-length batches only; small root feature conversion explicitly executes on host F32.

Configuration, backbone checkpoint prefix, actual weights, stats, LLM2Vec embeddings and normalized observations must come from a pinned official v1.2 fixture. Do not guess dimensions or silently substitute random weights. No checkpoint, licensed model assets or embedding data committed. The smoke test uses zero weights intentionally and tests plumbing, not model fidelity. No motion decoding, SOMA77 expansion, collision solver or text encoder yet.

## Local build
```sh
cargo nextest run -p candle-transformers --lib --release -E 'test(kimodo_cpu_smoke) | test(kimodo_ddim_schedule)'
# Only on kimodo_fixed_build_sm61, with Pascal-compatible CUDA toolkit:
CUDA_COMPUTE_CAP=61 cargo nextest run -p candle-transformers --lib --release --features cuda --run-ignored all -E 'test(kimodo_cuda_smoke)'
```

Next gate: export the official checkpoint config/stats plus identical nonzero noise, embeddings, masks and intermediate root/body/CFG/DDIM tensors from the NVIDIA reference, then compare full tensor arrays on CPU and SM61. No PASS claim before execution. No Extended or DKR workflow required.
