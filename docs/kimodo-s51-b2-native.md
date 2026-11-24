# S51-B2-native: direct official checkpoint, no Python requirement

The previous optional PyTorch oracle is not part of this qualification.
Use the current workspace hf-hub 1.0 blocking client, the original
F32 safetensors, YAML architecture and six NumPy normalization arrays.
No LLM2Vec download. No Extended/DKR, CUDA architecture override or new kernels.
The initial runner is unconditional unless an external embedding or normalized
constraints are supplied. It emits SOMA30 features, not a finished animation.

From Candle:
```sh
cargo run -p candle-transformers --release --features cuda,kimodo-cli --bin kimodo -- \
  --model nvidia/Kimodo-SOMA-RP-v1.2 --frames 24 --steps 3 \
  --out "$HOME/.cache/haricot/kimodo-native-b2"
```

Use --load-only to check weights without sampling; --cpu for CPU; --revision
for a specific commit; --model-dir for a pre-existing local checkpoint directory.
HF cache is reused. Remaining files are pinned to the config snapshot revision.
Use a fresh output directory on each successful sample.

Expected markers (not yet executed): KIMODO_NATIVE_CONFIG=PASS,
KIMODO_NATIVE_LOAD=PASS, KIMODO_NATIVE_SAMPLE=PASS.
The first capture is a functional test, not a quality or NVIDIA parity gate.
Next: decode cont6d and FK to SOMA77, build Macarena constraints and mesh solver.
