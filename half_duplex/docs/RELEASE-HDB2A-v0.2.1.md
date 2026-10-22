# HDB2-A v0.2.1 — Convolution normalization qualification fix

- Fixed: v0.2.0 rejected NeMo checkpoints with `conv_norm_type != batch_norm`.
- Supported: `batch_norm`, `layer_norm` (true channel-wise LayerNorm with
  original affine parameters, no fabricated BN statistics).
- Supported offline context: centered, causal, [left,right] convolution padding.
- Rejects unimplemented normalization modes, stating their exact value.
- `./run-hdb2a.sh repair-runtime` safely updates v0.2.0 bootstrap in place.
- Updated conversion metadata, test fixtures, README, STATE-MEMORY.md.
- Qualified: Python synthetic tests 5/5; Bash/Python syntax, ZIP integrity.
- NOT qualified: real checkpoint topology; Cargo/Candle build; ASR outputs.
