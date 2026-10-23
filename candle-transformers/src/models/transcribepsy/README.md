# TranscribePsy — standalone model branch (TP1)

This minimal standalone branch originates from `haricot/candle:main`.
It does not ship audio, checkpoints, conversion scripts, experiments, or
machine-specific optimizations.

## Native model building blocks

- `transcribepsy::encoder`: full-utterance NeMo-style FastConformer stack:
  - typed encoder configuration with explicit unsupported-mode checks;
  - QVAC-native depthwise-striding log-Mel subsampling and frame lengths;
  - relative-position multihead self-attention and finite padding masks;
  - Macaron FFNs, GLU/depthwise convolution and residuals;
  - explicit `layer_norm` / `batch_norm` and convolution context padding.
- `transcribepsy::Predictor`: original QVAC `prediction.dec_rnn.lstm`
  tensors, strict native LSTM biases; no flattened conversion aliases.
- `transcribepsy::Joint`: encoder/predictor joint output.
- `transcribepsy::tdt`: bounded greedy token/duration decision policy.

The encoder accepts log-Mel `[batch, frames, feat_in]`, per-sample lengths
in frames, and returns `[batch, subsampled_frames, d_model]` plus lengths.
Pass `vb.pp("encoder")` when loading native weights.

## Verification boundary

`cargo test -p candle-transformers --lib transcribepsy`

Small in-memory CPU tests cover shape contracts, mandatory native weights,
subsampling stage indexing/lengths, and padded attention execution. They
do **not** validate numerical equivalence with NeMo, a real checkpoint's
full inventory, CPU/CUDA parity, or correct spoken-text transcription.

Native source of truth:
`qvac/TranscribePsy-250M-EU-sub/model.safetensors` at revision
`ccf70de9ff9848f5cd5975adda1370ce4addf6f1`. Weights stay on HF Hub.

**Not part of TP1:** audio log-Mel DSP, tokenizer, complete ASR wrapper,
beam search or stream caches. TP2 must compare one public French clip
against the HDB2-A reference (Mel, encoder output, TDT decisions, text);
only identical decoding decisions with bounded tensor error can qualify
runtime integration.

Do not delete `half_duplex` or `half_duplex_standalone` before a
separate `haricot/half_duplex` application migration is qualified.
