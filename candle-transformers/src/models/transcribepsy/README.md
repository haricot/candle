# TranscribePsy — standalone model branch (TP0)

This branch is based directly on `haricot/candle:main`. It is intentionally
small enough to review upstream and contains no HDB2-A experiment harness,
NeMo conversion scripts, original checkpoint, personal audio, transcript,
recordings, or benchmark reports.

## Implemented (TP0)

- Rust model components in `candle-transformers/src/models/transcribepsy/`:
  - strict QVAC native LSTM predictor weights, including both LSTM biases;
  - joint encoder/predictor projection and joint output head;
  - token/duration greedy decision policy with explicit blank and frame logic.
- Config deserialization compatible with HDB2-A's normalized local
  `config.json` sections (`decoder`, `joint`, `decoding`).
- Tiny CPU tests using in-memory synthetic tensors; no Hugging Face access.
- No runtime, API, path, or machine-specific dependency on HDB2-A.

The original `qvac/TranscribePsy-250M-EU-sub/model.safetensors` remains the
source of truth for native tensor names. No Haricot weight repository is needed.

## Not implemented and not claimed

- NeMo log-Mel frontend and all FastConformer encoder blocks;
- a complete real-model loader, PCM/WAV microphone pipeline or tokenizer;
- real-weights inference, TDT textual decoding or CPU/CUDA numerical parity;
- migration of `voice_endpointing` or `half_duplex_standalone`.

A passing TP0 unit test means the Rust building blocks work on synthetic data,
**not** that TranscribePsy ASR is qualified.

## Next clean steps

1. Port FastConformer as a separate reusable module, checking the exact
   NeMo `layer_norm` / positional-mask semantics and using official QVAC keys.
2. Add a model wrapper connecting log-Mel, encoder, predictor, joint and TDT.
3. Run one *public* French WAV against the existing HDB2-A reference:
   Mel, last encoder tensor, TDT decisions and transcript; tensor tolerance,
   identical decoding decisions. Keep full 14-clip campaigns outside Candle.
4. Once qualified, promote the model to `extended`, preserving independent
   model tests. The separate `haricot/half_duplex` application will assemble
   TranscribePsy, FireRedVAD and Smart Turn.

Local command (model weights never downloaded by this test):

```bash
cargo test -p candle-transformers transcribepsy:: --lib
```

Do not delete `half_duplex` or `half_duplex_standalone` before complete
runtime parity and an application migration have been validated.
