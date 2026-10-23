# TranscribePsy — standalone model branch (TP2)

This intentionally small Candle model module is based on `haricot/candle:main`.
No user recordings, neural weights, checkpoints, HDB2 conversion scripts,
per-machine optimizations, or experiment results are bundled in this branch.

## Native QVAC model

- `transcribepsy::frontend::LogMel`: deterministic mono PCM to log-Mel,
  loading the original `preprocessor.featurizer.window` and `fb` tensors
  from QVAC Safetensors; no implicit resampling. The radix-2 CPU FFT is an
  unqualified baseline, not yet numerically identical to NeMo.
- `transcribepsy::encoder::FastConformer`: typed native encoder including
  depthwise-striding subsampling, relative-position attention, Macaron FFN,
  GLU/depthwise convolution and masked attention.
- `transcribepsy::{Predictor, Joint}`: native QVAC LSTM and joint weights.
- `transcribepsy::tdt`: bounded greedy duration/token decisions, including
  the blank-duration-zero frame advance required to prevent a decoding loop.
- `transcribepsy::pipeline::{ModelConfig, Model, Transducer, DecodeLimits}`:
  native `preprocessor.*`, `encoder.*`, `decoder.*`, `joint.*`
  construction. `Model::transcribe_mono` provides token IDs and a TDT
  decision trace. The caller decodes IDs with the **exact original QVAC
  tokenizer** via `DecodeTrace::decode_text`, keeping app dependencies out of
  `candle-transformers`.

Public narrow APIs support independent Mel, encoder and transducer parity
without incorporating HDB2-A research code into Candle.

## Verification

```sh
cargo test -p candle-transformers --lib transcribepsy
```

GitHub Actions runs only in-memory CPU tests: config rejection, layout,
mandatory native tensor names, mask/length shape semantics, FFT sanity,
TDT blank/duration behavior and a synthetic assembled decoder. CI requires
no model download, network credentials or private WAV files.

**Not yet validated:** full native QVAC checkpoint loading in the assembled
model, FFT/log-Mel numerical comparison against NeMo, full FastConformer
numerical parity, identical real TDT/token decisions, official tokenizer,
final French transcription, CUDA parity or streaming caches.

The next gate is *one licensed public French clip* and the unchanged official
Safetensors file: compare Mel and final encoder tensors within declared numeric
tolerances, and require exact TDT decisions and decoded text before qualifying
the integration. Keep full HDB2-A campaigns in `haricot/half_duplex`.

QVAC checkpoint source:
`qvac/TranscribePsy-250M-EU-sub/model.safetensors`,
revision `ccf70de9ff9848f5cd5975adda1370ce4addf6f1`.
Weights remain on Hugging Face Hub.

Do not delete historical Candle half-duplex branches before completing the
separate `haricot/half_duplex` app migration.
