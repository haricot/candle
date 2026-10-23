# TranscribePsy — standalone model branch (TP2)

This intentionally small Candle model module is based on `haricot/candle:main`.
No user recordings, neural weights, checkpoints, HDB2 conversion scripts,
per-machine optimizations, or experiment results are bundled in this branch.

## Native QVAC model

- `transcribepsy::frontend::LogMel`: deterministic mono PCM to log-Mel,
  loading the original `preprocessor.featurizer.window` and `fb` tensors
  from QVAC Safetensors; no implicit resampling. The radix-2 CPU FFT is an
  CPU implementation qualified within TP3 tolerances on one pinned utterance.
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

**TP3-R1 locally validated:** official unmodified QVAC checkpoint loading;
913 F32 tensors identical to NeMo; log-Mel and final FastConformer within the
original TP3 tolerances; all 16 real TDT decisions, token IDs and original
SentencePiece text exact on one licensed public French clip. Independent
NeMo-Mel-to-Candle-encoder and NeMo-encoder-to-Candle-TDT handoffs also pass.
See `tools/transcribepsy_tp3/README.md` for the measurements and workflow.

**Not qualified:** CUDA, streaming caches, other utterances or general ASR
quality. Bitwise tensor equality is not claimed. Keep full HDB2-A campaigns in
`haricot/half_duplex`; none of that app or its generators is added to this model.

QVAC checkpoint source:
`qvac/TranscribePsy-250M-EU-sub/model.safetensors`,
revision `ccf70de9ff9848f5cd5975adda1370ce4addf6f1`.
Weights remain on Hugging Face Hub.

Do not delete historical Candle half-duplex branches before completing the
separate `haricot/half_duplex` app migration.
