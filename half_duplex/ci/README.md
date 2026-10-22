# Haricot Half-Duplex, standalone branch `half_duplex`

Source snapshot: full HDB2-C2 v0.6.6 under `half_duplex/`. This is a hybrid stack:

- Rust + Candle FastConformer/TDT, QVAC TranscribePsy model converter, NeMo reference implementation and a NeMo-compatible Rust Mel frontend.
- FireRedVAD + Smart Turn v3.2: Python ONNX inference adapters (not pure Rust).
- `install_update.sh` builds an isolated candidate and never overwrites the existing runtime.

## Qualification boundaries

1. CPU source contracts: runs HDB1/HDB2-A/B/B2/C2 unit tests with Python, and shell syntax checks.
2. CPU Rust: clones exact pinned Parakeet upstream and compiles **real** C2 Rust source + `cargo test --lib`. Only a **synthetic configuration metadata file**, never fabricated weights, is used to allow compilation without downloading the 1 GB checkpoint. This job **does not infer speech**.
3. Dataset smoke: downloads the public French **CaFE/Oréau** subsets of [CAMEO](https://huggingface.co/datasets/amu-cai/CAMEO); checks PCM16 16 kHz mono and transcriptions. No microphone recordings uploaded.
4. Real ASR on CPU: heavy opt-in `[real-asr]` commit message or workflow dispatch, downloads **actual public** NeMo model, converts to Candle, performs NeMo/Candle/C2 A/B using CaFE (neutral + sad) and uploads only JSON metrics. This is **not** automatically called qualified; read RMSE/CER/WER.
5. CUDA cuDNN sm_61: separate gated workflow using a **self-hosted NVIDIA runner** with labels `linux`, `x64`, `cuda`, `sm61`; no CUDA-qualified GitHub runner is assumed to exist. Trigger via `[cuda-sm61]` commit message only when the runner has the needed local model/environment.

CaFE is emotional isolated speech, not a conversational turn-taking ground-truth corpus. FireRedVAD and Smart Turn model inference must be tested separately; a synthetic pause is only a VAD state-machine test, **not semantic Smart Turn validation**.

## Personal recordings

The source snapshot includes only `tests/fixtures/hdb0-tone-16k.wav`, a synthetic WAV. It contains **no personal microphone recordings**. Explicitly choose any additional recordings and their publication visibility before pushing them to a **public** repository. Use CaFE for automated evaluations.

No weights, NeMo checkpoint, personal audio or raw waveforms are published as CI artifacts.
