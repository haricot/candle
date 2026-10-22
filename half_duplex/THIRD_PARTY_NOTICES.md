# Third-party references and model assets

HDB1 includes no model weights. The install command retrieves publicly hosted
model files separately on the user's local machine. Review the models' terms
before deployment beyond local experimentation.

- FireRedVAD: https://github.com/FireRedTeam/FireRedVAD — Apache License 2.0.
  The streaming VAD endpoint transition design and CMVN conventions are from
  FireRedVAD's published code and accompanying ONNX adaptation.
- fireredvad-onnx: https://github.com/eschmidbauer/fireredvad-onnx — MIT License.
  The ONNX `feat`/`caches_in` contract, cache layout and Kaldi FBank processing
  informed this independently authored Python adapter. The original source
  and model files are not bundled in the ZIP.
- Smart Turn v3.2: https://github.com/pipecat-ai/smart-turn and
  https://huggingface.co/pipecat-ai/smart-turn-v3 — BSD 2-Clause License.
  The input preparation uses the published `WhisperFeatureExtractor` contract.
- Candle: https://github.com/huggingface/candle (Rust inference), with user's
  fork https://github.com/haricot/candle kept pinned by the HDB2-A/B loaders.
- TranscribePsy: https://huggingface.co/qvac/TranscribePsy-250M-EU-sub
  — user-downloaded checkpoint; not redistributed by this ZIP.

There are no network APIs in the runtime inference pipeline. `download-models`
contacts GitHub and Hugging Face only when explicitly invoked.

## HDB2-C numerical parity tools (2026-10-18)

This ZIP includes only new original HDB2-C Python orchestration, tests,
and an additional original Rust Cargo bin. It does **not** redistribute
NeMo Toolkit, PyTorch, pretrained QVAC model files or any user recordings.
NeMo Speech is obtained from NVIDIA's upstream public Python package;
PyTorch CPU from the public official CPU wheel repository. Follow upstream
license and dependency notices after installation.

- NVIDIA NeMo Speech: https://github.com/NVIDIA-NeMo/Speech
- PyTorch: https://github.com/pytorch/pytorch
- Original NeMo checkpoint referenced (not distributed): https://huggingface.co/qvac/TranscribePsy-250M-EU-sub

Neither original model weights nor runtime binaries are modified by C1.
The optional `hdb2c_probe` adds a separate local Rust Cargo binary rather than
patching the production Candle transcription executable.
