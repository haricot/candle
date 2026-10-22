# HDB2-C v0.6.2 — Isolated package index remediation

Observed v0.6.1 failure after successful Python 3.13, CPU Torch installation:
`nemo_toolkit[asr]==2.7.3` requires `setuptools>=79`, while the PyTorch CPU
index used by `uv` offers only `setuptools<=78.1.0`. The `uv` first-index
resolution policy masked PyPI. This is **not** a mathematical package conflict.

Fix: keep the existing `.venv-hdb2c-py313`, reuse `torch==2.12.1+cpu`,
install `setuptools>=79,<82` from PyPI, resolve pinned NeMo/ONNX/Protobuf
from PyPI only, and verify Python import plus torch CPU identity. No use of
`--index-strategy unsafe-best-match` and no rebuilding Torch/Candle.

Command: `./run-hdb2c.sh install-nemo && ./run-hdb2c.sh verify-nemo`

Native imports of pinned NeMo remain unqualified on user workstation until
user reports success. C1 and numeric parity remain NOT_RUN.
