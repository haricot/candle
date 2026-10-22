#!/usr/bin/env bash
# HDB2-C NeMo ORIGINAL vs Candle Port, same WAV + optional numerical tensors.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
usage() {
cat <<'TXT'
HDB2-C v0.6.6: Numerical Parity & ASR Fidelity
  ./run-hdb2c.sh selftest
  ./run-hdb2c.sh preflight [--wav /path/to/audio.wav]
  ./run-hdb2c.sh install-nemo   # isolated Python 3.13; uv required
  ./run-hdb2c.sh verify-nemo    # torch + ONNX + NeMo imports
  ./run-hdb2c.sh c1 /path/to/direct-micro.wav [--variant cuda-cudnn] [--probes] [--reference-file /path/to/verified.txt]
  ./run-hdb2c.sh probe /path/to/HDB2C_REPORT_DIR [--variant cpu] [--skip-build]
  ./run-hdb2c.sh na-ab /path/to/C1_PROBES_DIR [--variant cpu|cuda-cudnn] [--skip-build]
  ./run-hdb2c.sh c2 [--report /path/to/C1_PROBES_DIR] [--prepare-only] [--cuda-cudnn]

NeMo PyTorch must be isolated from existing HDB1 ONNX Python.
Install uses isolated Python 3.13 (do not reuse broken Python 3.14); uv required.
Existing CUDA+cuDNN Candle model and binaries are never altered.
A new Rust hdb2c_probe *additional Cargo bin* is built separately only on demand.
Env: HDB2C_NEMO, HDB2C_NEMO_PYTHON, HDB2A_RUNTIME_DIR,
     HDB2A_MODEL_DIR, HDB2A_REPORT_ROOT.
TXT
}
case "${1:-help}" in
  selftest)
    python3 -m unittest discover -s tests/hdb2c -v
    ;;
  install-nemo)
    # Never reuse the Python 3.14 environment that triggered the ONNX/Protobuf
    # source-build mismatch. The existing environment remains untouched.
    if ! command -v uv >/dev/null; then
      echo 'HDB2C=UV_REQUIRED: install uv (Arch: sudo pacman -S uv), then retry.' >&2
      exit 3
    fi
    # Find the venv from the previous v0.6.1 install even when this ZIP is
    # extracted into a NEW versioned directory. Do not download torch again.
    ENV_DIR="${HDB2C_ENV_DIR:-}"
    if [[ -z "$ENV_DIR" ]]; then
      for candidate in \
        "$ROOT/.venv-hdb2c-py313" \
        "$HOME/tmp/haricot-half-duplex/.venv-hdb2c-py313" \
        "$(dirname "$ROOT")/haricot-half-duplex-hdb2c-v0.6.1/.venv-hdb2c-py313"; do
        if [[ -x "$candidate/bin/python" ]]; then ENV_DIR="$candidate"; break; fi
      done
    fi
    [[ -n "$ENV_DIR" ]] || ENV_DIR="$ROOT/.venv-hdb2c-py313"
    echo "HDB2C_NEMO_ENV_DIR=$ENV_DIR"
    # Reuse the existing virtual environment and torch wheel.  uv venv can
    # otherwise overwrite a successful partial install after a failed resolve.
    if [[ -x "$ENV_DIR/bin/python" ]]; then
      echo 'HDB2C_NEMO_ENV=REUSE_EXISTING_PY313'
    else
      uv python install 3.13
      uv venv --python 3.13 --seed "$ENV_DIR"
      echo 'HDB2C_NEMO_ENV=CREATED_PY313'
    fi
    PY="$ENV_DIR/bin/python"
    "$PY" -c 'import sys; assert sys.version_info[:2] == (3,13), sys.version'
    # Only torch is installed from the PyTorch CPU index.  Using it as an
    # extra index when resolving NeMo masks new setuptools releases on PyPI.
    if "$PY" -c 'import torch; assert torch.__version__ == "2.12.1+cpu" and torch.version.cuda is None' >/dev/null 2>&1; then
      echo 'HDB2C_TORCH_CPU=REUSE_INSTALLED_2_12_1'
    else
      uv pip install --no-config --python "$PY" 'torch==2.12.1+cpu' \
        --default-index https://download.pytorch.org/whl/cpu
    fi
    # NeMo >=2.7.3 requires setuptools>=79 from PyPI.  This is compatible
    # with torch's setuptools<82 bound; do not search on the torch CPU index.
    uv pip install --no-config --python "$PY" \
      'setuptools>=79,<82' --default-index https://pypi.org/simple
    printf '%s\n' \
       'torch==2.12.1+cpu' \
       'nemo_toolkit==2.7.3' \
       'onnx==1.18.0' \
       'protobuf==5.29.6' \
       'setuptools>=79,<82' > "$ENV_DIR/hdb2c-constraints.txt"
    # uv favours already-installed torch+cpu. Pin its version in constraints
    # and use ONE index (PyPI) to avoid exposing unrelated deps to torch index.
    uv pip install --no-config --python "$PY" --only-binary onnx \
      --default-index https://pypi.org/simple \
      -c "$ENV_DIR/hdb2c-constraints.txt" \
      'nemo_toolkit[asr]==2.7.3' 'onnx==1.18.0' \
      'protobuf==5.29.6' 'numpy<3' 'soundfile>=0.12' 'safetensors>=0.4'
    "$PY" -c 'import torch; assert torch.__version__ == "2.12.1+cpu" and torch.version.cuda is None, (torch.__version__, torch.version.cuda); print("HDB2C_TORCH_CPU=PASS")'
    "$PY" -m pip check
    "$PY" -c 'import sys,torch,onnx,google.protobuf,nemo.collections.asr; assert sys.version_info[:2]==(3,13); assert torch.version.cuda is None; print("HDB2C_NEMO_IMPORT=PASS",sys.version.split()[0],torch.__version__,onnx.__version__,google.protobuf.__version__)'
    ;;
  verify-nemo)
    PY="${HDB2C_NEMO_PYTHON:-}"
    if [[ -z "$PY" ]]; then
      for candidate in \
        "$ROOT/.venv-hdb2c-py313/bin/python" \
        "$HOME/tmp/haricot-half-duplex/.venv-hdb2c-py313/bin/python" \
        "$(dirname "$ROOT")/haricot-half-duplex-hdb2c-v0.6.1/.venv-hdb2c-py313/bin/python"; do
        if [[ -x "$candidate" ]]; then PY="$candidate"; break; fi
      done
    fi
    [[ -n "$PY" && -x "$PY" ]] || { echo 'HDB2C=PY313_NOT_INSTALLED; run install-nemo' >&2; exit 3; }
    "$PY" -c 'import sys,torch,onnx,google.protobuf,nemo.collections.asr; assert sys.version_info[:2]==(3,13); assert torch.version.cuda is None; print("HDB2C_NEMO_IMPORT=PASS",sys.version.split()[0],torch.__version__,onnx.__version__,google.protobuf.__version__)'
    "$PY" -m pip check
    ;;
  preflight|c1|probe)
    ARGS=("$@")
    PY="${HDB2C_PYTHON:-$ROOT/.venv-hdb2c-py313/bin/python}"
    [[ -x "$PY" ]] || PY=python3
    "$PY" -m hdb2c.cli "${ARGS[@]}"
    ;;
  na-ab)
    shift
    PY="${HDB2C_PYTHON:-$ROOT/.venv-hdb2c-py313/bin/python}"
    [[ -x "$PY" ]] || PY=python3
    "$PY" -m hdb2c.na_ab "$@"
    ;;
  c2)
    shift
    exec "$ROOT/install_update.sh" "$@"
    ;;
  help|--help|-h)usage;;
  *)usage;exit 2;;
esac
