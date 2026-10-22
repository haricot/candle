# HDB2-C v0.6.1 — Python dependency remediation

Source: user log 2026-10-18. Native Arch Python 3.14 led to pip resolver backtracking NeMo 3.0.0 → 2.5.0, compiling ONNX and mismatching gencode 7.36.1 vs protobuf runtime 5.29.6. `preflight` failed after an incomplete install because NumPy was absent.

Resolution: separate Python 3.13 via `uv`, bounded NeMo `2.7.3`, ONNX `1.18.0`, protobuf `5.29.6` and CPU-only torch `2.12.1+cpu`; wheel-only ONNX; verify imports + `pip check`. This configuration is an experimental compatibility candidate, not proven until installed and loaded locally. No unpinned protobuf override.

Commands: `sudo pacman -S --needed uv` (if absent), `./run-hdb2c.sh install-nemo`, `./run-hdb2c.sh verify-nemo`, `./run-hdb2c.sh c1 "$HOME/tmp/hdb1-direct-micro.wav" --variant cuda-cudnn`. The original Python 3.14 environment is not deleted.
