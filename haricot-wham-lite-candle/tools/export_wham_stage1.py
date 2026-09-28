#!/usr/bin/env python3
"""Export only the WHAM Stage-1 recurrent core to safetensors.

Usage:
  python tools/export_wham_stage1.py --checkpoint checkpoint.pth --out haricot-wham-stage1.safetensors

The script intentionally excludes image-feature integration, SMPL mesh assets and trajectory
refinement. It preserves PyTorch LSTM key names so Candle's native candle-nn LSTM can load them.
"""

import argparse
from collections import OrderedDict

import torch
from safetensors.torch import save_file

PREFIXES = (
    "motion_encoder.",
    "trajectory_decoder.",
    "motion_decoder.",
)


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("gen_state_dict", "state_dict", "model", "network"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(obj, dict):
        return obj
    raise TypeError("checkpoint does not contain a recognizable state_dict")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = unwrap_state_dict(raw)
    out = OrderedDict()

    for name, tensor in state.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        if not name.startswith(PREFIXES):
            continue
        # Stage-1 lite deliberately removes the image feature integrator and refiner.
        if name.startswith("trajectory_refiner.") or name.startswith("integrator."):
            continue
        out[name] = tensor.detach().contiguous().cpu()

    if not out:
        raise RuntimeError("no WHAM Stage-1 tensors found in checkpoint")

    save_file(out, args.out)
    print(f"exported_tensors={len(out)}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
