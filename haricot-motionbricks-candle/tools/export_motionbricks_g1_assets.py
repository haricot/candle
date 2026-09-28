#!/usr/bin/env python3
"""Export the exact released MotionBricks G1Skeleton34 geometry + 418-D stats to JSON.

This intentionally does not copy MotionBricks model weights into this project. It extracts
only the small runtime bridge metadata from a local GR00T-WholeBodyControl checkout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

EXPECTED_PARENTS = [
    -1,
    0, 1, 2, 3, 4, 5, 6,
    0, 8, 9, 10, 11, 12, 13,
    0, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24,
    17, 26, 27, 28, 29, 30, 31, 32,
]


def find_motionbricks_root(p: Path) -> Path:
    p = p.resolve()
    candidates = [p, p / "motionbricks"]
    for c in candidates:
        if (c / "out/motionbricks_pose/version_1/skeleton/joints.p").is_file():
            return c
    raise FileNotFoundError(
        "Could not find motionbricks/out/motionbricks_pose/version_1 under " + str(p)
    )


def tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motionbricks-root", required=True,
                    help="GR00T-WholeBodyControl repo root OR its motionbricks/ directory")
    ap.add_argument("--out", default="assets/motionbricks-g1-v1.json")
    ap.add_argument("--source-rev", default="087f9ac01d46f6d8e4d0b73c01ae64799f292a38")
    args = ap.parse_args()

    root = find_motionbricks_root(Path(args.motionbricks_root))
    base = root / "out/motionbricks_pose/version_1"
    joints_path = base / "skeleton/joints.p"
    parents_path = base / "skeleton/parents.p"
    mean_path = base / "stats/motion/mean.npy"
    std_path = base / "stats/motion/std.npy"

    # These files are official MotionBricks release artifacts from a checkout the user supplied.
    # Explicit weights_only=False is required by modern PyTorch for these historical tensor pickles.
    joints = torch.load(joints_path, map_location="cpu", weights_only=False).squeeze()
    parents = torch.load(parents_path, map_location="cpu", weights_only=False).reshape(-1)
    mean = np.load(mean_path).reshape(-1)
    std = np.load(std_path).reshape(-1)

    if tuple(joints.shape) != (34, 3):
        raise RuntimeError(f"expected joints.p shape (34,3), got {tuple(joints.shape)}")
    parents_list = [int(x) for x in parents.tolist()]
    if parents_list != EXPECTED_PARENTS:
        raise RuntimeError(f"unexpected G1 parents: {parents_list}")
    if mean.size != 418 or std.size != 418:
        raise RuntimeError(f"expected 418-D stats, got mean={mean.size} std={std.size}")

    payload = {
        "source": f"NVlabs/GR00T-WholeBodyControl@{args.source_rev}:motionbricks_pose/version_1",
        "neutral_joints": tensor_to_list(joints),
        "parents": parents_list,
        "mean": np.asarray(mean, dtype=np.float32).tolist(),
        "std": np.asarray(std, dtype=np.float32).tolist(),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("=== MOTIONBRICKS G1 ASSET EXPORT V0.2 ===")
    print(f"root={root}")
    print(f"joints_shape={list(joints.shape)}")
    print(f"parents={len(parents_list)}")
    print(f"stats_dim={mean.size}")
    print(f"out={out}")
    print("MOTIONBRICKS_G1_ASSET_EXPORT=PASS")


if __name__ == "__main__":
    main()
