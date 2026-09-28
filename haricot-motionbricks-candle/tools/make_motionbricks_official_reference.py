#!/usr/bin/env python3
"""Run the pinned official MotionBricks feature implementation on a Haricot G1 sequence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PINNED_REV = "087f9ac01d46f6d8e4d0b73c01ae64799f292a38"
EXPECTED_SCHEMA = "haricot.motionbricks.g1-parity-input.v1"


def find_motionbricks_root(p: Path) -> Path:
    p = p.resolve()
    for c in (p, p / "motionbricks"):
        if (c / "motionbricks/motionlib/core/motion_reps/dual_root_global_joints.py").is_file() and (
            c / "out/motionbricks_pose/version_1/skeleton/joints.p"
        ).is_file():
            return c
    raise FileNotFoundError(f"Cannot locate MotionBricks package/release under {p}")


def verify_git_rev(repo_arg: Path, requested_rev: str) -> str:
    import subprocess

    candidates = [repo_arg.resolve(), repo_arg.resolve().parent]
    for c in candidates:
        if (c / ".git").exists():
            got = subprocess.check_output(
                ["git", "-C", str(c), "rev-parse", "HEAD"], text=True
            ).strip()
            if got != requested_rev:
                raise RuntimeError(
                    f"MotionBricks checkout mismatch: HEAD={got}, required={requested_rev}"
                )
            return got
    raise RuntimeError("Could not find .git to verify the pinned MotionBricks revision")


def as_list(x: torch.Tensor) -> list:
    return x.detach().cpu().to(torch.float32).numpy().tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motionbricks-root", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-rev", default=PINNED_REV)
    args = ap.parse_args()

    if args.source_rev != PINNED_REV:
        raise RuntimeError(f"v0.2.1 is pinned to {PINNED_REV}")

    repo_arg = Path(args.motionbricks_root)
    root = find_motionbricks_root(repo_arg)
    verified_rev = verify_git_rev(repo_arg, args.source_rev)
    sys.path.insert(0, str(root))

    # These are the actual released MotionBricks classes/functions from the pinned checkout.
    from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import (
        DualRootGlobalJoints,
    )
    from motionbricks.motionlib.core.skeletons import G1Skeleton34
    from motionbricks.motionlib.core.utils.stats import Stats

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if raw.get("schema") != EXPECTED_SCHEMA:
        raise RuntimeError(f"unexpected input schema: {raw.get('schema')}")
    fps = float(raw["fps"])
    frames = raw["frames"]
    if len(frames) < 2:
        raise RuntimeError("official parity needs at least two frames")

    positions = torch.tensor(
        [f["positions"] for f in frames], dtype=torch.float32
    ).unsqueeze(0)
    global_rots = torch.tensor(
        [f["global_rotations"] for f in frames], dtype=torch.float32
    ).unsqueeze(0)

    base = root / "out/motionbricks_pose/version_1"
    skeleton = G1Skeleton34(
        folder=str(base / "skeleton"),
        name="g1skel34",
        t_pose="capture",
    )
    stats = Stats(folder=str(base / "stats/motion"))
    motion_rep = DualRootGlobalJoints(
        fps=fps,
        skeleton=skeleton,
        name="g1skel34_dual_root_global_joints",
        stats=stats,
    )

    # MotionBricks requires an explicit lengths tensor for batched inputs,
    # even when B=1 and every sequence has the same full length.
    lengths = torch.tensor([len(frames)], dtype=torch.long, device=positions.device)

    # Omit foot_contacts intentionally: this exercises MotionBricks' official
    # foot_detect_from_pos_and_vel path, matching the Rust v0.2 bridge.
    with torch.no_grad():
        dual = motion_rep(
            {
                "posed_joints": positions,
                "global_joint_rots": global_rots,
            },
            to_normalize=False,
            lengths=lengths,
        )
        normalized_dual = motion_rep.normalize(dual)
        global_rep = motion_rep.get_feature_subset(dual, mode="global")
        local_rep = motion_rep.get_feature_subset(dual, mode="local")
        normalized_global = motion_rep.get_feature_subset(normalized_dual, mode="global")
        normalized_local = motion_rep.get_feature_subset(normalized_dual, mode="local")

    if tuple(dual.shape) != (1, len(frames), 418):
        raise RuntimeError(f"unexpected official dual shape {tuple(dual.shape)}")
    if tuple(global_rep.shape) != (1, len(frames), 414):
        raise RuntimeError(f"unexpected official global shape {tuple(global_rep.shape)}")
    if tuple(local_rep.shape) != (1, len(frames), 413):
        raise RuntimeError(f"unexpected official local shape {tuple(local_rep.shape)}")

    payload = {
        "schema": "haricot.motionbricks.official-feature-reference.v1",
        "source": f"NVlabs/GR00T-WholeBodyControl@{verified_rev}:official MotionBricks DualRootGlobalJoints",
        "source_rev": verified_rev,
        "fps": fps,
        "frames": len(frames),
        "dual": as_list(dual[0]),
        "global": as_list(global_rep[0]),
        "local": as_list(local_rep[0]),
        "normalized_dual": as_list(normalized_dual[0]),
        "normalized_global": as_list(normalized_global[0]),
        "normalized_local": as_list(normalized_local[0]),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("=== OFFICIAL MOTIONBRICKS FEATURE REFERENCE V0.2.1-R1 ===")
    print(f"script_path={Path(__file__).resolve()}")
    print(f"motionbricks_root={root}")
    print(f"source_rev={verified_rev}")
    print(f"frames={len(frames)}")
    print(f"fps={fps}")
    print(f"dual_shape={list(dual.shape)}")
    print(f"global_shape={list(global_rep.shape)}")
    print(f"local_shape={list(local_rep.shape)}")
    print(f"out={out}")
    print("OFFICIAL_MOTIONBRICKS_REFERENCE=PASS")


if __name__ == "__main__":
    main()
