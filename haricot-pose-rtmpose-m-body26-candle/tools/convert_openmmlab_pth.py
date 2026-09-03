#!/usr/bin/env python3
"""Convert official OpenMMLab RTMPose-m Halpe26 PTH to Candle-ready Safetensors.

This is an offline conversion tool only. Runtime inference is pure Rust/Candle.
ConvModule Conv+SyncBN pairs are folded into a single convolution weight+bias.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

OFFICIAL_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state_dict(path: Path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for candidate in ("state_dict", "model", "model_state_dict"):
            value = checkpoint.get(candidate)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"unsupported checkpoint container: {type(checkpoint)!r}")

    out = {}
    for key, value in checkpoint.items():
        if not hasattr(value, "detach"):
            continue
        if key.startswith("module."):
            key = key[len("module."):]
        out[key] = value.detach().cpu()
    return out


def fused_module_paths() -> list[str]:
    paths = ["backbone.stem.0", "backbone.stem.1", "backbone.stem.2"]
    blocks_per_stage = [2, 4, 4, 2]
    for stage, blocks in enumerate(blocks_per_stage, start=1):
        paths.append(f"backbone.stage{stage}.0")
        csp_index = 2 if stage == 4 else 1
        if stage == 4:
            paths.extend(["backbone.stage4.1.conv1", "backbone.stage4.1.conv2"])
        csp = f"backbone.stage{stage}.{csp_index}"
        paths.extend([f"{csp}.main_conv", f"{csp}.short_conv", f"{csp}.final_conv"])
        for i in range(blocks):
            b = f"{csp}.blocks.{i}"
            paths.extend([
                f"{b}.conv1",
                f"{b}.conv2.depthwise_conv",
                f"{b}.conv2.pointwise_conv",
            ])
    return paths


def fuse_conv_bn(sd, base: str, eps: float):
    import torch

    cw = sd[f"{base}.conv.weight"].float()
    cb = sd.get(f"{base}.conv.bias")
    if cb is None:
        cb = torch.zeros(cw.shape[0], dtype=torch.float32)
    else:
        cb = cb.float()
    gamma = sd[f"{base}.bn.weight"].float()
    beta = sd[f"{base}.bn.bias"].float()
    mean = sd[f"{base}.bn.running_mean"].float()
    var = sd[f"{base}.bn.running_var"].float()
    inv = gamma / torch.sqrt(var + eps)
    fw = cw * inv.reshape(-1, 1, 1, 1)
    fb = beta + (cb - mean) * inv
    return fw.contiguous(), fb.contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", type=Path, help="official .pth checkpoint")
    ap.add_argument("--download-official", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("rtmpose-m-halpe26-candle-f32.safetensors"))
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--bn-eps", type=float, default=1.0e-5)
    args = ap.parse_args()

    src = args.input
    if args.download_official:
        src = src or Path("rtmpose-m-halpe26-official.pth")
        if not src.exists():
            print(f"download: {OFFICIAL_URL}")
            urllib.request.urlretrieve(OFFICIAL_URL, src)
    if src is None:
        ap.error("provide INPUT.pth or --download-official")
    if not src.exists():
        raise FileNotFoundError(src)

    sd = load_state_dict(src)
    missing = []
    for base in fused_module_paths():
        for suffix in ("conv.weight", "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var"):
            key = f"{base}.{suffix}"
            if key not in sd:
                missing.append(key)

    required_head = [
        "head.final_layer.weight", "head.final_layer.bias",
        "head.mlp.0.g", "head.mlp.1.weight",
        "head.gau.ln.g", "head.gau.uv.weight", "head.gau.gamma", "head.gau.beta",
        "head.gau.o.weight", "head.gau.res_scale.scale",
        "head.cls_x.weight", "head.cls_y.weight",
    ]
    for key in required_head:
        if key not in sd:
            missing.append(key)
    for stage in range(1, 5):
        csp_index = 2 if stage == 4 else 1
        for suffix in ("weight", "bias"):
            key = f"backbone.stage{stage}.{csp_index}.attention.fc.{suffix}"
            if key not in sd:
                missing.append(key)
    if missing:
        print("checkpoint keys sample:")
        for key in sorted(sd)[:80]:
            print(" ", key)
        raise KeyError("missing required RTMPose-m keys:\n" + "\n".join(missing))

    exported = {}
    for base in fused_module_paths():
        weight, bias = fuse_conv_bn(sd, base, args.bn_eps)
        exported[f"{base}.fused.weight"] = weight
        exported[f"{base}.fused.bias"] = bias

    for stage in range(1, 5):
        csp_index = 2 if stage == 4 else 1
        prefix = f"backbone.stage{stage}.{csp_index}.attention.fc"
        exported[f"{prefix}.weight"] = sd[f"{prefix}.weight"].float().contiguous()
        exported[f"{prefix}.bias"] = sd[f"{prefix}.bias"].float().contiguous()

    for key in required_head:
        exported[key] = sd[key].float().contiguous()

    from safetensors.torch import save_file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(exported, str(args.output), metadata={
        "architecture": "RTMPose-m CSPNeXt-P5 + RTMCCHead",
        "dataset": "Halpe26",
        "input": "RGB NCHW 1x3x256x192",
        "simcc": "x=384,y=512,split_ratio=2.0",
        "source": OFFICIAL_URL,
        "conv_bn_fused": "true",
        "bn_eps": repr(args.bn_eps),
    })

    meta = args.meta or args.output.with_suffix(".json")
    info = {
        "schema": "haricot.rtmpose.body26.weights.v1",
        "source_url": OFFICIAL_URL,
        "source_file": str(src),
        "source_sha256": sha256(src),
        "output_file": str(args.output),
        "output_sha256": sha256(args.output),
        "tensor_count": len(exported),
        "bn_eps": args.bn_eps,
        "conv_bn_fused": True,
        "input_width": 192,
        "input_height": 256,
        "keypoints": 26,
        "simcc_x": 384,
        "simcc_y": 512,
    }
    meta.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
