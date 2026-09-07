#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    args = ap.parse_args()
    from safetensors import safe_open
    with safe_open(args.weights, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        shapes = {k: list(f.get_tensor(k).shape) for k in keys}
    required = {
        "head.final_layer.weight": [26, 768, 7, 7],
        "head.final_layer.bias": [26],
        "head.mlp.1.weight": [256, 48],
        "head.gau.uv.weight": [1152, 256],
        "head.gau.gamma": [2, 128],
        "head.gau.beta": [2, 128],
        "head.gau.o.weight": [256, 512],
        "head.gau.res_scale.scale": [256],
        "head.cls_x.weight": [384, 256],
        "head.cls_y.weight": [512, 256]
    }
    failures = []
    for k, shape in required.items():
        if shapes.get(k) != shape:
            failures.append({"key": k, "expected": shape, "got": shapes.get(k)})
    fused = [k for k in keys if k.endswith(".fused.weight")]
    if len(fused) != 57:
        failures.append({"fused_conv_count": len(fused), "expected": 57})
    report = {
        "schema": "haricot.pose.rtmpose.body26.weights.audit.v1",
        "status": "PASS" if not failures else "FAIL",
        "tensor_count": len(keys),
        "fused_conv_count": len(fused),
        "expected_fused_conv_count": 57,
        "failures": failures
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not failures else 2)


if __name__ == "__main__":
    main()
