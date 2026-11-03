#!/usr/bin/env python3
"""Compare native/reference results. Exact decisions; abs+rel activation bounds.

Numerical tolerances apply to CPU F32 qualification, not a CUDA parity claim.
Bitwise differences remain a diagnostic. Only public metrics belong in git.
"""
import argparse
import json
from pathlib import Path

import numpy as np

# Mel is a log spectrum: low-energy bins amplify FFT/window rounding. The
# absolute floor covers values near zero; the relative term scales with |log Mel|.
# Encoder bounds cover differing F32 matmul/normalization reduction orders.
# These bounds are deliberately checked per element, not only by cosine similarity.
BOUNDS = {"mel": (1e-4, 1e-4), "encoder": (1e-4, 1e-4)}


def compare(reference, candidate):
    a = json.loads((reference / "result.json").read_text())
    b = json.loads((candidate / "result.json").read_text())
    exact_keys = ("mode", "language", "lookahead", "samples", "sample_rate", "mel_frames",
                  "encoder_frames", "text", "tokens", "decisions")
    result = {"exact": {k: a[k] == b[k] for k in exact_keys}, "stages": {}}
    for name, (atol, rtol) in BOUNDS.items():
        am = a["activations"][name]
        bm = b["activations"][name]
        if am["shape"] != bm["shape"]:
            result["stages"][name] = dict(pass_=False, reference_shape=am["shape"],
                                         candidate_shape=bm["shape"])
            continue
        x = np.fromfile(reference / f"{name}.f32le", dtype="<f4").astype(np.float64)
        y = np.fromfile(candidate / f"{name}.f32le", dtype="<f4").astype(np.float64)
        if x.size != np.prod(am["shape"]) or x.shape != y.shape:
            raise ValueError(f"truncated or oversized {name} activation file")
        finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
        diff = np.abs(x - y)
        bound = atol + rtol * np.abs(x)
        norms = np.linalg.norm(x) * np.linalg.norm(y)
        result["stages"][name] = dict(
            pass_=finite and bool(np.all(diff <= bound)), finite=finite, shape=am["shape"],
            atol=atol, rtol=rtol, max_abs=float(diff.max()),
            rmse=float(np.sqrt(np.mean(diff ** 2))),
            cosine=float(np.dot(x, y) / norms) if norms else None,
            max_error_over_bound=float((diff / bound).max()),
            bitwise_equal_elements=int(np.count_nonzero(x == y)), elements=int(x.size),
        )
    if not result["exact"]["decisions"]:
        for i in range(max(len(a["decisions"]), len(b["decisions"]))):
            av = a["decisions"][i] if i < len(a["decisions"]) else None
            bv = b["decisions"][i] if i < len(b["decisions"]) else None
            if av != bv:
                result["first_divergent_decision"] = dict(index=i, reference=av, candidate=bv)
                break
    result["pass"] = all(result["exact"].values()) and all(v["pass_"] for v in result["stages"].values())
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = compare(args.reference, args.candidate)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
