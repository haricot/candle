#!/usr/bin/env python3
"""No optimistic parity: strict numerical gates and exact discrete decisions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ATOL = 1e-3
MIN_COSINE = 0.999


def numbers(a_dir, b_dir, part, ref, cand):
    ashape, bshape = ref[part]["shape"], cand[part]["shape"]
    a = np.fromfile(a_dir / (part + ".f32le"), dtype="<f4")
    b = np.fromfile(b_dir / (part + ".f32le"), dtype="<f4")
    if a.size != int(np.prod(ashape)) or b.size != int(np.prod(bshape)):
        raise ValueError("Malformed TP3 tensor buffer")
    a, b = a.reshape(ashape), b.reshape(bshape)
    va, vb = ref[part]["valid"], cand[part]["valid"]
    consistent = (len(ashape) == len(bshape) == 3 and
                  ashape[0] == bshape[0] == 1 and ashape[2] == bshape[2]
                  and va == vb and va > 0 and
                  va <= ashape[1] and vb <= bshape[1])
    summary = {"valid_frames_equal": va == vb, "shape_reference": ashape,
               "shape_candidate": bshape, "atol": ATOL,
               "min_cosine": MIN_COSINE, "valid_shapes_compatible": consistent}
    if not consistent:
        return dict(summary, numeric_pass=False, reason="SHAPE_OR_VALID_LENGTH_MISMATCH")
    a, b = a[:, :va].astype(np.float64), b[:, :vb].astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(summary, numeric_pass=False, reason="NONFINITE")
    diff = a - b
    max_abs = float(np.abs(diff).max())
    rmse = float(np.sqrt(np.mean(diff * diff)))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.sum(a * b) / norm) if norm else (1.0 if not np.any(diff) else 0.0)
    summary.update(max_abs=max_abs, rmse=rmse, cosine=cosine,
                   numeric_pass=max_abs <= ATOL and cosine >= MIN_COSINE,
                   bitwise_equal=bool(np.array_equal(
                       a.astype("<f4").view("<u4"), b.astype("<f4").view("<u4"))))
    return summary


def decisions(reference, candidate):
    blank = reference["blank"]
    actual = [dict(row, token=blank if row["token"] is None else row["token"])
              for row in candidate]
    keys = ("frame", "token", "duration_index", "duration", "advance", "forced_advance")
    mismatch = None
    for i, (r, c) in enumerate(zip(reference["decisions"], actual)):
        diff = {k: {"nemo": r[k], "candle": c[k]}
                for k in keys if r[k] != c[k]}
        if diff:
            mismatch = {"decision": i, "differences": diff}
            break
    if mismatch is None and len(reference["decisions"]) != len(actual):
        mismatch = {"decision": min(len(reference["decisions"]), len(actual)),
                    "reason": "DECISION_COUNT_MISMATCH"}
    return {"exact": mismatch is None,
            "reference_count": len(reference["decisions"]),
            "candidate_count": len(actual), "first_difference": mismatch}


def run(args):
    from sentencepiece import SentencePieceProcessor
    a = json.loads((args.reference / "trace.json").read_text())
    b = json.loads((args.candidate / "trace.json").read_text())
    fixture = json.loads((args.fixture / "fixture.json").read_text())
    if a["schema"] != "TP3-NEMO-QVAC-F32-V1" or b["schema"] != "TP3-CANDLE-QVAC-F32-V1":
        raise ValueError("TP3 incompatible trace schema")
    sp = SentencePieceProcessor(model_file=str(args.reference / "tokenizer.model"))
    proto = (args.reference / "tokenizer.model").read_bytes()
    if hashlib.sha256(proto).hexdigest() != a["tokenizer_sha256"]:
        raise ValueError("Tokenizer bytes changed after NeMo reference")
    decoded = sp.decode([int(t) for t in b["tokens"]]).strip()
    mel = numbers(args.reference, args.candidate, "mel", a, b)
    encoder = numbers(args.reference, args.candidate, "encoder", a, b)
    decision = decisions(a, b["decisions"])
    token_exact = a["tokens"] == b["tokens"]
    text_exact = decoded == a["text"]
    weights_ok = a["identical_float_tensors"] >= 900
    passing = (weights_ok and mel["numeric_pass"] and encoder["numeric_pass"]
               and decision["exact"] and token_exact and text_exact)
    result = {"schema": "TP3-ONE-CLIP-PARITY-V1",
              "status": "PASS_REAL_MODEL_PARITY" if passing else "REJECT_OR_NOT_QUALIFIED",
              "model": "qvac/TranscribePsy-250M-EU-sub",
              "safetensors_sha256": a["weights_sha256"],
              "nemo_sha256": a["nemo_sha256"],
              "identical_native_weight_tensors": a["identical_float_tensors"],
              "tokenizer_sha256": a["tokenizer_sha256"],
              "fixture": fixture,
              "mel": mel, "encoder": encoder,
              "tdt": decision, "tokens_exact": token_exact,
              "text_exact": text_exact,
              "text_nemo": a["text"], "text_candle": decoded,
              "scope": "ONE_PUBLIC_FRENCH_UTTERANCE_CPU_F32_NO_STREAMING_QUALIFICATION"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n")
    print("TP3_GATE=" + result["status"])
    print("TP3_MEL=" + json.dumps(mel))
    print("TP3_ENCODER=" + json.dumps(encoder))
    print("TP3_TDT=" + json.dumps(decision))
    print("TP3_TEXT_EXACT=" + str(text_exact))
    return passing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if not run(a):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
