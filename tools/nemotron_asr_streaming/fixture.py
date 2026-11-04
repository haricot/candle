#!/usr/bin/env python3
"""Prepare one public French CaFE/CAMEO qualification clip. Never commit audio.

Dataset amu-cai/CAMEO, CaFE split: CC-BY-NC-SA-4.0.
Source and speaker attribution: https://huggingface.co/datasets/amu-cai/CAMEO
"""
import argparse
import hashlib
import json
import subprocess
import urllib.request
import wave
from pathlib import Path

REVISION = "c7f2fc9f0538ca871d38e183ba8ad6c329120ecc"
PARQUET_SHA = "cef4c83676038c80841c8b9712be62bbf3d03c8dc148310203b634aad7c4c9e1"
PCM_SHA = "c333bd62e3224931788c8ec8b379161d5c62220d9794d8666636aee480bb8baf"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--parquet", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parquet = args.parquet or args.output / "cafe.parquet"
    if not parquet.exists():
        urllib.request.urlretrieve(
            f"https://huggingface.co/datasets/amu-cai/CAMEO/resolve/{REVISION}/data/cafe-00000-of-00001.parquet",
            parquet,
        )
    if hashlib.sha256(parquet.read_bytes()).hexdigest() != PARQUET_SHA:
        raise ValueError("pinned CAMEO Parquet hash mismatch")
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet).to_pylist()
    row = min((r for r in rows if r["speaker_id"] == "cafe_12" and r["emotion"] == "neutral"
               and r["language"].lower() == "french"), key=lambda r: r["file_id"])
    if row["dataset"] != "CaFE":
        raise ValueError("unexpected fixture dataset")
    source = args.output / "public-clip.flac"
    source.write_bytes(row["audio"]["bytes"])
    wav = args.output / "public-fr-16k.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-t", "15", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
    source.unlink()
    with wave.open(str(wav), "rb") as f:
        params = f.getparams()
        pcm = f.readframes(f.getnframes())
    if params[:3] != (1, 2, 16000) or len(pcm) != 2 * 55649 or hashlib.sha256(pcm).hexdigest() != PCM_SHA:
        raise ValueError("public PCM differs from the qualified fixture")
    with wave.open(str(args.output / "public-fr-repeat-16k.wav"), "wb") as f:
        f.setparams(params)
        f.writeframes(pcm + pcm)
    metadata = dict(dataset="amu-cai/CAMEO", revision=REVISION, split="cafe", license="CC-BY-NC-SA-4.0",
                    selection="speaker_id=cafe_12;emotion=neutral;min(file_id)", pcm_sha256=PCM_SHA,
                    sample_rate=16000, frames=55649, public_expected_transcription=row["transcription"])
    (args.output / "fixture.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
