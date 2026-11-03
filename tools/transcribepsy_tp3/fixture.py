#!/usr/bin/env python3
"""Extract exactly one public CAMEO French PCM fixture, never commit the WAV.

CAMEO overall: CC BY-NC-SA 4.0; CaFE split: CC BY-NC-SA 4.0.
Pinned Hugging Face revision and deterministic file_id selection.
"""
import argparse
import hashlib
import io
import json
import subprocess
import wave
from pathlib import Path

DATASET = "amu-cai/CAMEO"
REVISION = "c7f2fc9f0538ca871d38e183ba8ad6c329120ecc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    rows = pq.read_table(a.parquet).to_pylist()
    fr = [row for row in rows
          if row["speaker_id"] == "cafe_12"
          and row["emotion"] == "neutral"
          and row["language"].lower() == "french"]
    if not fr:
        raise RuntimeError("Pinned CaFE neutral French sample disappeared")
    row = min(fr, key=lambda r: r["file_id"])
    if row.get("dataset") != "CaFE":
        raise RuntimeError("Unexpected dataset in pinned CaFE split")
    b = row["audio"]["bytes"]
    if not b:
        raise RuntimeError("CAMEO Parquet omitted sample audio bytes")
    a.output.mkdir(parents=True, exist_ok=True)
    tmp = a.output / "public-clip.flac"
    tmp.write_bytes(b)
    wav = a.output / "public-fr-16k.wav"
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(tmp), "-t", "15", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(wav)], check=True)
    tmp.unlink()
    with wave.open(str(wav), "rb") as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (16000, 1, 2):
            raise RuntimeError("Unexpected public clip WAV format")
        frames = f.getnframes()
        pcm = f.readframes(frames)
    if frames < 4000 or frames > 16000 * 15:
        raise RuntimeError("Invalid selected public audio length")
    provenance = {
        "dataset": DATASET, "revision": REVISION, "split": "cafe",
        "license": "CC-BY-NC-SA-4.0",
        "selection": "speaker_id=cafe_12;emotion=neutral;min(file_id)",
        "file_id_sha256": hashlib.sha256(row["file_id"].encode()).hexdigest(),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
        "sample_rate": 16000, "frames": frames,
        "public_expected_transcription": row["transcription"],
    }
    (a.output / "fixture.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n")
    print("TP3_PUBLIC_FIXTURE=PASS source=CAMEO/cafe samples=1 frames=" + str(frames))


if __name__ == "__main__":
    main()
