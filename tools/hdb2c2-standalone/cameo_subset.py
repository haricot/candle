#!/usr/bin/env python3
"""Deterministic CAMEO CaFE/Oréau ASR benchmark sampler; never publish WAVs."""
import argparse
import hashlib
import json
import wave
from collections import Counter
from pathlib import Path

REPO = "amu-cai/CAMEO"
REV = "c7f2fc9f0538ca871d38e183ba8ad6c329120ecc"
EMOTIONS = ("anger","disgust","fear","happiness","neutral","sadness","surprise")

def select(rows, split, quota):
    counts = Counter()
    for row in rows:
        if str(row.get("language","")).casefold() != "french":
            continue
        source = str(row.get("dataset","")).casefold()
        if source not in (("cafe",) if split=="cafe" else ("oréau","oreau")):
            continue
        emotion = str(row.get("emotion","")).casefold()
        if emotion not in EMOTIONS or counts[emotion]>=quota:
            continue
        if not str(row.get("transcription","")).strip():
            continue
        counts[emotion] += 1
        yield row
        if all(counts[x] >= quota for x in EMOTIONS):
            return

def pcm16(audio):
    import numpy as np
    if not isinstance(audio,dict) or audio.get("sampling_rate")!=16000 or "array" not in audio:
        raise ValueError("Expected decoded CAMEO 16 kHz audio dict")
    data=np.asarray(audio["array"],dtype=np.float32)
    if data.ndim!=1 or len(data)<160 or not np.isfinite(data).all():
        raise ValueError("Invalid mono audio")
    return np.clip(np.rint(data*32768),-32768,32767).astype("<i2").tobytes()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--per-emotion",type=int,default=1)
    p.add_argument("--revision",default=REV)
    a=p.parse_args()
    if not 1<=a.per_emotion<=5: p.error("--per-emotion must be in 1..5")
    from datasets import load_dataset
    a.out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for split in ("cafe","oreau"):
        counts=Counter()
        ds=load_dataset(REPO,split=split,streaming=True,revision=a.revision)
        for row in select(ds,split,a.per_emotion):
            emotion=str(row["emotion"]).casefold()
            pcm=pcm16(row["audio"])
            digest=hashlib.sha256(pcm).hexdigest()
            name=f"{split}-{emotion}-{digest[:12]}"
            wav=a.out/(name+".wav")
            ref=a.out/(name+".txt")
            with wave.open(str(wav),"wb") as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(pcm)
            ref.write_text(row["transcription"].strip()+"\n",encoding="utf8")
            manifest.append(dict(dataset=REPO,revision=a.revision,split=split,
                id=name,emotion=emotion,speaker_id=str(row.get("speaker_id","")),
                license=str(row.get("license","")),pcm_sha256=digest,
                transcription=row["transcription"].strip(),
                wav=str(wav.resolve()),reference_file=str(ref.resolve())))
            counts[emotion]+=1
        if any(counts[x]<a.per_emotion for x in EMOTIONS):
            raise RuntimeError(f"Incomplete {split} coverage: {dict(counts)}")
    (a.out/"manifest.json").write_text(json.dumps(dict(dataset=REPO,
        revision=a.revision,rows=manifest),indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(f"HDB2C2_CAMEO_SAMPLES={len(manifest)}")

if __name__=="__main__":
    main()
