#!/usr/bin/env python3
"""Optional qualification only: execute the pinned, unmodified HF CPU F32 reference.

Runtime transcription is the Rust example; this script is never imported by it.
Transformers source: 944e22eb226890dec60299386579275a66b3b44d.
"""
import argparse
import hashlib
import json
import time
import wave
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoProcessor, Nemotron3_5AsrForRNNT


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError("expected mono PCM16 WAV at 16 kHz")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768


def save_tensor(output, name, tensor):
    value = tensor.detach().cpu().float().numpy()
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite {name}")
    value.astype("<f4").tofile(output / f"{name}.f32le")
    return {"shape": list(value.shape), "dtype": "f32le"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=["streaming", "offline"], default="streaming")
    ap.add_argument("--language", default="fr-FR")
    ap.add_argument("--lookahead", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    pcm = read_wav(args.input)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = Nemotron3_5AsrForRNNT.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    inputs = processor(pcm, sampling_rate=16000, language=args.language, return_tensors="pt")
    mel = inputs["input_features"]
    valid_mel = int(inputs["attention_mask"].sum())
    encoded = []

    def observe(_module, _inputs, result):
        encoded.append(result.last_hidden_state.detach().clone())

    hook = model.encoder.register_forward_hook(observe)
    start = time.perf_counter()
    common = dict(
        prompt_ids=inputs["prompt_ids"], num_lookahead_tokens=args.lookahead,
        do_sample=False, max_new_tokens=10000,
    )
    with torch.inference_mode():
        if args.mode == "offline":
            prediction = model.generate(
                input_features=mel, attention_mask=inputs["attention_mask"], **common
            )
            valid_encoder = int(model.encoder._get_subsampling_output_length(
                torch.tensor([valid_mel])
            )[0])
        else:
            # Exact official fixed-Mel-block API. The frontend is a centered pass;
            # only valid frames are sent, and the final block is padded with zero Mel.
            # This protocol is also produced online by the Rust PCM frontend.
            def chunks():
                offset = 0
                while offset < valid_mel:
                    n = 1 + 8 * args.lookahead if offset == 0 else 8 * (args.lookahead + 1)
                    chunk = mel[:, offset:min(offset + n, valid_mel)]
                    offset += n
                    yield torch.nn.functional.pad(chunk, (0, 0, 0, n - chunk.shape[1]))

            prediction = model.generate(input_features=chunks(), **common)
            valid_encoder = sum(t.shape[1] for t in encoded)
    elapsed = time.perf_counter() - start
    hook.remove()
    sequences = prediction.sequences[0].tolist()
    durations = prediction.durations[0].tolist()
    if sequences[0] != model.config.blank_token_id or durations[0] != 0:
        raise ValueError("unexpected decoder start token")
    frame = 0
    decisions = []
    tokens = []
    for token, advance in zip(sequences[1:], durations[1:], strict=True):
        blank = token == model.config.blank_token_id
        decisions.append(dict(frame=frame, token=token, advance=advance,
                              forced_advance=bool(advance and not blank)))
        if not blank:
            tokens.append(token)
        frame += advance
    if frame != valid_encoder:
        raise ValueError(f"generation stopped before encoder exhaustion: {frame} != {valid_encoder}")
    args.output.mkdir(parents=True, exist_ok=True)
    activations = dict(
        mel=save_tensor(args.output, "mel", mel),
        encoder=save_tensor(args.output, "encoder", torch.cat(encoded, dim=1)),
        mel_valid_frames=valid_mel, encoder_valid_frames=valid_encoder,
    )
    result = dict(
        model=str(args.model_dir), device="cpu", dtype="f32", mode=args.mode,
        language=args.language, lookahead=args.lookahead, samples=len(pcm), sample_rate=16000,
        wav_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        mel_frames=valid_mel, encoder_frames=valid_encoder, tokens=tokens, decisions=decisions,
        text=processor.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
        inference_seconds=elapsed, activations=activations,
        versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                      transformers_commit="944e22eb226890dec60299386579275a66b3b44d"),
    )
    (args.output / "activations.json").write_text(json.dumps(activations, indent=2) + "\n")
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("decisions", "activations")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
