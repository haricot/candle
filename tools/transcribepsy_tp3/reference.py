#!/usr/bin/env python3
"""TP3 single-clip QVAC reference: original NeMo, official weights, production TDT.

No patched NeMo kernel or synthetic encoder injections. A reference failure
means NOT_QUALIFIED, never a fabricated PASS.
"""
import argparse
import hashlib
import json
import tarfile
import wave
from pathlib import Path

import numpy as np

QVAC_SAFETENSORS_SHA256 = "481fec004f2074909a5d253cb674fe676a49ce856bca2ce0156136ea920c3168"
QVAC_NEMO_SHA256 = "0acdaf1e1ebf09094400263006d134d33b0ebb871ebb064253cfd3a7c528903c"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exact_native_weights(model, weights):
    """Prove saved F32 tensor bytes equal the original NeMo state."""
    from safetensors import safe_open
    state = model.state_dict()
    count = 0
    with safe_open(str(weights), framework="pt", device="cpu") as safe:
        official = set(safe.keys())
        for key, original in state.items():
            if not original.is_floating_point():
                continue
            if key not in official:
                raise RuntimeError("Missing official native tensor: " + key)
            native = safe.get_tensor(key)
            original = original.detach().cpu().contiguous()
            if native.shape != original.shape or native.dtype != original.dtype:
                raise RuntimeError("Native tensor dtype/shape mismatch: " + key)
            if not np.array_equal(native.numpy().view(np.uint8), original.numpy().view(np.uint8)):
                raise RuntimeError("Official QVAC weights differ from NeMo: " + key)
            count += 1
        unused = official - set(state)
        if unused:
            raise RuntimeError("Unmatched QVAC Safetensors tensors: " + str(len(unused)))
    if count < 900:
        raise RuntimeError("Native QVAC tensor inventory unexpectedly small")
    return count


def read_pcm(wav_path):
    with wave.open(str(wav_path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getcomptype()) != (16000, 1, 2, "NONE"):
            raise RuntimeError("TP3 only accepts mono PCM16 16k WAV")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def native_config(model, durations):
    from omegaconf import OmegaConf
    original = OmegaConf.to_container(model.cfg, resolve=True)
    cfg = {k: original[k] for k in ("preprocessor", "encoder", "decoder", "joint", "decoding")}
    # The saved YAML is sparse: e.g. preemph=0.97 and pad_to=16 are
    # constructor defaults, not YAML keys. Export the effective eval settings
    # so an independent implementation never silently chooses other defaults.
    f = model.preprocessor.featurizer
    for name in ("normalize", "preemph", "dither", "pad_to", "pad_value",
                 "exact_pad", "mag_power", "log", "log_zero_guard_type",
                 "log_zero_guard_value", "frame_splicing"):
        cfg["preprocessor"][name] = getattr(f, name)
    if f.use_grads:
        raise RuntimeError("TP3 expects evaluation STFT without the training sqrt guard")
    cfg["decoding"]["durations"] = list(durations)
    cfg["decoding"]["model_type"] = "tdt"
    # This value is implicit in some NeMo configs, explicit in native QVAC joint.
    cfg["joint"]["num_extra_outputs"] = len(durations)
    cfg["joint"]["num_classes"] = int(model.decoder.blank_idx)
    cfg["decoder"]["vocab_size"] = int(model.decoder.blank_idx)
    return cfg


def extracted_tokenizer(nemo_path, model, target):
    """Use exact SentencePiece model embedded in QVAC .nemo, not vocab heuristics."""
    runtime = model.tokenizer.tokenizer.serialized_model_proto()
    with tarfile.open(nemo_path, "r:*") as tar:
        members = [m for m in tar.getmembers()
                   if m.isfile() and m.name.rsplit("/", 1)[-1].endswith(".model")]
        matching = [m for m in members if tar.extractfile(m).read() == runtime]
        if len(matching) != 1:
            raise RuntimeError("Original QVAC SentencePiece asset not uniquely identified")
    target.write_bytes(runtime)
    return hashlib.sha256(runtime).hexdigest()


def capture_production_tdt(model, encoded, lengths):
    """Observe original GreedyBatchedTDTInfer; no decoder substitution."""
    import torch
    decoder, joint = model.decoder, model.joint
    infer = model.decoding.decoding
    if type(infer).__name__ != "GreedyBatchedTDTInfer":
        raise RuntimeError("Expected real NeMo GreedyBatchedTDTInfer")
    if infer.decoding_computer.has_fusion_models():
        raise RuntimeError("No fused LM permitted")
    blank = int(decoder.blank_idx)
    durations = [int(i) for i in infer.durations]
    maximum = infer.max_symbols
    cache = {}
    decisions = []
    original_after = joint.joint_after_projection
    original_include = (infer.include_duration, infer.decoding_computer.include_duration)

    def save_encoder(_module, _args, output):
        cache["projected_encoder"] = output.detach()
    def save_prediction(_module, _args, output):
        cache["projected_prediction"] = output.detach()
    def save_raw(_module, _args, output):
        cache["raw"] = output.detach()
    # Preserve original return shape! Some NeMo releases expect a 2D/3D output.
    # Avoid wrapping or replacing LSTM state; only observe actual joint calls.
    def after_shape_safe(f, g):
        projection = cache["projected_encoder"]
        matches = (projection.contiguous().view(torch.int32)
                   == f.contiguous().view(torch.int32)).all(dim=-1)
        indexes = matches[0].nonzero().flatten().tolist()
        if len(indexes) != 1:
            raise RuntimeError("Cannot attribute production TDT frame")
        actual = original_after(f, g)
        raw = cache["raw"].flatten()
        flat = actual.flatten()
        token = int(flat[:-len(durations)].argmax())
        duration_index = int(flat[-len(durations):].argmax())
        if (token != int(raw[:-len(durations)].argmax())
                or duration_index != int(raw[-len(durations):].argmax())):
            raise RuntimeError("Actual TDT argmax differs from raw head")
        decisions.append(dict(frame=int(indexes[0]), token=token,
                              duration_index=duration_index,
                              duration=durations[duration_index]))
        return actual

    hooks = [joint.enc.register_forward_hook(save_encoder),
             joint.pred.register_forward_hook(save_prediction),
             joint.joint_net.register_forward_hook(save_raw)]
    try:
        joint.joint_after_projection = after_shape_safe
        infer.include_duration = infer.decoding_computer.include_duration = True
        with torch.inference_mode():
            hypothesis = infer(encoder_output=encoded, encoded_lengths=lengths)[0][0]
    finally:
        joint.joint_after_projection = original_after
        infer.include_duration, infer.decoding_computer.include_duration = original_include
        for h in hooks:
            h.remove()

    def integers(x):
        return [int(i) for i in x.cpu().tolist() if i is not None] if isinstance(x, torch.Tensor) else [int(i) for i in x]
    ids = integers(hypothesis.y_sequence)
    frames = integers(hypothesis.timestamp)
    emission_durations = integers(hypothesis.token_duration)
    emitted = [d for d in decisions if d["token"] != blank]
    if (ids != [d["token"] for d in emitted]
            or frames != [d["frame"] for d in emitted]
            or emission_durations != [d["duration"] for d in emitted]):
        raise RuntimeError("Production NeMo observed decisions differ from its hypothesis")
    symbols = 0
    for i, d in enumerate(decisions):
        count = symbols + (d["token"] != blank)
        effective = max(1, d["duration"]) if d["token"] == blank else d["duration"]
        forced = effective == 0 and maximum is not None and count >= maximum
        next_frame = d["frame"] + effective + int(forced)
        d["advance"] = next_frame - d["frame"]
        d["forced_advance"] = bool(forced)
        symbols = 0 if effective or forced else count
        if i + 1 < len(decisions) and decisions[i + 1]["frame"] != next_frame:
            raise RuntimeError("Recorded actual NeMo TDT duration/frame transition drift")
    return dict(decisions=decisions, tokens=ids, text=model.tokenizer.ids_to_text(ids).strip(),
                blank=blank, durations_config=durations, max_symbols=maximum,
                production_algorithm=type(infer).__module__ + "." + type(infer).__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if digest(a.weights) != QVAC_SAFETENSORS_SHA256:
        raise RuntimeError("Official QVAC Safetensors SHA256 mismatch")
    if digest(a.nemo) != QVAC_NEMO_SHA256:
        raise RuntimeError("Official QVAC .nemo SHA256 mismatch")
    a.output.mkdir(parents=True, exist_ok=True)

    import torch
    from nemo.collections.asr.models import ASRModel
    torch.set_num_threads(2)
    model = ASRModel.restore_from(str(a.nemo), map_location="cpu")
    model.to("cpu").float().eval()
    if any(m.training for m in model.modules()):
        raise RuntimeError("Reference NeMo encoder not recursively in eval mode")
    weights = exact_native_weights(model, a.weights)
    audio = torch.from_numpy(read_pcm(a.wav).copy()).unsqueeze(0)
    with torch.inference_mode():
        mel, mel_lengths = model.preprocessor(
            input_signal=audio, length=torch.tensor([audio.shape[1]], dtype=torch.long))
        encoded, encoder_lengths = model.encoder(audio_signal=mel, length=mel_lengths)

    def save(name, tensor):
        data = np.ascontiguousarray(tensor.detach().cpu().numpy().astype("<f4"))
        if not np.isfinite(data).all():
            raise RuntimeError("Nonfinite reference activation " + name)
        data.tofile(a.output / (name + ".f32le"))
        return list(data.shape)

    mel_shape = save("mel", mel.transpose(1, 2))
    encoder_shape = save("encoder", encoded.transpose(1, 2))
    tokenizer_sha = extracted_tokenizer(a.nemo, model, a.output / "tokenizer.model")
    reference = capture_production_tdt(model, encoded, encoder_lengths)
    config = native_config(model, reference["durations_config"])
    (a.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    manifest = dict(schema="TP3-NEMO-QVAC-F32-V1",
                    weights_sha256=QVAC_SAFETENSORS_SHA256,
                    nemo_sha256=QVAC_NEMO_SHA256,
                    identical_float_tensors=weights,
                    tokenizer_sha256=tokenizer_sha,
                    wav_sha256=digest(a.wav),
                    mel={"shape": mel_shape, "valid": int(mel_lengths[0])},
                    encoder={"shape": encoder_shape, "valid": int(encoder_lengths[0])},
                    **reference)
    (a.output / "trace.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("TP3_REFERENCE=REAL_NEMO_CAPTURED identical_weights=" + str(weights)
          + " mel_valid=" + str(manifest["mel"]["valid"])
          + " encoder_valid=" + str(manifest["encoder"]["valid"])
          + " decisions=" + str(len(manifest["decisions"])))


if __name__ == "__main__":
    main()
