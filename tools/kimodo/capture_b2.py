#!/usr/bin/env python3
"""S51-B2: capture a real checkpoint's F32 oracle without loading LLM2Vec.
Run inside the official Kimodo Python environment. Outputs remain local.
Synthetic nonzero embeddings/conditions test numerical execution, not text semantics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

MODEL = "nvidia/Kimodo-SOMA-RP-v1.2"
SOURCE = "e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954"

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--revision", required=True, help="Exact 40-hex Hugging Face revision")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if not (2 <= a.frames <= 300 and 2 <= a.steps <= 50 and a.threads > 0):
        p.error("invalid frames, steps or threads")
    if len(a.revision) != 40 or any(c not in "0123456789abcdef" for c in a.revision):
        p.error("revision must be an exact lowercase Hugging Face commit SHA")
    upstream = a.upstream.resolve()
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE:
        raise RuntimeError(f"expected NVIDIA source {SOURCE}, got {head}")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError("NVIDIA tracked source is dirty")
    out = a.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    sys.path.insert(0, str(upstream))
    import torch
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from safetensors.torch import save_file
    from kimodo.model.loading import instantiate_from_dict

    torch.set_num_threads(a.threads)
    torch.manual_seed(314159)
    torch.set_float32_matmul_precision("highest")
    # Avoid fused transformer paths for a transparent F32 oracle.
    torch.backends.mha.set_fastpath_enabled(False)
    snapshot = Path(snapshot_download(repo_id=MODEL, revision=a.revision))
    if snapshot.name != a.revision:
        raise RuntimeError(f"unexpected snapshot revision: {snapshot}")
    conf_path = snapshot / "config.yaml"
    conf = OmegaConf.to_container(OmegaConf.merge(
        OmegaConf.load(conf_path), OmegaConf.create({"checkpoint_dir": str(snapshot)})
    ), resolve=True)
    conf.pop("checkpoint_dir", None)
    conf["text_encoder"] = None
    ref = instantiate_from_dict(conf, overrides={"device": "cpu"}).eval().float()
    net = ref.denoiser.model
    rep = ref.motion_rep
    if net.motion_mask_mode != "concat" or (
        rep.motion_rep_dim, rep.global_root_dim, rep.local_root_dim, ref.skeleton.nbjoints
    ) != (369, 5, 4, 30):
        raise RuntimeError("unsupported checkpoint layout; do not silently adapt")
    if not 2 <= a.steps <= ref.diffusion.num_base_steps:
        raise RuntimeError("invalid step count")
    root, body = net.root_model, net.body_model
    fields = ("latent_dim", "ff_size", "num_layers", "num_heads", "use_text_mask",
              "input_first_heading_angle", "norm_first", "activation")
    cfg = {k: getattr(root, k) for k in fields}
    if any(getattr(root, k) != getattr(body, k) for k in fields):
        raise RuntimeError("root/body transformer configurations differ")
    cfg.update(text_dim=int(root.llm_shape[-1]), text_tokens=int(root.num_text_tokens),
               num_base_steps=int(ref.diffusion.num_base_steps), fps=float(rep.fps))
    if int(body.llm_shape[-1]) != cfg["text_dim"] or body.num_text_tokens != cfg["text_tokens"]:
        raise RuntimeError("root/body text configuration differs")
    if cfg["text_dim"] != 4096 or cfg["fps"] != 30.0:
        raise RuntimeError(f"unexpected SOMA checkpoint config {cfg}")
    stats = {}
    for name, st in (("global", rep.global_root_stats), ("local", rep.local_root_stats)):
        stats[name+"_mean"] = st.mean.flatten().tolist()
        stats[name+"_std"] = st.std.flatten().tolist()
        if st.eps != 1e-5:
            raise RuntimeError("unexpected stats epsilon")
    if any(len(stats[k]) != (5 if k.startswith("global") else 4) for k in stats):
        raise RuntimeError("unexpected statistics shape")
    # Export exactly the loaded backbone's state dict, with the prefix removed
    # by NVIDIA's checkpoint loader; no quantization or synthetic weights.
    weights = {k: v.detach().cpu().float().contiguous().clone()
               for k, v in net.state_dict().items() if k.startswith(("root_model.", "body_model."))}
    if not weights or not all(torch.isfinite(v).all().item() for v in weights.values()):
        raise RuntimeError("missing or nonfinite backbone weights")
    params = sum(v.numel() for v in weights.values())
    if params < 100_000_000:
        raise RuntimeError("checkpoint unexpectedly small")
    t, d = a.frames, rep.motion_rep_dim
    ntext = min(2, cfg["text_tokens"])
    noise = torch.randn((1, t, d), dtype=torch.float32)
    text = torch.randn((1, ntext, cfg["text_dim"]), dtype=torch.float32) * 0.1
    text_mask = torch.ones((1, ntext), dtype=torch.bool)
    pad = torch.ones((1, t), dtype=torch.bool)
    heading = torch.tensor([0.37], dtype=torch.float32)
    observed = torch.zeros_like(noise)
    mask = torch.zeros_like(noise)
    mask[:, 0, :5] = 1
    mask[:, t//2, :25] = 1
    observed[mask.bool()] = noise[mask.bool()] * 0.1
    tensors = dict(noise=noise, text=text, observed=observed, mask=mask)
    use, mapping = ref.diffusion.space_timesteps(a.steps)
    steps = [int(mapping[i]) for i in range(a.steps)]
    with torch.inference_mode():
        ts = torch.tensor([steps[-1]], dtype=torch.long)
        direct = net(noise, pad, text, text_mask, ts, heading, mask, observed)
        tensors["direct"] = direct
        tensors["local_root"] = rep.global_root_to_local_root(
            direct[..., :5], normalized=True, lengths=pad.sum(-1))
        x = noise
        for i in reversed(range(a.steps)):
            ts = torch.tensor([steps[i]], dtype=torch.long)
            clean = ref.denoiser([2.0, 2.0], x, pad, text, text_mask, ts,
                                 first_heading_angle=heading, motion_mask=mask,
                                 observed_motion=observed, cfg_type="separated")
            nxt = ref.sampler(use, x, clean, torch.tensor([i], dtype=torch.long))
            tensors[f"input_{i}"] = x
            tensors[f"clean_{i}"] = clean
            tensors[f"next_{i}"] = nxt
            x = nxt
        tensors["final"] = x
    if not all(torch.isfinite(v).all().item() for v in tensors.values()):
        raise RuntimeError("nonfinite NVIDIA reference")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Publish only complete captures. Leave no manifest on error.
    with tempfile.TemporaryDirectory(prefix=".kimodo-b2-", dir=out.parent) as tmp:
        stage = Path(tmp) / "fixture"
        stage.mkdir()
        save_file(weights, str(stage/"weights.safetensors"))
        save_file({k:v.detach().cpu().float().contiguous().clone() for k,v in tensors.items()},
                  str(stage/"oracle.safetensors"))
        manifest = dict(schema="haricot.kimodo.b2.v1", model=MODEL, model_revision=a.revision,
                        upstream=SOURCE, config=cfg, stats=stats, steps=steps,
                        heading=heading.tolist(), text_lengths=[ntext],
                        frames=t, parameters=params, embedding_kind="synthetic_nonzero",
                        torch_version=str(torch.__version__), config_sha256=sha(conf_path),
                        weights_sha256=sha(stage/"weights.safetensors"),
                        oracle_sha256=sha(stage/"oracle.safetensors"))
        (stage/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        os.rename(stage, out)
    print(f"KIMODO_B2_CAPTURE=PASS model={MODEL} revision={a.revision} params={params} frames={t} steps={a.steps}")
    print(f"KIMODO_B2_FIXTURE={out}")

if __name__ == "__main__":
    main()
