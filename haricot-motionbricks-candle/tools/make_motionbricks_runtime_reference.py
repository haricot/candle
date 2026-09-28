#!/usr/bin/env python3
"""Torch-only MotionBricks numerical oracle for Haricot v0.3-r2.

Dependencies: torch + safetensors only.

This file intentionally does NOT import MotionBricks, Hydra, OmegaConf, or
PyTorch Lightning.  It consumes the canonical safetensors produced by the
Rust v0.3-r1 checkpoint exporter and independently reimplements the fixed-16
runtime profile with torch functional operators.

The oracle is independent from the Candle operator implementation.  Weight
extraction/remapping remains the responsibility of the Rust exporter and is
covered by its pinned-revision + structural-anchor audit.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

SCHEMA = "haricot-motionbricks-torch-only-reference.v0.3-r2-fix1"
PINNED_REV = "087f9ac01d46f6d8e4d0b73c01ae64799f292a38"

NUM_TOKENS = 16
FRAMES_PER_TOKEN = 4
FRAMES = NUM_TOKENS * FRAMES_PER_TOKEN
POSE_HEADS = 8
CODES_PER_HEAD = 10
POSE_DIM = 304
POSE_DECODER_DIM = 413
GLOBAL_ROOT_DIM = 5
LOCAL_ROOT_DIM = 4


class Weights:
    def __init__(self, path: Path, device: torch.device):
        self.path = path
        self.t = load_file(str(path), device=str(device))
        if len(self.t) != 410:
            raise RuntimeError(f"expected 410 runtime tensors, got {len(self.t)}")
        self.require("vqvae.codebook", (8, 10, 32))
        self.require(
            "pose._transformer_model.layers.0.self_attn.in_proj_weight",
            (3072, 1024),
        )
        self.require(
            "root._shared_transformer_model.layers.0.self_attn.in_proj_weight",
            (1536, 512),
        )
        self.require("vqvae.decoder.model.0.weight", (512, 256, 3))
        self.require("root._proj_local_pose.weight", (256, 304))
        self.require("pose._proj_local_pose.weight", (160, 304))
        self.require("vqvae.decoder.target_cond_blocks.0.weight", (128, 304))
        self.require("vqvae.decoder.model.6.weight", (413, 512, 3))
        for name, value in self.t.items():
            if value.dtype != torch.float32:
                raise RuntimeError(f"runtime tensor {name} is {value.dtype}, expected float32")

    def require(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        if name not in self.t:
            raise RuntimeError(f"missing runtime tensor: {name}")
        value = self.t[name]
        if tuple(value.shape) != shape:
            raise RuntimeError(
                f"runtime tensor {name} shape={tuple(value.shape)}, expected={shape}"
            )
        return value

    def get(self, name: str) -> torch.Tensor:
        try:
            return self.t[name]
        except KeyError as exc:
            raise RuntimeError(f"missing runtime tensor: {name}") from exc


def det(shape: tuple[int, ...], freq: float, scale: float, device: torch.device) -> torch.Tensor:
    n = math.prod(shape)
    a = torch.arange(n, dtype=torch.float32, device=device)
    return (torch.sin(a * freq) * scale).reshape(shape)


def linear(w: Weights, prefix: str, x: torch.Tensor) -> torch.Tensor:
    weight = w.get(prefix + ".weight")
    bias = w.t.get(prefix + ".bias")
    return F.linear(x, weight, bias)


def embedding(w: Weights, prefix: str, ids: torch.Tensor) -> torch.Tensor:
    return F.embedding(ids, w.get(prefix + ".weight"))


def conv1d(
    w: Weights,
    prefix: str,
    x: torch.Tensor,
    *,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    weight = w.get(prefix + ".weight")
    bias = w.t.get(prefix + ".bias")
    return F.conv1d(x, weight, bias, stride=1, padding=padding, dilation=dilation, groups=1)


def fc_block(w: Weights, prefix: str, x: torch.Tensor, num_layers: int = 2) -> torch.Tensor:
    h = x
    for i in range(num_layers):
        h = F.leaky_relu(linear(w, f"{prefix}.fc_layers.{i}", h), negative_slope=0.01)
    return linear(w, f"{prefix}.forward_projection", h)


def self_attention(
    w: Weights,
    prefix: str,
    x: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    b, s, d = x.shape
    if d % n_heads != 0:
        raise RuntimeError(f"attention dim {d} not divisible by {n_heads}")
    head_dim = d // n_heads
    qkv = F.linear(
        x,
        w.get(prefix + ".in_proj_weight"),
        w.get(prefix + ".in_proj_bias"),
    )
    q, k, v = qkv.split(d, dim=-1)
    q = q.reshape(b, s, n_heads, head_dim).transpose(1, 2)
    k = k.reshape(b, s, n_heads, head_dim).transpose(1, 2)
    v = v.reshape(b, s, n_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(q * (1.0 / math.sqrt(head_dim)), k.transpose(2, 3))
    probs = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(probs, v).transpose(1, 2).contiguous().reshape(b, s, d)
    return linear(w, prefix + ".out_proj", ctx)


def transformer_layer(
    w: Weights,
    prefix: str,
    x: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    attn = self_attention(w, prefix + ".self_attn", x, n_heads)
    x = F.layer_norm(
        x + attn,
        (x.shape[-1],),
        w.get(prefix + ".norm1.weight"),
        w.get(prefix + ".norm1.bias"),
        1e-5,
    )
    ff = linear(w, prefix + ".linear2", F.relu(linear(w, prefix + ".linear1", x)))
    return F.layer_norm(
        x + ff,
        (x.shape[-1],),
        w.get(prefix + ".norm2.weight"),
        w.get(prefix + ".norm2.bias"),
        1e-5,
    )


def transformer(
    w: Weights,
    prefix: str,
    x: torch.Tensor,
    *,
    n_layers: int,
    n_heads: int,
) -> torch.Tensor:
    h = x
    for i in range(n_layers):
        h = transformer_layer(w, f"{prefix}.layers.{i}", h, n_heads)
    return h


def resnet1d(
    w: Weights,
    prefix: str,
    x: torch.Tensor,
    *,
    depth: int = 4,
    dilation_growth_rate: int = 3,
) -> torch.Tensor:
    # Official runtime: reverse_dilation=True => [27, 9, 3, 1] for depth=4.
    dilations = [dilation_growth_rate**i for i in range(depth)][::-1]
    h = x
    for i, dilation in enumerate(dilations):
        p = f"{prefix}.model.{i}"
        residual = h
        h = conv1d(w, p + ".conv1", F.relu(h), padding=dilation, dilation=dilation)
        h = conv1d(w, p + ".conv2", F.relu(h), padding=0, dilation=1)
        h = h + residual
    return h


def double_cond_decoder(
    w: Weights,
    prefix: str,
    x: torch.Tensor,
    *,
    external_cond: torch.Tensor,
    target_cond: torch.Tensor,
    has_target_cond: torch.Tensor,
    input_emb_width: int,
    external_cond_dim: int,
    target_cond_dim: int,
    down_t: int = 2,
    width: int = 512,
    depth: int = 4,
) -> torch.Tensor:
    if x.shape[1] != 256 and prefix == "vqvae.decoder":
        raise RuntimeError(f"vqvae decoder input channels={x.shape[1]}, expected 256")
    if target_cond.shape[-1] != target_cond_dim:
        raise RuntimeError("target condition feature width mismatch")
    if external_cond.shape[-1] != external_cond_dim:
        raise RuntimeError("external condition feature width mismatch")

    h = F.relu(conv1d(w, prefix + ".model.0", x, padding=1))
    b = h.shape[0]

    for i in range(down_t):
        frames_per_position = 1 << (down_t - i)
        num_positions = h.shape[-1]
        timesteps = num_positions * frames_per_position
        if timesteps != target_cond.shape[1] or timesteps != external_cond.shape[1]:
            raise RuntimeError(
                f"decoder stage {i}: timesteps={timesteps}, target={target_cond.shape[1]}, "
                f"external={external_cond.shape[1]}"
            )

        target = F.relu(linear(w, f"{prefix}.target_cond_blocks.{i * 2}", target_cond))
        frame = h.transpose(1, 2).contiguous().reshape(
            b, timesteps, width // frames_per_position
        )
        mask = has_target_cond[:, :timesteps, None].bool()
        h = torch.where(mask, target[:, :timesteps], frame)
        h = h.reshape(b, num_positions, width).transpose(1, 2)

        ext = external_cond[:, :timesteps].reshape(b, num_positions, -1)
        h = torch.cat([h.transpose(1, 2), ext], dim=-1)
        h = F.relu(linear(w, f"{prefix}.external_cond_blocks.{i * 2}", h)).transpose(1, 2)

        stage_prefix = f"{prefix}.model.{i + 2}"
        h = resnet1d(w, stage_prefix + ".0", h, depth=depth)
        h = h.repeat_interleave(2, dim=2)
        h = conv1d(w, stage_prefix + ".2", h, padding=1)

    h = F.relu(conv1d(w, f"{prefix}.model.{2 + down_t}", h, padding=1))
    h = conv1d(w, f"{prefix}.model.{2 + down_t + 2}", h, padding=1)
    if h.shape[1] != input_emb_width:
        raise RuntimeError(
            f"decoder {prefix} output channels={h.shape[1]}, expected={input_emb_width}"
        )
    return h


def root_forward(
    w: Weights,
    global_root_values: torch.Tensor,
    local_root_values: torch.Tensor,
    poses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, frames, _ = poses.shape
    if frames != 8:
        raise RuntimeError("fixed16 root oracle expects exactly 8 constraint frames")

    pose = linear(w, "root._proj_local_pose", poses)
    local = linear(w, "root._proj_local_root_value", local_root_values)
    global_ = linear(w, "root._proj_global_root_value", global_root_values)
    merged = torch.cat([pose, local, global_], dim=-1)

    start = fc_block(w, "root._proj_start_input", merged[:, :4])
    end = fc_block(w, "root._proj_end_input", merged[:, 4:])
    frame_emb = torch.cat([start, end], dim=1)
    positioned = frame_emb + w.get("root._input_position_emb.weight")[None, :, :]

    # MotionBricks min_tokens=6, fixed profile num_tokens=16 => embedding index 10.
    num_idx = torch.full((b,), 10, dtype=torch.long, device=poses.device)
    time_emb = embedding(w, "root._proj_input_num_tokens", num_idx).reshape(b, 1, 512)
    first_input = torch.cat([time_emb, positioned], dim=1)
    first_out = transformer(
        w,
        "root._shared_transformer_model",
        first_input,
        n_layers=3,
        n_heads=16,
    )
    num_token_logits = linear(w, "root._proj_num_token_output_logit", first_out[:, 0, :])

    middle = embedding(w, "root._middle_token_emb", num_idx).reshape(b, 1, 512)
    pos = w.get("root._position_emb.embed").reshape(1, NUM_TOKENS, 512).expand(b, -1, -1)
    second_input = torch.cat([middle, positioned, pos], dim=1)
    second = transformer(
        w,
        "root._root_token_transformer_model",
        second_input,
        n_layers=3,
        n_heads=16,
    )
    token_hidden = second[:, 9 : 9 + NUM_TOKENS, :]

    no_frame = w.get("root._conv_no_frame_emb").reshape(1, 1, 512).expand(
        b, FRAMES - 8, -1
    )
    dense_frame = torch.cat([frame_emb[:, :4], no_frame, frame_emb[:, 4:]], dim=1)
    zeros = torch.zeros(
        (b, FRAMES - 8, GLOBAL_ROOT_DIM),
        dtype=global_root_values.dtype,
        device=global_root_values.device,
    )
    dense_global = torch.cat(
        [global_root_values[:, :4], zeros, global_root_values[:, 4:]], dim=1
    )
    has_target = torch.zeros((b, FRAMES), dtype=torch.bool, device=poses.device)
    has_target[:, :4] = True
    has_target[:, -4:] = True

    pred = double_cond_decoder(
        w,
        "root._conv_output",
        token_hidden.transpose(1, 2),
        external_cond=dense_frame,
        target_cond=dense_global,
        has_target_cond=has_target,
        input_emb_width=GLOBAL_ROOT_DIM,
        external_cond_dim=512,
        target_cond_dim=GLOBAL_ROOT_DIM,
    ).transpose(1, 2)

    return num_token_logits, pred


def pose_forward(
    w: Weights,
    pose_tokens: torch.Tensor,
    root_values: torch.Tensor,
    pose_cond: torch.Tensor,
    has_pose_cond: torch.Tensor,
) -> torch.Tensor:
    b, positions, heads = pose_tokens.shape
    if (positions, heads) != (NUM_TOKENS, POSE_HEADS):
        raise RuntimeError("fixed16 pose oracle expects pose_tokens [B,16,8]")

    root = linear(w, "pose._proj_local_root_values", root_values.reshape(b, 16, 16))
    cond = linear(w, "pose._proj_local_pose", pose_cond)

    offsets = (
        torch.arange(POSE_HEADS, dtype=torch.long, device=pose_tokens.device)
        .reshape(1, 1, POSE_HEADS)
        * 11
    )
    ids = pose_tokens + offsets
    token = F.embedding(ids, w.get("pose._pose_token_emb.weight")).reshape(b, 16, 256)
    token = fc_block(w, "pose._proj_pose_token_emb", token).reshape(b, FRAMES, 160)
    mask = has_pose_cond.reshape(b, FRAMES, 1).to(dtype=cond.dtype)
    pose = (cond * mask + token * (1.0 - mask)).reshape(b, 16, 640)

    num_idx = torch.full((b,), 10, dtype=torch.long, device=pose_tokens.device)
    num = F.embedding(num_idx, w.get("pose._proj_num_valid_positions.weight"))
    num = num.reshape(b, 1, 128).expand(b, 16, 128)

    merged = torch.cat([pose, root, num], dim=-1)
    pos = w.get("pose._position_emb.embed").reshape(1, 16, 1024).expand(b, -1, -1)
    h = F.relu(linear(w, "pose._proj_input.0", merged)) + pos
    h = transformer(
        w,
        "pose._transformer_model",
        h,
        n_layers=16,
        n_heads=16,
    )
    return linear(w, "pose._proj_pose_output_logit", h).reshape(b, 16, 8, 10)


def vqvae_forward(
    w: Weights,
    pose_tokens: torch.Tensor,
    target_cond: torch.Tensor,
    has_target_cond: torch.Tensor,
    external_cond: torch.Tensor,
) -> torch.Tensor:
    b, tokens, heads = pose_tokens.shape
    if heads != POSE_HEADS:
        raise RuntimeError("vqvae decoder expects 8 pose heads")
    codebook = w.get("vqvae.codebook")
    chunks = []
    for h in range(POSE_HEADS):
        idx = pose_tokens[:, :, h].reshape(-1)
        chunks.append(codebook[h].index_select(0, idx).reshape(b, tokens, 32))
    quant = torch.cat(chunks, dim=-1).transpose(1, 2)
    return double_cond_decoder(
        w,
        "vqvae.decoder",
        quant,
        external_cond=external_cond,
        target_cond=target_cond,
        has_target_cond=has_target_cond,
        input_emb_width=POSE_DECODER_DIM,
        external_cond_dim=2,
        target_cond_dim=POSE_DIM,
    ).transpose(1, 2)


def assert_output(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != shape:
        raise RuntimeError(f"{name} shape={tuple(value.shape)}, expected={shape}")
    if value.is_floating_point() and not torch.isfinite(value).all().item():
        raise RuntimeError(f"{name} contains NaN/Inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Rust-exported motionbricks-runtime-v1.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu, ...")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false; use --device cpu")

    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    w = Weights(weights_path, device)

    # Exact deterministic profile retained from v0.3-r1's official-reference generator.
    root_global = det((1, 8, 5), 0.071, 0.20, device)
    root_local = det((1, 8, 4), 0.053, 0.10, device)
    root_poses = det((1, 8, POSE_DIM), 0.017, 0.12, device)

    pose_tokens = torch.empty((1, 16, 8), dtype=torch.long, device=device)
    for t in range(16):
        for h in range(8):
            pose_tokens[0, t, h] = (t * 3 + h * 5) % 11  # includes MASK=10
    pose_root = det((1, 64, 4), 0.029, 0.08, device)
    pose_cond = det((1, 64, POSE_DIM), 0.011, 0.10, device)
    has_pose = torch.zeros((1, 64), dtype=torch.bool, device=device)
    has_pose[:, :4] = True
    has_pose[:, -4:] = True

    target_cond = det((1, 64, POSE_DIM), 0.013, 0.09, device)
    has_target = torch.zeros((1, 64), dtype=torch.bool, device=device)
    has_target[:, :4] = True
    has_target[:, -4:] = True
    external_cond = det((1, 64, 2), 0.037, 0.05, device)

    with torch.inference_mode():
        root_num_logits, root_pred = root_forward(w, root_global, root_local, root_poses)
        pose_logits = pose_forward(w, pose_tokens, pose_root, pose_cond, has_pose)
        vq_tokens = pose_logits.argmax(dim=-1).to(torch.long)
        recon = vqvae_forward(w, vq_tokens, target_cond, has_target, external_cond)

    assert_output("root.num_token_logits", root_num_logits, (1, 12))
    assert_output("root.pred_global_root_values", root_pred, (1, 64, 5))
    assert_output("pose.pose_logits", pose_logits, (1, 16, 8, 10))
    assert_output("vqvae.recon_state", recon, (1, 64, POSE_DECODER_DIM))

    tensors = {
        "input.root.global_root_values": root_global.float().cpu(),
        "input.root.local_root_values": root_local.float().cpu(),
        "input.root.poses": root_poses.float().cpu(),
        "input.pose.pose_tokens": pose_tokens.long().cpu(),
        "input.pose.root_values": pose_root.float().cpu(),
        "input.pose.pose_cond": pose_cond.float().cpu(),
        "input.pose.has_pose_cond": has_pose.float().cpu(),
        "input.vqvae.pose_tokens": vq_tokens.long().cpu(),
        "input.vqvae.target_cond": target_cond.float().cpu(),
        "input.vqvae.has_target_cond": has_target.float().cpu(),
        "input.vqvae.external_cond": external_cond.float().cpu(),
        "output.root.num_token_logits": root_num_logits.float().cpu(),
        "output.root.pred_global_root_values": root_pred.float().cpu(),
        "output.pose.pose_logits": pose_logits.float().cpu(),
        "output.vqvae.recon_state": recon.float().cpu(),
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(out),
        metadata={
            "schema": SCHEMA,
            "source": "Rust canonical runtime safetensors; Torch functional oracle",
            "source_rev": PINNED_REV,
            "profile": "fixed16-no-text-all-root-constraints",
            "motionbricks_import": "false",
            "hydra": "false",
            "lightning": "false",
        },
    )

    print("=== HARICOT MOTIONBRICKS TORCH-ONLY NUMERICAL ORACLE V0.3-R2-FIX1 ===")
    print(f"source_rev={PINNED_REV}")
    print(f"torch_version={torch.__version__}")
    print(f"device={device}")
    print("dependencies=torch,safetensors")
    print("motionbricks_import=false")
    print("hydra=false")
    print("omegaconf=false")
    print("lightning=false")
    print("python_checkpoint_pickle=false")
    print("profile=fixed16-no-text-all-root-constraints")
    print("tokens=16")
    print("frames=64")
    print(f"runtime_tensor_count={len(w.t)}")
    print("root.num_token_logits=[1,12]")
    print("root.pred_global_root_values=[1,64,5]")
    print("pose.pose_logits=[1,16,8,10]")
    print("vqvae.recon_state=[1,64,413]")
    print(f"out={out}")
    print("MOTIONBRICKS_TORCH_ONLY_ORACLE=PASS")


if __name__ == "__main__":
    main()
