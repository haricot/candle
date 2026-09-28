#!/usr/bin/env python3
"""Generate a self-contained PyTorch numerical-parity case for HaricotWHAM-lite.

This does not import WHAM, SMPL, torchvision, or CUDA. It reconstructs only the
Stage-1 Linear/LSTM recurrent core from the exported safetensors weights, runs a
deterministic F32 CPU input, and writes both inputs and expected outputs to one
safetensors file consumed by the Rust `--numerical-gate` mode.
"""

import argparse
import math
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file, save_file

N_JOINTS = 17
INPUT_DIM = 37
EMBED_DIM = 512
CONTEXT_DIM = 512 + 17 * 3
N_LAYERS = 3
POSE_JOINTS = 24
POSE_DIM = POSE_JOINTS * 6
KP3D_DIM = N_JOINTS * 3
MAIN_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def linear(x: torch.Tensor, w: Dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return torch.nn.functional.linear(x, w[prefix + ".weight"], w[prefix + ".bias"])


def neural_init(
    x: torch.Tensor,
    w: Dict[str, torch.Tensor],
    prefix: str,
    hidden_dim: int,
    n_layers: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    x = torch.relu(linear(x, w, prefix + ".linear1"))
    x = torch.relu(linear(x, w, prefix + ".linear2"))
    x = linear(x, w, prefix + ".linear3")
    x = x.reshape(x.shape[0], 2, n_layers, hidden_dim)
    return [(x[:, 0, i].clone(), x[:, 1, i].clone()) for i in range(n_layers)]


def zero_state(batch: int, hidden_dim: int, n_layers: int):
    return [
        (
            torch.zeros(batch, hidden_dim, dtype=torch.float32),
            torch.zeros(batch, hidden_dim, dtype=torch.float32),
        )
        for _ in range(n_layers)
    ]


def stacked_lstm_step(
    x: torch.Tensor,
    state: List[Tuple[torch.Tensor, torch.Tensor]],
    w: Dict[str, torch.Tensor],
    prefix: str,
) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    out = x
    next_state = []
    for layer_idx, (h, c) in enumerate(state):
        base = f"{prefix}.rnn"
        w_ih = w[f"{base}.weight_ih_l{layer_idx}"]
        w_hh = w[f"{base}.weight_hh_l{layer_idx}"]
        b_ih = w[f"{base}.bias_ih_l{layer_idx}"]
        b_hh = w[f"{base}.bias_hh_l{layer_idx}"]
        gates = torch.nn.functional.linear(out, w_ih, b_ih) + torch.nn.functional.linear(h, w_hh, b_hh)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        out = h_next
        next_state.append((h_next, c_next))
    return out, next_state


def regressor_step(
    x: torch.Tensor,
    inits: List[torch.Tensor],
    state,
    w,
    prefix: str,
    n_heads: int,
):
    xc = torch.cat([x, *inits], dim=-1)
    hidden, next_state = stacked_lstm_step(xc, state, w, prefix)
    preds = [linear(hidden, w, f"{prefix}.declayer{i}") for i in range(n_heads)]
    return preds, hidden, next_state


def neutral_pose6d(joints: int) -> torch.Tensor:
    one = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    return one.repeat(joints).reshape(1, joints * 6)


def make_inputs(frames: int):
    # Non-trivial but bounded deterministic sequence. No RNG is involved.
    pose = torch.empty((1, frames, INPUT_DIM), dtype=torch.float32)
    for t in range(frames):
        for j in range(INPUT_DIM):
            pose[0, t, j] = math.sin((t * INPUT_DIM + j) * 0.017) * 0.25

    init_kp3d = torch.empty((1, KP3D_DIM), dtype=torch.float32)
    for i in range(KP3D_DIM):
        init_kp3d[0, i] = math.cos(i * 0.031) * 0.05

    cam = torch.empty((1, frames, 6), dtype=torch.float32)
    for t in range(frames):
        for j in range(6):
            cam[0, t, j] = math.sin((t * 6 + j) * 0.013) * 0.002

    return {
        "input.pose2d": pose,
        "input.init_kp3d": init_kp3d,
        "input.init_pose_rot6d": neutral_pose6d(POSE_JOINTS),
        "input.init_root_rot6d": neutral_pose6d(1),
        "input.cam_angvel": cam,
    }


def forward(w: Dict[str, torch.Tensor], inp: Dict[str, torch.Tensor]):
    pose2d = inp["input.pose2d"]
    init_kp3d = inp["input.init_kp3d"]
    init_pose = inp["input.init_pose_rot6d"]
    init_root = inp["input.init_root_rot6d"]
    cam = inp["input.cam_angvel"]
    batch, frames, _ = pose2d.shape

    # Motion encoder initialization.
    init = torch.cat([init_kp3d, pose2d[:, 0]], dim=-1)
    me_state = neural_init(init, w, "motion_encoder.neural_init", EMBED_DIM, N_LAYERS)
    prev_kp = init_kp3d

    # Trajectory starts with a zero recurrent state.
    tr_state = zero_state(batch, CONTEXT_DIM, N_LAYERS)
    prev_root = init_root

    # Motion decoder neural initialization from WHAM's 20 main SMPL joints.
    pose_jointed = init_pose.reshape(batch, POSE_JOINTS, 6)
    main_pose = pose_jointed[:, MAIN_JOINTS].reshape(batch, len(MAIN_JOINTS) * 6)
    md_state = neural_init(main_pose, w, "motion_decoder.neural_init", CONTEXT_DIM, N_LAYERS)
    prev_pose = init_pose

    kp_frames = []
    body_frames = []
    roots = [init_root]
    vel_frames = []
    contact_frames = []
    shape_frames = []
    camera_frames = []

    for t in range(frames):
        embedded = linear(pose2d[:, t], w, "motion_encoder.embed_layer")
        preds, hidden, me_state = regressor_step(
            embedded,
            [prev_kp],
            me_state,
            w,
            "motion_encoder.regressor",
            1,
        )
        pred_kp = preds[0]
        context = torch.cat([hidden, pred_kp], dim=-1)
        prev_kp = pred_kp

        preds, _, tr_state = regressor_step(
            context,
            [prev_root, cam[:, t]],
            tr_state,
            w,
            "trajectory_decoder.regressor",
            2,
        )
        pred_vel, pred_root = preds
        prev_root = pred_root

        preds, _, md_state = regressor_step(
            context,
            [prev_pose],
            md_state,
            w,
            "motion_decoder.regressor",
            4,
        )
        pred_pose, pred_shape, pred_cam, pred_contact = preds
        prev_pose = pred_pose

        kp_frames.append(pred_kp.reshape(batch, N_JOINTS, 3))
        body_frames.append(pred_pose)
        roots.append(pred_root)
        vel_frames.append(pred_vel)
        contact_frames.append(pred_contact)
        shape_frames.append(pred_shape)
        camera_frames.append(pred_cam)

    return {
        "output.joints_3d": torch.stack(kp_frames, dim=1),
        "output.body_rot6d": torch.stack(body_frames, dim=1),
        "output.root_rot6d": torch.stack(roots, dim=1),
        "output.root_velocity": torch.stack(vel_frames, dim=1),
        "output.contact_logits": torch.stack(contact_frames, dim=1),
        "output.shape": torch.stack(shape_frames, dim=1),
        "output.weak_camera": torch.stack(camera_frames, dim=1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="exported HaricotWHAM Stage-1 safetensors")
    ap.add_argument("--out", required=True, help="output parity-case safetensors")
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    if args.frames < 1:
        raise SystemExit("--frames must be >= 1")

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    weights = {k: v.float().contiguous() for k, v in load_file(args.weights, device="cpu").items()}
    inputs = make_inputs(args.frames)
    outputs = forward(weights, inputs)
    payload = {**inputs, **outputs}
    payload = {k: v.detach().float().contiguous() for k, v in payload.items()}
    save_file(payload, args.out)

    print("=== HARICOT WHAM LITE PYTORCH REFERENCE V0.1 ===")
    print(f"frames={args.frames}")
    print(f"tensor_count={len(payload)}")
    print(f"out={args.out}")
    for name, tensor in payload.items():
        print(f"{name} shape={list(tensor.shape)}")


if __name__ == "__main__":
    main()
