# WHAM-lite Candle — independent staging branch

**Status: source import pending.** This is an independent branch rooted at the previously GPU-validated Candle SM61 commit `ac983d16750a136a8563def01926ee83a7831d22`. It contains an auditable import contract **but not WHAM source code**.

## Source of truth before import
- Historical full source package: `haricot-wham-lite-candle-v0.3.4-intermediate-parity-r1.zip` (or the original v0.3.4 local-shapes package if later code is unavailable).
- Local checkout historically used: `/home/np/tmp/haricot-wham-lite-candle-v0.1`. The ZIP and weights are local to the project; GitHub currently has no WHAM implementation.
- Historical Candle pin: `cuda_sm61_runner@55e14acc615b4a0850c3d11046348a7e57c6d9de`; do not re-import that historical Candle core.
- Source scope: WHAM stage1 and WHAM frontend audit only, including stateful processing. Do NOT bundle RTMPose or MotionBricks runtime into this feature branch.
- Input contract for isolated profiling: `precomputed_pose2d_no_yolo_no_rtmpose`.
- External weights only: `haricot-wham-stage1.safetensors`. Do not commit the model weights or private intermediate traces.
- Historical `--stream-benchmark` result was baseline-only: 3 epochs PASS but drift across epochs; Nsight census was HOLD. It does **not** certify runtime performance on the new Candle base.
- Output interface target: `HaricotMotionFrame { joints, velocities, accelerations, confidence, ground, body_basis }`. The interface is a planned adapter, not evidence of an existing implementation in this branch.

## Promotion prerequisites
1. Import and hash a complete WHAM source tree from the local archive, extracting MotionBricks to its own branch.
2. Replace historical Candle Git/path pins with this branch's local Candle libraries (without copying old Candle internals).
3. Validate offline CPU behavior and parity against local reference; capture exact source/weights fingerprints.
4. Hosted CUDA 12.9.2 / Ubuntu 26.04 compile, then deliberate physical SM61 run if quality gates pass.
5. Enable source_imported in EXTENDED_FEATURE.json only after source files and manifest are actually committed.
