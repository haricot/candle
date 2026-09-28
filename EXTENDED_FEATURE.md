# Standalone RTMPose Body26 feature branch

- Branch: `rtmpose_body26_standalone`
- Validated foundation: `cuda_asd_runner_v2@ac983d16750a136a8563def01926ee83a7831d22`
- RTMPose-only source donor: `rtmpose_body26_candle@f6a3e36a219e38687f121c45d769a8fe28aeb64b`, directory `haricot-pose-rtmpose-m-body26-candle/`.
- Only the model crate, converter, contract and notices are transplanted; **no** historical Candle core, cuDNN or CUDA code is copied from the donor branch.
- Cargo dependency overrides are **local** (`../candle-core`, `../candle-nn`) and this nested package is its own workspace.
- Intended input: RGB top-down ROI, RTMPose-m Halpe26 256×192; output: 26-keypoint SimCC (X 26×384, Y 26×512).
- Detector, Kinect acquisition, WHAM and MotionBricks are **not** included in this feature branch.
- Historical weights and measurements come from the donor's former Candle revision and are **not** revalidated on this branch merely by transplantation.

Validation gates before promotion:
1. `cargo check --manifest-path haricot-pose-rtmpose-m-body26-candle/Cargo.toml --no-default-features` (CPU).
2. Hosted CUDA 12.9.2 / Ubuntu 26.04 compile with `--features 'cuda cudnn'` and `CUDA_COMPUTE_CAP=61`.
3. Model-contract and weight-conversion audits; a real SM61 numerical/latency check requires separately provisioned checkpoint and image. Never report performance PASS without these external assets.
