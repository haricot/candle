# HaricotPose — native RTMPose-m Body26 for Candle

Native Rust/Candle implementation of **RTMPose-m Halpe26 256×192**, developed for HaricotPose and validated on NVIDIA Pascal/sm61.

## Model

- RTMPose-m / Halpe26 (26 keypoints)
- CSPNeXt-P5 medium backbone
- RTMCC + SimCC head
- RGB NCHW `B×3×256×192`
- SimCC X `[B,26,384]`, Y `[B,26,512]`
- single-pass inference (no flip TTA)
- no ONNX Runtime / TFLite dependency at runtime

Raw SimCC peaks use the official `normalize=false` semantics and are **not probabilities**.

## Candle base

The standalone crate is pinned to:

```text
haricot/candle
branch: cuda_sm61_grouped_cudnn_probe
commit: 19e061d1c9615a21cb3a60870a6c2a746f2e985b
```

The grouped-cuDNN implementation is being prepared separately for upstream review. The model code itself does not depend on a detector and accepts one or more top-down person ROIs.

## Performance evidence — GTX 1080 Max-Q / sm61

The original Candle grouped-convolution implementation decomposed `groups=N` into `N` single-group convolutions plus concatenation. A layer audit showed RTMPose depthwise 5×5 convolutions consuming about 94.5% of the measured forward time.

Native cuDNN grouped convolution using `cudnnSetConvolutionGroupCount` changed the RTMPose B1 F32 result from approximately:

```text
~503.7 ms median  →  25.996 ms median
                     30.042 ms p95
                     38.47 poses/s
```

The isolated grouped-convolution validation produced exact F32 parity (`max_abs=0`, `max_rel=0`) and these measured speedups on the same GPU:

```text
groups=2                      2.762x
groups=4                      3.413x
groups=8, batch=3             7.041x
depthwise C=48               42.996x
depthwise C=96, batch=3      89.066x
depthwise C=384             402.470x
```

An earlier isolated C=384 run measured 445.265x. These are workload- and GPU-specific measurements, not a claim that all Candle workloads improve by those factors.

## Build

```text
export CUDA_COMPUTE_CAP=61
export ALLOW_LEGACY=bf16,fp8

cargo run --release \
  --features "cuda cudnn" \
  --bin haricot-rtmpose-body26 -- \
  --weights rtmpose-m-halpe26-candle-f32.safetensors \
  --image person.png \
  --warmup 10 \
  --runs 20 \
  --batch 1
```

CPU F32 is also supported with `--cpu`.

## Weights

The official OpenMMLab checkpoint is not redistributed. Convert it once with:

```text
python tools/convert_openmmlab_pth.py \
  --download-official \
  --output rtmpose-m-halpe26-candle-f32.safetensors
```

Then audit the converted Safetensors:

```text
python tools/audit_safetensors.py rtmpose-m-halpe26-candle-f32.safetensors
```

The converter folds Conv+SyncBN pairs into inference-ready convolution weights/biases. Expected fused convolution count: 57.

## Haricot integration target

```text
RGB/Kinect frame
      ↓
person detector / track
      ↓
SAFE TRACK ROI
      ↓
1..3 ROI batch
      ↓
RTMPose-m Body26 Candle
      ↓
HaricotPoseFrame
      ↓
Kinect depth fusion
      ↓
HaricotMotionFrame
```

The downstream Haricot motion/scoring/rendering layers remain independent of the pose backend and joint-count implementation.
