# Third-party notices

This branch contains an independent Rust/Candle implementation of the RTMPose-m Body26 inference architecture.

## OpenMMLab MMPose / MMDetection

Architecture, preprocessing, RTMCC/SimCC behavior, Halpe26 metadata and checkpoint naming were derived from the public OpenMMLab MMPose and MMDetection projects, distributed under Apache-2.0.

- MMPose: https://github.com/open-mmlab/mmpose
- MMDetection: https://github.com/open-mmlab/mmdetection

The OpenMMLab checkpoint is **not redistributed**. `tools/convert_openmmlab_pth.py` downloads it from OpenMMLab's official model hosting only when explicitly requested.

## Candle

The runtime targets the Haricot Candle fork at the exact commit recorded in `CANDLE_BASE.txt`. Candle is licensed under MIT OR Apache-2.0.
