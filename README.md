# WHAM-lite stage1 — ASD V2 migration

Sources: user-provided haricot-wham-lite-v034-candle-asd-r1.zip. No model weights are distributed. Package contains only WHAM stage1 (LSTM/linear), Body26→COCO17 adapter, HaricotMotionFrame and WHAM export scripts. Historical MotionBricks applications are excluded.

Candle baseline: cuda_asd_runner_v2@ac983d16750a136a8563def01926ee83a7831d22; nested package links ../candle-core and ../candle-nn, with no old Git/path pin.

ASD V2 is the underlying Candle dispatcher. WHAM stage1 has NO Conv1D: existing ASD V2 exact grouped/transpose decisions do not demonstrate WHAM acceleration. No optimization is promoted here. Validate CPU compile, hosted CUDA 12.9.2/cuDNN 9.10.2.21 compile, then actual WHAM weights+pose fixtures with independent Torch oracle. Only then introduce dedicated GPU tests and benchmarking.
