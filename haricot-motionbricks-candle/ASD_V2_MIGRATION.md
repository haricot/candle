# MotionBricks fixed16 / ASD V2 migration

Sources: user-provided haricot-wham-lite-v034-candle-asd-r1.zip. Standalone runtime, G1 bridge, parity, tools; no WHAM stage1 model. HaricotMotionFrame is a serialization-compatible struct subset to decouple independent branches.

Candle baseline: cuda_asd_runner_v2@ac983d16750a136a8563def01926ee83a7831d22; nested package links ../candle-core and ../candle-nn, not the former local v0.1 overlay. The former candle_nn::asd_conv1d module does not exist on the validated v2 base. The r1 MotionBricks policy was tested at cuDNN 9.1.0.2; this build targets cuDNN 9.10.2.21, so all Conv1D operations use baseline Candle and legacy CLI selection flags FAIL CLOSED.

New --asd-v2-audit is an opt-in CUDA-only Torch numerical gate (max_abs=5e-4, rmse=1e-4) and does NOT select a faster kernel. Historical MotionBricks ASD policy retained uncompiled as motionbricks_asd_r1_read_only.rs. Old K1 GEMM / K3 cuDNN Direct code remains for *offline review only*; no production path can select it through the CLI/pipeline.

Next isolated campaign: CPU+hosted CUDA compile; actual model parity with local runtime/reference safetensors; capture 42 Conv1D module invocations (15 signatures; two non-contiguous input occurrences), then isolated same-input operator oracle, cuDNN 9.10.2.21 microbenchmark and A/B/A under no GPU contention. An independent V2 exact-signature implementation requires a separately reviewed source commit and physical SM61 proof; no reuse of 9.1.0.2 performance claims.
