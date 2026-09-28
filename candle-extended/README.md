# Candle ASD runner Extended — imported and isolated source candidates

Validated six-feature foundation (unmodified): `cuda_asd_runner_v2@ac983d16750a136a8563def01926ee83a7831d22`.

These branches now contain **actual standalone sources**, imported from the user-supplied `haricot-wham-lite-v034-candle-asd-r1.zip`, and are independently versioned. They are NOT merged together here.

- `rtmpose_body26_standalone@6e729d2fc1a1558e322abf89295ab3d93915bcb4`: RTMPose-m Halpe26 implementation from its older branch; no obsolete Candle core modifications imported.
- `wham_lite_candle@261eaf084d2e3fab140933ada5f88fe3b10efdde`: WHAM stage1 recurrent model, Body26→COCO17 adapter, and HaricotMotionFrame conversion; **no MotionBricks runtime**.
- `motionbricks_candle@a5d60c5aea6167d5c86130ee2153aaf0ffafd326`: MotionBricks fixed16 decoder/root/pose, G1 418-D bridge, Torch reference tools; **no WHAM stage1 model**. Legacy `candle_nn::asd_conv1d` call is removed and old cuDNN 9.1.0.2 routes fail closed. `--asd-v2-audit` exercises only baseline Candle Conv1D with unchanged Torch numerical tolerances.

All three branches descend from exactly the same CUDA 12.9.2 / Ubuntu 26.04 / SM61 physically validated foundation; none has independently completed its new-base CPU+CUDA compile or new physical model-level numerical acceptance.

Use `mirror_orch:.github/workflows/candle-asd-extended.yml` manually, binding `feature`, exact `candidate_sha`, `validation=audit|cpu|cpu_cuda`. It verifies the source inventory for WHAM/MotionBricks and compiles in GitHub-hosted jobs. It does **not** claim GPU execution, route performance, or commit promotion.

Once the compile gates pass: run WHAM and MotionBricks fixture-level numerical oracles. MotionBricks requires new SM61 CUDA + cuDNN 9.10.2.21 isolated same-input Conv1D parity and model-level A/B/A before proposing any per-signature ASD V2 production rule. This manifest deliberately remains `ready_for_merge=false`, `ready_for_gpu=false`, `auto_promote=false` until those proofs exist.

The original WHAM/MotionBricks code, adapted standalone snapshots and file-hash inventories are also saved as `haricot-wham-motionbricks-asd-v2-source.zip` in the user's Library.
