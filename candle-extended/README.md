# Candle ASD runner Extended — staged integration manifest

This branch starts at the **physically GPU-validated six-feature** `cuda_asd_runner_v2@ac983d16750a136a8563def01926ee83a7831d22`. It is an **integration staging branch**, not a combined RTMPose/WHAM/MotionBricks implementation and not a new GPU-certified target.

`candle-extended/sources.json` pins the three **separate** feature branch SHAs and truthfully marks import/validation states. RTMPose has source imported but is not revalidated on the new base; WHAM and MotionBricks branches contain source-import contracts only, no executable model sources.

The opt-in `mirror_orch:.github/workflows/candle-asd-extended.yml` audits each exact ref and can compile RTMPose CPU and hosted CUDA. It does not attempt physical GPU execution or automatic promotion. Preserve the existing six-source integration manifest, old branches, model weights and local traces.

Only after all three source-import and independent numerical/compile gates pass: prepare a NEW merge proposal and source inventory (3 exact feature commits), check Rust/Candle lockfile and tensor interface compatibility, review the proposed aggregate, run fresh CPU/CUDA Candidate CI, and *then* arrange a new explicit SM61 GPU test. Do not amend the six-source promotion evidence or claim a PASS from source stubs.
