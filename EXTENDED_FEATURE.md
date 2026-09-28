# MotionBricks Candle — independent staging branch

**Status: source import pending.** This branch derives only from validated Candle `ac983d16750a136a8563def01926ee83a7831d22`; it does **not** include RTMPose or WHAM stage1 code.

## Source of truth before import
- The MotionBricks runtime is historically packaged within `haricot-wham-lite-candle-v0.3.4-intermediate-parity-r1.zip`; a standalone MotionBricks source ZIP was not located.
- Extract runtime code and its direct model dependencies only. Keep a clear interface for a WHAM-derived `HaricotMotionFrame`, G1 retarget, fixed16 and optional 418/414/413 block path; verify source before treating those optional paths as implemented.
- Historical numerical gate: `fixed16-no-text-all-root-constraints`; CPU/CUDA model-level Root/Pose/VQ-VAE and Torch oracle PASS. Captures observed 42 Conv1D invocations, 15 shapes, groups=1, two non-contiguous input occurrences, no ConvTranspose.
- **No isolated-operator parity or performance proof was established** by the 42 model-level trace events; `dispatch_eligible=false` and `numeric_parity_validated=false` in the shape catalog.
- External assets, not to commit: `motionbricks-runtime-v1.safetensors`, `motionbricks-runtime-reference-r2-fix1.safetensors`, optional other historical model assets.
- Keep MotionBricks isolated from upstream RTMPose; fixture-only pose/motion input is sufficient for initial CPU acceptance.

## Promotion prerequisites
1. Import the actual extracted runtime sources and commit source hash manifest.
2. Local Candle-only path dependencies on this verified base.
3. CPU fixed16 model-level parity, CUDA 12.9.2 compilation, then physical SM61 model-level parity with protected local weights/reference.
4. A separate isolated Conv1D oracle and stable benchmark before any ASD dispatch rule promotion.
5. Change EXTENDED_FEATURE.json source_imported only when executable runtime and build manifest exist.
