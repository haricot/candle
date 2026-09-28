# Candle ASD Extended — isolated features

Foundation: `cuda_asd_runner_v2@ac983d16750a136a8563def01926ee83a7831d22` (six-function physically GPU-validated, untouched).

| Branch | Source state | Exact initial candidate SHA |
|---|---|---|
| `rtmpose_body26_standalone` | RTMPose-m Halpe26 model crate, converter and contract imported from old RTMPose branch; **old Candle core changes excluded** | `6e729d2fc1a1558e322abf89295ab3d93915bcb4` |
| `wham_lite_candle` | Import manifest only; source ZIP needed | `26f66481b0d82077092113866d70ffeb0214cfbb` |
| `motionbricks_candle` | Import manifest only; extract runtime from historical WHAM ZIP | `bd59f206698d3a8108570343a31d82edb6343fb5` |

Run the opt-in `.github/workflows/candle-asd-extended.yml` on `mirror_orch` with `feature`, the EXACT live feature-branch `candidate_sha`, and `validation=audit|cpu|cpu_cuda`.

For RTMPose, CPU and CUDA steps check **compilation only**. Historical performance and numerical parity refer to the donor's previous Candle revision and must be verified afresh with external model weights/ROI fixtures before any physical GPU campaign or promotion. The CPU step creates an ephemeral Cargo.lock and uploads it; pin a committed lock before demanding a fully reproducible release.

For WHAM/MotionBricks, `audit` reports `SOURCE_REQUIRED`. CPU/CUDA builds fail closed until actual source trees are imported and their independent adapters configured. Historical motion-level CPU/CUDA parity and 42 Conv1D intermediate events do not prove isolated operator parity or performance.

This workflow never launches a physical self-hosted GPU runner, promotes any branch, or triggers cleanup. Add an extension manifest and a separate explicit GPU gate only when the independently imported components pass.
