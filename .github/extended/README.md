# Candle ASD Extended — isolated features

Foundation: `cuda_asd_runner_v2@ac983d16750a136a8563def01926ee83a7831d22` (six-function physically GPU-validated, untouched).

| Branch | Source state | Exact initial candidate SHA |
|---|---|---|
| `rtmpose_body26_standalone` | RTMPose-m Halpe26 model crate, converter and contract imported from old RTMPose branch; **old Candle core changes excluded** | `6e729d2fc1a1558e322abf89295ab3d93915bcb4` |
| `wham_lite_candle` | Import manifest only; source ZIP needed | `26f66481b0d82077092113866d70ffeb0214cfbb` |
| `motionbricks_candle` | Import manifest only; extract runtime from historical WHAM ZIP | `bd59f206698d3a8108570343a31d82edb6343fb5` |

The historical opt-in `candle-asd-extended.yml` workflow has been retired from
the Unified default branch in R3.2. Do not mistake these old candidate SHAs for
qualified Unified sources. New ASD integrations must first be added to the
manifest-driven Extended/Unified composition, then pass its independent CPU,
CUDA, and physical SM61 gates before promotion.

For RTMPose, the retired CPU and CUDA steps checked **compilation only**. Historical performance and numerical parity refer to the donor's previous Candle revision and must be verified afresh with external model weights/ROI fixtures before any physical GPU campaign or promotion. The old CPU step created an ephemeral Cargo.lock and uploaded it; pin a committed lock before demanding a fully reproducible release.

For WHAM/MotionBricks, the retired `audit` reported `SOURCE_REQUIRED`. CPU/CUDA builds must continue to fail closed until actual source trees are imported and their independent adapters configured. Historical motion-level CPU/CUDA parity and 42 Conv1D intermediate events do not prove isolated operator parity or performance.

The old isolated compilation checks never constituted physical GPU qualification.
Preserve the historical evidence, and add a manifest and explicit end-to-end
qualification only after the imported components pass.
