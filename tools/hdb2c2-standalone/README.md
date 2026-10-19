# HDB2-C2 standalone CPU parity

Branch: `haricot/candle:hdb2c2_cpu_standalone` (forked from `main`). All modifications are additive; no Candle core source, CUDA, Extended, or orchestrator branches are changed.

## Evidence boundaries

- `pinned-upstream-smoke`: builds/tests the exact public `n0madic/parakeet-candle` source revision `53c0d3065f9dd8de0356b25391e60ed3a471192b`. **A green run does NOT qualify HDB2-C2**.
- `full-cpu-parity`: runs only if the complete source archive `haricot-half-duplex-hdb2c-c2-standalone-v0.6.6.zip` exists in this folder. The script validates its pinned SHA256; installs isolated Python 3.13 NeMo CPU; downloads the original public `qvac/TranscribePsy-250M-EU-sub` NeMo checkpoint; builds the isolated local Candle runtime; converts weights; runs real NeMo and Candle ASR on one public reference WAV; runs the real C2 Rust frontend and numeric comparison. It uploads sanitized JSON reports, not weights, binaries or microphone recordings.

Expected ZIP SHA256: `9eac6aa6ca7f1ed567849aceb9a27baaebc4a7d78d98f235ca101f05076bedb9`.

**Status:** The standalone branch's source ZIP has not yet been committed. It is available in the ChatGPT conversation and user's local machine. Therefore the full C2 CPU qualification job is intentionally SKIPPED. Do not conflate an upstream smoke build with NeMo/Candle parity.

On a new branch, GitHub's `workflow_dispatch` UI generally requires the workflow to exist on the default branch; this workflow instead triggers via push to this standalone branch. No private Bluetooth audio is copied to GitHub.

All test assertions are fail-closed; the successful creation of a report does not imply that numerical or transcription parities have passed. Review `c2_ab_cpu.json` and `c2c3_exact_cpu.json` before promotion.
