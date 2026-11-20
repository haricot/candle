# R1.24 S5.1 — Sealed catalogue and residual hardware audit

Reference: qualified S4.4 archive at production baseline d1929bea2a57beb82d466c9654c1d5f5d6ff616c.
The sorted compact JSON map family -> PTX SHA-256 from the successful S4.4
log has SHA-256 619c8fc36501aa3e5268ebcde5976b8b4a50b7bbdd7b651cb8877ac4e7399805.

S5.1 is a read-only qualification step, not a claim of full architecture neutrality.
The local seal records PTX and manifest hashes, entry inventories, entry maps,
optional index hash, source revision, archive and evidence hashes. Manifest/index
file hashes are newly recorded, not inferred from previous PTX-only evidence.
It is a reproducibility fingerprint, not a signed attestation.

| Classification | Residual coupling |
|---|---|
| Active recipe | NativeRecipe::S31NativeV1 maps to frozen FP8/FP4/BF16/routed Rust descriptors; embedded admission compares against them. |
| Active recipe | MMQ geometry and shared memory formula remain in dkr_native_catalog.rs and dkr_s31_contract.rs. |
| Build/schema adapter | Legacy validator requires S31 source revision, flags, target and entry maps. Preserve this for S31; separate adapter from generic consumer. |
| Static compatibility | Legacy launcher/recipe paths remain for non-strict builds; strict admission never allows missing provider. |
| Optional runtime | cuDNN failure policy still tests compute capability; not part of current strict feature qualification. |
| Generic | Device target admission and CUDA limits are valid generic safety checks; do not remove them. |
| Historical labels | sm61 in diagnostic/custom-op names is not itself a dispatch condition. |

The automated source audit records anchored findings and lexical candidates.
It is not a proof of call-graph reachability or a complete audit of every
possible feature combination. No new device is qualified.

Use headless-relogin after compiling/archiving outside the graphical shutdown.
The S5.1 headless replay uses the existing archive, samples device initialization
three times, records GPU state, and does not invoke a Cargo build. It is a smoke
test, not a cold CUDA JIT benchmark. S5.3 must isolate JIT cache, thermal state,
warmup, multiple rounds and kernel timing from process/driver initialization.
No kernel source, package, engine production code or Mistral pins change here.

Follow-up architecture gate: move fixed S31 descriptors and MMQ geometry to
validated catalogue data / package-side adapter, keeping generic checked plan
evaluation in Candle and the exact qualified artifact identities. Do not declare
neutrality closed merely because static PTX and FFI compilation are absent.
