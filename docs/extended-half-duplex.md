# Half Duplex: ninth standalone Extended component

## Current R15 policy — functional CI at three levels

The default Extended input is `half_duplex_profile=auto`. The assembler records
the resolved choice in the immutable candidate's
`candle-integration/half-duplex-ci.json`. CPU and physical CUDA use this same
choice; users cannot downgrade a candidate requiring a larger campaign.

- `short`: four hash-pinned public clips, original CPU NeMo reference and both
  native R9 decoding paths. No detailed FP64 replay or CPU MKL/oneDNN/SLEEF build.
- `full`: all 42 distinct R14 clips on CPU and CUDA, with encoder intermediate
  diagnostics. Automatically selected when the actual engine/generator/conversion
  or shared core/nn/kernel source diff changes.
- `audit`: explicit complete campaign plus R14 FP64 operator replays on the GPU
  stage. Its legacy numerical qualification remains visible separately.

Exact tokens/text/frames/durations/TDT decisions, correct device/config/lengths,
finite observed tensors, and the original final-encoder limits remain mandatory.
Finite intermediate discrepancies and cross-backend bits are diagnostic, with
no fitted relative tolerances. Missing clips, missing traces or errors fail.
NeMo/PyTorch remain a CPU-only CI reference; the native application is unchanged.

The migration baseline is candidate
`50ed7c1636e93fa5c8bf0a16e90bcce6a5ad7cc1`, measured in run `39815786458`:
42/42 exact CUDA results, 965 exact decisions per execution path, all final
encoder gates pass (worst max_abs `1.564621925354004e-6`), and 14/14 CPU R13
numerical passes. The historical run remains failed/unpromoted for intermediate
thresholds. The user accepted this functional scope before R15 implementation.
After a successful R15 promotion, the last promoted Extended tree is the baseline.

Force the complete campaign or audit using workflow dispatch:

```bash
gh workflow run candle-extended-composite.yml \
  --repo haricot/candle --ref orch/extended-v1 \
  -f campaign="hdb-full-$(date -u +%Y%m%dT%H%M%S)" \
  -f publish=true -f half_duplex_profile=full
```

Use a new campaign name and `half_duplex_profile=audit` for detailed FP64.
The existing local run-bound V6.3 runner remains responsible for physical GPU
execution. Promotion still waits for every CPU, CUDA compile and physical gate.
See the source's `half_duplex/ci/README-R15-CI-PROFILES.md` for details.

## Historical R14 — FP64 operator audit and expanded corpus

The Half Duplex source is pinned to
`7351b3e9695d76cbb7484a6632b290c8b32d6c5c`. It retains R9 native Rust math
and the original numerical gates while adding exact decoding controls and
CPU FP64 replays on observed operands. The integrated CPU profile keeps full
R13 qualification on the original 14 and also exercises the FP64 oracle on
their real inputs. The physical CUDA profile measures those same 14 PCM hashes
plus 28 additional distinct CAMEO utterances, with separate cohort summaries.

Relative tolerances are surveyed element by element; none is automatically
fitted or approved from the GPU outputs. Cross-backend float bits remain
diagnostic. FP64 references are local operator replays, not a full FP64 ASR
reference or an activation substitute. An audit can complete all 42 clips and
still block promotion because original numerical gates remain mandatory.

Local preparation validation: 194 software contracts pass, actual installed
NeMo modules pass 23 FP64 reference replays and 41 candidate-layout replays,
and all 22 generated R9 Rust source files are byte-identical to the previous
candidate. No physical GPU was executed during this preparation.

The user separately confirmed native CUDA microphone transcription outside
Podman/CI with candidate `711776ce60b3558b3f338f90ae7687552f0a0c84`. This is
user-reported functional evidence, not a WER or numerical qualification. The
local R9 command requires `HDB2C6_MASK_MODE=both`, as the CI launcher supplies.
No private recording or transcription is published. See the source's
`half_duplex/ci/README-R14-CUDA-AUDIT.md` for evidence and scope.

## Historical nine-source integration

The user requested hosted preparation/CPU/CUDA compilation now and a later
physical GPU run started with the existing local unified runner.
`half_duplex_standalone` is a main-based source copy of the application on
`half_duplex@79430e051bc6c7a2ac4d10cb101199313b4bb2ff`, without importing its
older Candle numerical implementation into the composed candidate.

Every native STT consumer binds Candle dependencies to the exact prepared or
integrated checkout. Standalone checks run the181 Python contracts and real
native consumer compilation; integrated CPU runs a fresh complete R13
qualification on14 real CaFE/Oréau inputs. Hosted CUDA compiles the native STT
consumer for CUDA12.9.2/cuDNN/SM61 without executing a GPU kernel.

The queued physical Extended job measures the native R9 CUDA/cuDNN path on
all14 clips, keeping NeMo/PyTorch CPU-only. R13's CPU-specific MKL/oneDNN/norm
kernels are not represented as a CUDA port. Original strict gates remain
mandatory; the numerical GPU result must be measured and can block promotion.
STT JSON metrics are preserved separately on both success and failure.

The V6.3 runner's existing candle/extended profile accepts7–32 components and
requires every independent CPU job plus the exact integrated CPU/CUDA proof.
Use its existing commands after the physical job appears in the queue:

```bash
cd "$HOME/tmp/haricot-sm61-runner-v6.3"
./run-once.sh check candle/extended --run-id RUN_ID
./run-once.sh start candle/extended --run-id RUN_ID
```

No runner archive/image/profile change or physical GPU launch is performed by
the hosted preparation. The legacy dedicated Half Duplex GPU route stays
paused. Source contract, source generation, workflow structure and exact-delta
checks precede publication; real CUDA execution remains the user's next step.

The previous campaign promoted EmbeddingGemma2 and Lean4 to canonical refs
without changing their trees. Renew their strict reviewed commit pins from
0b9b993/ d175523 to806a44b/1e19833 respectively; verified tree identities are
955c17eb6d50b6e10b234e1824718ca5c9ed673c and
08e646e0eba20a6258e7833170b19b045314d97a. No source content is changed by
those two pin renewals. Earlier exact-stage Legacy/ASD resolution pins remain
unchanged.

The standalone source carries the same ZIP9.0.3 dependency/API migration as
the other reviewed sources (Cargo.toml, candle-core/src/npy.rs and pickle.rs).
This avoids main's yanked ZIP8.6.0 without changing neural numerical kernels.

## Hosted CUDA toolchain repair — 2026-10-22

Run 39738701484 passed all 9 independent CPU jobs and the integrated CPU gate.
The actual integrated R13 artifact 12410272308 contains 29 JSON files and
confirms 14/14 PASS_ALL_DECLARED_NUMERIC_GATES with unchanged thresholds.
Its ZIP SHA256 is `4b9281ba1e2549ce8e4e7eb9d5f9fd2aaab3013558519b529bccd7b6d4c759c3`.

Hosted CUDA stopped before native STT compilation: the application requests
Rust 1.99.0/rustfmt/clippy, while the compiler image installed only stable 1.99.1.
Rustup attempted installation into root-owned /opt/rustup from the non-root
Docker job, failing with permission denied. The physical GPU job was skipped.

Preinstall the exact 1.99.0 toolchain and both requested components while
building the hosted image. A new image-build preflight uses UID 65534 and an
unreachable Rust distribution server to resolve the same complete toolchain
file and run cargo/rustc/rustfmt/clippy without installation or write access.
Local real Rustup testing confirms all four commands resolve offline against
the application's actual toolchain file; the local host maps only UID 0, so
the non-root check is explicitly performed by the hosted image build.

The model/application sources, numerical thresholds and standalone source
pins remain unchanged. Launch a fresh Extended campaign so its immutable
orchestration checkout uses the repaired image. Old-run retries would retain
the broken recipe; the existing run-bound CPU gates and runner selection
remain enforced for the new campaign. The user's local runner image does
not require this hosted-image repair.
