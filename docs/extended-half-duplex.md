# Half Duplex: ninth standalone Extended component

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
