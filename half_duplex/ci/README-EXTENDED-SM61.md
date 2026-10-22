# Half Duplex standalone in Candle Extended

`half_duplex_standalone` is based on fork `main` and contains the complete
Half Duplex source from `half_duplex@79430e051bc6c7a2ac4d10cb101199313b4bb2ff`.
It adds only its application folder and the two existing application workflows;
it does not import an older Candle core into Extended. The old dedicated GPU
workflow/launcher remain paused. The explicitly requested GPU route is the
manifest-driven `Candle Extended` workflow on `orch/extended-v1`.

The independent source CPU gate runs the existing Python contracts and compiles
the actual native R9 STT consumer against its exact prepared Candle tree.
The integrated CPU gate reruns the full R13 qualification on all14 real clips,
again using the exact integrated Candle source. The CPU result previously
measured against the pinned historical dependency is retained as historical
evidence, not substituted for this new integrated CPU gate.

Hosted CUDA compilation includes the native STT consumer with CUDA12.9.2,
cuDNN and target `sm_61`, without executing a GPU kernel. A successful build
enqueues `SM61 Extended runtime` with label `extended-RUN_ID`.

The physical job checks the complete Extended composition and runs native
R9 CUDA/cuDNN inference, the independent R7/R8B controls and the original
numerical/text/decision gates on all14 clips. The original NeMo/PyTorch
reference stays CPU-only. Official PyTorch CUDA support for Pascal is therefore
not required. No CUDA-to-CPU fallback is accepted in the native model.

The CPU-specific MKL/oneDNN/ordered-norm kernels of R13 are not a CUDA port.
This GPU campaign measures the separate native CUDA path, including the R9
tokenizer correction. Its numerical result remains to be established on the
actual card; unchanged strict gates may fail and block promotion. Build success
and exact text do not by themselves establish full numerical or bitwise parity.
Metrics are uploaded even when numerical qualification fails.

After hosted jobs succeed and `SM61 Extended runtime` is waiting, use the
existing V6.3 unified runner on the laptop:

```bash
cd "$HOME/tmp/haricot-sm61-runner-v6.3"
./run-once.sh check candle/extended --run-id RUN_ID
./run-once.sh start candle/extended --run-id RUN_ID
```

The existing rootless image, shared GPU lock, ASD store and ephemeral token
prompt are reused. No change to the runner archive/image/profiles is needed.
The runner validates every hosted gate, all9 source components, the immutable
candidate SHA and the exact queued job before registration. The user starts
the runner; the assistant does not start a process on the physical laptop.

Artifacts are separated into independent source CPU proofs, combined CPU
proof, `*-half-duplex-cpu`, hosted CUDA proof, physical SM61 proof and
`*-half-duplex-gpu`. Only JSON metrics are uploaded for the STT corpus.
Microphone/streaming tests remain separate. Extended's existing atomic
promotion is reachable only after the actual physical runtime job succeeds.
