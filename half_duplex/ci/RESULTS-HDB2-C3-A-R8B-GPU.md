# R8B — completed physical sm61 measurement

Source: `8e827fadddfb6b5e555289836728aee52c6b9bb1`.
[GPU run 39598582574, job 119216990721](https://github.com/haricot/candle/actions/runs/39598582574/job/119216990721)
completed all 14 real CaFE/Oreau clips without execution errors. The strict
ASR qualification fails; successful execution is not numerical qualification.

Physical host: NVIDIA GTX 1080 Max-Q, compute capability 6.1, driver
580.178.04, CUDA 12.9.2 and cuDNN 9.10.2.21. The native candidate uses real
CUDA/cuDNN; NeMo remains the independent CPU reference. Native frontend FFT
and separate primitive replays use host arithmetic, as declared by R8.

Artifact: `half-duplex-r8-cuda-sm61-reports`, ID `12346604616`, 359392 bytes.
ZIP SHA256: `7d2651eb672a2c84ee179b44cdc62f79f23481fd8b86f2b7d3254c0b1541fc18`.

| Check | Observed result |
| --- | --- |
| R7/R8B same-backend byte controls | pass, 14/14 |
| Frontend numerical gates | pass, 14/14 |
| Five convolutions and three ReLUs on identical inputs | pass, 112/112 |
| Final linear on identical inputs | fail, 14/14; worst max_abs 0.0068359375 |
| Valid final encoder numerical gate | pass, 14/14; worst max_abs 9.462237358093262e-7 |
| Discrete TDT decisions | exact, all 283 decisions |
| Exact final text | 12/14; known unknown-token rendering mismatch |
| Complete internal-stage numerical qualification | not established |
| Global FP32 bit equality | fail, 14/14 |

Autonomous pre_encode reaches max_abs 0.00732421875, amplified to 0.185546875
after scaling. Shared-Mel still fails pre_encode on 14/14, worst
0.00634765625. These measurements are retained for later comparison, not used
to start another GPU campaign.

R9 subsequently corrects text on CPU without changing neural arithmetic; it
has not been measured on GPU. The user now requires all CPU numerical gates
to pass before any further GPU test. The GPU workflow is manual-only and no
replacement GPU run has been requested or queued.
