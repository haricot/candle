# Kimodo S51-B2 — real checkpoint parity (candidate)

S51-B1: user reported 3/3 CPU/DDIM/CUDA synthetic smoke tests PASS on Pascal.
B2 introduces a real-weight oracle capture and full-array tests. No Python runtime
in Candle. No change to the model implementation, CUDA kernels, Extended or DKR.
No CUDA architecture environment override is needed on the local GTX 1080 Max-Q.

Capture uses exact HF model revision and exact NVIDIA source commit. The Python
environment must provide the upstream requirements and PyTorch CPU.
It bypasses the LLM2Vec encoder entirely; the fixed nonzero embeddings and sparse
normalized observations are numerical fixtures, not a Macarena choreography.
Backbone weights are exported F32 from the actual loaded official checkpoint.
Expect around another checkpoint's worth of local disk storage. No weights in Git.

See tools/kimodo/capture_b2.py --help. Determine model revision with
huggingface_hub.HfApi().model_info("nvidia/Kimodo-SOMA-RP-v1.2").sha.
The capture refuses overwrite and saves hashes in the manifest; check them before
running parity if copying the files between machines.

From the Candle repo, with KIMODO_FIXTURE pointing to the resulting directory:

    cargo nextest run -p candle-transformers --test kimodo_real --release --features cuda --run-ignored all --test-threads 1 -E 'test(kimodo_real_cpu) | test(kimodo_real_cuda)' --no-capture

The independent root/body, root bridge, teacher-forced CFG/DDIM and full chained
sampling checks report max abs error, RMSE and element-level tolerance violations.
Any violation fails the test. Initial thresholds are explicit engineering gates,
not measured or claimed upstream equivalence. B2 needs actual execution.
