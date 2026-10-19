# Dataset and implementation status — HDB2-C2 CPU

**Benchmark source:** [amu-cai/CAMEO](https://huggingface.co/datasets/amu-cai/CAMEO), immutable revision `c7f2fc9f0538ca871d38e183ba8ad6c329120ecc`.

- `cafe`: 936 French emotional utterances, CC BY-NC-SA 4.0.
- `oreau`: 502 French emotional utterances, CC BY 4.0.
- Use a fixed quota of one example per emotion (7 × 2 corpora = 14). Write a WAV and the actual orthographic transcription for each. Every C1 comparison supplies `--reference-file`; C2 numerical probes use one example from each corpus. Do not republish audio or weights in CI artifacts. All WER/CER refer to the corpus annotation, not a synthetic reference.
- FireRedVAD and Smart Turn need separate end-of-turn tests. CaFE/CAMEO contain individual emotional utterances, **not annotated natural conversation turn-completion ground truth**. Silence inserted between clips may test VAD state transitions but cannot qualify Smart Turn semantic decisions.
- **Publication gap:** GitHub currently contains this CI harness and dataset sampler, but not the complete HDB2-C2 v0.6.6 source archive. The archive has the Rust NeMo frontend (`rust/hdb2c_nemo.rs`), the TranscribePsy NeMo-to-Candle port (`scripts/convert_nemo_hdb2a.py`, `scripts/prepare_candle_runtime.py`), and the Python FireRedVAD/Smart Turn ONNX integration (`hdb1/{firered_onnx,smart_turn,pipeline}.py`). The complete C2 job remains deliberately skipped until these sources are committed and their SHA256 validated.
- A successful `pinned-upstream-smoke` job means only **public upstream Parakeet compiles**. It does **not** validate the user's TranscribePsy port, VAD, or Smart Turn.

External model/dataset licenses apply; no private audio should be uploaded.
