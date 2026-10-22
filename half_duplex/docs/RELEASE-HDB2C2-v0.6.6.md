# HDB2-C2 v0.6.6 — Rust test constructor fix (E0063)

- Symptom: binaries build in release; `cargo test --release --lib hdb2c_nemo::tests` fails in `src/parakeet/model.rs::time_ratio_positive`.
- Root cause: `cargo test --lib FILTER` still compiles all cfg(test) unit tests. Historical model unit test did not initialize eight new HDB2-C2 `PreprocessArgs` fields and the two HDB2-A `ConformerArgs` additions.
- Correction: generated isolated candidate updates only the matching upstream `time_ratio_positive` constructor literals; explicit values and pin guards, no generic `Default`.
- Existing live runtime preserved; new candidate at `runtime-hdb2c-c2-v0.6.6`; no Python or model reinstall.
- Requires real Rust compilation and actual NeMo/Candle A/B to qualify. `install_update.sh` aborts on compiler/test failure.
- Primary command: `./install_update.sh --report "$HOME/tmp/haricot-half-duplex/reports/hdb2c-20261018T195644Z-87019-000"`.
