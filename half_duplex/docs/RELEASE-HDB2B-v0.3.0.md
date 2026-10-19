# HDB2-B v0.3.0 — standalone CUDA/Pascal evidence gate

New files: `run-hdb2b.sh`, `scripts/hdb2b_runtime_patch.py`,
`scripts/hdb2b_driver.py`, `tests/hdb2b/test_hdb2b.py`.

Baseline HDB2-A source preserved, all model assets excluded. HDB2-B patches
only the already cloned pinned runtime and builds variant-specific binaries;
no mutation of haricot/candle or haricot/mistral.rs.

Performance is measured on the user's computer only. The supplied ZIP has
no observed CUDA timings or performance wins. FP16 is not yet enabled.
