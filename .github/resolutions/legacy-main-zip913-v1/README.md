# Reviewed current-main compatibility

Main: `cfcbd7186d13271843ce7eb41d25659bd0835850`.

The manifest freezes each complete source patch SHA256, every conflict-stage blob and the entire prepared Git tree. Unlisted sources, moved main, additional conflicts or changed kernels are rejected. Original and prepared patch hashes remain separate in campaign evidence.

Keep upstream Zip 9.1.3 and its NPY/Pickle APIs. Keep every feature change outside those obsolete compatibility edits. Extended also retains ASD sha2 and grouped-transpose exports together with upstream Metal MoE exports. Clean voice source migrations change only full-index patch blob headers.

Aggregate resolutions retain upstream package version 0.11.1 and Metal indexed MoE while preserving existing CUDA kernel resolutions. Source branches remain unchanged until the existing physical validation and atomic promotion gates.

`legacy-main-selftest.py` reproduces original inputs, verifies the full reviewed trees, rejects changed main/source/kernel and replays the five-source Legacy union with its exact-stage guards.
