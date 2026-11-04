# Candle Extended standalone publication

Each standalone is prepared on the selected fork `main` with only its own source
delta and explicitly reviewed upstream compatibility changes. Extended composes
these independent prepared commits. After the hosted CPU/CUDA and physical SM61
gates pass, the promotion publishes every prepared standalone and `extended`.
The previous source commits are archived; all branch changes use one atomic push
with exact source leases. `main` is never rewritten by promotion.

`publication.update_canonical_feature_refs` is enabled for new Extended campaigns.
The promoter still respects the immutable policy of historical candidates.
Strictly pinned sources accept successive exact-delta rebases and complete trees
from the registered Zip/Metal migration. Unreviewed source changes remain rejected.

The publication-only workflow repairs the missing standalone publication for
successful qualification 41731722946. It verifies the original run/attempt/jobs,
dedicated physical GPU labels, artifact ZIP digests, hosted executable hashes,
CPU/CUDA/GPU evidence, candidate/main ancestry, isolated prepared commits and all
current leases. It then publishes the 13 original prepared commits without a
compilation, GPU execution, candidate change or rewrite of the original reports.
The original manifest keeps its `false` policy; a separate publication receipt
records the requested policy change and the source qualification.

This publishes standalone sources prepared as part of the qualified Extended
composition. It does not claim a separate physical GPU run of each standalone.
