"""42 distinct CAMEO excerpts: the original 14 and 28 held-out utterances."""
from collections import Counter
import json
from pathlib import Path

from .c3_r8 import EMOTIONS

DATASET = 'amu-cai/CAMEO'
REVISION = 'c7f2fc9f0538ca871d38e183ba8ad6c329120ecc'


def select_expanded(manifest, *, anchors=None):
    from .audio import describe
    if manifest.get('dataset') != DATASET or manifest.get('revision') != REVISION:
        raise ValueError('R14 requires the pinned real CAMEO corpus')
    rows = manifest['rows']
    expected = {(s, e, i) for s in ('cafe', 'oreau') for e in EMOTIONS for i in range(3)}
    observed = [(r['split'], r['emotion'], r['selection_index']) for r in rows]
    if len(rows) != 42 or set(observed) != expected:
        raise ValueError('R14 requires exactly three distinct clips in each of 14 strata')
    if len({r['pcm_sha256'] for r in rows}) != 42 or len({r['id'] for r in rows}) != 42:
        raise ValueError('R14 refuses duplicate PCM or clip identity')
    if anchors is None:
        path = Path(__file__).resolve().parents[1]/'ci/results/r13-ci-full-cpu-20261021.json'
        anchors = [r['clip'] for r in json.loads(path.read_text())['campaign_summary']['results']]
    fixed = {(r['split'], r['emotion']): r['pcm_sha256'] for r in anchors}
    if len(fixed) != 14:
        raise ValueError('R14 incomplete original regression identities')
    tagged = []
    for row in rows:
        if row.get('dataset') != DATASET or row.get('revision') != REVISION:
            raise ValueError('R14 mixed corpus provenance')
        if describe(Path(row['wav']))['pcm_sha256'] != row['pcm_sha256']:
            raise ValueError('R14 corpus PCM changed: '+row['id'])
        regression = row['selection_index'] == 0
        if regression and row['pcm_sha256'] != fixed[(row['split'], row['emotion'])]:
            raise ValueError('R14 original 14 regression inputs changed')
        tagged.append({**row, 'cohort': 'regression14' if regression else 'additional28'})
    # Finish the original 14 before evaluating the two additional utterances.
    return sorted(tagged, key=lambda r: (r['selection_index'], r['split'], r['emotion']))


def coverage(rows):
    counts = Counter((r['cohort'], r['split'], r['emotion']) for r in rows)
    return {'dataset': DATASET, 'revision': REVISION, 'clips': len(rows),
            'cohorts': dict(Counter(r['cohort'] for r in rows)),
            'strata': [{'cohort': c, 'split': s, 'emotion': e, 'clips': n}
                       for (c, s, e), n in sorted(counts.items())],
            'pcm_sha256': sorted(r['pcm_sha256'] for r in rows),
            'speaker_ids': sorted({r['speaker_id'] for r in rows if r.get('speaker_id')}),
            'generalization_limit': 'additional utterances from the same datasets; not speaker-disjoint or open-domain WER qualification'}
