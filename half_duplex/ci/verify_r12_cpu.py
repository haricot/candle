"""Gate the isolated CUDA replay on actual CPU bitwise evidence and source identity."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = ('hdb2c/c3_r12_layernorm.py', 'hdb2c/layernorm_operator.py', 'ci/r12_layernorm.sh')
SOURCE = 'ci/experimental/cpu_parity/ordered_layernorm.cpp'


def verify(report):
    expected = {'scope': 'BLOCK0_NORM_FF1_COMMON_INPUT_ONLY_NOT_FULL_STT',
                'backend_requested': 'cpu', 'execution_status': 'COMPLETE',
                'qualification': 'PASS_BITWISE_14_COMMON_INPUTS',
                'expected_clips': 14, 'completed_clips': 14, 'cpu_bitwise_passed': 14,
                'gpu_execution': 'NOT_RUN', 'model_operator_replaced': False, 'errors': []}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError('R12 CPU qualification differs: '+key)
    reference = report['metadata']['reference']
    checkpoint = json.loads((ROOT/'ci/r7_verified_native_sources.json').read_text())['checkpoint_sha256']
    if (reference['checkpoint_sha256'] != checkpoint or reference['torch'] != '2.12.1+cpu'
            or reference['nemo_toolkit'] != '2.7.3'):
        raise ValueError('R12 CPU checkpoint/reference version differs')
    for name in CORE:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != report['metadata']['source_sha256'][name]:
            raise ValueError('R12 CPU-qualified core source changed: '+name)
    if hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest() != report['metadata']['cpu']['source_sha256']:
        raise ValueError('R12 CPU-qualified native LayerNorm source changed')
    rows = report['results']
    emotions = ('anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise')
    expected_pairs = {(split, emotion) for split in ('cafe', 'oreau') for emotion in emotions}
    if (len(rows) != 14 or len({r['clip']['pcm_sha256'] for r in rows}) != 14
            or {(r['clip']['split'], r['clip']['emotion']) for r in rows} != expected_pairs):
        raise ValueError('R12 CPU requires the complete distinct 14-clip corpus')
    for row in rows:
        for field in ('instrumentation_control', 'cpu_vs_nemo'):
            if row[field]['bitwise_equal'] is not True or row[field]['mismatching_f32_words'] != 0:
                raise ValueError('R12 CPU common-input bitwise control failed')
        if row['epsilon'] != 1e-5 or row['lanes'] != 8:
            raise ValueError('R12 CPU normalization configuration changed')
    return {'status': 'PASS_ACTUAL_CPU_14_AND_UNCHANGED_OPERATOR', 'checkpoint_sha256': checkpoint,
            'scope': expected['scope']}


if __name__ == '__main__':
    print(json.dumps(verify(json.loads(Path(sys.argv[1]).read_text()))))
