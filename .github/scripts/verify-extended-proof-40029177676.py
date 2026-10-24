#!/usr/bin/env python3
"""Exact evidence verification for the user-authorized 40029177676 finalization."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def verify(root, request):
    for artifact in request['artifacts']:
        path = root / (artifact['kind'] + '.zip')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256'], 'artifact digest'
        target = root / artifact['kind']
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
            z.extractall(target)
    cpu = json.loads((root / 'cpu/report.json').read_text())
    cuda = json.loads((root / 'cuda/report.json').read_text())
    assert cpu['status'] == 'CPU_PASSED' and cuda['status'] == 'CUDA_COMPILE_PASSED'
    for result in (cpu, cuda):
        assert result['candidate_sha'] == request['candidate_sha']
        assert result['campaign'] == request['campaign']
    assert cuda['target'] == 'sm_61' and cuda['exit_code'] == 0
    assert cpu['lock_sha256'] == cuda['lock_sha256']
    assert hashlib.sha256((root / 'cpu/Cargo.lock').read_bytes()).hexdigest() == cpu['lock_sha256']
    assert cpu['manifest_sha256'] == hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['feature_count'] == 12 and len(manifest['features']) == 12
    assert manifest['main_sha'] == request['main_sha'] and manifest['campaign'] == request['campaign']
    assert manifest['old_integration_sha'] == request['old_extended_sha']
    assert sorted(cpu['feature_scope']) == sorted(f['feature'] for f in manifest['features'])
    assert [{'branch': f['target_ref'], 'old_sha': f['source_sha'], 'new_sha': f['prepared_sha']}
            for f in manifest['features']] == request['feature_targets']
    summary = (root / 'gpu/summary.txt').read_text().splitlines()
    for expected in [
        'GATE_nemotron_asr_cuda_real_parity=PASS',
        'EXTENDED_SM61_CANDIDATE=' + request['candidate_sha'],
        'EXTENDED_SM61_CAMPAIGN=' + request['campaign'],
        'EXTENDED_SM61_VALIDATION=PASS',
    ]:
        assert expected in summary, expected
    fields = ['wav_sha256', 'mode', 'language', 'lookahead', 'samples', 'sample_rate',
              'mel_frames', 'encoder_frames', 'text', 'tokens', 'decisions']
    results = []
    for lang in ['fr-FR', 'auto']:
        base = root / 'gpu/nemotron-real-parity'
        a = json.loads((base / ('cpu-' + lang) / 'result.json').read_text())
        b = json.loads((base / ('cuda-cudnn-' + lang) / 'result.json').read_text())
        assert a['device'] == 'Cpu' and b['device'].startswith('Cuda(')
        assert a['dtype'] == b['dtype'] == 'f32'
        assert a['language'] == b['language'] == lang
        assert a['mode'] == b['mode'] == 'streaming' and a['lookahead'] == 3
        assert a['samples'] == 55649 and a['sample_rate'] == 16000
        assert a['wav_sha256'] == 'c2d42a97888961145f0c549809382c7eb0deadae33bf7a9a72da66563124c974'
        assert len(a['tokens']) == 12 and len(a['decisions']) == 60
        assert all(a[k] == b[k] for k in fields)
        assert b['exact_expected_result'] is True and b['different_fields'] == []
        results.append({'language': lang, 'tokens': len(b['tokens']), 'decisions': len(b['decisions']),
                        'text': b['text'], 'device': b['device']})
    receipt = {'status': 'EXACT_QUALIFICATION_VERIFIED', 'candidate_sha': request['candidate_sha'],
               'source_run_id': request['gpu_run_id'], 'source_attempt': request['gpu_attempt'],
               'lock_sha256': cpu['lock_sha256'], 'nemotron': results}
    (root / 'evidence-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    print(json.dumps(verify(Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())), ensure_ascii=False))
