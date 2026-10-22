"""R15 functional regression: fixed 4-clip CI or complete 42-clip campaign.

Native R9 inference is unchanged. Intermediate numeric/bitwise comparisons are
diagnostics. Exact decisions, finite observed tensors, device routing and the
original final-encoder limits are mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

from .audio import describe
from .cli import ROOT, settings, take_run_dir
from .c3_r3 import read_tensor
from .c3_r7 import encoder_audit
from .c3 import STAGES as ENCODER_STAGES
from .c3_r3 import STAGES as SUBSAMPLING_STAGES
from .c3_r4 import STAGES as BLOCK_STAGES
from .c3_r7_metrics import tensor_metrics, compare_decisions
from .c3_r8 import native, write_json
from .c3_r9_source import original_metadata
from .c3_r14_corpus import select_expanded, DATASET, REVISION

SIGNATURE = 'hdb2c3-r15-functional-regression-v0.7.15'
SHORT_STRATA = (('cafe', 'anger'), ('cafe', 'neutral'),
                ('oreau', 'fear'), ('oreau', 'surprise'))
FINAL_LIMITS = {'max_abs': 0.001, 'min_cosine': 0.999}
REQUIRED_STAGES = (*SUBSAMPLING_STAGES, *ENCODER_STAGES, 'block0_pos_emb',
                   *BLOCK_STAGES, 'encoder_pad_mask', 'encoder_att_mask')


def select_rows(manifest, profile, *, anchors=None):
    if profile not in ('short', 'full'):
        raise ValueError('R15 profile must be short or full; FP64 uses the explicit audit route')
    if manifest.get('dataset') != DATASET or manifest.get('revision') != REVISION:
        raise ValueError('R15 requires the pinned public CAMEO revision')
    if profile == 'full':
        return select_expanded(manifest, anchors=anchors)
    if anchors is None:
        archived = json.loads((ROOT/'ci/results/r13-ci-full-cpu-20261021.json').read_text())
        anchors = [r['clip'] for r in archived['campaign_summary']['results']]
    fixed = {(r['split'], r['emotion']): r['pcm_sha256'] for r in anchors}
    selected = []
    for split, emotion in SHORT_STRATA:
        rows = [r for r in manifest['rows'] if r['split'] == split and
                r['emotion'] == emotion and r.get('selection_index') == 0]
        if len(rows) != 1:
            raise ValueError('R15 missing/duplicate fixed clip: '+split+'/'+emotion)
        row = rows[0]
        if (row.get('dataset') != DATASET or row.get('revision') != REVISION or
                row['pcm_sha256'] != fixed[(split, emotion)] or
                describe(Path(row['wav']))['pcm_sha256'] != row['pcm_sha256']):
            raise ValueError('R15 fixed clip identity changed')
        selected.append({**row, 'cohort': 'regression4'})
    if len({r['pcm_sha256'] for r in selected}) != 4 or len({r['id'] for r in selected}) != 4:
        raise ValueError('R15 refuses duplicate regression identities')
    return selected


def finite_trace(folder):
    """Check every observed f32 tensor, including padded frames and TDT logits."""
    files = sorted(folder.rglob('*.f32le'))
    if not files:
        raise ValueError('R15 missing floating-point trace')
    elements = 0
    for path in files:
        # read_tensor checks shape, storage length, optional digest and NaN/Inf.
        elements += read_tensor(path).size
    return {'all_finite': True, 'tensors_checked': len(files),
            'elements_checked': int(elements), 'scope': 'ALL_OBSERVED_F32_TENSORS'}


def assess(reference, candidate, *, variant, mode):
    ref = json.loads((reference/'manifest.json').read_text())
    out = json.loads((candidate/'result.json').read_text())
    decisions = compare_decisions(ref['decisions'],
        json.loads((candidate/'tdt/tdt_trace.json').read_text()))
    if not ref['decisions']:
        raise ValueError('R15 missing production TDT decisions')
    required = ['pcm.f32le', 'mel.f32le', 'encoder.f32le']
    required += ['encoder_trace/'+name+'.f32le' for name in REQUIRED_STAGES]
    required += [f'tdt/{i:06}/{name}.f32le' for i in range(len(ref['decisions']))
                 for name in ('enc', 'g', 'h', 'c', 'logits')]
    if any(not (candidate/name).is_file() for name in required):
        raise ValueError('R15 incomplete native observation coverage')
    exact = {k: out[k] == ref[k] for k in ('ids', 'frames', 'durations', 'text')}
    valid = ref['encoder_valid']
    ref_encoder = reference/'encoder.f32le'
    if not ref_encoder.is_file():
        ref_encoder = reference/'encoder_trace/layer_23.f32le'  # R14 audit layout
    a, b = read_tensor(ref_encoder), read_tensor(candidate/'encoder.f32le')
    if (a.ndim != 3 or b.ndim != 3 or a.shape[0] != 1 or b.shape[0] != 1 or
            valid <= 0 or min(a.shape[1], b.shape[1]) < valid):
        raise ValueError('R15 malformed/missing final encoder frames')
    final = tensor_metrics(a[:, :valid], b[:, :valid],
                           atol=FINAL_LIMITS['max_abs'], min_cosine=FINAL_LIMITS['min_cosine'])
    lengths = (out['encoder_valid_frames'] == valid and out['mel_valid_frames'] == ref['mel_valid'])
    decoder = (out['blank'] == ref['blank'] and out['durations_config'] == ref['durations_config']
               and out['max_symbols'] == ref['max_symbols'])
    scopes = {'native': 'NATIVE_PCM_FRONTEND_ENCODER_TDT', 'shared_mel': 'SHARED_NEMO_MEL_ENCODER_TDT'}
    scope = out['scope'] == scopes[mode]
    device = str(out['device']).lower().startswith('cuda(' if variant == 'cuda-cudnn' else 'cpu')
    finite = finite_trace(candidate)
    passed = (all(exact.values()) and decisions['exact'] and final['numeric_pass'] and
              lengths and decoder and scope and device and finite['all_finite'])
    return {'pass': bool(passed), 'output_exact': exact, 'decision_counts': decisions,
            'valid_lengths_exact': lengths, 'decoder_config_exact': decoder,
            'expected_scope': scope, 'device_routing_exact': device,
            'finite': finite, 'final_encoder': final,
            'intermediate_numeric_and_cross_backend_bits_are_blocking': False}


def prepare_runtime(s, variant, work):
    """Build exactly the already-qualified R9 source, without older probe builds."""
    spec = importlib.util.spec_from_file_location('hdb15_build', ROOT/'ci/extended_build.py')
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    core = Path(os.environ.get('HDB2A_CANDLE_ROOT', ROOT.parent)).resolve()
    source = builder.prepare_build(core, work)
    expected = json.loads((ROOT/'ci/results/r13-ci-full-cpu-20261021.json').read_text())[
        'campaign_summary']['source_manifest']['tokenizer_metadata']
    if original_metadata(s['nemo'], s['model']) != expected:
        raise ValueError('R15 original tokenizer differs from the qualified R9 source')
    target = work/'cargo-target'
    env = {**os.environ, 'CARGO_TARGET_DIR': str(target), 'CUDA_COMPUTE_CAP': '61'}
    builder.run(s['python'], ROOT/'scripts/hdb2a_cargo_lock.py',
                '--manifest-path', source/'Cargo.toml')
    command = ['cargo', 'build', '--release', '--manifest-path', str(source/'Cargo.toml'),
               '--bin', 'hdb2c_r9_probe']
    if variant == 'cuda-cudnn':
        command += ['--features', 'cuda cudnn']
    subprocess.run(command, check=True, env=env)
    return target/'release/hdb2c_r9_probe', {
        'source_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((source/'src').rglob('*.rs'))},
        'cargo_lock_sha256': hashlib.sha256((source/'Cargo.lock').read_bytes()).hexdigest(),
        'candle_source': 'LOCAL_EXACT_CANDIDATE',
        'candidate_sha': subprocess.check_output(['git', '-C', str(core), 'rev-parse', 'HEAD'], text=True).strip()}


def summarize(results, errors, expected):
    complete = not errors and len(results) == expected
    passed = complete and all(r['pass'] for r in results)
    return {'expected_clips': expected, 'completed_clips': len(results),
            'execution_status': 'COMPLETE' if complete else 'INCOMPLETE',
            'qualification': 'PASS_FUNCTIONAL_REGRESSION' if passed else 'FAIL_FUNCTIONAL_REGRESSION',
            'pass': bool(passed), 'results': results, 'errors': errors}


def main(corpus, variant, profile, *, keep_raw=False):
    # Reject invalid profiles before preparing environments or loading any model.
    rows = select_rows(json.loads(corpus.read_text()), profile)
    s = settings(); s['reports'].mkdir(parents=True, exist_ok=True)
    evidence = json.loads((ROOT/'ci/r7_verified_native_sources.json').read_text())
    checkpoint_sha = hashlib.file_digest(s['nemo'].open('rb'), 'sha256').hexdigest()
    if (checkpoint_sha != evidence['checkpoint_sha256'] or
            json.loads((s['model']/'conversion.json').read_text())['source_nemo_sha256'] != checkpoint_sha):
        raise ValueError('R15 checkpoint/conversion identity differs')
    run = take_run_dir(s['reports'])
    published = Path(os.environ.get('GITHUB_WORKSPACE', str(run)))/'halfduplex-ci-published'
    published.mkdir(parents=True, exist_ok=True)
    binary, source = prepare_runtime(s, variant, run/'build')
    from .c3_r15_reference import Reference
    teacher = Reference(s['nemo'])  # Restore once for the entire corpus.
    results, errors = [], []
    for i, row in enumerate(rows):
        folder = run/f'clip-{i:02}'; folder.mkdir()
        try:
            reference = folder/'nemo'
            ref = teacher.run(Path(row['wav']), reference, detailed=profile == 'full')
            reference_finite = finite_trace(reference)
            modes, diagnostics = {}, {}
            for mode in ('native', 'shared_mel'):
                target = folder/mode
                native(binary, s['model'], Path(row['wav']), target, variant,
                       reference=reference if mode == 'shared_mel' else None)
                modes[mode] = assess(reference, target, variant=variant, mode=mode)
                if profile == 'full':
                    diagnostics[mode] = encoder_audit(reference/'encoder_trace', target, ref['encoder_valid'])
            result = {'clip': {k: row[k] for k in ('id', 'split', 'emotion', 'pcm_sha256')},
                      'cohort': row['cohort'], 'pass': all(m['pass'] for m in modes.values()),
                      'reference_finite': reference_finite, 'modes': modes,
                      'intermediate_diagnostics': diagnostics,
                      'intermediate_diagnostics_block_promotion': False}
            write_json(published/f'r15-{variant}-{i:02}.json', result)
            results.append(result)
            print(f'HDB_R15_CLIP={i+1}/{len(rows)} pass={result["pass"]}', flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            errors.append({'index': i, 'id': row['id'], 'error': type(exc).__name__+': '+str(exc)})
        finally:
            if not keep_raw:
                shutil.rmtree(folder)
    summary = {'signature': SIGNATURE, 'profile': profile, 'variant': variant,
               'checkpoint_sha256': checkpoint_sha, 'dataset_revision': REVISION,
               'source': source, 'fp64_operator_audit': 'NOT_REQUESTED',
               'final_encoder_limits': FINAL_LIMITS,
               'production_tensor_injection': 'NONE',
               'limits': 'Offline batch=1 on fixed public excerpts; not general WER or streaming qualification.',
               **summarize(results, errors, len(rows))}
    write_json(published/f'r15-{variant}-summary.json', summary)
    write_json(run/'summary.json', summary)
    print('HDB_R15='+summary['qualification'], flush=True)
    return 0 if summary['pass'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--variant', choices=('cpu', 'cuda-cudnn'), required=True)
    parser.add_argument('--profile', choices=('short', 'full'), required=True)
    parser.add_argument('--keep-raw', action='store_true')
    args = parser.parse_args()
    raise SystemExit(main(args.corpus, args.variant, args.profile, keep_raw=args.keep_raw))
