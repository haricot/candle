"""R7: full native pipeline byte audit and production NeMo TDT parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .c3 import STAGES
from .c3_r3 import STAGES as DW_STAGES, read_tensor
from .c3_r4 import STAGES as BLOCK_STAGES
from .c3_r5 import digest
from .c3_r6 import exact_control
from .c3_r7_metrics import tensor_metrics, compare_decisions
from .c3_r7_source import transform_model

SIGNATURE = 'hdb2c3-r7-global-bits-production-tdt-v0.7.7'


def build_candidate(reference, target):
    reference, target = reference.resolve(), target.resolve()
    if reference == target or reference in target.parents or target in reference.parents:
        raise ValueError('R7 requires an isolated source tree')
    parent = json.loads((reference/'HDB2C3-R6-MANIFEST.json').read_text())
    if parent['signature'] != 'hdb2c3-r6-encoder-padding-mask-ab-v0.7.6':
        raise ValueError('R7 requires the verified R6 parent')
    for name, sha in parent['target_sha256'].items():
        if digest(reference/name) != sha:
            raise ValueError('R7 parent source changed: '+name)
    root = Path(__file__).resolve().parents[1]
    outputs = {'src/parakeet/model.rs': transform_model((reference/'src/parakeet/model.rs').read_text(),
                                                       (root/'rust/hdb2c_tdt.rs').read_text()),
               'src/bin/hdb2c_r7_probe.rs': (root/'rust/hdb2c_r7_probe.rs').read_text()}
    manifest = {'signature': SIGNATURE, 'parent': str(reference),
                'parent_source_sha256': {name: digest(reference/name) for name in
                    ('src/parakeet/model.rs', 'src/parakeet/conformer.rs', 'src/parakeet/attention.rs')},
                'target_sha256': {name: hashlib.sha256(value.encode()).hexdigest() for name, value in outputs.items()},
                'scope': 'CPU offline R6 both masks; native full PCM path plus distinct common-input TDT operators'}
    if target.exists():
        if json.loads((target/'HDB2C3-R7-MANIFEST.json').read_text()) != manifest:
            raise ValueError('R7 existing candidate manifest differs')
        for name, sha in manifest['target_sha256'].items():
            if digest(target/name) != sha:
                raise ValueError('R7 generated source changed: '+name)
        return manifest
    staging = target.with_name(target.name+'.creating')
    if staging.exists():
        raise ValueError('R7 stale staging directory')
    try:
        shutil.copytree(reference, staging, ignore=lambda _, names: [n for n in names if n in
                       {'target', '.git', '__pycache__', '.venv'}])
        for name, value in outputs.items():
            path = staging/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        (staging/'HDB2C3-R7-MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def compile_candidate(reference, candidate, *, skip_build=False):
    target = candidate/'target/r7-cpu'
    control = reference/'target/r6-cpu/release/hdb2c_r6_probe'
    before = digest(control)
    if not skip_build:
        env = {**os.environ, 'CARGO_TARGET_DIR': str(target)}
        subprocess.run(['cargo', 'build', '--release', '--manifest-path', str(candidate/'Cargo.toml'),
                        '--bin', 'hdb2c_r7_probe'], check=True, env=env)
        subprocess.run(['cargo', 'test', '--release', '--manifest-path', str(candidate/'Cargo.toml'),
                        '--lib', 'hdb2c7_tests'], check=True, env=env)
    if digest(control) != before:
        raise RuntimeError('R7 build modified the R6 executable')
    return target


def metric_files(a, b, name, **kw):
    return tensor_metrics(read_tensor(a/f'{name}.f32le'), read_tensor(b/f'{name}.f32le'), **kw)


def operator_comparison(nemo, native, ref):
    rows = []
    for case in ref['predictor_cases']:
        a, b = nemo/'predictor'/case['id'], native/'predictor'/case['id']
        if case['has_state']:
            for name in ('input_h', 'input_c'):
                if not metric_files(a, b, name)['bitwise_equal']:
                    raise ValueError('R7 native predictor consumed different state bytes')
        rows.append({'id': case['id'], 'kind': 'predictor', 'token': case['token'],
                     'metrics': {name: metric_files(a, b, name, atol=1e-4, min_cosine=.999999) for name in ('g', 'h', 'c')}})
    for case in ref['decisions']:
        a, b = nemo/'joint'/case['id'], native/'joint'/case['id']
        for name in ('enc', 'g'):
            if not metric_files(a, b, name)['bitwise_equal']:
                raise ValueError('R7 native joint consumed different input bytes')
        raw = read_tensor(b/'logits.f32le').ravel()
        split = ref['blank']+1
        rows.append({'id': case['id'], 'kind': 'joint',
                     'argmax_exact': int(raw[:split].argmax()) == case['token'] and int(raw[split:].argmax()) == case['duration_index'],
                     'metrics': {'logits': metric_files(a, b, 'logits', atol=1e-3, min_cosine=.999999)}})
    exact = all(m['bitwise_equal'] for row in rows for m in row['metrics'].values())
    numeric = all(m['numeric_pass'] for row in rows for m in row['metrics'].values())
    decisions = all(row.get('argmax_exact', True) for row in rows)
    return {'scope': 'COMMON_INPUT_OPERATORS_NOT_GLOBAL_PIPELINE', 'cases': rows,
            'shared_inputs': 'BIT_IDENTICAL_STATE_ENCODER_AND_PREDICTOR_INPUTS',
            'bitwise_equal': exact, 'numeric_pass': numeric, 'argmax_exact': decisions}


def trajectory_comparison(nemo, native, ref):
    rows = json.loads((native/'tdt/tdt_trace.json').read_text())
    decisions = compare_decisions(ref['decisions'], rows)
    result = json.loads((native/'result.json').read_text())
    outputs = {key: result[key] == ref[key] for key in ('ids', 'frames', 'durations', 'text')}
    activations = []
    # Different trajectories cannot be paired by a misleading array index.
    if decisions['exact']:
        for row in ref['decisions']:
            pred = nemo/'predictor'/row['predictor_id']
            joint = nemo/'joint'/row['id']
            target = native/'tdt'/row['id']
            metrics = {name: metric_files(pred, target, name, atol=1e-4, min_cosine=.999999) for name in ('g', 'h', 'c')}
            metrics['logits'] = metric_files(joint, target, 'logits', atol=1e-3, min_cosine=.999999)
            activations.append({'id': row['id'], 'metrics': metrics})
    return {'scope': result['scope'], 'decisions': decisions, 'output_exact': outputs,
            'candidate_output': result, 'activations': activations,
            'numeric_pass': bool(activations) and all(m['numeric_pass'] for row in activations for m in row['metrics'].values()),
            'bitwise_equal': bool(activations) and all(m['bitwise_equal'] for row in activations for m in row['metrics'].values()),
            'tdt_discrete_exact': decisions['exact'] and all(outputs.values())}


def encoder_audit(nemo, native, valid):
    names = list(DW_STAGES)+list(STAGES)+['block0_pos_emb']+list(BLOCK_STAGES)
    rows = [{'name': name, **metric_files(nemo, native/'encoder_trace', name)} for name in names]
    a = read_tensor(nemo/'layer_23.f32le')[:, :valid]
    b = read_tensor(native/'encoder_trace/layer_23.f32le')[:, :valid]
    return {'scope': 'ALL_FRAMES_INCLUDING_PADDING', 'stages': rows,
            'first_bitwise_difference': next((row['name'] for row in rows if not row['bitwise_equal']), None),
            'bitwise_equal': all(row['bitwise_equal'] for row in rows),
            'valid_final_output': tensor_metrics(a, b)}


def execute(report, *, skip_build=False):
    from .cli import settings
    s = settings(); report = report.resolve()
    original = json.loads((report/'nemo_report.json').read_text())
    r6 = json.loads((report/'c3_padding_cpu.json').read_text())
    if r6['checkpoint_sha256'] != original['checkpoint_sha256'] or digest(s['nemo']) != original['checkpoint_sha256']:
        raise ValueError('R7 checkpoint identity changed')
    if json.loads((s['model']/'conversion.json').read_text())['source_nemo_sha256'] != original['checkpoint_sha256']:
        raise ValueError('R7 model conversion identity differs')
    reference = s['runtime'].with_name(s['runtime'].name+'-hdb2c-c3-r6-v0.7.6')
    candidate = s['runtime'].with_name(s['runtime'].name+'-hdb2c-c3-r7-v0.7.7')
    manifest = build_candidate(reference, candidate)
    target = compile_candidate(reference, candidate, skip_build=skip_build)
    executable = target/'release/hdb2c_r7_probe'
    out = report/'c3_cpu/r7'; out.mkdir(exist_ok=True)
    nemo = out/'nemo'
    subprocess.run([str(s['python']), '-m', 'hdb2c.c3_r7_nemo', '--nemo', str(s['nemo']),
                    '--model-dir', str(s['model']), '--report', str(report), '--output', str(nemo)], check=True)
    ref = json.loads((nemo/'manifest.json').read_text())
    with np.load(report/'nemo_mel.npz') as data:
        mel_valid = int(data['valid_frames'][0])
    valid = int(np.load(report/'nemo_encoder.npz')['valid_frames'][0])
    # No inherited Mel-injection env can contaminate the native PCM path.
    env = {key: value for key, value in os.environ.items() if not key.startswith(('HDB2C3_NEMO_MEL', 'HDB2C7_TDT_TRACE', 'HDB2C3_TRACE_DIR'))}
    env['HDB2C6_MASK_MODE'] = 'both'
    for mode in ('shared_mel', 'native_pcm'):
        folder = out/mode; folder.mkdir(exist_ok=True)
        args = [str(executable), '--model-dir', str(s['model']), '--input', str(report/'input-16k.wav'), '--output', str(folder)]
        if mode == 'shared_mel':
            args += ['--shared-mel', str(nemo), '--mel-valid', str(mel_valid)]
        subprocess.run(args, check=True, env={**env, 'HDB2C3_TRACE_DIR': str(folder/'encoder_trace'),
                                            'HDB2C7_TDT_TRACE_DIR': str(folder/'tdt')})
    operators = out/'operators'
    subprocess.run([str(executable), '--model-dir', str(s['model']), '--input', str(report/'input-16k.wav'),
                    '--output', str(operators), '--tdt-cases', str(nemo)], check=True, env=env)
    shared, native = out/'shared_mel', out/'native_pcm'
    control = exact_control(report/'c3_cpu/candle_r6_both_trace', shared/'encoder_trace', STAGES, label='R7 unchanged R6 encoder')
    if not metric_files(nemo, shared, 'mel')['bitwise_equal']:
        raise ValueError('R7 common Mel bytes changed')
    native_result = json.loads((native/'result.json').read_text())
    if native_result['scope'] != 'NATIVE_PCM_FRONTEND_ENCODER_TDT':
        raise ValueError('R7 native PCM scope not acknowledged')
    frontend = {name: metric_files(nemo, native, name) for name in ('pcm', 'mel')}
    encoder = {mode: encoder_audit(report/'c3_cpu/nemo_trace', out/mode, valid) for mode in ('shared_mel', 'native_pcm')}
    tdt = {mode: trajectory_comparison(nemo, out/mode, ref) for mode in ('shared_mel', 'native_pcm')}
    common = operator_comparison(nemo, operators, ref)
    pipeline_exact = all(m['bitwise_equal'] for m in frontend.values()) and encoder['native_pcm']['bitwise_equal'] and tdt['native_pcm']['bitwise_equal'] and tdt['native_pcm']['tdt_discrete_exact']
    tdt_pass = common['numeric_pass'] and common['argmax_exact'] and all(r['tdt_discrete_exact'] and r['numeric_pass'] for r in tdt.values())
    result = {'signature': SIGNATURE, 'revision': 'HDB2-C3-A-R7',
              'status': 'TDT_NUMERIC_AND_DECISION_PARITY' if tdt_pass else 'TDT_PARITY_NOT_ESTABLISHED',
              'global_bitwise_status': 'PASS_BITWISE' if pipeline_exact else 'FAIL_DIFFERENT_F32_BITS',
              'global_bitwise_equal': pipeline_exact, 'pcm_sha256': original['pcm_sha256'],
              'checkpoint_sha256': original['checkpoint_sha256'],
              'reference': {k: v for k, v in ref.items() if k not in ('predictor_cases', 'decisions')},
              'reference_decisions': ref['decisions'], 'frontend_native_pcm': frontend,
              'encoder': encoder, 'tdt': tdt, 'common_input_operators': common,
              'encoder_r6_control': control, 'source_manifest': {k: v for k, v in manifest.items() if k != 'parent'},
              'native_pipeline_internal_tensor_injection': 'NONE',
              'limits': 'Two real CaFE clips, batch 1, offline CPU; no streaming/GPU qualification; exact decisions do not imply exact f32.'}
    (report/'c3_tdt_cpu.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('HDB2C3_R7='+result['status']+' global='+result['global_bitwise_status'], flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('report', type=Path); p.add_argument('--skip-build', action='store_true')
    a = p.parse_args(); execute(a.report, skip_build=a.skip_build)
