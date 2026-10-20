"""R5: real NeMo positional encoding vs the native Candle implementation.

Independent common-input operator checks are distinct from the uninjected
FastConformer run. Only numerical summaries are published by CI.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np

from .c3_r3 import _dump, read_tensor
from .c3_r4 import compare_valid, compare_r4

SIGNATURE = 'hdb2c3-r5-positional-f32-order-v0.7.5'
MARKER = 'HDB2C3_R5_POSITIONAL_F32_ORDER_V1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_r5(reference: Path, candidate: Path, *, skip_build=False):
    """Keep root-crate Cargo outputs disjoint, including the library artifacts."""
    import os
    import subprocess
    target = candidate/'target/r5-cpu'
    control = reference/'target/r4-cpu/release/hdb2c_probe'
    before = digest(control)
    if not skip_build:
        subprocess.run(['cargo', 'build', '--release', '--manifest-path', str(candidate/'Cargo.toml'),
                        '--bin', 'hdb2c_r5_probe', '--bin', 'hdb2c_positional'],
                       check=True, env={**os.environ, 'CARGO_TARGET_DIR': str(target)})
    if digest(control) != before:
        raise RuntimeError('R5 compilation modified the R4 control executable')
    return target


def transform_r5_attention(source):
    """Match NeMo's f32 multiply THEN exp, preserving native Rust sin/cos."""
    if MARKER in source:
        raise ValueError('R5 positional treatment already installed')
    old = '''        for i in 0..(d_model / 2) {
            let exp = -((10000.0f64).ln() / d_model as f64) * (2 * i) as f64;
            div_term.push(exp.exp() as f32);
        }'''
    new = '''        // HDB2C3_R5_POSITIONAL_F32_ORDER_V1
        // NeMo 2.7.3: arange(f32) * scalar(f32), followed by exp(f32).
        let exponent_factor = (-((10000.0f64).ln() / d_model as f64)) as f32;
        for i in 0..(d_model / 2) {
            let exponent = (2 * i) as f32 * exponent_factor;
            div_term.push(exponent.exp());
        }'''
    if source.count(old) != 1:
        raise ValueError('R5 pinned positional frequency construction changed')
    return source.replace(old, new, 1)


def build_r5_candidate(reference: Path, target: Path):
    reference, target = reference.resolve(), target.resolve()
    if reference == target or reference in target.parents or target in reference.parents:
        raise ValueError('R5 candidate must be separate from the R4 control')
    parent = json.loads((reference/'HDB2C3-R4-MANIFEST.json').read_text())
    conf = reference/'src/parakeet/conformer.rs'
    attention = reference/'src/parakeet/attention.rs'
    if digest(conf) != parent['generated_conformer_sha256']:
        raise ValueError('R5 parent R4 conformer differs from its manifest')
    probe = reference/'src/bin/hdb2c_probe.rs'
    root = Path(__file__).resolve().parents[1]
    outputs = {
        'src/parakeet/attention.rs': transform_r5_attention(attention.read_text()),
        'src/bin/hdb2c_r5_probe.rs': probe.read_text(),
        'src/bin/hdb2c_positional.rs': (root/'rust/hdb2c_positional.rs').read_text(),
    }
    manifest = {
        'signature': SIGNATURE, 'parent': str(reference),
        'parent_conformer_sha256': digest(conf),
        'parent_attention_sha256': digest(attention),
        'parent_probe_sha256': digest(probe),
        'target_sha256': {p: hashlib.sha256(s.encode()).hexdigest() for p, s in outputs.items()},
        'scope': 'isolated CPU f32 frequency order; native sin/cos; no activation/table injection into encoder',
    }
    if target.exists():
        path = target/'HDB2C3-R5-MANIFEST.json'
        if not path.is_file() or json.loads(path.read_text()) != manifest:
            raise RuntimeError('R5 candidate exists with a different manifest')
        if digest(target/'src/parakeet/conformer.rs') != manifest['parent_conformer_sha256']:
            raise RuntimeError('R5 candidate conformer changed')
        for name, want in manifest['target_sha256'].items():
            if digest(target/name) != want:
                raise RuntimeError('R5 candidate source changed: '+name)
        return manifest
    temp = target.with_name(target.name+'.creating')
    if temp.exists():
        raise RuntimeError('Stale R5 candidate staging directory')
    try:
        shutil.copytree(reference, temp, ignore=lambda _, names: [
            n for n in names if n in {'target', '.git', '__pycache__', '.venv'}])
        for name, content in outputs.items():
            path = temp/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        (temp/'HDB2C3-R5-MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
        temp.rename(target)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return manifest


def attach_nemo_r5_hooks(encoder, folder: Path):
    pos = encoder.pos_enc
    if type(pos).__name__ != 'RelPositionalEncoding':
        raise RuntimeError('R5 requires the actual NeMo RelPositionalEncoding')
    if any(m.training for m in encoder.modules()):
        raise RuntimeError('R5 encoder has active training modules')
    seen = set()
    def save(name, tensor):
        if name in seen:
            raise RuntimeError('R5 duplicate positional hook: '+name)
        _dump(folder, name, tensor, 'B,T,D')
        seen.add(name)
    def pre(module, args, kwargs):
        if (args and 'x' in kwargs) or (not args and 'x' not in kwargs):
            raise RuntimeError('R5 missing or ambiguous positional input')
        x = args[0] if args else kwargs['x']
        offset = args[1] if len(args) > 1 else kwargs.get('cache_len', 0)
        if offset != 0:
            raise RuntimeError('R5 live offline encoder unexpectedly uses a cache')
        save('pos_input', x)
        state = {
            'class': type(module).__module__+'.'+type(module).__name__,
            'd_model': int(module.d_model), 'xscale': module.xscale,
            'input_shape': list(x.shape), 'input_dtype': str(x.dtype),
            'input_device': str(x.device), 'cache_len': int(offset),
            'buffer_shape': list(module.pe.shape),
            'dropout_training': bool(module.dropout.training),
            'dropout_p': float(module.dropout.p),
            'dropout_emb_training': (bool(module.dropout_emb.training)
                                     if module.dropout_emb is not None else None),
        }
        if str(x.dtype) != 'torch.float32' or str(x.device) != 'cpu':
            raise RuntimeError('R5 requires a real f32 CPU positional input')
        (folder/'nemo_position_state.json').write_text(json.dumps(state, indent=2)+'\n')
    def post(_module, _args, result):
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError('R5 unexpected positional output')
        save('pos_output', result[0])
    return [pos.register_forward_pre_hook(pre, with_kwargs=True),
            pos.register_forward_hook(post)], seen


def prepare_positional_cases(pos, trace: Path):
    """Run the genuine NeMo class, in eval, on exact shared f32 inputs.

    The actual-input case uses the live positional input. Other cases are
    deterministic operator fixtures; they are never used for ASR claims.
    """
    import torch
    if any(m.training for m in pos.modules()):
        raise RuntimeError('R5 positional reference is not in eval mode')
    actual = read_tensor(trace/'pos_input.f32le')
    d = int(pos.d_model)
    if d <= 0 or d % 2 or actual.shape[2] != d:
        raise ValueError('R5 unsupported positional width')
    scale_input = bool(pos.xscale)
    if scale_input and not math.isclose(float(pos.xscale), math.sqrt(d), rel_tol=1e-12):
        raise ValueError('R5 NeMo xscale is not sqrt(d_model)')
    out = trace/'positional_nemo'
    out.mkdir(exist_ok=True)
    specs = [('actual_shared', actual, scale_input, 0, int(pos.pe.shape[1]//2+1))]
    for length in (1, 2, 7, 513):
        values = ((np.arange(length*d, dtype=np.int64) % 257)-128).astype(np.float32)
        values = (values*np.float32(0.03125)).reshape(1, length, d)
        for scale in (False, True):
            for offset in (0, 3):
                specs.append((f't{length}_s{int(scale)}_o{offset}', values, scale, offset, 2))
    cases = []
    with torch.inference_mode():
        for name, values, scale, offset, initial in specs:
            module = pos if name == 'actual_shared' else type(pos)(
                d_model=d, dropout_rate=float(pos.dropout.p), max_len=initial,
                xscale=math.sqrt(d) if scale else None,
                dropout_rate_emb=float(pos.dropout_emb.p) if pos.dropout_emb is not None else 0.)
            module.eval()
            x = torch.from_numpy(values.copy())
            module.extend_pe(values.shape[1]+offset, device=x.device, dtype=x.dtype)
            y, pe = module(x=x, cache_len=offset)
            y2, pe2 = module(x=x, cache_len=offset)
            if not torch.equal(y.view(torch.int32), y2.view(torch.int32)) or not torch.equal(
                    pe.contiguous().view(torch.int32), pe2.contiguous().view(torch.int32)):
                raise RuntimeError('R5 NeMo position reference is not bit reproducible')
            folder = out/name
            folder.mkdir(exist_ok=True)
            for key, value in (('input', x), ('output', y), ('pos_emb', pe)):
                _dump(folder, key, value, 'B,T,D')
            if name == 'actual_shared':
                for key, live in (('output', 'pos_output'), ('pos_emb', 'block0_pos_emb')):
                    if (folder/f'{key}.f32le').read_bytes() != (trace/f'{live}.f32le').read_bytes():
                        raise RuntimeError('R5 live and replayed NeMo position disagree')
            cases.append({'id': name, 'length': int(values.shape[1]), 'd_model': d,
                          'offset': offset, 'max_len': initial, 'scale_input': scale,
                          'input_sha256': digest(folder/'input.f32le')})
    manifest = {'signature': SIGNATURE, 'reference_class': type(pos).__module__+'.'+type(pos).__name__,
                'dtype': 'float32', 'device': 'cpu', 'all_modules_eval': True,
                'reference_bit_reproducible': True, 'cases': cases}
    (out/'cases.json').write_text(json.dumps(manifest, indent=2)+'\n')


def compare_exact(a, b, *, valid=None, atol=1e-5):
    if a.shape != b.shape:
        return {'status': 'SHAPE_MISMATCH', 'nemo_shape': list(a.shape),
                'candle_shape': list(b.shape), 'bitwise_equal': False}
    count = int(a.shape[1]) if valid is None else int(valid)
    result = compare_valid(a, b, count, atol=atol, min_cosine=0.999999)
    aa = np.ascontiguousarray(a[:, :count, :], dtype='<f4')
    bb = np.ascontiguousarray(b[:, :count, :], dtype='<f4')
    result['bitwise_equal'] = aa.tobytes() == bb.tobytes()
    result['mismatching_f32_bits'] = int(np.count_nonzero(aa.view(np.uint32) != bb.view(np.uint32)))
    return result


def summarize_r5(nemo: Path, control: Path, candidate: Path, operator: Path, valid: int):
    from .c3 import compare_stages
    # The treatment changes position frequencies only; subsampling must stay exact.
    pre = compare_exact(read_tensor(control/'pre_encode.f32le'),
                        read_tensor(candidate/'pre_encode.f32le'), atol=0.)
    if not pre['bitwise_equal']:
        raise RuntimeError('R5 unexpectedly changed the R4 pre-encoder')
    ref = json.loads((nemo/'positional_nemo/cases.json').read_text())
    native = json.loads((operator/'manifest.json').read_text())
    if ref['signature'] != SIGNATURE or native['signature'] != SIGNATURE:
        raise ValueError('R5 operator evidence signature mismatch')
    if native['cases'] != ref['cases'] or len(ref['cases']) != 17:
        raise ValueError('R5 missing, duplicate, or different operator cases')
    rows = []
    for case in ref['cases']:
        name = case['id']
        n, c = nemo/'positional_nemo'/name, operator/name
        if digest(n/'input.f32le') != case['input_sha256'] or digest(c/'input.f32le') != case['input_sha256']:
            raise ValueError('R5 native operator did not consume the byte-identical shared input')
        rows.append({**case, 'shared_input': 'BIT_IDENTICAL_F32LE',
                     'output': compare_exact(read_tensor(n/'output.f32le'), read_tensor(c/'output.f32le'), atol=1e-3),
                     'pos_emb': compare_exact(read_tensor(n/'pos_emb.f32le'), read_tensor(c/'pos_emb.f32le'))})
    numeric = all(row[key]['status'] == 'PASS_NUMERIC_PARITY' for row in rows for key in ('output', 'pos_emb'))
    exact = all(row[key]['bitwise_equal'] for row in rows for key in ('output', 'pos_emb'))
    full = compare_stages(nemo, candidate, valid)
    old = compare_stages(nemo, control, valid)
    block = compare_r4(nemo, candidate, valid, old, full)
    live = {}
    for name, filename in (('pos_input', 'pre_encode'), ('pos_output', 'post_pos_enc'),
                           ('pos_emb', 'block0_pos_emb')):
        a = read_tensor(nemo/f'{name if name != "pos_emb" else "block0_pos_emb"}.f32le')
        b = read_tensor(candidate/f'{filename}.f32le')
        live[name] = compare_exact(a, b, valid=None if name == 'pos_emb' else valid,
                                   atol=1e-5 if name == 'pos_emb' else 1e-3)
    boundary = compare_exact(read_tensor(nemo/'pos_output.f32le'),
                             read_tensor(nemo/'block0_input.f32le'), atol=0.)
    if not boundary['bitwise_equal']:
        raise RuntimeError('R5 NeMo block0 input differs from actual positional output')
    return {
        'stage': 'HDB2C3_A_R5_POSITIONAL_ENCODING_EXACT_PARITY', 'revision': 'HDB2-C3-A-R5',
        'signature': SIGNATURE, 'status': 'POSITIONAL_NUMERIC_PARITY' if numeric else 'POSITIONAL_NUMERIC_MISMATCH',
        'positional_exact_parity': 'PASS_BITWISE' if exact else 'NOT_ESTABLISHED',
        'criteria': {'pos_emb_max_abs': 1e-5, 'common_input_output_max_abs': 1e-3, 'cosine_min': .999999,
                     'bitwise': 'ALL_COMPARED_F32_BYTES_EQUAL_INCLUDING_SIGNED_ZERO'},
        'nemo_eval_state': json.loads((nemo/'nemo_eval_state.json').read_text()),
        'nemo_position_state': json.loads((nemo/'nemo_position_state.json').read_text()),
        'reference_class': ref['reference_class'], 'reference_bit_reproducible': ref['reference_bit_reproducible'],
        'operator_cases': rows, 'live_uninjected_encoder_position': live,
        'r4_r5_pre_encode_control': pre, 'nemo_pos_to_block0_boundary': boundary,
        'baseline_r4_pos_emb': compare_exact(read_tensor(nemo/'block0_pos_emb.f32le'),
                                            read_tensor(control/'block0_pos_emb.f32le')),
        'first_encoder_stage_r4': old['first_divergence'],
        'full_encoder_numeric_parity': full['status'], 'first_encoder_stage_r5': full['first_divergence'],
        'encoder_stages': full['stages'],
        'first_block0_stage_r5': block['first_block0_divergence'], 'block0_stages': block['stages'],
        'tdt_numeric_parity': 'NOT_TESTED', 'gpu_parity': 'NOT_TESTED',
        'note': 'Native Rust frequencies, sin/cos and scaling. Common-input operator fixtures do not inject activations or positional tables into full encoder inference.',
    }


def execute(report: Path, *, skip_build=False):
    import os
    import subprocess
    from .cli import settings
    s = settings()
    report = report.resolve()
    c1 = json.loads((report/'nemo_report.json').read_text())
    c1_eval = c1.get('probes', {}).get('inference_state', {})
    if c1_eval.get('status') != 'ALL_MODULES_EVAL_VERIFIED':
        raise ValueError('R5 requires fresh C1 probes measured in eval mode')
    conversion = json.loads((s['model']/'conversion.json').read_text())
    if conversion.get('source_nemo_sha256') != c1['checkpoint_sha256'] or digest(s['nemo']) != c1['checkpoint_sha256']:
        raise ValueError('R5 source checkpoint identity changed')
    original = json.loads((report/'candle_report.json').read_text())
    if original['pcm_sha256'] != c1['pcm_sha256']:
        raise ValueError('R5 source PCM identity changed')
    r4 = s['runtime'].with_name(s['runtime'].name+'-hdb2c-c3-r4-v0.7.4')
    r5 = s['runtime'].with_name(s['runtime'].name+'-hdb2c-c3-r5-v0.7.5')
    manifest = build_r5_candidate(r4, r5)
    target = compile_r5(r4, r5, skip_build=skip_build)
    out = report/'c3_cpu'
    trace = out/'candle_r5_trace'
    probe = out/'candle_r5_probe'
    trace.mkdir(exist_ok=True)
    probe.mkdir(exist_ok=True)
    with np.load(report/'nemo_mel.npz') as source:
        values = source['values']
        mel_valid = int(source['valid_frames'][0])
        total = int(values.shape[2])
        expected = values.transpose(0, 2, 1).astype('<f4').tobytes()
    with np.load(report/'nemo_encoder.npz') as source:
        valid = int(source['valid_frames'][0])
    mel = out/'nemo_mel_injected_BTD.f32le'
    if mel.read_bytes() != expected:
        raise ValueError('R5 shared Mel bytes changed since the NeMo reference')
    env = {**os.environ, 'HDB2C3_TRACE_DIR': str(trace),
           'HDB2C3_NEMO_MEL_PATH': str(mel), 'HDB2C3_NEMO_MEL_FRAMES': str(total),
           'HDB2C3_NEMO_MEL_VALID_FRAMES': str(mel_valid)}
    subprocess.run([str(target/'release/hdb2c_r5_probe'), '--model-dir', str(s['model']),
                    '--input', str(report/'input-16k.wav'), '--output', str(probe), '--cpu'],
                   check=True, env=env)
    meta = json.loads((probe/'candle_probe.json').read_text())
    if ((probe/'candle_mel.f32le').read_bytes() != expected
            or meta.get('injected_mel_valid_frames') != mel_valid
            or meta.get('encoder_valid_frames') != valid
            or meta.get('dw_mask_propagation') != 'NEMO_MASKED_CONV_SEQUENTIAL'
            or meta.get('encoder_length_argument') != 'EXPLICIT_NEMO_VALID_FRAMES_IF_INJECTED'):
        raise ValueError('R5 lost shared input, exact lengths, or R3C mask controls')
    operator = out/'positional_candle'
    subprocess.run([str(target/'release/hdb2c_positional'),
                    '--cases', str(out/'nemo_trace/positional_nemo/cases.json'),
                    '--output', str(operator)], check=True)
    result = summarize_r5(out/'nemo_trace', out/'candle_r4_trace', trace, operator, valid)
    result.update(pcm_sha256=c1['pcm_sha256'], checkpoint_sha256=c1['checkpoint_sha256'],
                  c1_eval_state=c1_eval, source_manifest={k: v for k, v in manifest.items() if k != 'parent'},
                  mel_input='EXACT_SHARED_NEMO_F32LE_VERIFIED',
                  encoder_positional_injection='NONE_NATIVE_COMPUTATION_ONLY')
    path = report/'c3_positional_cpu.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('HDB2C3_R5='+result['status']+' exact='+result['positional_exact_parity']
          +' first_encoder='+str(result['first_encoder_stage_r5'])
          +' first_block0='+str(result['first_block0_stage_r5']), flush=True)
    return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--skip-build', action='store_true')
    args = parser.parse_args()
    execute(args.report, skip_build=args.skip_build)


if __name__ == '__main__':
    main()
