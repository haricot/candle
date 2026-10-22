"""R6: 2x2 causal comparison of real encoder attention/convolution padding masks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .c3_r3 import _dump, read_tensor
from .c3_r4 import STAGES as BLOCK_STAGES, compare_valid
from .c3_r5 import digest, compare_exact
from .c3_r6_source import transform_attention, transform_conformer

SIGNATURE = 'hdb2c3-r6-encoder-padding-mask-ab-v0.7.6'
MODES = ('none', 'attention', 'convolution', 'both')


def expected_masks(lengths, total):
    lengths = np.asarray(lengths)
    if (lengths.ndim != 1 or lengths.dtype.kind not in 'iu' or total <= 0
            or not len(lengths) or np.any(lengths <= 0) or np.any(lengths > total)):
        raise ValueError('R6 invalid encoder lengths')
    pad = np.arange(total)[None, :] >= lengths[:, None]
    return pad[:, :, None], pad[:, :, None] | pad[:, None, :]


def attach_nemo_r6_hooks(encoder, folder):
    """Observe actual masks passed to block 0 and actual post-mask conv input."""
    import torch
    if any(m.training for m in encoder.modules()):
        raise RuntimeError('R6 NeMo encoder is not recursively eval')
    if (encoder.self_attention_model != 'rel_pos' or encoder.att_context_style != 'regular'
            or list(encoder.att_context_size) != [-1, -1] or not encoder.use_pad_mask):
        raise RuntimeError('R6 requires enabled padding masks and regular global relative attention')
    layer = encoder.layers[0]
    if layer.conv.pointwise_activation != 'glu_':
        raise RuntimeError('R6 requires the actual GLU convolution path')
    seen = set()
    def save(name, tensor):
        if name in seen:
            raise RuntimeError('R6 duplicate NeMo hook '+name)
        _dump(folder, name, tensor, 'B,T,D')
        seen.add(name)
    def block_pre(_module, args, kwargs):
        pad, att = kwargs.get('pad_mask'), kwargs.get('att_mask')
        if (not isinstance(pad, torch.Tensor) or not isinstance(att, torch.Tensor)
                or pad.dtype != torch.bool or att.dtype != torch.bool
                or pad.ndim != 2 or tuple(att.shape) != (pad.shape[0], pad.shape[1], pad.shape[1])):
            raise RuntimeError('R6 actual NeMo block masks missing or unexpected')
        save('encoder_pad_mask', pad.unsqueeze(2))
        save('encoder_att_mask', att)
        state = {'self_attention_model': encoder.self_attention_model,
                 'att_context_style': encoder.att_context_style,
                 'att_context_size': list(encoder.att_context_size),
                 'use_pad_mask': bool(encoder.use_pad_mask),
                 'use_pytorch_sdpa': bool(layer.self_attn.use_pytorch_sdpa),
                 'mask_dtype': str(pad.dtype), 'mask_true_means': 'BLOCKED',
                 'padded_frames': int(pad.sum()), 'encoder_total_frames': int(pad.shape[1]),
                 'conv_mask_boundary': 'AFTER_GLU_BEFORE_DEPTHWISE_CONV'}
        (folder/'nemo_padding_state.json').write_text(json.dumps(state, indent=2)+'\n')
    def depthwise_pre(_module, args, kwargs):
        if (args and 'x' in kwargs) or (not args and 'x' not in kwargs):
            raise RuntimeError('R6 missing or ambiguous NeMo depthwise input')
        x = args[0] if args else kwargs['x']
        save('block0_conv_after_mask', x.transpose(1, 2))
    return [layer.register_forward_pre_hook(block_pre, with_kwargs=True),
            layer.conv.depthwise_conv.register_forward_pre_hook(depthwise_pre, with_kwargs=True)], seen


def build_candidate(reference, target):
    reference, target = reference.resolve(), target.resolve()
    if reference == target or reference in target.parents or target in reference.parents:
        raise ValueError('R6 must have a separate source tree')
    parent = json.loads((reference/'HDB2C3-R5-MANIFEST.json').read_text())
    if parent['signature'] != 'hdb2c3-r5-positional-f32-order-v0.7.5':
        raise ValueError('R6 requires the verified R5 parent')
    if digest(reference/'src/parakeet/conformer.rs') != parent['parent_conformer_sha256']:
        raise ValueError('R6 parent conformer changed')
    for name, sha in parent['target_sha256'].items():
        if digest(reference/name) != sha:
            raise ValueError('R6 parent source changed: '+name)
    helper = (Path(__file__).resolve().parents[1]/'rust/hdb2c_masking.rs').read_text()
    outputs = {
        'src/parakeet/conformer.rs': transform_conformer((reference/'src/parakeet/conformer.rs').read_text()),
        'src/parakeet/attention.rs': transform_attention((reference/'src/parakeet/attention.rs').read_text(), helper),
        'src/bin/hdb2c_r6_probe.rs': (reference/'src/bin/hdb2c_r5_probe.rs').read_text(),
    }
    manifest = {'signature': SIGNATURE, 'parent': str(reference),
                'parent_source_sha256': {p: digest(reference/p) for p in
                    ('src/parakeet/conformer.rs', 'src/parakeet/attention.rs', 'src/bin/hdb2c_r5_probe.rs')},
                'target_sha256': {p: hashlib.sha256(s.encode()).hexdigest() for p, s in outputs.items()},
                'scope': 'CPU offline regular global rel_pos; 2x2 mask ablation; R5 positional math and weights retained'}
    if target.exists():
        path = target/'HDB2C3-R6-MANIFEST.json'
        if not path.is_file() or json.loads(path.read_text()) != manifest:
            raise RuntimeError('R6 existing candidate manifest differs')
        for name, sha in manifest['target_sha256'].items():
            if digest(target/name) != sha:
                raise RuntimeError('R6 candidate source changed: '+name)
        return manifest
    temp = target.with_name(target.name+'.creating')
    if temp.exists():
        raise RuntimeError('Stale R6 staging directory')
    try:
        shutil.copytree(reference, temp, ignore=lambda _, names: [
            n for n in names if n in {'target', '.git', '__pycache__', '.venv'}])
        for name, text in outputs.items():
            path = temp/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        (temp/'HDB2C3-R6-MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
        temp.rename(target)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return manifest


def compile_candidate(reference, candidate, *, skip_build=False):
    target = candidate/'target/r6-cpu'
    control = reference/'target/r5-cpu/release/hdb2c_r5_probe'
    before = digest(control)
    if not skip_build:
        env = {**os.environ, 'CARGO_TARGET_DIR': str(target)}
        subprocess.run(['cargo', 'build', '--release', '--manifest-path', str(candidate/'Cargo.toml'),
                        '--bin', 'hdb2c_r6_probe'], check=True, env=env)
        subprocess.run(['cargo', 'test', '--release', '--manifest-path', str(candidate/'Cargo.toml'),
                        '--lib', 'hdb2c6_tests'], check=True, env=env)
    if digest(control) != before:
        raise RuntimeError('R6 compilation modified the R5 control executable')
    return target


def exact_control(control, candidate, names, *, label):
    for name in names:
        row = compare_exact(read_tensor(control/f'{name}.f32le'),
                            read_tensor(candidate/f'{name}.f32le'), atol=0.)
        if not row['bitwise_equal']:
            raise RuntimeError('R6 '+label+' differs at '+name+' '+json.dumps(row))
    return {'status': 'PASS_BITWISE_ALL_FRAMES', 'stages_checked': len(names)}


def compare_mode(nemo, trace, valid):
    from .c3 import compare_stages
    full = compare_stages(nemo, trace, valid)
    blocks = full['stages'][2:]
    pad = read_tensor(nemo/'encoder_pad_mask.f32le')[:, :, 0].astype(bool)
    before = read_tensor(trace/'block0_conv_before_mask.f32le')
    after = read_tensor(trace/'block0_conv_after_mask.f32le')
    return {'encoder_stages': full['stages'],
            'all_encoder_stage_numeric_parity': full['status'],
            'encoder_blocks_numeric_parity': all(x['status'] == 'PASS_NUMERIC_PARITY' for x in blocks),
            'first_encoder_divergence': full['first_divergence'],
            'first_encoder_block_divergence': next((x['name'] for x in blocks if x['status'] != 'PASS_NUMERIC_PARITY'), None),
            'encoder_last': blocks[-1],
            'block0_stages': [{'name': name, **compare_valid(read_tensor(nemo/f'{name}.f32le'),
                read_tensor(trace/f'{name}.f32le'), valid)} for name in BLOCK_STAGES],
            'conv_depthwise_input': compare_exact(read_tensor(nemo/'block0_conv_after_mask.f32le'), after, atol=1e-3),
            'conv_padded_max_before': float(np.max(np.abs(before[pad]))) if pad.any() else 0.,
            'conv_padded_max_after': float(np.max(np.abs(after[pad]))) if pad.any() else 0.}


def execute(report, *, skip_build=False):
    from .cli import settings
    from .c3 import STAGES
    s = settings()
    report = report.resolve()
    r5_report = json.loads((report/'c3_positional_cpu.json').read_text())
    original = json.loads((report/'nemo_report.json').read_text())
    conversion = json.loads((s['model']/'conversion.json').read_text())
    if (r5_report['pcm_sha256'] != original['pcm_sha256'] or
            r5_report['checkpoint_sha256'] != original['checkpoint_sha256'] or
            digest(s['nemo']) != original['checkpoint_sha256'] or
            conversion['source_nemo_sha256'] != original['checkpoint_sha256']):
        raise ValueError('R6 checkpoint/PCM identity changed')
    if r5_report['nemo_eval_state']['after_eval']['training_module_count'] != 0:
        raise ValueError('R6 requires the verified eval-mode reference')
    reference = s['runtime'].with_name(s['runtime'].name+'-hdb2c-c3-r5-v0.7.5')
    candidate = s['runtime'].with_name(s['runtime'].name+'-hdb2c-c3-r6-v0.7.6')
    manifest = build_candidate(reference, candidate)
    target = compile_candidate(reference, candidate, skip_build=skip_build)
    out = report/'c3_cpu'
    nemo = out/'nemo_trace'
    with np.load(report/'nemo_mel.npz') as data:
        mel_values, mel_valid = data['values'], int(data['valid_frames'][0])
        shared = mel_values.transpose(0, 2, 1).astype('<f4').tobytes()
        mel_total = int(mel_values.shape[2])
    with np.load(report/'nemo_encoder.npz') as data:
        valid = int(data['valid_frames'][0])
    mel = out/'nemo_mel_injected_BTD.f32le'
    if mel.read_bytes() != shared:
        raise ValueError('R6 shared Mel differs from NeMo')
    state = json.loads((nemo/'nemo_padding_state.json').read_text())
    total = int(state['encoder_total_frames'])
    expected = expected_masks([valid], total)
    for name, want in zip(('encoder_pad_mask', 'encoder_att_mask'), expected):
        if not np.array_equal(read_tensor(nemo/f'{name}.f32le'), want.astype(np.float32)):
            raise ValueError('R6 actual NeMo mask differs from documented length semantics')
    baseline = out/'candle_r5_trace'
    rows = []
    for mode in MODES:
        trace, probe = out/f'candle_r6_{mode}_trace', out/f'candle_r6_{mode}_probe'
        trace.mkdir(exist_ok=True); probe.mkdir(exist_ok=True)
        env = {**os.environ, 'HDB2C6_MASK_MODE': mode, 'HDB2C3_TRACE_DIR': str(trace),
               'HDB2C3_NEMO_MEL_PATH': str(mel), 'HDB2C3_NEMO_MEL_FRAMES': str(mel_total),
               'HDB2C3_NEMO_MEL_VALID_FRAMES': str(mel_valid)}
        subprocess.run([str(target/'release/hdb2c_r6_probe'), '--model-dir', str(s['model']),
                        '--input', str(report/'input-16k.wav'), '--output', str(probe), '--cpu'], check=True, env=env)
        meta = json.loads((probe/'candle_probe.json').read_text())
        if ((probe/'candle_mel.f32le').read_bytes() != shared or
                meta.get('injected_mel_valid_frames') != mel_valid or meta.get('encoder_valid_frames') != valid or
                meta.get('dw_mask_propagation') != 'NEMO_MASKED_CONV_SEQUENTIAL'):
            raise ValueError('R6 consumed input or lengths differ')
        for name in ('encoder_pad_mask', 'encoder_att_mask'):
            if not compare_exact(read_tensor(nemo/f'{name}.f32le'), read_tensor(trace/f'{name}.f32le'), atol=0.)['bitwise_equal']:
                raise ValueError('R6 native mask differs from actual NeMo '+name)
        unchanged = exact_control(baseline, trace, ['pre_encode', 'post_pos_enc', 'block0_pos_emb'], label='upstream control')
        off = exact_control(baseline, trace, STAGES, label='mask-OFF control') if mode == 'none' else None
        no_pad = exact_control(baseline, trace, STAGES, label='no-padding control') if valid == total else None
        row = {'mode': mode, 'mask_bits': 'EXACT_NEMO_PAD_AND_ATTENTION_MASKS',
               'upstream_control': unchanged, 'off_control': off, 'no_padding_control': no_pad,
               **compare_mode(nemo, trace, valid)}
        if mode in ('convolution', 'both') and row['conv_padded_max_after'] != 0.:
            raise ValueError('R6 convolution input padding was not zeroed')
        rows.append(row)
        print('HDB2C3_R6_MODE='+mode+' final='+row['encoder_last']['status']+
              ' max_abs='+str(row['encoder_last']['max_abs'])+' cosine='+str(row['encoder_last']['cosine']), flush=True)
    full_pass = rows[-1]['encoder_last']['status'] == 'PASS_NUMERIC_PARITY'
    result = {'signature': SIGNATURE, 'revision': 'HDB2-C3-A-R6',
              'status': 'MASKING_FULL_OUTPUT_NUMERIC_PARITY' if full_pass else 'MASKING_OUTPUT_NUMERIC_PARITY_NOT_ESTABLISHED',
              'pcm_sha256': original['pcm_sha256'], 'checkpoint_sha256': original['checkpoint_sha256'],
              'valid_encoder_frames': valid, 'total_encoder_frames': total, 'padded_encoder_frames': total-valid,
              'nemo_padding_state': state, 'modes': rows,
              'criteria': {'activation_max_abs': 1e-3, 'activation_min_cosine': .999},
              'source_manifest': {k: v for k, v in manifest.items() if k != 'parent'},
              'full_encoder_exact_parity': 'NOT_ESTABLISHED', 'tdt_numeric_parity': 'NOT_TESTED',
              'gpu_parity': 'NOT_TESTED', 'encoder_internal_activation_or_positional_injection': 'NONE',
              'note': '2x2 native-mask experiment; shared NeMo Mel only. Final-output parity does not qualify all intermediate stages.'}
    path = report/'c3_padding_cpu.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print('HDB2C3_R6='+result['status'], flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--skip-build', action='store_true')
    args = parser.parse_args()
    execute(args.report, skip_build=args.skip_build)


if __name__ == '__main__':
    main()
