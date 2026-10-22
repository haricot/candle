"""Real NeMo reference and observation-only primitive traces for R8.

Functional frontend operations are observed by inserting saves into the installed
NeMo forward source. An independent uninstrumented forward must match all bytes.
Common-input replays are diagnostic executions, never a substitute for inference.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import textwrap
import types
from pathlib import Path

import numpy as np

from .audio import read_pcm, describe
from .c3_nemo import save_stage
from .c3_r3 import _dump as _raw_dump, attach_nemo_hooks, verify_nemo_stages, STAGES as DW_STAGES
from .c3_r4 import attach_nemo_r4_hooks, verify_r4_hooks
from .c3_r5 import attach_nemo_r5_hooks
from .c3_r6 import attach_nemo_r6_hooks
from .c3_r7_nemo import capture
from .c3_r7_metrics import tensor_metrics
from .c3_r8_source import once
from .nemo_runner import load_nemo, ensure_probe_eval

FRONTEND_STAGES = ('preemphasis', 'window', 'filterbank', 'frames', 'fft_complex',
                   'magnitude', 'power', 'mel_projection', 'log_mel', 'normalized', 'mel_output')


def _dump(folder, name, tensor, layout):
    folder.mkdir(parents=True, exist_ok=True)
    _raw_dump(folder, name, tensor, layout)


def instrument_frontend(source):
    """Only add observers; preserve every installed NeMo arithmetic expression."""
    source = textwrap.dedent(source)
    source = once(source, '    # disable autocast to get full range of stft values',
        '    _r8_save("preemphasis", x)\n'
        '    _r8_save("window", self.window)\n'
        '    _r8_save("filterbank", self.fb[0])\n'
        '    # disable autocast to get full range of stft values')
    source = once(source, '        x = self.stft(x)',
        '        _r8_frames(self, x)\n        x = self.stft(x)\n'
        '        _r8_save("fft_complex", torch.view_as_real(x)[0].permute(1, 0, 2))')
    source = once(source, '    x = torch.sqrt(x.pow(2).sum(-1) + guard)',
        '    x = torch.sqrt(x.pow(2).sum(-1) + guard)\n'
        '    _r8_save("magnitude", x[0].transpose(0, 1))')
    source = once(source, '    # return plain spectrogram if required',
        '    _r8_save("power", x[0].transpose(0, 1))\n    # return plain spectrogram if required')
    source = once(source, '        x = torch.matmul(self.fb.to(x.dtype), x)',
        '        x = torch.matmul(self.fb.to(x.dtype), x)\n        _r8_save("mel_projection", x[0])')
    source = once(source, '    # frame splicing if required',
        '    _r8_save("log_mel", x[0])\n    # frame splicing if required')
    source = once(source, '    # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)',
        '    _r8_save("normalized", x[0])\n'
        '    # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)')
    return source


def frontend_reference(model, pcm, output):
    import torch
    f = model.preprocessor.featurizer
    if type(f).__name__ != 'FilterbankFeatures' or f.exact_pad or f.training:
        raise ValueError('R8 only qualifies actual eval FilterbankFeatures, exact_pad=False')
    x = torch.from_numpy(pcm.copy())[None, :]
    length = torch.tensor([len(pcm)], dtype=torch.int64)
    with torch.inference_mode():
        baseline, baseline_length = model.preprocessor(input_signal=x, length=length)
    emitted = set()
    def save(name, tensor):
        if name in emitted:
            raise ValueError('duplicate R8 frontend observation '+name)
        _dump(output, name, tensor, 'R8_CANONICAL')
        emitted.add(name)
    def frames(featurizer, waveform):
        window = torch.nn.functional.pad(featurizer.window,
            ((featurizer.n_fft-featurizer.win_length)//2,
             featurizer.n_fft-featurizer.win_length-(featurizer.n_fft-featurizer.win_length)//2))
        padded = torch.nn.functional.pad(waveform, (featurizer.n_fft//2, featurizer.n_fft//2))
        framed = padded[0].unfold(0, featurizer.n_fft, featurizer.hop_length)*window
        save('frames', framed)
        # A derived framing view is only attributable to the actual STFT after
        # its independent FFT reproduces that STFT's output. Check below.
        _dump(output, 'derived_frames_fft', torch.view_as_real(torch.fft.rfft(framed)), 'T,K,RI')
    original = f.forward
    source = inspect.getsource(type(f).forward)
    namespace = {**type(f).forward.__globals__, '_r8_save': save, '_r8_frames': frames}
    exec(compile(instrument_frontend(source), '<R8-observed-installed-NeMo-forward>', 'exec'), namespace)
    try:
        f.forward = types.MethodType(namespace['forward'], f)
        with torch.inference_mode():
            observed, observed_length = model.preprocessor(input_signal=x, length=length)
        save('mel_output', observed.transpose(1, 2))
    finally:
        f.forward = original
    if emitted != set(FRONTEND_STAGES):
        raise ValueError('incomplete R8 frontend observations: '+str(emitted))
    control = tensor_metrics(baseline.numpy(), observed.numpy())
    if not control['bitwise_equal'] or not torch.equal(baseline_length, observed_length):
        raise RuntimeError('R8 frontend instrumentation changed actual NeMo output')
    from .c3_r3 import read_tensor
    fft_control = tensor_metrics(read_tensor(output/'fft_complex.f32le'), read_tensor(output/'derived_frames_fft.f32le'))
    if not fft_control['bitwise_equal']:
        raise RuntimeError('R8 derived common frames do not reproduce actual torch.stft bytes: '+json.dumps(fft_control))
    return baseline, baseline_length, {'instrumentation_control': control,
        'derived_frames_fft_control': fft_control,
        'installed_forward_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'common_frames_scope': 'DERIVED_FRAMES_VERIFIED_AGAINST_ACTUAL_STFT_BITS'}


def operator_hooks(enc, folder):
    import torch
    hooks, emitted = [], set()
    def attach(module, name):
        def before(_module, args):
            if name in emitted:
                raise ValueError('R8 duplicate subsampling op '+name)
            _dump(folder/name, 'input', args[0], 'NATIVE')
        def after(_module, _args, output):
            _dump(folder/name, 'output', output, 'NATIVE')
            if name == 'out':
                # Supplemental accuracy reference; never returned to the encoder.
                double = torch.nn.functional.linear(_args[0].double(),
                    _module.weight.double(),
                    None if _module.bias is None else _module.bias.double())
                _dump(folder/name, 'output_fp64_rounded', double.float(), 'NATIVE')
            emitted.add(name)
        hooks.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
        for attr in ('weight', 'bias'):
            value = getattr(module, attr, None)
            if value is not None:
                _dump(folder/name, attr, value, 'NATIVE_WEIGHT')
    for index in (0, 2, 3, 5, 6):
        attach(enc.pre_encode.conv[index], f'conv_{index:02}')
    attach(enc.pre_encode.out, 'out')
    counter = [0]
    def pre_relu(_module, args):
        name = ('relu_00', 'relu_03', 'relu_06')[counter[0]]
        _dump(folder/name, 'input', args[0], 'B,C,T,F')
    def post_relu(_module, _args, output):
        name = ('relu_00', 'relu_03', 'relu_06')[counter[0]]
        _dump(folder/name, 'output', output, 'B,C,T,F')
        emitted.add(name); counter[0] += 1
    seen = set()
    for index in (1, 4, 7):
        module = enc.pre_encode.conv[index]
        if not isinstance(module, torch.nn.ReLU):
            raise ValueError('unexpected real NeMo ReLU')
        if id(module) not in seen:
            seen.add(id(module))
            hooks.extend([module.register_forward_pre_hook(pre_relu), module.register_forward_hook(post_relu)])
    return hooks, emitted


def tokenizer_diagnostic(model, config, result):
    """Describe a text mismatch without publishing the transcript or token IDs."""
    from sentencepiece import sentencepiece_model_pb2
    tokenizer = model.tokenizer
    sp = tokenizer.tokenizer
    proto = sentencepiece_model_pb2.ModelProto.FromString(sp.serialized_model_proto())
    vocabulary = config['joint']['vocabulary']
    if vocabulary != [p.piece for p in proto.pieces]:
        raise ValueError('R8B converted vocabulary is not the actual SentencePiece vocabulary')
    ids = result['ids']
    legacy = ''.join(vocabulary[i].replace('▁',' ') for i in ids).strip()
    reference = result['text']
    counts = {name:0 for name in ('normal','unknown','control','user_defined','unused','byte')}
    types = {1:'normal',2:'unknown',3:'control',4:'user_defined',5:'unused',6:'byte'}
    for i in ids: counts[types[proto.pieces[i].type]] += 1
    first = next((i for i,(a,b) in enumerate(zip(reference,legacy)) if a!=b),
                 min(len(reference),len(legacy)) if reference!=legacy else None)
    def point(text):
        return None if first is None or first>=len(text) else f'U+{ord(text[first]):04X}'
    return {'scope':'ACTUAL_TOKENIZER_ON_IDENTICAL_EMITTED_IDS',
        'tokenizer_class':type(tokenizer).__name__,'legacy':bool(tokenizer.legacy),
        'sentencepiece_model_sha256':hashlib.sha256(sp.serialized_model_proto()).hexdigest(),
        'vocabulary_bits_verified':True,'piece_categories':counts,
        'vocabulary_join_exact':legacy==reference,'first_difference_character':first,
        'reference_codepoint':point(reference),'vocabulary_join_codepoint':point(legacy),
        'reference_characters':len(reference),'vocabulary_join_characters':len(legacy),
        'unknown_surface_codepoints':[f'U+{ord(c):04X}' for c in proto.trainer_spec.unk_surface],
        'add_dummy_prefix':proto.normalizer_spec.add_dummy_prefix,
        'remove_extra_whitespaces':proto.normalizer_spec.remove_extra_whitespaces,
        'has_denormalizer':bool(proto.denormalizer_spec.precompiled_charsmap)}


def execute(nemo, model_dir, wav, output):
    import torch
    from safetensors import safe_open
    model, _ = load_nemo(nemo)
    eval_state = ensure_probe_eval(model)
    if len(model.encoder.layers) != 24:
        raise ValueError('R8 requires the actual 24-block TranscribePsy encoder')
    pcm = read_pcm(wav)
    output.mkdir(parents=True)
    mel, mel_length, front_control = frontend_reference(model, pcm, output/'frontend')
    _dump(output, 'pcm', torch.from_numpy(pcm.copy()).reshape(1,-1,1), 'B,T,1')
    _dump(output, 'mel', mel.transpose(1,2), 'B,T,F')
    np.savez(output/'mel.npz', values=mel.numpy(), valid_frames=mel_length.numpy())
    with torch.inference_mode():
        baseline, baseline_length = model.encoder(audio_signal=mel, length=mel_length)
    trace = output/'encoder_trace'; trace.mkdir()
    hooks, stages = [], set()
    dw, dw_emitted = attach_nemo_hooks(model.encoder, trace); hooks.extend(dw)
    r4, r4_emitted = attach_nemo_r4_hooks(model.encoder, trace); hooks.extend(r4)
    r5, r5_emitted = attach_nemo_r5_hooks(model.encoder, trace); hooks.extend(r5)
    r6, r6_emitted = attach_nemo_r6_hooks(model.encoder, trace); hooks.extend(r6)
    ops, ops_emitted = operator_hooks(model.encoder, output/'subsampling'); hooks.extend(ops)
    def observe(name):
        def hook(_module, _args, result):
            if name in stages: raise ValueError('duplicate R8 encoder stage')
            save_stage(trace, name, result); stages.add(name)
        return hook
    hooks.append(model.encoder.pre_encode.register_forward_hook(observe('pre_encode')))
    hooks.append(model.encoder.pos_enc.register_forward_hook(observe('post_pos_enc')))
    hooks.extend(layer.register_forward_hook(observe(f'layer_{i:02}')) for i, layer in enumerate(model.encoder.layers))
    try:
        with torch.inference_mode():
            encoded, encoded_length = model.encoder(audio_signal=mel, length=mel_length)
    finally:
        for hook in hooks: hook.remove()
    verify_nemo_stages(dw_emitted); verify_r4_hooks(r4_emitted)
    if (ops_emitted != set(DW_STAGES) or len(stages)!=26 or
            r5_emitted!={'pos_input','pos_output'} or
            r6_emitted!={'encoder_pad_mask','encoder_att_mask','block0_conv_after_mask'}):
        raise ValueError('incomplete R8 real NeMo encoder/operator trace')
    control = tensor_metrics(baseline.numpy(), encoded.numpy())
    if not control['bitwise_equal'] or not torch.equal(encoded_length, baseline_length):
        raise RuntimeError('R8 encoder instrumentation changed actual NeMo output')
    weights = []
    with safe_open(str(model_dir/'model.safetensors'), framework='pt', device='cpu') as exported:
        for name, value in model.state_dict().items():
            if name.startswith(('decoder.', 'joint.')) and value.is_floating_point():
                original = value.detach().cpu().contiguous().float()
                converted = exported.get_tensor(name)
                if original.shape != converted.shape or original.numpy().tobytes()!=converted.numpy().tobytes():
                    raise ValueError('R8 converted TDT weight differs: '+name)
                weights.append({'name':name,'sha256':hashlib.sha256(original.numpy().tobytes()).hexdigest()})
    if not weights: raise ValueError('R8 no TDT weights checked')
    result = capture(model, baseline, baseline_length, output)
    result['tokenizer_diagnostic'] = tokenizer_diagnostic(model,
        json.loads((model_dir/'config.json').read_text()), result)
    result.update({'pcm_sha256':describe(wav)['pcm_sha256'],
        'checkpoint_sha256':hashlib.file_digest(nemo.open('rb'),'sha256').hexdigest(),
        'mel_valid':int(mel_length[0]),'encoder_valid':int(baseline_length[0]),
        'eval_state':eval_state,'frontend_control':front_control,'encoder_instrumentation_control':control,
        'converted_tdt_weights':weights,
        'preprocessor_state':{'pad_to':int(model.preprocessor.featurizer.pad_to),'pad_value':float(model.preprocessor.featurizer.pad_value)},
        'reference_scope':'ACTUAL_UNREPLACED_NEMO_GREEDY_BATCHED_TDT_ON_REAL_PCM'})
    if os.environ.get('HDB2C_R14_FP64_REFERENCE') == '1':
        from .c3_r14_precision import Fp64Auditor
        from .c3_r8 import write_json
        ensure_probe_eval(model)
        control = Fp64Auditor(model).evaluate(output)
        control['checkpoint_sha256'] = result['checkpoint_sha256']
        control['pcm_sha256'] = result['pcm_sha256']
        published = Path(os.environ.get('GITHUB_WORKSPACE', str(output.parent)))/'halfduplex-ci-published'
        write_json(published/('r14-fp64-reference-'+result['pcm_sha256'][:16]+'.json'), control)
        if control['execution_status'] != 'COMPLETE':
            raise RuntimeError('R14 real FP64 reference operator control incomplete')
        result['r14_fp64_reference_control'] = {'execution_status': 'COMPLETE',
            'local_operator_replays': len(control['local_operator_replays']),
            'reference_dtype': 'float64', 'original_model_dtype_preserved': True}
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('HDB2C3_R8_NEMO=REAL_REFERENCE_AND_PRIMITIVES decisions='+str(len(result['decisions'])),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ('nemo','model-dir','wav','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args(); execute(a.nemo,a.model_dir,a.wav,a.output)
