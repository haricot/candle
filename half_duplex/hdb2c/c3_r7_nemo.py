"""Observe the checkpoint's actual batched NeMo TDT decoder, without replacing it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audio import read_pcm
from .c3_r3 import _dump
from .c3_r7_metrics import advance
from .nemo_runner import load_nemo, ensure_probe_eval


def capture(model, encoded, lengths, folder):
    import torch
    decoder, joint = model.decoder, model.joint
    infer = model.decoding.decoding
    if type(infer).__name__ != 'GreedyBatchedTDTInfer':
        raise RuntimeError('R7 requires the checkpoint production batched TDT decoder')
    if any(m.training for m in model.modules()):
        raise RuntimeError('R7 requires recursive eval')
    if infer.decoding_computer.has_fusion_models():
        raise RuntimeError('R7 does not qualify fused LM/biasing')
    blank = int(decoder.blank_idx)
    durations = [int(v) for v in infer.durations]
    maximum = infer.max_symbols
    original_predict, original_after = decoder.predict, joint.joint_after_projection
    include = (infer.include_duration, infer.decoding_computer.include_duration)
    cache, predictors, decisions = {}, [], []
    hooks = []
    folder.mkdir(parents=True, exist_ok=True)
    def hook_enc(_module, _args, output):
        cache['encoder_projected'] = output.detach().clone()
    def hook_pred(_module, _args, output):
        cache['predictor_projected'] = output.detach().clone()
    def hook_raw(_module, _args, output):
        cache['raw'] = output.detach().clone()
    def predict(y=None, state=None, add_sos=True, batch_size=None):
        if add_sos:
            raise RuntimeError('R7 production decoder unexpectedly prepends SOS')
        number = len(predictors)
        name = f'{number:06}'
        out = folder/'predictor'/name
        out.mkdir(parents=True)
        token = int(y.item()) if y is not None else None
        if state is not None:
            for key, tensor in zip(('input_h', 'input_c'), state):
                _dump(out, key, tensor, 'layers,batch,hidden')
        result = original_predict(y=y, state=state, add_sos=add_sos, batch_size=batch_size)
        g, (h, c) = result
        for key, tensor in (('g', g), ('h', h), ('c', c)):
            _dump(out, key, tensor, 'native')
        predictors.append({'id': name, 'token': token, 'has_state': state is not None})
        cache['g'] = g.detach().clone()
        cache['predictor_id'] = name
        return result
    def after(f, g):
        projection = cache['encoder_projected']
        matches = (projection.contiguous().view(torch.int32) == f.contiguous().view(torch.int32)).all(dim=-1)
        indices = matches[0].nonzero().flatten().tolist()
        if len(indices) != 1:
            raise RuntimeError('R7 cannot identify the actual encoder frame uniquely')
        frame = int(indices[0])
        if not torch.equal(g.contiguous().view(torch.int32), cache['predictor_projected'].contiguous().view(torch.int32)):
            raise RuntimeError('R7 actual joint predictor does not match the observed projection')
        result = original_after(f, g)
        raw = cache['raw']
        raw_vector, vector = raw.flatten(), result.flatten()
        token = int(vector[:-len(durations)].argmax())
        duration_index = int(vector[-len(durations):].argmax())
        if token != int(raw_vector[:-len(durations)].argmax()) or duration_index != int(raw_vector[-len(durations):].argmax()):
            raise RuntimeError('R7 normalization changed an argmax')
        name = f'{len(decisions):06}'
        out = folder/'joint'/name
        out.mkdir(parents=True)
        for key, tensor in (('enc', encoded.transpose(1, 2)[:, frame:frame+1]),
                            ('g', cache['g']), ('logits', raw),
                            ('projected_enc', f), ('projected_pred', g)):
            _dump(out, key, tensor, 'native')
        decisions.append({'id': name, 'predictor_id': cache['predictor_id'], 'frame': frame,
                          'token': token, 'duration_index': duration_index, 'duration': durations[duration_index]})
        return result
    try:
        hooks = [joint.enc.register_forward_hook(hook_enc), joint.pred.register_forward_hook(hook_pred),
                 joint.joint_net.register_forward_hook(hook_raw)]
        decoder.predict, joint.joint_after_projection = predict, after
        infer.include_duration = infer.decoding_computer.include_duration = True
        with torch.inference_mode():
            hypothesis = infer(encoder_output=encoded, encoded_lengths=lengths)[0][0]
    finally:
        decoder.predict, joint.joint_after_projection = original_predict, original_after
        infer.include_duration, infer.decoding_computer.include_duration = include
        for hook in hooks:
            hook.remove()
    def integers(value):
        if isinstance(value, torch.Tensor):
            value = value.cpu().tolist()
        return [int(i) for i in value]
    ids, frames, emitted_durations = integers(hypothesis.y_sequence), integers(hypothesis.timestamp), integers(hypothesis.token_duration)
    symbols = 0
    for i, row in enumerate(decisions):
        next_frame, symbols, forced = advance(row['frame'], row['token'], blank, row['duration'], symbols, maximum)
        row['advance'] = next_frame-row['frame']
        row['forced_advance'] = forced
        if i+1 < len(decisions) and next_frame != decisions[i+1]['frame']:
            raise RuntimeError('R7 transition oracle differs from the observed production frame sequence')
    emitted = [r for r in decisions if r['token'] != blank]
    if (ids != [r['token'] for r in emitted] or frames != [r['frame'] for r in emitted]
            or emitted_durations != [r['duration'] for r in emitted]):
        raise RuntimeError('R7 observed production decisions differ from its returned hypothesis')
    result = {'reference_class': type(infer).__module__+'.'+type(infer).__name__,
              'production_algorithm': type(infer.decoding_computer).__name__, 'decoder_replaced': False,
              'blank': blank, 'durations_config': durations, 'max_symbols': maximum,
              'ids': ids, 'frames': frames, 'durations': emitted_durations,
              'text': model.tokenizer.ids_to_text(ids).strip(),
              'predictor_cases': predictors, 'decisions': decisions}
    (folder/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def execute(nemo, model_dir, report, output):
    import torch
    from safetensors import safe_open
    model, _ = load_nemo(nemo)
    eval_state = ensure_probe_eval(model)
    # Prove decoder/joint conversion bytes before comparing arithmetic.
    state = model.state_dict()
    weights = []
    with safe_open(str(model_dir/'model.safetensors'), framework='pt', device='cpu') as exported:
        for name, tensor in state.items():
            if not name.startswith(('decoder.', 'joint.')) or not tensor.is_floating_point():
                continue
            native = exported.get_tensor(name)
            original = tensor.detach().cpu().contiguous().float()
            if original.shape != native.shape or original.numpy().tobytes() != native.numpy().tobytes():
                raise RuntimeError('R7 converted TDT weights differ: '+name)
            weights.append({'name': name, 'shape': list(original.shape),
                            'sha256': hashlib.sha256(original.numpy().tobytes()).hexdigest()})
    if not weights:
        raise RuntimeError('R7 missing TDT weights')
    with np.load(report/'nemo_encoder.npz') as data:
        enc = torch.from_numpy(data['values'].copy())
        lengths = torch.from_numpy(data['valid_frames'].copy())
    with np.load(report/'nemo_mel.npz') as data:
        mel = torch.from_numpy(data['values'].copy()).transpose(1, 2)
    output.mkdir(parents=True, exist_ok=True)
    _dump(output, 'mel', mel, 'B,T,F')
    _dump(output, 'pcm', torch.from_numpy(read_pcm(report/'input-16k.wav')).reshape(1, -1, 1), 'B,T,1')
    reference = capture(model, enc, lengths, output)
    original = json.loads((report/'nemo_report.json').read_text())
    if reference['text'] != original['text']:
        raise RuntimeError('R7 production decode on the saved encoder differs from C1 text')
    reference.update({'eval_state': eval_state, 'weights': weights,
                      'blank_embedding_exact_zero': bool(torch.count_nonzero(model.decoder.prediction['embed'].weight[model.decoder.blank_idx]) == 0),
                      'torch_version': torch.__version__, 'joint_log_softmax': model.joint.log_softmax})
    (output/'manifest.json').write_text(json.dumps(reference, ensure_ascii=False, indent=2)+'\n')
    print('HDB2C3_R7_NEMO=ACTUAL_PRODUCTION_TDT_OBSERVED decisions='+str(len(reference['decisions'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('nemo', 'model-dir', 'report', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    execute(a.nemo, a.model_dir, a.report, a.output)
