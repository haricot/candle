#!/usr/bin/env python3
"""Execute actual installed NeMo modules in FP64 on small operator fixtures.

This verifies oracle execution/dtypes and preservation of the FP32 original;
it does not claim model/corpus/GPU inference or ASR accuracy.
"""
import json
import math
import shutil
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer
from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionalEncoding

from hdb2c.c3_r3 import _dump, read_tensor
from hdb2c.c3_r4 import attach_nemo_r4_hooks, verify_r4_hooks
from hdb2c.c3_r6 import attach_nemo_r6_hooks
from hdb2c.c3_r8_nemo import operator_hooks
from hdb2c.c3_r14_precision import Fp64Auditor


def main():
    torch.manual_seed(14)
    model = nn.Module()
    enc = nn.Module()
    model.encoder = enc
    enc.self_attention_model = 'rel_pos'
    enc.att_context_style = 'regular'
    enc.att_context_size = [-1, -1]
    enc.use_pad_mask = True
    enc.pre_encode = nn.Module()
    enc.pre_encode.conv = nn.ModuleList([
        nn.Conv2d(1, 2, 3, padding=1), nn.ReLU(), nn.Conv2d(2, 2, 1),
        nn.Conv2d(2, 2, 3, padding=1, groups=2), nn.ReLU(), nn.Conv2d(2, 2, 1),
        nn.Conv2d(2, 2, 3, padding=1, groups=2), nn.ReLU()])
    enc.pre_encode.out = nn.Linear(8, 16)
    enc.pos_enc = RelPositionalEncoding(16, 0., max_len=64, xscale=math.sqrt(16), dropout_rate_emb=0.)
    enc.pos_enc.extend_pe(16, device=torch.device('cpu'), dtype=torch.float32)
    enc.layers = nn.ModuleList([ConformerLayer(16, 32, n_heads=2, conv_kernel_size=3,
        conv_norm_type='layer_norm', dropout=0., dropout_att=0.)]+[nn.Identity() for _ in range(23)])
    model.eval()
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        trace = root/'encoder_trace'
        trace.mkdir()
        ops, emitted = operator_hooks(enc, root/'subsampling')
        hooks, stages = attach_nemo_r4_hooks(enc, trace)
        masks, _ = attach_nemo_r6_hooks(enc, trace)
        try:
            with torch.inference_mode():
                x = torch.linspace(-1., 1., 16*4).reshape(1, 1, 16, 4)
                for module in enc.pre_encode.conv:
                    x = module(x)
                x = enc.pre_encode.out(x.transpose(1, 2).reshape(1, 16, 8))
                _dump(trace, 'pre_encode', x, 'B,T,D')
                x, pos = enc.pos_enc(x)
                _dump(trace, 'post_pos_enc', x, 'B,T,D')
                pad = torch.arange(16)[None, :] >= 14
                att = pad[:, :, None] | pad[:, None, :]
                y = enc.layers[0](x=x, att_mask=att, pos_emb=pos, pad_mask=pad)
                assert y.dtype == torch.float32
        finally:
            for hook in [*ops, *hooks, *masks]:
                hook.remove()
        verify_r4_hooks(stages)
        assert len(emitted) == 9
        auditor = Fp64Auditor(model)
        control = auditor.evaluate(root)
        if control['execution_status'] != 'COMPLETE':
            raise RuntimeError(json.dumps(control['errors']))
        assert len(control['local_operator_replays']) == 23
        assert all(row['reference_dtype'] == 'float64' for row in control['local_operator_replays'])
        # Exercise the full native trace layout using CPU-only fixtures. A
        # changed operand/output pair must be replayed on its own input, not
        # on the original NeMo input. No ASR or CUDA result is fabricated.
        candidate = root/'candidate'
        shutil.copytree(trace, candidate/'encoder_trace')
        for name in emitted:
            for suffix in ('.json', '.f32le'):
                shutil.copyfile(root/'subsampling'/name/('input'+suffix),
                                candidate/'encoder_trace'/('input_'+name+suffix))
                shutil.copyfile(root/'subsampling'/name/('output'+suffix),
                                candidate/'encoder_trace'/(name+suffix))
        changed = torch.from_numpy(read_tensor(root/'subsampling/conv_00/input.f32le')).clone()+.25
        with torch.inference_mode():
            changed_result = enc.pre_encode.conv[0](changed)
        _dump(candidate/'encoder_trace', 'input_conv_00', changed, 'B,C,T,F')
        _dump(candidate/'encoder_trace', 'conv_00', changed_result, 'B,C,T,F')
        replay = auditor.evaluate(root, candidate, root)
        if replay['execution_status'] != 'COMPLETE':
            raise RuntimeError(json.dumps(replay['errors']))
        assert len(replay['local_operator_replays']) == 41
        own = next(r for r in replay['local_operator_replays']
                   if r['name'] == 'conv_00' and r['scope'] == 'AUTONOMOUS_FP32_ON_OWN_OPERANDS')
        assert own['fp32_vs_fp64']['max_abs'] < 1e-5
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())
        assert all(p.dtype == torch.float32 for p in model.parameters())
    print('HDB2C3_R14_NEMO_FP64_MODULES=PASS_23_REFERENCE_41_TRACE_LAYOUT_REPLAYS_NO_ASR_NO_GPU', flush=True)


if __name__ == '__main__':
    main()
