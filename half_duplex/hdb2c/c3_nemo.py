"""Capture actual NeMo subsampling/positional/24-block encoder activations.

Model is restored from the pinned original NeMo checkpoint. Input is the saved
NeMo log-Mel tensor from the exact C1 report: NO second preprocessing pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .nemo_runner import load_nemo


def save_stage(folder:Path, name:str, output) -> None:
    import torch
    tensor=output[0] if isinstance(output,(tuple,list)) else output
    if not isinstance(tensor,torch.Tensor):
        raise TypeError(f'NeMo {name} did not return a Tensor')
    a=tensor.detach().to('cpu',dtype=torch.float32).contiguous().numpy()
    if a.ndim!=3 or a.shape[0]!=1:
        raise ValueError(f'NeMo {name}: require B,T,D tensor, got {list(a.shape)}')
    if not np.isfinite(a).all():raise ValueError(f'Nonfinite NeMo stage {name}')
    raw=a.astype('<f4',copy=False)
    file=folder/f'{name}.f32le';file.write_bytes(raw.tobytes(order='C'))
    (folder/f'{name}.json').write_text(json.dumps({'shape':list(a.shape),'layout':'B,T,D','dtype':'f32le',
        'sha256':hashlib.sha256(file.read_bytes()).hexdigest()},indent=2)+'\n')


def execute(nemo:Path,report:Path,output:Path):
    import torch
    model,_=load_nemo(nemo)
    obj=np.load(report/'nemo_mel.npz')
    mel=np.asarray(obj['values'],dtype=np.float32)
    valid=np.asarray(obj['valid_frames'],dtype=np.int64)
    if mel.ndim!=3 or mel.shape[1]!=128 or mel.shape[0]!=1 or valid.shape!=(1,):
        raise ValueError('Expected actual B,128,T NeMo C1 Mel with one valid length')
    enc=model.encoder
    if not hasattr(enc,'pre_encode') or not hasattr(enc,'pos_enc') or len(enc.layers)!=24:
        raise RuntimeError('Unexpected NeMo FastConformer architecture; no fake/partial C3')
    output.mkdir(parents=True,exist_ok=True)
    hooks=[];emitted=set()
    def hook(name):
        def fn(_module,_args,result):
            if name in emitted: raise RuntimeError(f'Duplicate stage {name}')
            save_stage(output,name,result);emitted.add(name)
        return fn
    try:
        hooks.append(enc.pre_encode.register_forward_hook(hook('pre_encode')))
        hooks.append(enc.pos_enc.register_forward_hook(hook('post_pos_enc')))
        for i,layer in enumerate(enc.layers):
            hooks.append(layer.register_forward_hook(hook(f'layer_{i:02}')))
        with torch.inference_mode():
            feat,feat_len=enc(audio_signal=torch.from_numpy(mel.copy()),length=torch.tensor(valid))
    finally:
        for h in hooks:h.remove()
    expected={'pre_encode','post_pos_enc'}|{f'layer_{i:02}' for i in range(24)}
    if emitted!=expected:raise RuntimeError(f'NeMo missing stages: {sorted(expected-emitted)}')
    # Compare the final forward output with original NeMo C1 encoder trace.
    base=np.load(report/'nemo_encoder.npz')
    orig=np.asarray(base['values'],dtype=np.float32)
    new=feat.detach().cpu().numpy()
    if orig.shape!=new.shape or not np.allclose(orig,new,rtol=0,atol=1e-6):
        raise RuntimeError('NeMo C3 encoder forward differs from its C1 reference')
    if int(feat_len[0])!=int(base['valid_frames'][0]):
        raise RuntimeError('NeMo C3 valid output length differs from C1')
    meta={'stage':'HDB2C3_NEMO_LAYER_TRACE','status':'REAL_26_CHECKPOINTS',
        'stages':sorted(emitted),'source_nemo_sha256':hashlib.sha256(nemo.read_bytes()).hexdigest(),
        'encoder_valid_frames':int(feat_len[0])}
    (output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('HDB2C3_NEMO_TRACE=26_STAGES_AND_C1_OUTPUT_VERIFIED',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--nemo',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();execute(a.nemo,a.report,a.output)

if __name__=='__main__':main()