"""Capture actual NeMo subsampling/positional/24-block encoder activations.

Model is restored from the pinned original NeMo checkpoint. Input is the saved
NeMo log-Mel tensor from the exact C1 report: NO second preprocessing pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .nemo_runner import load_nemo, ensure_probe_eval
from .c3_r3 import attach_nemo_hooks, verify_nemo_stages
from .c3_r4 import attach_nemo_r4_hooks, verify_r4_hooks
from .c3_r5 import attach_nemo_r5_hooks, prepare_positional_cases


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



def compare_nemo_reference(original, observed, valid_frames: int, *,
                           max_abs_limit: float = 1e-4,
                           min_cosine: float = 0.999999):
    """Test NeMo's own reproducibility on valid frames; retain padded-frame diagnostics."""
    a=np.asarray(original,dtype=np.float32)
    b=np.asarray(observed,dtype=np.float32)
    result={'stage':'HDB2C3_NEMO_SELF_REFERENCE','original_shape':list(a.shape),
            'observed_shape':list(b.shape),'valid_frames':int(valid_frames),
            'thresholds':{'max_abs':max_abs_limit,'min_cosine':min_cosine}}
    if a.ndim!=3 or a.shape!=b.shape or valid_frames<=0 or valid_frames>a.shape[-1]:
        return {**result,'status':'SHAPE_OR_VALID_LENGTH_MISMATCH'}
    x=a[...,:valid_frames].astype(np.float64)
    y=b[...,:valid_frames].astype(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return {**result,'status':'NONFINITE_VALID_FRAMES'}
    delta=np.abs(x-y)
    norm=np.linalg.norm(x)*np.linalg.norm(y)
    cosine=float(np.sum(x*y)/norm) if norm>0 else (1.0 if not delta.any() else 0.0)
    err=float(delta.max())
    return {**result,'max_abs':err,'rmse':float(np.sqrt(np.mean((x-y)**2))),
            'cosine':cosine,
            'max_abs_all_frames':float(np.abs(a.astype(np.float64)-b.astype(np.float64)).max()),
            'status':('PASS_VALID_FRAMES_NUMERIC_REPRODUCIBILITY'
                      if err<=max_abs_limit and cosine>=min_cosine
                      else 'NEMO_SELF_REFERENCE_DIVERGENCE')}

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
    # C1 transcribes before probing the restored model; reproduce this state.
    wav=report/'input-16k.wav'
    if not wav.is_file():raise FileNotFoundError(f'Missing normalized C1 WAV: {wav}')
    from .nemo_runner import transcribe
    warm_text=transcribe(model,wav)
    prior=json.loads((report/'nemo_report.json').read_text())
    if warm_text!=prior.get('text'):
        raise RuntimeError('NeMo C3 warmup transcript differs from C1; refuse mixed reference')
    c1_eval=prior.get('probes',{}).get('inference_state',{})
    if c1_eval.get('status')!='ALL_MODULES_EVAL_VERIFIED':
        raise RuntimeError('R5 requires a fresh C1 --probes reference with recursive eval proof')
    eval_state=ensure_probe_eval(model)
    output.mkdir(parents=True,exist_ok=True)
    (output/'nemo_eval_state.json').write_text(json.dumps(eval_state,indent=2)+'\n')
    hooks=[];emitted=set()
    dw_hooks,dw_emitted=attach_nemo_hooks(enc,output)
    r4_hooks,r4_emitted=attach_nemo_r4_hooks(enc,output)
    r5_hooks,r5_emitted=attach_nemo_r5_hooks(enc,output)
    def hook(name):
        def fn(_module,_args,result):
            if name in emitted: raise RuntimeError(f'Duplicate stage {name}')
            save_stage(output,name,result);emitted.add(name)
        return fn
    try:
        hooks.extend(dw_hooks)
        hooks.extend(r4_hooks)
        hooks.extend(r5_hooks)
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
    verify_nemo_stages(dw_emitted)
    verify_r4_hooks(r4_emitted)
    if r5_emitted!={'pos_input','pos_output'}:
        raise RuntimeError('R5 positional boundary hooks incomplete')
    # Compare the final forward output with original NeMo C1 encoder trace.
    base=np.load(report/'nemo_encoder.npz')
    orig=np.asarray(base['values'],dtype=np.float32)
    new=feat.detach().cpu().numpy()
    valid_frames=int(base['valid_frames'][0])
    result=compare_nemo_reference(orig,new,valid_frames)
    result['warmup_transcription_matched_c1']=True
    result['c3_valid_frames']=int(feat_len[0])
    (output/'nemo_reference_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('HDB2C3_NEMO_SELF_REFERENCE='+result['status']+
          ' max_abs='+str(result.get('max_abs'))+
          ' rmse='+str(result.get('rmse'))+
          ' cosine='+str(result.get('cosine')),flush=True)
    if int(feat_len[0])!=valid_frames:
        raise RuntimeError('NeMo C3 valid length differs from C1: '+str(result))
    if result['status']!='PASS_VALID_FRAMES_NUMERIC_REPRODUCIBILITY':
        raise RuntimeError('NeMo C3 valid frames differ from C1: '+str(result))
    prepare_positional_cases(enc.pos_enc,output)
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
