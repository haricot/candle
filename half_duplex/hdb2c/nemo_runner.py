"""Isolated **real NeMo** inference, CPU, original .nemo; optional internal probes.

Never loads Candle weights. Fail closed on restoration/preprocessor errors.
"""
import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from .audio import sha256, describe, read_pcm, report


def transcribe(model,path):
    # NeMo 2.x accepts paths; avoids assumptions about encoder/decoder shapes.
    out=model.transcribe(audio=[str(path)],batch_size=1)
    if not out: raise RuntimeError('NeMo returned empty hypothesis list')
    obj=out[0]
    return str(getattr(obj,'text',obj)).strip()


def load_nemo(path):
    import torch
    from nemo.collections.asr.models import ASRModel
    model=ASRModel.restore_from(restore_path=str(path),map_location='cpu')
    model=model.to('cpu').eval()
    # Exactly what got instantiated, not a model inferred from filenames.
    return model,torch


def ensure_probe_eval(model):
    """Transcription teardown can unfreeze/train children of an eval parent.

    torch.inference_mode() controls autograd, not dropout or BatchNorm.
    Record the actual module tree, then recursively restore inference semantics.
    """
    def state():
        training=[name or '<model>' for name,module in model.named_modules() if module.training]
        encoder=getattr(model,'encoder',None)
        pos=getattr(encoder,'pos_enc',None)
        drops={}
        for attr in ('dropout','dropout_emb'):
            module=getattr(pos,attr,None)
            if module is not None:
                drops[attr]={'p':float(module.p),'training':bool(module.training)}
        return {'model_training':bool(model.training),
                'encoder_training':bool(encoder.training) if encoder is not None else None,
                'training_module_count':len(training),'training_module_names':training[:12],
                'position_dropout':drops}
    before=state()
    model.eval()
    after=state()
    if after['training_module_count']:
        raise RuntimeError('NeMo probe still has training modules after eval: '+repr(after))
    return {'status':'ALL_MODULES_EVAL_VERIFIED','before_eval':before,'after_eval':after}


def make_probes(model,torch,wav,output):
    import numpy as np
    eval_state=ensure_probe_eval(model)
    (output/'nemo_eval_state.json').write_text(json.dumps(eval_state,indent=2)+'\n')
    x=read_pcm(wav)
    audio=torch.from_numpy(x.copy())[None,:]
    lengths=torch.tensor([len(x)],dtype=torch.int64)
    with torch.inference_mode():
        mel,mel_len=model.preprocessor(input_signal=audio,length=lengths)
        mel=mel.detach().to('cpu',dtype=torch.float32)
        mellen=int(mel_len[0].item())
        # Always save full torch padded output and valid frames independently.
        # NeMo: [1,features,frames]
        np.savez_compressed(str(output/'nemo_mel.npz'),values=mel.numpy(),valid_frames=np.array([mellen],dtype=np.int64))
        (output/'nemo_mel.json').write_text(json.dumps({'tensor_shape':list(mel.shape),'valid_frames':mellen,
                            'layout':'batch,features,frames','dtype':'float32'},indent=2)+'\n')
        feat,feat_len=model.encoder(audio_signal=mel,length=mel_len)
        feat=feat.detach().to('cpu',dtype=torch.float32)
        flen=int(feat_len[0].item())
        np.savez_compressed(str(output/'nemo_encoder.npz'),values=feat.numpy(),valid_frames=np.array([flen],dtype=np.int64))
        (output/'nemo_encoder.json').write_text(json.dumps({'tensor_shape':list(feat.shape),'valid_frames':flen,
                             'layout':'NeMo encoder native (inspect axes)','dtype':'float32'},indent=2)+'\n')
    return {'inference_state':eval_state,
            'mel':{'tensor_shape':list(mel.shape),'valid_frames':mellen},
            'encoder':{'tensor_shape':list(feat.shape),'valid_frames':flen}}


def run(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    nemo=Path(args.nemo).resolve();wav=Path(args.wav).resolve()
    if not nemo.is_file():raise FileNotFoundError('Missing original NeMo checkpoint '+str(nemo))
    a=describe(wav)
    start=time.perf_counter();model,torch=load_nemo(nemo)
    restore_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter();text=transcribe(model,wav)
    asr_ms=(time.perf_counter()-start)*1000
    if not text:raise RuntimeError('NeMo produced empty transcription')
    result={'schema_version':1,'stage':'HDB2C_NEMO_ORIGINAL','status':'REAL_NEMO_ASR',
            'model_class':f'{type(model).__module__}.{type(model).__name__}',
            'checkpoint':str(nemo),'checkpoint_sha256':sha256(nemo),
            'pcm_sha256':a['pcm_sha256'],'device':'cpu','torch_version':torch.__version__,
            'python_version':platform.python_version(),'restore_ms':round(restore_ms,3),
            'asr_ms':round(asr_ms,3),'text':text,'probe_status':'NOT_REQUESTED'}
    (output/'nemo_text.txt').write_text(text+'\n',encoding='utf-8')
    if args.probe:
        result['probes']=make_probes(model,torch,wav,output)
        result['probe_status']='REAL_NEMO_MEL_AND_ENCODER'
    report(output/'nemo_report.json',result)
    print('HDB2C_NEMO=REAL_CPU_INFERENCE'+(' PROBES=REAL' if args.probe else ''))
    print('NEMO_TEXT='+text)
    print('HDB2C_NEMO_REPORT='+str(output/'nemo_report.json'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--nemo',required=True,type=Path)
    p.add_argument('--wav',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--probe',action='store_true')
    a=p.parse_args()
    try:run(a)
    except Exception as exc:
        # Evidence: no faux success on import/restoration/version errors.
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        report(out/'nemo_report.json',{'stage':'HDB2C_NEMO_ORIGINAL','status':'NOT_QUALIFIED',
               'error':str(exc),'traceback':traceback.format_exc()[-10000:]})
        traceback.print_exc();sys.exit(1)
if __name__=='__main__':main()
