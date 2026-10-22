"""C1/C2/C3 gates. Compare ONLY real checkpoint inference and real tensors."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audio import report
from .metrics import wer, compare_arrays, normalize_text


def read_nemo(path,name):
    data=np.load(path/f'nemo_{name}.npz')
    return data['values'], int(data['valid_frames'][0])


def read_candle(path,name,probe):
    m=probe[name]
    raw=np.fromfile(path/m['filename'],dtype='<f4')
    shape=m['shape']
    if raw.size!=int(np.prod(shape)):
        raise ValueError(f'{name}: f32 raw values length mismatch for shape {shape}')
    return raw.reshape(shape)


def compare_probes(nemo_dir,candle_dir,output):
    nemo_dir,candle_dir=Path(nemo_dir),Path(candle_dir)
    nm=json.loads((nemo_dir/'nemo_report.json').read_text())
    cp=json.loads((candle_dir/'candle_probe.json').read_text())
    if nm.get('status')!='REAL_NEMO_ASR' or nm.get('probe_status')!='REAL_NEMO_MEL_AND_ENCODER':
        raise ValueError('NeMo real inference and probes not qualified')
    if cp.get('status')!='REAL_CANDLE_MEL_AND_ENCODER':
        raise ValueError('Candle real tensor probes not qualified')
    result={'stage':'HDB2C_REAL_NUMERICAL_PROBES','status':'NOT_QUALIFIED',
            'notes':['Compare tensor outputs, not model names or test doubles.']}
    for name in ('mel','encoder'):
        nt,nvalid=read_nemo(nemo_dir,name)
        ct=read_candle(candle_dir,name,cp)
        if nt.ndim!=3 or ct.ndim!=3 or nt.shape[0]!=1 or ct.shape[0]!=1:
            result[name]={'status':'UNEXPECTED_TENSOR_RANK','nemo_shape':list(nt.shape),
                          'candle_shape':list(ct.shape)};continue
        # NeMo outputs [batch, channels, frames], Candle [batch, frames, channels].
        n_aligned=nt.transpose(0,2,1)
        cvalid=int(cp.get('nemo_valid_mel_frames',ct.shape[1])) if name=='mel' else int(cp['encoder_valid_frames'])
        if not (0 < nvalid <= n_aligned.shape[1]) or not (0 < cvalid <= ct.shape[1]):
            result[name]={'status':'INVALID_VALID_LENGTH', 'nemo_valid_frames':nvalid,
                          'candle_valid_frames':cvalid, 'nemo_shape':list(nt.shape),
                          'candle_shape':list(ct.shape)}
            continue
        # Compare only valid frames; padded NeMo output must not be compared.
        if nvalid!=cvalid or n_aligned.shape[-1]!=ct.shape[-1]:
            result[name]={'status':'VALID_SHAPE_MISMATCH','nemo_valid_frames':nvalid,
                'candle_valid_frames':cvalid,'nemo_shape':list(nt.shape),'candle_shape':list(ct.shape)}
            continue
        r=compare_arrays(n_aligned[:,:nvalid,:],ct[:,:cvalid,:],name,atol=1e-3,min_cosine=0.999)
        r['nemo_padded_frames']=nt.shape[-1]-nvalid
        result[name]=r
    if result.get('mel',{}).get('status')!='PASS_NUMERIC_PARITY':
        result['encoder_attribution']='BLOCKED_BY_UNQUALIFIED_LOGMEL'
    else:
        result['encoder_attribution']='MEL_QUALIFIED_FOR_ENCODER_COMPARISON'
    if result.get('mel',{}).get('status')=='PASS_NUMERIC_PARITY' and result.get('encoder',{}).get('status')=='PASS_NUMERIC_PARITY':
        result['status']='PASS_NUMERIC_MEL_AND_ENCODER_ONLY'
    else:
        result['status']='NUMERIC_PARITY_NOT_ESTABLISHED'
    report(output,result)
    return result


def compare_texts(nemo_report,candle_report,output,ref_path=None):
    n=json.loads(Path(nemo_report).read_text())
    c=json.loads(Path(candle_report).read_text())
    if n.get('status')!='REAL_NEMO_ASR' or c.get('status')!='REAL_CANDLE_ASR':
        raise ValueError('Both real inferences required for C1')
    if n['pcm_sha256']!=c['pcm_sha256']:
        raise ValueError('PCM mismatch; refuse any C1 comparison')
    result={'stage':'HDB2C_SAME_WAV_NEMO_CANDLE','nemo_text':n['text'],'candle_text':c['text'],
        'pcm_sha256':n['pcm_sha256'],'nemo_checkpoint_sha256':n['checkpoint_sha256'],
        'candle_checkpoint_source_sha256':c.get('checkpoint_source_nemo_sha256'),
        'text_identical_nfc':normalize_text(n['text'])==normalize_text(c['text']),
        'cross_model_wer_nemo_as_reference':wer(n['text'],c['text']),
        'ground_truth_status':'NOT_PROVIDED', 'status':'COMPARISON_OBSERVED_NOT_NUMERIC_PARITY'}
    if result['candle_checkpoint_source_sha256'] is None:
        result['status']='CHECKPOINT_SOURCE_UNVERIFIED'
    elif result['candle_checkpoint_source_sha256']!=n['checkpoint_sha256']:
        result['status']='CHECKPOINT_SOURCE_MISMATCH'
    if ref_path is not None:
        reference=Path(ref_path).read_text(encoding='utf-8').strip()
        result['ground_truth_status']='USER_SUPPLIED_REFERENCE_NOT_INDEPENDENTLY_VERIFIED'
        result['reference_text']=reference
        result['nemo_wer']=wer(reference,n['text'])
        result['candle_wer']=wer(reference,c['text'])
    report(output,result)
    return result


def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='task',required=True)
    c=sp.add_parser('text');c.add_argument('--nemo',required=True);c.add_argument('--candle',required=True)
    c.add_argument('--reference-file');c.add_argument('--output',required=True)
    a=sp.add_parser('tensors');a.add_argument('--nemo-dir',required=True)
    a.add_argument('--candle-dir',required=True);a.add_argument('--output',required=True)
    x=p.parse_args()
    if x.task=='text':r=compare_texts(x.nemo,x.candle,x.output,x.reference_file)
    else:r=compare_probes(x.nemo_dir,x.candle_dir,x.output)
    print('HDB2C_STATUS='+r['status'])
    print('HDB2C_REPORT='+x.output)
if __name__=='__main__':main()
