"""HDB2-C3-A-R3: actual NeMo/Candle DW-striding weight + operation parity.

Reference tensors remain in the temporary CI report directory. Only numeric
metrics, tensor shape and checkpoint hashes are uploaded to GitHub Actions.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

CONV_IDS=(0,2,3,5,6)
RELU_IDS=(0,3,6)
WEIGHTS=tuple(x for k in (*CONV_IDS,"out") for x in (f"weight_{k:02}" if isinstance(k,int) else "weight_out",f"bias_{k:02}" if isinstance(k,int) else "bias_out"))
STAGES=("conv_00","relu_00","conv_02","conv_03","relu_03","conv_05","conv_06","relu_06","out")

def transform_dw_source(src: str) -> str:
    """Instrument the exact pinned Candle subsampler; never alter its math."""
    anchor='''        let mut x = x.unsqueeze(1)?; // (B, 1, T, F)
        for (idx, conv) in self.conv.iter().enumerate() {
            x = conv.forward(&x)?;
            if idx == 0 || idx % 2 == 0 {
                x = x.relu()?;
            }
        }
        let x = x.transpose(1, 2)?; // (B, T, C, F)
        let (b, t, c, f) = x.dims4()?;
        let x = x.reshape((b, t, c * f))?;
        let x = self.out.forward(&x)?;
'''
    if src.count(anchor)!=1:
        raise ValueError('R3 requires exact pinned DW-striding forward; refuse unknown source')
    new='''        // HDB2C3_R3_DW_TRACE_CPU: observation only; do not change computation.
        if self.conv.len()!=5 {
            return Err(candle::Error::Msg("Expected 5 DW striding convolutions".into()));
        }
        let mut x = x.unsqueeze(1)?; // (B, 1, T, F)
        for (idx, conv) in self.conv.iter().enumerate() {
            let nemo_idx = if idx==0 {0} else {2+3*((idx-1)/2)+((idx-1)%2)};
            hdb2c3_trace_stage(&format!("weight_{nemo_idx:02}"),conv.weight())?;
            if let Some(bias) = conv.bias() {
                hdb2c3_trace_stage(&format!("bias_{nemo_idx:02}"),bias)?;
            } else {
                return Err(candle::Error::Msg("Expected DW conv bias".into()));
            }
            x = conv.forward(&x)?;
            hdb2c3_trace_stage(&format!("conv_{nemo_idx:02}"),&x)?;
            if idx == 0 || idx % 2 == 0 {
                x = x.relu()?;
                hdb2c3_trace_stage(&format!("relu_{nemo_idx:02}"),&x)?;
            }
        }
        let x = x.transpose(1, 2)?; // (B, T, C, F)
        let (b, t, c, f) = x.dims4()?;
        let x = x.reshape((b, t, c * f))?;
        hdb2c3_trace_stage("weight_out",self.out.weight())?;
        if let Some(bias)=self.out.bias(){
            hdb2c3_trace_stage("bias_out",bias)?;
        } else {
            return Err(candle::Error::Msg("Expected DW projection bias".into()));
        }
        let x = self.out.forward(&x)?;
        hdb2c3_trace_stage("out",&x)?;
'''
    return src.replace(anchor,new,1)

def _dump(folder:Path, name:str, tensor, layout:str):
    import torch
    if not isinstance(tensor,torch.Tensor):raise TypeError(f'{name}: expected PyTorch tensor')
    a=tensor.detach().to('cpu',dtype=torch.float32).contiguous().numpy()
    if a.size==0 or not np.isfinite(a).all():raise ValueError(f'{name}: invalid tensor')
    p=folder/f'{name}.f32le'
    data=a.astype('<f4',copy=False).tobytes(order='C')
    p.write_bytes(data)
    p.with_suffix('.json').write_text(json.dumps({
        'shape':list(a.shape),'layout':layout,'dtype':'f32le',
        'sha256':hashlib.sha256(data).hexdigest()},indent=2)+'\n')

def attach_nemo_hooks(enc,folder:Path):
    """Hook each conv and unique ReLU module (NeMo reuses one ReLU instance)."""
    import torch
    conv=enc.pre_encode.conv
    if len(conv)!=8 or not all(isinstance(conv[i],torch.nn.Conv2d) for i in CONV_IDS):
        raise RuntimeError('Unexpected DW striding NeMo conv layout')
    if not all(isinstance(conv[i],torch.nn.ReLU) for i in (1,4,7)):
        raise RuntimeError('Unexpected NeMo ReLU layout')
    folder.mkdir(parents=True,exist_ok=True)
    emitted=set()
    hooks=[]
    def save(name,result,layout):
        if name in emitted:raise RuntimeError('R3 duplicate NeMo stage '+name)
        _dump(folder,name,result,layout);emitted.add(name)
    for i in CONV_IDS:
        module=conv[i]
        _dump(folder,f'weight_{i:02}',module.weight,'OIHW')
        if module.bias is None:raise RuntimeError('Missing NeMo conv bias')
        _dump(folder,f'bias_{i:02}',module.bias,'O')
        hooks.append(module.register_forward_hook(
            lambda _m,_a,y,j=i:save(f'conv_{j:02}',y,'B,C,T,F')))
    _dump(folder,'weight_out',enc.pre_encode.out.weight,'OI')
    if enc.pre_encode.out.bias is None:raise RuntimeError('Missing NeMo linear bias')
    _dump(folder,'bias_out',enc.pre_encode.out.bias,'O')
    hooks.append(enc.pre_encode.out.register_forward_hook(
        lambda _m,_a,y:save('out',y,'B,T,D')))
    relu_count=[0]
    def relu_hook(_m,_a,y):
        n=relu_count[0]
        if n>=len(RELU_IDS):raise RuntimeError('Unexpected extra NeMo ReLU invocation')
        relu_count[0]+=1
        save(f'relu_{RELU_IDS[n]:02}',y,'B,C,T,F')
    seen=set()
    for i in (1,4,7):
        m=conv[i]
        if id(m) not in seen:
            seen.add(id(m))
            hooks.append(m.register_forward_hook(relu_hook))
    return hooks,emitted

def verify_nemo_stages(emitted):
    missing=set(STAGES)-set(emitted)
    if missing or len(emitted)!=len(STAGES):
        raise RuntimeError(f'Incomplete NeMo R3 trace: missing={sorted(missing)}')

def read_tensor(path:Path):
    meta=json.loads(path.with_suffix('.json').read_text())
    shape=meta['shape']
    if not isinstance(shape,list) or not shape or any(type(v) is not int or v<=0 for v in shape):
        raise ValueError(f'R3 invalid tensor shape: {path}')
    raw=path.read_bytes()
    if 'sha256' in meta and hashlib.sha256(raw).hexdigest()!=meta['sha256']:
        raise ValueError(f'R3 corrupted trace {path}')
    arr=np.frombuffer(raw,dtype='<f4')
    if arr.size!=int(np.prod(shape)) or not np.isfinite(arr).all():
        raise ValueError(f'R3 invalid raw tensor {path}')
    return arr.reshape(shape)

def stage_valid_frames(mel_valid:int):
    if mel_valid<=0:raise ValueError('mel_valid must be positive')
    one=(mel_valid+1)//2
    two=(one+1)//2
    three=(two+1)//2
    return dict(conv_00=one,relu_00=one,conv_02=two,conv_03=two,
                relu_03=two,conv_05=three,conv_06=three,relu_06=three,out=three)

def metrics(a,b,axis=None,valid=None,tolerance=1e-3,min_cosine=.999):
    result={'nemo_shape':list(a.shape),'candle_shape':list(b.shape)}
    if a.shape!=b.shape:
        return {**result,'status':'SHAPE_MISMATCH'}
    if axis is not None:
        if valid is None or valid<=0 or valid>a.shape[axis]:
            return {**result,'status':'INVALID_VALID_LENGTH','valid_frames':valid}
        sl=[slice(None)]*a.ndim;sl[axis]=slice(0,valid)
        a=a[tuple(sl)];b=b[tuple(sl)]
        result['valid_frames']=valid
    x=a.astype(np.float64);y=b.astype(np.float64)
    d=np.abs(x-y)
    norm=np.linalg.norm(x)*np.linalg.norm(y)
    cosine=float(np.sum(x*y)/norm) if norm>0 else (1.0 if not d.any() else 0.0)
    err=float(d.max())
    return {**result,'max_abs':err,'rmse':float(np.sqrt(np.mean((x-y)**2))),
            'cosine':cosine,'status':'PASS_NUMERIC_PARITY' if
            err<=tolerance and cosine>=min_cosine else 'NUMERIC_MISMATCH'}

def compare_dw(nemo:Path,candle:Path,mel_valid:int,encoder_valid:int):
    lengths=stage_valid_frames(mel_valid)
    if lengths['out']!=encoder_valid:
        raise ValueError('NeMo encoder lengths disagree with DW subsampling formula')
    weights=[];wfirst=None
    for name in WEIGHTS:
        n=read_tensor(nemo/f'{name}.f32le')
        c=read_tensor(candle/f'{name}.f32le')
        m=metrics(n,c,tolerance=1e-6,min_cosine=.999999)
        weights.append({'name':name,**m})
        if m['status']!='PASS_NUMERIC_PARITY' and wfirst is None:wfirst=name
    stages=[];first=None
    for name in STAGES:
        n=read_tensor(nemo/f'{name}.f32le')
        c=read_tensor(candle/f'{name}.f32le')
        axis=1 if name=='out' else 2
        m=metrics(n,c,axis=axis,valid=lengths[name])
        stages.append({'name':name,**m})
        if m['status']!='PASS_NUMERIC_PARITY' and first is None:first=name
    return {'stage':'HDB2C3_A_R3_DW_STRIDING_EXACT_ATTRIBUTION',
            'revision':'HDB2-C3-A-R3','status':'ALL_DW_STAGES_NUMERIC_PARITY' if first is None and wfirst is None else 'FIRST_DW_DIVERGENCE_LOCALIZED',
            'weights_status':'PASS_NUMERIC_PARITY' if wfirst is None else 'WEIGHT_MISMATCH',
            'first_weight_mismatch':wfirst,'first_operation_divergence':first,
            'stage_valid_frames':lengths,'weights':weights,'operations':stages,
            'criteria':{'weight_max_abs':1e-6,'activation_max_abs':1e-3,'activation_min_cosine':.999},
            'tdt_numeric_parity':'NOT_TESTED','note':'No DW math changed; masks are attribution hypotheses, not yet repaired.'}
