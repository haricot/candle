"""HDB2-C3-A-R4: instrument the actual FastConformer block 0 (NeMo/Candle).

Observation only: separate isolated clone of R3-C-R1. Never uploads raw
waveforms, weights or activation dumps. Results are numeric JSON only.
"""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from .c3_r3 import _dump, read_tensor

SIGNATURE="hdb2c3-r4-block0-operation-evidence-v0.7.4"
MARKER="HDB2C3_R4_BLOCK0_INTERNAL_TRACE_V1"
STAGES=(
    "block0_input", "block0_norm_ff1", "block0_ff1", "block0_after_ff1",
    "block0_norm_att", "block0_attention", "block0_after_attention",
    "block0_norm_conv", "block0_convolution", "block0_after_convolution",
    "block0_norm_ff2", "block0_ff2", "block0_after_ff2", "block0_norm_out",
)

def transform_r4_conformer(source:str)->str:
    """Instrument the exact post-R3C source; leave mathematical expressions unchanged."""
    if MARKER in source:
        raise ValueError("R4 duplicate trace")
    if "HDB2C3_R3C_EXACT_MASK_PROPAGATION_CPU_V1" not in source:
        raise ValueError("R4 requires verified R3C masked conformer")
    start=source.find("impl ConformerBlock {")
    end=source.find("\n#[derive(Debug, Clone)]\nstruct DwStridingSubsampling",start)
    if start<0 or end<=start:
        raise ValueError("R4 pinned ConformerBlock boundaries absent")
    head,block,tail=source[:start],source[start:end],source[end:]
    def one(src,old,new,label):
        if src.count(old)!=1:
            raise ValueError(f"R4 block source changed: {label}, found {src.count(old)}")
        return src.replace(old,new,1)
    signature='''        pos_emb: Option<&Tensor>,
        mut cache: Option<&mut dyn CacheLike>,
    ) -> Result<Tensor> {'''
    block=one(block,signature,'''        pos_emb: Option<&Tensor>,
        mut cache: Option<&mut dyn CacheLike>,
        r4_trace: bool,
    ) -> Result<Tensor> {''',"forward signature")
    block=one(block,"        let mut x = x.clone();",'''        // HDB2C3_R4_BLOCK0_INTERNAL_TRACE_V1: no math changed.
        let mut x = x.clone();
        if r4_trace {hdb2c3_trace_stage("block0_input",&x)?;}''',"input")
    block=one(block,"        let ff1 = self.ff1.forward(&self.norm_ff1.forward(&x)?)?;",'''        let ff1_norm=self.norm_ff1.forward(&x)?;
        if r4_trace {hdb2c3_trace_stage("block0_norm_ff1",&ff1_norm)?;}
        let ff1 = self.ff1.forward(&ff1_norm)?;
        if r4_trace {hdb2c3_trace_stage("block0_ff1",&ff1)?;}''',"ff1")
    block=one(block,"        x = (&x + &(&ff1 * 0.5)?)?;",'''        x = (&x + &(&ff1 * 0.5)?)?;
        if r4_trace {hdb2c3_trace_stage("block0_after_ff1",&x)?;}''',"residual ff1")
    block=one(block,"        let x_norm = self.norm_self_att.forward(&x)?;",'''        let x_norm = self.norm_self_att.forward(&x)?;
        if r4_trace {hdb2c3_trace_stage("block0_norm_att",&x_norm)?;}''',"norm att")
    block=one(block,"        x = (&x + &attn_out)?;",'''        if r4_trace {hdb2c3_trace_stage("block0_attention",&attn_out)?;}
        x = (&x + &attn_out)?;
        if r4_trace {hdb2c3_trace_stage("block0_after_attention",&x)?;}''',"attention residual")
    block=one(block,"        let norm_x = self.norm_conv.forward(&x)?;",'''        let norm_x = self.norm_conv.forward(&x)?;
        if r4_trace {hdb2c3_trace_stage("block0_norm_conv",&norm_x)?;}''',"norm conv")
    block=one(block,"        let conv_out = self.conv.forward(&norm_x, cache)?;",'''        let conv_out = self.conv.forward(&norm_x, cache)?;
        if r4_trace {hdb2c3_trace_stage("block0_convolution",&conv_out)?;}''',"conv")
    block=one(block,"        x = (&x + &conv_out)?;",'''        x = (&x + &conv_out)?;
        if r4_trace {hdb2c3_trace_stage("block0_after_convolution",&x)?;}''',"conv residual")
    block=one(block,"        let ff2 = self.ff2.forward(&self.norm_ff2.forward(&x)?)?;",'''        let ff2_norm=self.norm_ff2.forward(&x)?;
        if r4_trace {hdb2c3_trace_stage("block0_norm_ff2",&ff2_norm)?;}
        let ff2 = self.ff2.forward(&ff2_norm)?;
        if r4_trace {hdb2c3_trace_stage("block0_ff2",&ff2)?;}''',"ff2")
    block=one(block,"        x = (&x + &(&ff2 * 0.5)?)?;",'''        x = (&x + &(&ff2 * 0.5)?)?;
        if r4_trace {hdb2c3_trace_stage("block0_after_ff2",&x)?;}''',"ff2 residual")
    block=one(block,"        self.norm_out.forward(&x)",'''        let y=self.norm_out.forward(&x)?;
        if r4_trace {hdb2c3_trace_stage("block0_norm_out",&y)?;}
        Ok(y)''',"norm out")
    source=head+block+tail
    source=one(source,'x = layer.forward(&x, pos_emb.as_ref(), Some(cache))?;',
       'x = layer.forward(&x, pos_emb.as_ref(), Some(cache), false)?;', "cache call")
    source=one(source,'x = layer.forward(&x, pos_emb.as_ref(), None)?;',
       'x = layer.forward(&x, pos_emb.as_ref(), None, index==0)?;', "non-cache call")
    source=one(source,'            pos_emb = Some(pos);',
       '''            hdb2c3_trace_stage("block0_pos_emb",&pos)?;
            pos_emb = Some(pos);''',"positional trace")
    return source

def attach_nemo_r4_hooks(encoder,folder:Path):
    """Capture actual layer-0 module outputs + input of next module (residual)."""
    import torch
    if len(encoder.layers)!=24:
        raise RuntimeError("R4 expects NeMo encoder with exactly 24 blocks")
    layer=encoder.layers[0]
    folder.mkdir(parents=True,exist_ok=True)
    hooks=[];seen=set()
    def save(name,tensor):
        if name in seen:raise RuntimeError("R4 duplicate NeMo hook "+name)
        if isinstance(tensor,(tuple,list)):
            tensor=tensor[0]
        _dump(folder,name,tensor,'B,T,D')
        seen.add(name)
    def pre(name):
        def fn(_module,args):
            if len(args)<1:raise RuntimeError("R4 hook without input: "+name)
            save(name,args[0])
        return fn
    def post(name):
        def fn(_module,args,out):
            save(name,out)
        return fn
    hooks.append(layer.register_forward_pre_hook(pre('block0_input')))
    for name,attr in (
        ('block0_norm_ff1','norm_feed_forward1'),
        ('block0_ff1','feed_forward1'),
        ('block0_norm_att','norm_self_att'),
        ('block0_attention','self_attn'),
        ('block0_norm_conv','norm_conv'),
        ('block0_convolution','conv'),
        ('block0_norm_ff2','norm_feed_forward2'),
        ('block0_ff2','feed_forward2'),
        ('block0_norm_out','norm_out')):
        if not hasattr(layer,attr):
            raise RuntimeError("R4 unexpected NeMo block attribute "+attr)
        hooks.append(getattr(layer,attr).register_forward_hook(post(name)))
    for name,attr in (
        ('block0_after_ff1','norm_self_att'),
        ('block0_after_attention','norm_conv'),
        ('block0_after_convolution','norm_feed_forward2'),
        ('block0_after_ff2','norm_out')):
        hooks.append(getattr(layer,attr).register_forward_pre_hook(pre(name)))
    def pos(_module,_args,out):
        if not isinstance(out,(tuple,list)) or len(out)<2:
            raise RuntimeError("R4 expected NeMo positional encoding (x,pos_emb)")
        save('block0_pos_emb',out[1])
    hooks.append(encoder.pos_enc.register_forward_hook(pos))
    return hooks,seen

def verify_r4_hooks(seen:set):
    required=set(STAGES)|{'block0_pos_emb'}
    if seen!=required:
        raise RuntimeError("R4 NeMo hook coverage mismatch: "+repr(sorted(required-seen)))

def compare_valid(n,c,valid:int,atol=1e-3,min_cosine=.999):
    result={'nemo_shape':list(n.shape),'candle_shape':list(c.shape)}
    if n.ndim!=3 or c.ndim!=3 or n.shape[0]!=c.shape[0] or n.shape[2]!=c.shape[2]:
        return {**result,'status':'SHAPE_MISMATCH'}
    if valid<=0 or min(n.shape[1],c.shape[1])<valid:
        return {**result,'status':'INVALID_VALID_LENGTH'}
    a=n[:,:valid,:].astype(np.float64);b=c[:,:valid,:].astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite R4 block activations")
    diff=np.abs(a-b)
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    cosine=float(np.sum(a*b)/norm) if norm>0 else (1. if not diff.any() else 0.)
    absmax=float(np.max(diff))
    return {**result,'valid_frames':valid,'max_abs':absmax,
            'rmse':float(np.sqrt(np.mean((a-b)**2))),'cosine':cosine,
            'status':'PASS_NUMERIC_PARITY' if absmax<=atol and cosine>=min_cosine else 'NUMERIC_MISMATCH'}

def compare_r4(nemo:Path,candle:Path,encoder_frames:int,masked_layers:dict,r4_layers:dict):
    if len(masked_layers.get('stages',[]))!=26 or len(r4_layers.get('stages',[]))!=26:
        raise ValueError('R4 candidate must preserve all 26 encoder stages')
    # 26-stage self-parity is a hard gate: instrumentation must not alter math.
    self_checks=[]
    for before,after in zip(masked_layers['stages'],r4_layers['stages']):
        if before['name']!=after['name']:
            raise ValueError('R4 encoder stage identities changed')
    # Separate masked-vs-R4 exact trace equality is verified by caller using raw files.
    stages=[];first=None
    for name in STAGES:
        a=read_tensor(nemo/f'{name}.f32le')
        b=read_tensor(candle/f'{name}.f32le')
        row={'name':name,**compare_valid(a,b,encoder_frames)}
        stages.append(row)
        if first is None and row['status']!='PASS_NUMERIC_PARITY':
            first=name
    pos_a=read_tensor(nemo/'block0_pos_emb.f32le')
    pos_b=read_tensor(candle/'block0_pos_emb.f32le')
    positional={'nemo_shape':list(pos_a.shape),'candle_shape':list(pos_b.shape)}
    if pos_a.shape==pos_b.shape and pos_a.ndim==3:
        positional=compare_valid(pos_a,pos_b,pos_a.shape[1])
    else:
        positional['status']='POSITIONAL_SHAPE_MISMATCH'
    return {'stage':'HDB2C3_A_R4_BLOCK0_EXACT_ATTRIBUTION',
        'revision':'HDB2-C3-A-R4', 'status':'ALL_BLOCK0_STAGES_NUMERIC_PARITY' if
            first is None and positional['status']=='PASS_NUMERIC_PARITY' else
            'BLOCK0_NUMERIC_PARITY_NOT_ESTABLISHED',
        'first_block0_divergence':first,'positional_evidence':positional,
        'input_numeric_status':stages[0]['status'],
        'input_residual_caveat':('BLOCK0_INPUT_ALREADY_DIFFERENT_NO_INDEPENDENT_BLOCK_ROOT_CAUSE'
              if stages[0]['status']!='PASS_NUMERIC_PARITY' else 'INPUT_WITHIN_GATE'),
        'criteria':{'activation_max_abs':1e-3,'activation_min_cosine':.999},
        'stages':stages,'tdt_numeric_parity':'NOT_TESTED',
        'note':'No correction applied to block 0; raw layer traces kept private on runner.'}

def compare_r4_control(masked_dir:Path,r4_dir:Path,encoder_frames:int):
    from .c3 import STAGES as MAIN_STAGES
    checks=[];first=None
    for name in MAIN_STAGES:
        n=read_tensor(masked_dir/f'{name}.f32le')
        c=read_tensor(r4_dir/f'{name}.f32le')
        row={'name':name,**compare_valid(n,c,encoder_frames,atol=0.,min_cosine=.999999999)}
        checks.append(row)
        if row['status']!='PASS_NUMERIC_PARITY' and first is None:first=name
    if first is not None:
        raise RuntimeError('R4 instrumentation changed masked reference output at '+first)
    return {'status':'EXACT_26_STAGE_R3C_R4_INSTRUMENTATION_PARITY',
            'stages_checked':len(checks),'raw_values_exact_on_valid_frames':True}

def build_r4_candidate(reference:Path,target:Path):
    reference=reference.resolve();target=target.resolve()
    if reference==target or reference in target.parents or target in reference.parents:
        raise ValueError('R4 candidate must not overwrite R3C control')
    manifest=reference/'HDB2C3-R3C-MANIFEST.json'
    if not manifest.is_file():raise ValueError('Missing independently manifested R3C source')
    conformer=reference/'src/parakeet/conformer.rs'
    if not conformer.is_file():raise FileNotFoundError(str(conformer))
    generated=transform_r4_conformer(conformer.read_text())
    want={'signature':SIGNATURE,'parent':str(reference),
        'parent_conformer_sha256':hashlib.sha256(conformer.read_bytes()).hexdigest(),
        'generated_conformer_sha256':hashlib.sha256(generated.encode()).hexdigest(),
        'scope':'CPU block0 evidence only; R3C and C2 untouched'}
    if target.exists():
        p=target/'HDB2C3-R4-MANIFEST.json'
        if not p.is_file() or json.loads(p.read_text())!=want:
            raise RuntimeError('R4 candidate manifest changed; refuse overwrite')
        if hashlib.sha256((target/'src/parakeet/conformer.rs').read_bytes()).hexdigest()!=want['generated_conformer_sha256']:
            raise RuntimeError('R4 candidate source modified')
        print('HDB2C3_R4_CANDIDATE=REUSED_VERIFIED_COPY',flush=True)
        return
    temp=target.with_name(target.name+'.creating')
    if temp.exists():raise RuntimeError('Stale R4 candidate temp dir')
    def ignore(_,names):return [n for n in names if n in {'target','.git','__pycache__','.venv'}]
    try:
        shutil.copytree(reference,temp,ignore=ignore)
        (temp/'src/parakeet/conformer.rs').write_text(generated)
        (temp/'HDB2C3-R4-MANIFEST.json').write_text(json.dumps(want,indent=2)+'\n')
        temp.rename(target)
    finally:
        if temp.exists():shutil.rmtree(temp)
    print('HDB2C3_R4_CANDIDATE=CREATED_ISOLATED_COPY',flush=True)
