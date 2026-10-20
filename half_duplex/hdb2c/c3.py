"""HDB2-C3-A: actual NeMo-vs-Candle layer trace with EXACT shared Mel input.

Requires a genuine HDB2-C2 --probes C1 report and an already-built C2 candidate.
C3 produces an ISOLATED Rust checkout, does not mutate C2 or the production runtime.
A completed diagnostic is not automatically numeric parity qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from .cli import settings
from .c3_r3 import transform_dw_source, compare_dw
from .c3_r3c import build_masked_candidate, summarize_ab

SIGNATURE = 'hdb2c3-r3-dw-weight-operation-traces-v0.7.2'
STAGES = ['pre_encode', 'post_pos_enc'] + [f'layer_{i:02}' for i in range(24)]
HELPER = r'''
// HDB2C3_TRACE_INSTRUMENTATION_OFFLINE_CPU_V1
// Raw internal activations are local evidence only, NEVER publish in artifacts.
fn hdb2c3_trace_stage(name: &str, tensor: &Tensor) -> Result<()> {
    let Some(folder) = std::env::var_os("HDB2C3_TRACE_DIR") else { return Ok(()); };
    let dir = std::path::PathBuf::from(folder);
    std::fs::create_dir_all(&dir).map_err(|e| candle::Error::Msg(e.to_string()))?;
    let cpu = tensor.to_device(&candle::Device::Cpu)?.to_dtype(DType::F32)?;
    let shape = cpu.dims().to_vec();
    let flat = cpu.flatten_all()?.to_vec1::<f32>()?;
    let mut raw = Vec::with_capacity(flat.len() * 4);
    for n in flat { raw.extend_from_slice(&n.to_le_bytes()); }
    std::fs::write(dir.join(format!("{name}.f32le")), raw)
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
    let layout=match shape.len(){4=>"B,C,T,F",3=>"B,T,D",_=>"WEIGHT"};
    let meta = serde_json::json!({"shape":shape,"dtype":"f32le","layout":layout});
    let encoded = serde_json::to_vec_pretty(&meta)
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
    std::fs::write(dir.join(format!("{name}.json")), encoded)
        .map_err(|e| candle::Error::Msg(e.to_string()))?;
    Ok(())
}
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform_conformer(source: str) -> str:
    if 'HDB2C3_TRACE_INSTRUMENTATION_OFFLINE_CPU_V1' in source:
        raise ValueError('C3 trace already installed; avoid instrumentation duplication')
    def once(src, needle, replacement):
        n=src.count(needle)
        if n!=1:
            raise ValueError(f'Unexpected pinned conformer source: anchor count {n}: {needle[:90]!r}')
        return src.replace(needle,replacement,1)
    source=once(source,'        let mut pos_emb = None;\n',
        '        hdb2c3_trace_stage("pre_encode", &x)?;\n        let mut pos_emb = None;\n')
    source=once(source,'        if let Some(cache) = cache.as_mut() {\n            for (layer, cache)',
        '        hdb2c3_trace_stage("post_pos_enc", &x)?;\n        if let Some(cache) = cache.as_mut() {\n            for (layer, cache)')
    source=once(source,
        '            for layer in &self.layers {\n                x = layer.forward(&x, pos_emb.as_ref(), None)?;\n            }',
        '            for (index, layer) in self.layers.iter().enumerate() {\n'
        '                x = layer.forward(&x, pos_emb.as_ref(), None)?;\n'
        '                hdb2c3_trace_stage(&format!("layer_{index:02}"), &x)?;\n'
        '            }')
    source=transform_dw_source(source)
    return source+'\n'+HELPER


def transform_probe(source: str) -> str:
    original='    let mel=get_logmel(&audio,&pre,&dev)?;'
    if source.count(original)!=1:
        raise ValueError('Expected exactly one C2 probe Mel computation site')
    injection=r'''    // The SAME NeMo Mel is injected into Candle to attribute encoder errors.
    // The file contains [B=1,T,128] f32le; it is derived from nemo_mel.npz.
    let mel=if let Some(path) = std::env::var_os("HDB2C3_NEMO_MEL_PATH") {
        let frames:usize=std::env::var("HDB2C3_NEMO_MEL_FRAMES")
            .context("HDB2C3_NEMO_MEL_FRAMES missing")?.parse()?;
        let bytes=fs::read(path)?;
        anyhow::ensure!(bytes.len()==frames*128*4, "Injected Mel size mismatch");
        let floats:Vec<f32>=bytes.chunks_exact(4).map(|b|f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();
        anyhow::ensure!(floats.iter().all(|x|x.is_finite()),"Nonfinite injected Mel");
        Tensor::from_vec(floats,(1,frames,128),&dev)?
    } else {
        get_logmel(&audio,&pre,&dev)?
    };'''
    # R2 preserves the exact injected Mel and ALSO propagates its real NeMo
    # valid frame count into Candle. This modifies only the C3 probe executable,
    # never the C2 runtime or its normal ASR path.
    source=source.replace(original,injection,1)
    length_anchor='    let (features,lengths)=match &mut model {'
    if source.count(length_anchor)!=1:
        raise ValueError('Missing pinned C2 encoder invocation anchor')
    lengths_code='''    // HDB2-C3-A-R2: exact NeMo valid-Mel frame count; never derive from padding.
    let injected_mel_valid_frames:Option<usize> =
        if std::env::var_os("HDB2C3_NEMO_MEL_PATH").is_some() {
            let valid:usize=std::env::var("HDB2C3_NEMO_MEL_VALID_FRAMES")
                .context("Missing HDB2C3_NEMO_MEL_VALID_FRAMES")?.parse()?;
            let (batch, time, features)=mel.dims3()?;
            anyhow::ensure!(batch==1 && features==128 && valid>0 && valid<=time,
                "Invalid injected NeMo Mel valid frame count: {valid} / {time}");
            Some(valid)
        } else {
            anyhow::ensure!(std::env::var_os("HDB2C3_NEMO_MEL_VALID_FRAMES").is_none(),
                "NeMo valid-frame env supplied without NeMo Mel injection");
            None
        };
    let encoder_input_lengths=if let Some(valid)=injected_mel_valid_frames {
        Some(Tensor::from_vec(vec![valid as i64],(1,),&dev)?)
    } else {None};
'''
    source=source.replace(length_anchor,lengths_code+length_anchor,1)
    # C2 already supplies its own sample-derived valid_t; R2 checks
    # NeMo's actual preprocessor length rather than silently reusing that
    # estimate. Keep the C2 expression as the non-injected fallback.
    forward_anchor='m.encoder.forward(&mel,Some(&valid_t))?'
    if source.count(forward_anchor)!=1:
        raise ValueError('C3 requires the exact post-C2 TDT encoder invocation')
    if source.count('let valid_t=Tensor::from_vec(vec![valid as i64],(1,),&dev)?;')!=1:
        raise ValueError('C3 requires the pinned C2 sample-derived input length')
    source=source.replace(forward_anchor,
        'm.encoder.forward(&mel,encoder_input_lengths.as_ref().or(Some(&valid_t)))?',1)
    meta_anchor='        "mel":mel_meta,"encoder":feat_meta,"encoder_valid_frames":lengths[0],'
    if source.count(meta_anchor)!=1:
        raise ValueError('Pinned C2 probe metadata anchor changed')
    source=source.replace(meta_anchor,
        '        "mel":mel_meta,"encoder":feat_meta,"encoder_valid_frames":lengths[0],\n'
        '        "injected_mel_valid_frames":injected_mel_valid_frames,'
        '        "encoder_length_argument":"EXPLICIT_NEMO_VALID_FRAMES_IF_INJECTED",',1)
    return source


def build_candidate(c2: Path, c3: Path) -> None:
    c2=c2.resolve();c3=c3.resolve()
    if c2==c3 or c2 in c3.parents or c3 in c2.parents:
        raise ValueError('C3 candidate must be a separate checkout')
    if not (c2/'HDB2C2-MANIFEST.json').is_file():
        raise ValueError('C3 requires a real, previously qualified C2 source checkout')
    original=c2/'src/parakeet/conformer.rs'
    probe=c2/'src/bin/hdb2c_probe.rs'
    if not original.is_file() or not probe.is_file():
        raise FileNotFoundError('Missing pinned Conformer or actual C2 Rust probe')
    conformer=transform_conformer(original.read_text())
    probe_src=transform_probe(probe.read_text())
    generated={'src/parakeet/conformer.rs':hashlib.sha256(conformer.encode()).hexdigest(),
        'src/bin/hdb2c_probe.rs':hashlib.sha256(probe_src.encode()).hexdigest()}
    manifest={'signature':SIGNATURE,'parent':str(c2),
        'source_hashes':{'conformer':sha(original),'probe':sha(probe)},
        'generated_hashes':generated,'scope':'CPU exact NeMo logmel + explicit NeMo Mel valid length; no C2 model mutation'}
    if c3.exists():
        stored=c3/'HDB2C3-MANIFEST.json'
        if not stored.is_file() or json.loads(stored.read_text())!=manifest:
            raise RuntimeError('C3 candidate exists but manifest changed; refuse replacement')
        for path,digest in generated.items():
            if sha(c3/path)!=digest:raise RuntimeError(f'C3 candidate modified: {path}')
        print('HDB2C3_CANDIDATE=REUSED_VERIFIED_COPY',flush=True)
        return
    tmp=c3.with_name(c3.name+'.creating')
    if tmp.exists():raise RuntimeError('Aborted stale C3 candidate exists: '+str(tmp))
    def ignore(_,names):return [v for v in names if v in {'target','.git','__pycache__','.venv'}]
    try:
        shutil.copytree(c2,tmp,ignore=ignore)
        (tmp/'src/parakeet/conformer.rs').write_text(conformer)
        (tmp/'src/bin/hdb2c_probe.rs').write_text(probe_src)
        (tmp/'HDB2C3-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
        tmp.rename(c3)
    finally:
        if tmp.exists():shutil.rmtree(tmp)
    print('HDB2C3_CANDIDATE=CREATED_ISOLATED_COPY',flush=True)


def artifact(path: Path) -> np.ndarray:
    meta=json.loads(path.with_suffix('.json').read_text())
    shape=meta['shape']
    if len(shape)!=3 or shape[0]!=1 or any(not isinstance(n,int) or n<=0 for n in shape):
        raise ValueError(f'Not a B,T,D tensor: {path}, {shape}')
    a=np.fromfile(path,dtype='<f4')
    if a.size!=np.prod(shape):raise ValueError(f'Trace byte length does not match shape: {path}')
    a=a.reshape(shape)
    if not np.isfinite(a).all():raise ValueError(f'Nonfinite trace: {path}')
    return a


def compare_stages(ne_dir:Path, ca_dir:Path, valid_frames:int, atol=0.001, min_cosine=0.999) -> dict:
    if valid_frames<=0:raise ValueError('Invalid number of valid subsampled frames')
    stages=[];first=None
    for name in STAGES:
        n=artifact(ne_dir/f'{name}.f32le');c=artifact(ca_dir/f'{name}.f32le')
        result={'name':name,'nemo_shape':list(n.shape),'candle_shape':list(c.shape)}
        if n.shape[0]!=c.shape[0] or n.shape[2]!=c.shape[2] or min(n.shape[1],c.shape[1])<valid_frames:
            result.update(status='SHAPE_OR_LENGTH_MISMATCH')
        else:
            # NeMo and Candle can pad their time axes differently. Compare VALID positions only.
            a=n[:,:valid_frames,:].astype(np.float64);b=c[:,:valid_frames,:].astype(np.float64)
            diff=np.abs(a-b)
            denom=np.linalg.norm(a)*np.linalg.norm(b)
            cosine=float(np.sum(a*b)/denom) if denom>0 else (1. if not diff.any() else 0.)
            result.update(valid_frames=valid_frames,max_abs=float(diff.max()),
              rmse=float(np.sqrt(np.mean((a-b)**2))),cosine=cosine,
              status='PASS_NUMERIC_PARITY' if float(diff.max())<=atol and cosine>=min_cosine else 'NUMERIC_MISMATCH')
        if result['status']!='PASS_NUMERIC_PARITY' and first is None:first=name
        stages.append(result)
    return {'stage':'HDB2C3_A_FASTCONFORMER_LAYER_ATTRIBUTION',
        'status':'ALL_STAGES_NUMERIC_PARITY' if first is None else 'FIRST_DIVERGENCE_LOCALIZED',
        'first_divergence':first,'criteria':{'max_abs_atol':atol,'cosine_min':min_cosine},
        'valid_encoder_frames':valid_frames,'stages':stages,
        'tdt_parity':'NOT_EVALUATED_ENCODER_FIRST',
        'note':'NeMo log-Mel injected bit-identically to Candle; raw traces deliberately kept local'}


def validate_mel_valid_frames(frames: np.ndarray, total: int) -> int:
    values=np.asarray(frames)
    if values.shape!=(1,) or not np.issubdtype(values.dtype,np.integer):
        raise ValueError('NeMo C1 Mel valid lengths must be one integer')
    valid=int(values[0])
    if not (0<valid<=total):
        raise ValueError(f'NeMo Mel valid frames invalid: {valid}/{total}')
    return valid


def execute(reportdir:Path, *, skip_build:bool=False)->Path:
    s=settings();reportdir=reportdir.resolve()
    r=json.loads((reportdir/'nemo_report.json').read_text())
    assert r.get('probe_status')=='REAL_NEMO_MEL_AND_ENCODER','Requires real C1 --probes tensors'
    original=json.loads((reportdir/'candle_report.json').read_text())
    if r['pcm_sha256']!=original['pcm_sha256']:
        raise ValueError('Different PCM in NeMo/Candle original reports')
    conv=json.loads((s['model']/'conversion.json').read_text())
    if conv.get('source_nemo_sha256')!=r['checkpoint_sha256'] or sha(s['nemo'])!=r['checkpoint_sha256']:
        raise ValueError('Checkpoint identity does not match C1/source conversion')
    c2=s['runtime'].with_name(s['runtime'].name+'-hdb2c-c2-v0.6.6')
    c3=s['runtime'].with_name(s['runtime'].name+'-hdb2c-c3-r3-v0.7.2')
    c3c=s['runtime'].with_name(s['runtime'].name+'-hdb2c-c3-r3c-v0.7.3')
    build_candidate(c2,c3)
    build_masked_candidate(c3,c3c)
    target=c3/'target'/'c3-cpu'
    exe=target/'release/hdb2c_probe'
    target_mask=c3c/'target'/'c3c-cpu'
    exe_mask=target_mask/'release/hdb2c_probe'
    if not skip_build:
        env=os.environ.copy();env['CARGO_TARGET_DIR']=str(target)
        cargo=['cargo','build','--release','--manifest-path',str(c3/'Cargo.toml'),'--bin','hdb2c_probe']
        print('HDB2C3_RUST_BUILD='+' '.join(cargo),flush=True)
        subprocess.run(cargo,check=True,env=env)
    if not exe.is_file():raise FileNotFoundError('C3 actual Rust probe was not compiled')
    if not skip_build:
        env=os.environ.copy()
        env['CARGO_TARGET_DIR']=str(target_mask)
        cargo=['cargo','build','--release','--manifest-path',str(c3c/'Cargo.toml'),'--bin','hdb2c_probe']
        print('HDB2C3_R3C_RUST_BUILD='+' '.join(cargo),flush=True)
        subprocess.run(cargo,check=True,env=env)
    if not exe_mask.is_file():raise FileNotFoundError('C3-R3C masked Rust probe was not compiled')
    out=reportdir/'c3_cpu';out.mkdir(exist_ok=True)
    nemo_dir=out/'nemo_trace';candle_dir=out/'candle_trace';nemo_dir.mkdir(exist_ok=True);candle_dir.mkdir(exist_ok=True)
    # This Python subprocess uses the pinned NeMo isolated environment.
    subprocess.run([str(s['python']),'-m','hdb2c.c3_nemo','--nemo',str(s['nemo']),
        '--report',str(reportdir),'--output',str(nemo_dir)],check=True,
        env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])})
    mel=np.load(reportdir/'nemo_mel.npz')['values']
    if mel.ndim!=3 or mel.shape[0]!=1 or mel.shape[1]!=128:
        raise ValueError('Unexpected NeMo Mel layout; require B,128,T')
    mel_valid_arr=np.load(reportdir/'nemo_mel.npz')['valid_frames']
    mel_valid=validate_mel_valid_frames(mel_valid_arr,int(mel.shape[2]))
    mel_path=out/'nemo_mel_injected_BTD.f32le'
    mel.transpose(0,2,1).astype('<f4').tofile(mel_path)
    env=os.environ.copy();env.update(HDB2C3_TRACE_DIR=str(candle_dir),
         HDB2C3_NEMO_MEL_PATH=str(mel_path),HDB2C3_NEMO_MEL_FRAMES=str(mel.shape[2]),
         HDB2C3_NEMO_MEL_VALID_FRAMES=str(mel_valid))
    probe_out=out/'candle_probe';probe_out.mkdir(exist_ok=True)
    subprocess.run([str(exe),'--model-dir',str(s['model']), '--input',str(reportdir/'input-16k.wav'),
      '--output',str(probe_out),'--cpu'],check=True,env=env)
    # Verify that the *actual* Candle probe consumed byte-identical Mel.
    injected=np.fromfile(mel_path,dtype='<f4')
    recorded=np.fromfile(probe_out/'candle_mel.f32le',dtype='<f4')
    if not np.array_equal(injected,recorded):
        raise ValueError('NeMo Mel injection did not reach the real Candle encoder')
    probe_meta=json.loads((probe_out/'candle_probe.json').read_text())
    if probe_meta.get('injected_mel_valid_frames')!=mel_valid:
        raise ValueError('Candle C3 did not acknowledge the exact NeMo valid Mel length')
    if probe_meta.get('encoder_length_argument')!='EXPLICIT_NEMO_VALID_FRAMES_IF_INJECTED':
        raise ValueError('Candle C3 encoder length was not passed explicitly')
    valid=int(np.load(reportdir/'nemo_encoder.npz')['valid_frames'][0])
    if probe_meta.get('encoder_valid_frames')!=valid:
        raise ValueError(f'Encoder valid-frame lengths diverge NeMo={valid} Candle={probe_meta.get("encoder_valid_frames")}')
    result=compare_stages(nemo_dir,candle_dir,valid)
    result['input_lengths']={'nemo_mel_valid_frames':mel_valid,
                             'nemo_mel_total_frames':int(mel.shape[2]),
                             'nemo_encoder_valid_frames':valid,
                             'candle_encoder_valid_frames':probe_meta['encoder_valid_frames'],
                             'state':'EXPLICIT_SHARED_NEMO_VALID_MEL_FRAMES_VERIFIED'}
    result['padding_mask_parity']='R3_BASELINE_MASKS_OFF'
    result['pcm_sha256']=r['pcm_sha256'];result['checkpoint_sha256']=r['checkpoint_sha256']
    result['mel_input']='EXACT_SHARED_NEMO_F32LE_VERIFIED'
    result['revision']='HDB2-C3-A-R3'
    dw=compare_dw(nemo_dir,candle_dir,mel_valid,valid)
    dw['checkpoint_sha256']=r['checkpoint_sha256']
    dw['pcm_sha256']=r['pcm_sha256']
    dw_path=reportdir/'c3_dw_cpu.json'
    dw_path.write_text(json.dumps(dw,ensure_ascii=False,indent=2)+'\n')
    print(f'HDB2C3_R3={dw["status"]} weights={dw["weights_status"]} first_op={dw["first_operation_divergence"]}',flush=True)
    path=reportdir/'c3_layers_cpu.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    # Treatment: same weights and NeMo Mel, isolated Rust with exact per-op masks.
    masked_dir=out/'candle_masked_trace'
    masked_dir.mkdir(exist_ok=True)
    mask_probe=out/'candle_masked_probe'
    mask_probe.mkdir(exist_ok=True)
    mask_env={**env,'HDB2C3_TRACE_DIR':str(masked_dir)}
    subprocess.run([str(exe_mask),'--model-dir',str(s['model']),
        '--input',str(reportdir/'input-16k.wav'),'--output',str(mask_probe),'--cpu'],
        check=True,env=mask_env)
    mask_meta=json.loads((mask_probe/'candle_probe.json').read_text())
    if mask_meta.get('dw_mask_propagation')!='NEMO_MASKED_CONV_SEQUENTIAL':
        raise ValueError('Masked probe did not attest its NeMo DW mask semantics')
    if mask_meta.get('injected_mel_valid_frames')!=mel_valid or mask_meta.get('encoder_valid_frames')!=valid:
        raise ValueError('R3C masked probe valid-frame lengths differ from NeMo')
    recorded_mask=np.fromfile(mask_probe/'candle_mel.f32le',dtype='<f4')
    if not np.array_equal(injected,recorded_mask):
        raise ValueError('R3C masked probe did not consume bit-identical NeMo Mel')
    masked_dw=compare_dw(nemo_dir,masked_dir,mel_valid,valid)
    masked_layers=compare_stages(nemo_dir,masked_dir,valid)
    summary=summarize_ab(dw,masked_dw,result,masked_layers,r['checkpoint_sha256'],r['pcm_sha256'])
    abpath=reportdir/'c3_mask_ab_cpu.json'
    abpath.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(f'HDB2C3_R3C={summary["status"]} first_masked={summary["first_operation_masked"]} pre_encode={summary["pre_encode_status"]} encoder={summary["full_encoder_status"]}',flush=True)
    print(f'HDB2C3_A={result["status"]} first_divergence={result["first_divergence"]}',flush=True)
    print(f'HDB2C3_REPORT={path}',flush=True)
    return path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('report',type=Path)
    ap.add_argument('--skip-build',action='store_true')
    a=ap.parse_args();execute(a.report,skip_build=a.skip_build)

if __name__=='__main__':main()