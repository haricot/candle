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

SIGNATURE = 'hdb2c3-exact-nemo-mel-layer-trace-v0.7.0'
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
    let meta = serde_json::json!({"shape":shape,"dtype":"f32le","layout":"B,T,D"});
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
    return source.replace(original,injection,1)


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
        'generated_hashes':generated,'scope':'CPU exact same NeMo logmel, no model mutation'}
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
    c3=s['runtime'].with_name(s['runtime'].name+'-hdb2c-c3-v0.7.0')
    build_candidate(c2,c3)
    target=c3/'target'/'c3-cpu'
    exe=target/'release/hdb2c_probe'
    if not skip_build:
        env=os.environ.copy();env['CARGO_TARGET_DIR']=str(target)
        cargo=['cargo','build','--release','--manifest-path',str(c3/'Cargo.toml'),'--bin','hdb2c_probe']
        print('HDB2C3_RUST_BUILD='+' '.join(cargo),flush=True)
        subprocess.run(cargo,check=True,env=env)
    if not exe.is_file():raise FileNotFoundError('C3 actual Rust probe was not compiled')
    out=reportdir/'c3_cpu';out.mkdir(exist_ok=True)
    nemo_dir=out/'nemo_trace';candle_dir=out/'candle_trace';nemo_dir.mkdir(exist_ok=True);candle_dir.mkdir(exist_ok=True)
    # This Python subprocess uses the pinned NeMo isolated environment.
    subprocess.run([str(s['python']),'-m','hdb2c.c3_nemo','--nemo',str(s['nemo']),
        '--report',str(reportdir),'--output',str(nemo_dir)],check=True,
        env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])})
    mel=np.load(reportdir/'nemo_mel.npz')['values']
    if mel.ndim!=3 or mel.shape[0]!=1 or mel.shape[1]!=128:
        raise ValueError('Unexpected NeMo Mel layout; require B,128,T')
    mel_path=out/'nemo_mel_injected_BTD.f32le'
    mel.transpose(0,2,1).astype('<f4').tofile(mel_path)
    env=os.environ.copy();env.update(HDB2C3_TRACE_DIR=str(candle_dir),
         HDB2C3_NEMO_MEL_PATH=str(mel_path),HDB2C3_NEMO_MEL_FRAMES=str(mel.shape[2]))
    probe_out=out/'candle_probe';probe_out.mkdir(exist_ok=True)
    subprocess.run([str(exe),'--model-dir',str(s['model']), '--input',str(reportdir/'input-16k.wav'),
      '--output',str(probe_out),'--cpu'],check=True,env=env)
    # Verify that the *actual* Candle probe consumed byte-identical Mel.
    injected=np.fromfile(mel_path,dtype='<f4')
    recorded=np.fromfile(probe_out/'candle_mel.f32le',dtype='<f4')
    if not np.array_equal(injected,recorded):
        raise ValueError('NeMo Mel injection did not reach the real Candle encoder')
    valid=int(np.load(reportdir/'nemo_encoder.npz')['valid_frames'][0])
    result=compare_stages(nemo_dir,candle_dir,valid)
    result['pcm_sha256']=r['pcm_sha256'];result['checkpoint_sha256']=r['checkpoint_sha256']
    result['mel_input']='EXACT_SHARED_NEMO_F32LE_VERIFIED'
    path=reportdir/'c3_layers_cpu.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'HDB2C3_A={result["status"]} first_divergence={result["first_divergence"]}',flush=True)
    print(f'HDB2C3_REPORT={path}',flush=True)
    return path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('report',type=Path)
    ap.add_argument('--skip-build',action='store_true')
    a=ap.parse_args();execute(a.report,skip_build=a.skip_build)

if __name__=='__main__':main()