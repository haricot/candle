"""CPU-only numerical qualification with original checkpoint buffers and MKL.

The independent R7/R8B pipeline and actual NeMo reference are retained. This
profile does not modify the live runtime or qualify its default backend.
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

from .cli import ROOT,settings,take_run_dir
from .c3_r5 import digest
from .c3_r7 import encoder_audit,trajectory_comparison,operator_comparison
from .c3_r8 import (prepare_sources,selected_corpus,native,execute_clip,
                   comparisons,files,write_json)
from .c3_r9_source import original_metadata
from .c3_r9_text import strict_numeric
from .c3_r10_source import outputs

SIGNATURE='hdb2c3-r10-native-cpu-mkl-v0.7.10'


def prepare_candidate(parent,root,metadata,python):
    sdk=json.loads(subprocess.check_output([str(python),'-c',
        'import sys,json,importlib.metadata as m;print(json.dumps({"root":sys.prefix,"mkl":m.version("mkl"),"headers":m.version("mkl-include")}))'],text=True))
    if sdk['mkl']!='2024.2.2' or sdk['headers']!='2024.2.2':
        raise ValueError('R10 requires separately installed mkl==2024.2.2 and mkl-include==2024.2.2')
    sdkroot=Path(sdk['root'])
    libraries=sorted((sdkroot/'lib').glob('libmkl*.so.2'))
    if not libraries or not (sdkroot/'include/mkl_dfti.h').is_file():
        raise ValueError('R10 standalone MKL libraries/headers missing')
    sdk['library_sha256']={p.name:digest(p) for p in libraries}
    generated=outputs(parent,root,metadata)
    names=sorted(p.relative_to(parent).as_posix() for p in (parent/'src').rglob('*.rs'))+['Cargo.toml','Cargo.lock']
    manifest={'signature':SIGNATURE,'sdk':sdk,'tokenizer_metadata':metadata,
        'parent_source_sha256':{n:digest(parent/n) for n in names},
        'target_sha256':{n:hashlib.sha256(c.encode()).hexdigest() for n,c in generated.items()}}
    target=parent.with_name(parent.name+'-r10-cpu-mkl-v0.7.10')
    if target.exists():
        if json.loads((target/'R10-CPU-MANIFEST.json').read_text())!=manifest:
            raise ValueError('R10 cached source/SDK differs; use a fresh candidate directory')
        for n,sha in {**manifest['parent_source_sha256'],**manifest['target_sha256']}.items():
            if digest(target/n)!=sha:raise ValueError('R10 cached candidate changed: '+n)
    else:
        shutil.copytree(parent,target,ignore=shutil.ignore_patterns('target','.git','__pycache__','.venv'))
        for n,c in generated.items():
            p=target/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(c)
        write_json(target/'R10-CPU-MANIFEST.json',manifest)
    build=target/'target/r10-cpu'
    env={**os.environ,'CARGO_TARGET_DIR':str(build),'HDB2C_MKL_ROOT':str(sdkroot)}
    for command in (['build','--bin','hdb2c_r10_probe'],['test','--lib','hdb2c10']):
        subprocess.run(['cargo',command[0],'--release','--manifest-path',str(target/'Cargo.toml'),*command[1:]],env=env,check=True)
    for n,sha in manifest['parent_source_sha256'].items():
        if digest(parent/n)!=sha:raise ValueError('R10 parent changed during build')
    public={**manifest,'sdk':{k:v for k,v in sdk.items() if k!='root'}}
    return build/'release/hdb2c_r10_probe',public


def execute_candidate(s,binary,row,report,baseline,manifest):
    reference=report/'nemo';ref=json.loads((reference/'manifest.json').read_text())
    for mode in ('native','shared_mel','operators','tdt_operators'):
        native(binary,s['model'],Path(row['wav']),report/('r10_'+mode),'cpu',
            reference=reference if mode!='native' else None,frontend_trace=mode=='native',
            operators=mode=='operators',tdt=mode=='tdt_operators')
    target=report/'r10_native'
    if not files(reference,target,'pcm')['bitwise_equal']:
        raise ValueError('R10 PCM changed')
    if not files(reference,report/'r10_shared_mel','mel')['bitwise_equal']:
        raise ValueError('R10 shared-Mel input changed')
    result=comparisons(reference,target,report/'r10_operators')
    encoder={mode:encoder_audit(reference/'encoder_trace',report/('r10_'+mode),ref['encoder_valid'])
        for mode in ('native','shared_mel')}
    tdt={mode:trajectory_comparison(reference,report/('r10_'+mode),ref) for mode in ('native','shared_mel')}
    common=operator_comparison(reference,report/'r10_tdt_operators',ref)
    numeric=strict_numeric({**result,'common_input_tdt':common},encoder,tdt)
    bits=(all(r['bitwise_equal'] for r in result['frontend']) and encoder['native']['bitwise_equal']
        and tdt['native']['bitwise_equal'] and tdt['native']['tdt_discrete_exact'])
    public_tdt={}
    for mode,item in tdt.items():
        public_tdt[mode]={k:v for k,v in item.items() if k not in ('candidate_output','decisions')}
        public_tdt[mode]['decision_counts']={k:v for k,v in item['decisions'].items() if k!='first_difference'}
    return {**result,'signature':SIGNATURE,'revision':'HDB2-C3-A-R10-CPU',
        'clip':baseline['clip'],'execution_status':'REAL_CPU_PROFILE_COMPLETE',
        'qualification':'PASS_ALL_DECLARED_NUMERIC_GATES' if numeric else 'NUMERIC_PARITY_NOT_ESTABLISHED',
        'global_bitwise_status':'PASS_BITWISE' if bits else 'FAIL_DIFFERENT_F32_BITS',
        'encoder':encoder,'tdt':public_tdt,'common_input_tdt':common,
        'cpu_backend':json.loads((target/'cpu_backend.json').read_text()),
        'checkpoint_sha256':baseline['checkpoint_sha256'],'source_manifest':manifest,
        'independent_r7_r8b_control':baseline['instrumentation_control'],
        'native_pipeline_internal_tensor_injection':'NONE',
        'limits':'Isolated CPU profile; default backend, other CPUs, GPU and microphone are not qualified.'}


def main(corpus,require_numeric=False,keep_raw=False):
    s=settings();root=take_run_dir(s['reports']);rows=selected_corpus(json.loads(corpus.read_text()))
    binaries,parent_manifest=prepare_sources(s,'cpu');parent=binaries[1].parents[3]
    baseline_hashes={str(p):digest(p) for p in binaries}
    binary,manifest=prepare_candidate(parent,ROOT,original_metadata(s['nemo'],s['model']),s['python'])
    if any(digest(Path(p))!=sha for p,sha in baseline_hashes.items()):raise ValueError('R10 build changed baseline executable')
    published=Path(os.environ.get('GITHUB_WORKSPACE',str(root)))/'halfduplex-ci-published'
    published.mkdir(parents=True,exist_ok=True);results=[];errors=[]
    for index,row in enumerate(rows):
        report=root/f'clip-{index:02}';report.mkdir()
        try:
            # The original reference, observations and independent math controls
            # execute on this same PCM before the new profile is evaluated.
            baseline=execute_clip(s,binaries,row,report,'cpu',parent_manifest)
            write_json(published/f'r8b-r10-control-cpu-{index:02}.json',baseline)
            result=execute_candidate(s,binary,row,report,baseline,manifest)
            write_json(published/f'r10-cpu-{index:02}.json',result)
            results.append({'clip':result['clip'],'qualification':result['qualification'],
                'frontend_bitwise':all(t['bitwise_equal'] for t in result['frontend']),
                'common_projection_bitwise':next(t for t in result['common_input_subsampling'] if t['name']=='out')['common_input_operator']['bitwise_equal'],
                'tdt_exact_numeric':all(t['tdt_discrete_exact'] and t['numeric_pass'] for t in result['tdt'].values()),
                'final_encoder':result['encoder']['native']['valid_final_output']})
            print(f'HDB2C3_R10_CLIP={index+1}/14 '+result['qualification'],flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc();errors.append({'index':index,'id':row['id'],'error':type(exc).__name__+': '+str(exc)})
            write_json(published/f'r10-cpu-{index:02}-error.json',errors[-1])
        finally:
            if not keep_raw:shutil.rmtree(report)
    summary={'signature':SIGNATURE,'scope':'REAL_14_CLIP_CPU_PROFILE_ONLY','expected_clips':14,
        'completed_clips':len(results),'results':results,'errors':errors,
        'execution_status':'COMPLETE' if len(results)==14 and not errors else 'INCOMPLETE',
        'qualification':'PASS_ALL_DECLARED_NUMERIC_GATES' if len(results)==14 and not errors
            and all(r['qualification']=='PASS_ALL_DECLARED_NUMERIC_GATES' for r in results) else 'NUMERIC_PARITY_NOT_ESTABLISHED',
        'gpu':'NOT_RUN','microphone':'NOT_RUN'}
    write_json(published/'r10-cpu-summary.json',summary);write_json(root/'summary.json',summary)
    print('HDB2C3_R10='+summary['execution_status']+' '+summary['qualification'],flush=True)
    print('HDB2C3_R10_REPORT='+str(root),flush=True)
    return 1 if errors else 2 if require_numeric and summary['qualification']!='PASS_ALL_DECLARED_NUMERIC_GATES' else 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--variant',choices=('cpu',),default='cpu');p.add_argument('--keep-raw',action='store_true')
    p.add_argument('--require-numeric',action='store_true');a=p.parse_args()
    sys.exit(main(a.corpus,a.require_numeric,a.keep_raw))
