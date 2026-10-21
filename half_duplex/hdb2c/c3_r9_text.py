"""R9 isolated native token-text correction, with full R8B numerical controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .cli import ROOT, settings, take_run_dir
from .c3_r5 import digest
from .c3_r7 import encoder_audit, trajectory_comparison
from .c3_r8 import (prepare_sources, selected_corpus, native, instrumentation_control,
                   execute_clip, files, write_json)
from .c3_r8_nemo import FRONTEND_STAGES
from .c3_r9_source import original_metadata, outputs

SIGNATURE='hdb2c3-r9-original-tokenizer-text-v0.7.9'


def prepare_text_candidate(parent,root,metadata,variant):
    target=parent.with_name(parent.name+'-r9-text-v0.7.9')
    generated=outputs(parent,root,metadata)
    # Parent manifests cover generated observation helpers; also retain every
    # original native math source, including the decoder before text changes.
    names=sorted(p.relative_to(parent).as_posix() for p in (parent/'src').rglob('*.rs'))+['Cargo.toml','Cargo.lock']
    manifest={'signature':SIGNATURE,'tokenizer_metadata':metadata,
        'parent_source_sha256':{name:digest(parent/name) for name in names},
        'target_sha256':{name:hashlib.sha256(value.encode()).hexdigest() for name,value in generated.items()}}
    if target.exists():
        if json.loads((target/'R9-TEXT-MANIFEST.json').read_text())!=manifest:
            raise ValueError('R9 existing candidate differs')
        for name,sha in {**manifest['parent_source_sha256'],**manifest['target_sha256']}.items():
            if digest(target/name)!=sha:raise ValueError('R9 candidate modified: '+name)
    else:
        shutil.copytree(parent,target,ignore=shutil.ignore_patterns('target','.git','__pycache__','.venv'))
        for name,value in generated.items():
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
        write_json(target/'R9-TEXT-MANIFEST.json',manifest)
    cargo_target=target/'target'/('r9-'+variant)
    env={**os.environ,'CARGO_TARGET_DIR':str(cargo_target),'CUDA_COMPUTE_CAP':'61'}
    for command in (['build','--bin','hdb2c_r9_probe'],['test','--lib','hdb2c']):
        args=['cargo',command[0],'--release','--manifest-path',str(target/'Cargo.toml'),*command[1:]]
        if variant=='cuda-cudnn':args+=['--features','cuda cudnn']
        subprocess.run(args,env=env,check=True)
    for name,sha in manifest['parent_source_sha256'].items():
        if digest(parent/name)!=sha:raise ValueError('R9 parent source changed')
    return cargo_target/'release/hdb2c_r9_probe',manifest


def strict_numeric(baseline,encoder,tdt):
    return (all(r['numeric_pass'] for r in baseline['frontend']+baseline['common_input_frontend'])
        and all(r['common_input_operator']['numeric_pass'] for r in baseline['common_input_subsampling'])
        and all(r['numeric_pass'] for mode in encoder.values() for r in mode['stages'])
        and baseline['common_input_tdt']['numeric_pass'] and baseline['common_input_tdt']['argmax_exact']
        and all(r['numeric_pass'] and r['tdt_discrete_exact'] for r in tdt.values()))


def execute_text_clip(s,binary,row,report,variant,baseline,manifest):
    reference=report/'nemo';ref=json.loads((reference/'manifest.json').read_text())
    controls={};encoder={};tdt={}
    for mode in ('native','shared_mel'):
        target=report/('r9_'+mode)
        native(binary,s['model'],Path(row['wav']),target,variant,
            reference=reference if mode=='shared_mel' else None,frontend_trace=mode=='native')
        controls[mode]=instrumentation_control(report/mode,target,allow_text_change=True)
        encoder[mode]=encoder_audit(reference/'encoder_trace',target,ref['encoder_valid'])
        tdt[mode]=trajectory_comparison(reference,target,ref)
    for name in FRONTEND_STAGES:
        if not files(report/'native/frontend_trace',report/'r9_native/frontend_trace',name)['bitwise_equal']:
            raise ValueError('R9 changed frontend primitive '+name)
    public_tdt={}
    for mode,item in tdt.items():
        public_tdt[mode]={k:v for k,v in item.items() if k not in ('candidate_output','decisions')}
        public_tdt[mode]['decision_counts']={k:v for k,v in item['decisions'].items() if k!='first_difference'}
    numeric=strict_numeric(baseline,encoder,tdt)
    bits=(all(x['bitwise_equal'] for x in baseline['frontend']) and encoder['native']['bitwise_equal']
        and tdt['native']['bitwise_equal'] and tdt['native']['tdt_discrete_exact'])
    return {'signature':SIGNATURE,'revision':'HDB2-C3-A-R9-TEXT','clip':baseline['clip'],
        'execution_status':'REAL_NATIVE_TEXT_CORRECTION_AND_R8B_CONTROLS_COMPLETE','variant':variant,
        'checkpoint_sha256':baseline['checkpoint_sha256'],
        'qualification':'PASS_ALL_DECLARED_NUMERIC_GATES' if numeric else 'NUMERIC_PARITY_NOT_ESTABLISHED',
        'global_bitwise_status':'PASS_BITWISE' if bits else 'FAIL_DIFFERENT_F32_BITS',
        'math_control':controls,'frontend_primitive_control':'PASS_BITWISE_ALL_11_R8B_R9',
        'tokenizer_metadata':manifest['tokenizer_metadata'],'source_manifest':manifest,
        'encoder':encoder,'tdt':public_tdt,
        'baseline_r8b_qualification':baseline['qualification'],
        'native_pipeline_internal_tensor_injection':'NONE',
        'limits':'Isolated candidate, original numerical gates, no production promotion or streaming claim.'}


def main(corpus,variant,require_numeric=False):
    s=settings();s['reports'].mkdir(parents=True,exist_ok=True);root=take_run_dir(s['reports'])
    rows=selected_corpus(json.loads(corpus.read_text()))
    binaries,parent_manifest=prepare_sources(s,variant)
    metadata=original_metadata(s['nemo'],s['model'])
    binary,manifest=prepare_text_candidate(binaries[1].parents[3],ROOT,metadata,variant)
    published=Path(os.environ.get('GITHUB_WORKSPACE',str(root)))/'halfduplex-ci-published'
    published.mkdir(parents=True,exist_ok=True);results=[];errors=[]
    for index,row in enumerate(rows):
        report=root/f'clip-{index:02}';report.mkdir()
        try:
            baseline=execute_clip(s,binaries,row,report,variant,parent_manifest)
            write_json(published/f'r8b-r9-control-{variant}-{index:02}.json',baseline)
            result=execute_text_clip(s,binary,row,report,variant,baseline,manifest)
            write_json(published/f'r9-text-{variant}-{index:02}.json',result)
            results.append({'clip':result['clip'],'qualification':result['qualification'],
                'global_bitwise_status':result['global_bitwise_status'],
                'math_control':result['math_control'],
                'text_exact':all(r['output_exact']['text'] for r in result['tdt'].values()),
                'tdt_exact':all(r['tdt_discrete_exact'] for r in result['tdt'].values()),
                'tdt_numeric':all(r['numeric_pass'] for r in result['tdt'].values()),
                'decisions':result['tdt']['native']['decision_counts'],
                'final_encoder':result['encoder']['native']['valid_final_output']})
            print(f'HDB2C3_R9_CLIP={index+1}/{len(rows)} text_exact={results[-1]["text_exact"]}',flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc();errors.append({'index':index,'id':row['id'],'error':type(exc).__name__+': '+str(exc)})
            write_json(published/f'r9-text-{variant}-{index:02}-error.json',errors[-1])
        finally:shutil.rmtree(report)
    summary={'signature':SIGNATURE,'variant':variant,'expected_clips':len(rows),'completed_clips':len(results),
        'execution_status':'COMPLETE' if len(results)==len(rows) and not errors else 'INCOMPLETE',
        'qualification':'PASS_ALL_DECLARED_NUMERIC_GATES' if not errors and len(results)==len(rows)
            and all(r['qualification']=='PASS_ALL_DECLARED_NUMERIC_GATES' for r in results) else 'NUMERIC_PARITY_NOT_ESTABLISHED',
        'results':results,'errors':errors,'microphone':'NOT_RUN','streaming':'NOT_RUN'}
    write_json(published/f'r9-text-{variant}-summary.json',summary);write_json(root/'summary.json',summary)
    print('HDB2C3_R9='+summary['execution_status']+' '+summary['qualification'],flush=True)
    return 1 if errors else 2 if require_numeric and summary['qualification']!='PASS_ALL_DECLARED_NUMERIC_GATES' else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--variant',choices=('cpu','cuda-cudnn'),default='cpu')
    parser.add_argument('--require-numeric',action='store_true');a=parser.parse_args()
    sys.exit(main(a.corpus,a.variant,a.require_numeric))
