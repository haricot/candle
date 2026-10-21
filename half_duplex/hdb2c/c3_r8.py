"""R8: frontend attribution, common-input DW operators, real 14-clip campaign.

The report's execution status and numerical/bitwise qualifications are distinct.
All declared numerical thresholds remain unchanged from R7. No raw activations,
WAVs, transcripts or local paths are published by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .cli import ROOT, settings, take_run_dir
from .c2 import prepare
from .c3 import build_candidate as build_r3, STAGES
from .c3_r3 import read_tensor, STAGES as DW_STAGES
from .c3_r3c import build_masked_candidate
from .c3_r4 import build_r4_candidate
from .c3_r5 import build_r5_candidate, digest
from .c3_r6 import build_candidate as build_r6
from .c3_r7 import build_candidate as build_r7, encoder_audit, operator_comparison, trajectory_comparison
from .c3_r7_focused import verify_native_sources
from .c3_r7_metrics import tensor_metrics, compare_decisions
from .c3_r8_source import outputs, probe
from .c3_r8_nemo import FRONTEND_STAGES

SIGNATURE = 'hdb2c3-r8-frontend-common-dw-corpus-v0.7.8'
EMOTIONS = ('anger','disgust','fear','happiness','neutral','sadness','surprise')


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def metric(a, b):
    result = tensor_metrics(a, b)
    if a.shape == b.shape:
        x, y = np.asarray(a,dtype=np.float64), np.asarray(b,dtype=np.float64)
        scale = float(np.max(np.abs(x)))
        result['reference_max_abs'] = scale
        result['max_error_over_reference_max_abs'] = float(np.max(np.abs(x-y))/max(scale,1e-30))
        # Supplementary diagnostics only: this never changes numeric_pass.
        result['relative_l2'] = float(np.linalg.norm(x-y)/max(float(np.linalg.norm(x)),1e-30))
    return result


def files(a, b, name, candidate_name=None):
    return metric(read_tensor(a/(name+'.f32le')),read_tensor(b/((candidate_name or name)+'.f32le')))


def selected_corpus(manifest):
    rows = manifest['rows']
    expected = {(split,emotion) for split in ('cafe','oreau') for emotion in EMOTIONS}
    observed = [(r['split'],r['emotion']) for r in rows]
    if len(rows)!=14 or len(set(observed))!=14 or set(observed)!=expected:
        raise ValueError('R8 requires exactly the 14 real CaFE/Oreau split/emotion pairs')
    if len({r['pcm_sha256'] for r in rows})!=14:
        raise ValueError('R8 refuses duplicate PCM excerpts')
    from .audio import describe
    for row in rows:
        if describe(Path(row['wav']))['pcm_sha256']!=row['pcm_sha256']:
            raise ValueError('R8 corpus PCM changed: '+row['id'])
    return rows


def prepare_sources(s, variant):
    runtime=s['runtime']
    def path(suffix): return runtime.with_name(runtime.name+suffix)
    c2=path('-hdb2c-c2-v0.6.6'); r3=path('-hdb2c-c3-r3-v0.7.2')
    r3c=path('-hdb2c-c3-r3c-v0.7.3'); r4=path('-hdb2c-c3-r4-v0.7.4')
    r5=path('-hdb2c-c3-r5-v0.7.5'); r6=path('-hdb2c-c3-r6-v0.7.6')
    r7=path('-hdb2c-c3-r7-v0.7.7'); r8=path('-hdb2c-c3-r8-v0.7.8')
    evidence=json.loads((ROOT/'ci/r7_verified_native_sources.json').read_text())
    if digest(s['nemo'])!=evidence['checkpoint_sha256']:
        raise ValueError('R8 exact qualified TranscribePsy checkpoint required')
    if json.loads((s['model']/'conversion.json').read_text())['source_nemo_sha256']!=evidence['checkpoint_sha256']:
        raise ValueError('R8 converted model source identity differs')
    prepare(runtime,c2,ROOT,s['model'])
    subprocess.run([str(s['python']),str(ROOT/'scripts/hdb2a_cargo_lock.py'),'--manifest-path',str(c2/'Cargo.toml')],check=True)
    build_r3(c2,r3);build_masked_candidate(r3,r3c);build_r4_candidate(r3c,r4)
    build_r5_candidate(r4,r5);build_r6(r5,r6);build_r7(r6,r7)
    verify_native_sources(r7,evidence)
    generated=outputs(r7,ROOT)
    manifest={'signature':SIGNATURE,'parent_source_sha256':evidence['native_source_sha256'],
        'target_sha256':{name:hashlib.sha256(value.encode()).hexdigest() for name,value in generated.items()}}
    if r8.exists():
        if json.loads((r8/'HDB2C3-R8-MANIFEST.json').read_text())!=manifest:
            raise ValueError('R8 existing candidate differs; refusing overwrite')
        for name,sha in manifest['target_sha256'].items():
            if digest(r8/name)!=sha:raise ValueError('R8 candidate source modified: '+name)
    else:
        shutil.copytree(r7,r8,ignore=shutil.ignore_patterns('target','.git','__pycache__','.venv'))
        for name,value in generated.items():
            destination=r8/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(value)
        write_json(r8/'HDB2C3-R8-MANIFEST.json',manifest)
    # A device-selectable baseline probe changes only host/device routing. Its
    # encoder, frontend and TDT sources are byte-identical to qualified R7.
    baseline=path('-hdb2c-c3-r8-baseline-v0.7.8')
    routed=probe((ROOT/'rust/hdb2c_r7_probe.rs').read_text(),operators=False)
    baseline_manifest={'parent_source_sha256':evidence['native_source_sha256'],
        'probe_sha256':hashlib.sha256(routed.encode()).hexdigest()}
    if not baseline.exists():
        shutil.copytree(r7,baseline,ignore=shutil.ignore_patterns('target','.git','__pycache__','.venv'))
        (baseline/'src/bin/hdb2c_r7_probe.rs').write_text(routed)
        write_json(baseline/'R8-BASELINE.json',baseline_manifest)
    elif (json.loads((baseline/'R8-BASELINE.json').read_text())!=baseline_manifest or
          digest(baseline/'src/bin/hdb2c_r7_probe.rs')!=baseline_manifest['probe_sha256']):
        raise ValueError('R8 baseline modified')
    for name,sha in evidence['native_source_sha256'].items():
        if name!='src/bin/hdb2c_r7_probe.rs' and digest(baseline/name)!=sha:
            raise ValueError('R8 baseline native math changed')
    targets=[]
    for candidate,binary in ((baseline,'hdb2c_r7_probe'),(r8,'hdb2c_r8_probe')):
        target=candidate/'target'/('r8-'+variant)
        env={**os.environ,'CARGO_TARGET_DIR':str(target),'CUDA_COMPUTE_CAP':'61'}
        args=['cargo','build','--release','--manifest-path',str(candidate/'Cargo.toml'),'--bin',binary]
        if variant=='cuda-cudnn':args+=['--features','cuda cudnn']
        subprocess.run(args,check=True,env=env)
        if candidate==r8:
            tests=['cargo','test','--release','--manifest-path',str(candidate/'Cargo.toml'),'--lib','hdb2c']
            if variant=='cuda-cudnn':tests+=['--features','cuda cudnn']
            subprocess.run(tests,check=True,env=env)
        targets.append(target/'release'/binary)
    verify_native_sources(r7,evidence)
    return targets,manifest


def clean_env():
    env={k:v for k,v in os.environ.items() if not k.startswith(('HDB2C3_NEMO_MEL','HDB2C7_TDT_TRACE','HDB2C3_TRACE_DIR','HDB2C8_FRONTEND_TRACE'))}
    return {**env,'HDB2C6_MASK_MODE':'both'}


def native(executable, model, wav, output, variant, *, reference=None, frontend_trace=False, operators=False, tdt=False):
    args=[str(executable),'--model-dir',str(model),'--input',str(wav),'--output',str(output),'--device',variant]
    env=clean_env()
    if operators:args+=['--operator-cases',str(reference)]
    elif tdt:args+=['--tdt-cases',str(reference)]
    else:
        env.update({'HDB2C3_TRACE_DIR':str(output/'encoder_trace'),'HDB2C7_TDT_TRACE_DIR':str(output/'tdt')})
        if reference:
            ref=json.loads((reference/'manifest.json').read_text())
            args+=['--shared-mel',str(reference),'--mel-valid',str(ref['mel_valid'])]
        if frontend_trace:env['HDB2C8_FRONTEND_TRACE_DIR']=str(output/'frontend_trace')
    subprocess.run(args,check=True,env=env)


def input_for(name, native_dir):
    trace=native_dir/'encoder_trace'
    if name=='conv_00':return read_tensor(native_dir/'mel.f32le')[:,None,:,:]
    parents={'conv_02':'relu_00','conv_03':'conv_02','conv_05':'relu_03','conv_06':'conv_05',
             'relu_00':'conv_00','relu_03':'conv_03','relu_06':'conv_06','out':'relu_06'}
    x=read_tensor(trace/(parents[name]+'.f32le'))
    return x.transpose(0,2,1,3).reshape(x.shape[0],x.shape[2],-1) if name=='out' else x


def comparisons(reference, candidate, replay):
    frontend=[{'name':name,**files(reference/'frontend',candidate/'frontend_trace',name)} for name in FRONTEND_STAGES]
    common_front=[]
    for name in ('window','filterbank','fft_complex','magnitude','power','mel_projection','log_mel'):
        common_front.append({'name':name,**files(reference/'frontend',replay/'frontend',name)})
    for name in ('frames','fft_complex','magnitude','power','filterbank','mel_projection'):
        if not files(reference/'frontend',replay/'frontend',name,'input_'+name)['bitwise_equal']:
            raise ValueError('R8 frontend replay consumed changed input bytes: '+name)
    common_dw=[]
    for name in DW_STAGES:
        a,b=reference/'subsampling'/name,replay/'subsampling'/name
        if not files(a,b,'input')['bitwise_equal']:
            raise ValueError('R8 subsampling replay consumed changed input bytes')
        for weight in ('weight','bias'):
            if (a/(weight+'.f32le')).exists() and not files(a,b,weight)['bitwise_equal']:
                raise ValueError('R8 native subsampling weights differ: '+name+'/'+weight)
        local=files(a,b,'output')
        end=files(reference/'encoder_trace',candidate/'encoder_trace',name)
        inherited=metric(read_tensor(a/'input.f32le'),input_for(name,candidate))
        common_dw.append({'name':name,'common_input_operator':local,'autonomous_pipeline':end,
            'input_difference':inherited,'input_bytes_verified':True,
            'attribution':('LOCAL_OPERATOR_DIFFERENCE_OBSERVED' if not local['numeric_pass'] else
                'INHERITED_DIFFERENCE_OBSERVED' if not end['numeric_pass'] else 'PASS_NUMERIC_PARITY')})
    return {'frontend':frontend,'common_input_frontend':common_front,'common_input_subsampling':common_dw,
        'first_frontend_bit_difference':next((r['name'] for r in frontend if not r['bitwise_equal']),None),
        'first_frontend_numeric_mismatch':next((r['name'] for r in frontend if not r['numeric_pass']),None),
        'common_inputs':'BIT_IDENTICAL_REAL_NEMO_INPUTS_VERIFIED',
        'operator_execution_scope':'SEPARATE_DIAGNOSTIC_REPLAY_NEVER_PRODUCTION_INJECTION'}


def instrumentation_control(reference, observed):
    names={p.stem for p in (reference/'encoder_trace').glob('*.f32le')}
    observed_names={p.stem for p in (observed/'encoder_trace').glob('*.f32le')}
    required=set(DW_STAGES)|set(STAGES)|{'encoder_pad_mask','encoder_att_mask'}
    if names!=observed_names or not required.issubset(names):
        raise ValueError('R8 baseline/instrumented trace coverage differs')
    for name in names:
        observed_metric=files(reference/'encoder_trace',observed/'encoder_trace',name)
        if not observed_metric['bitwise_equal']:
            raise ValueError('R8 instrumentation changed R7 stage '+name+': '+json.dumps(observed_metric))
    for name in ('pcm','mel','encoder'):
        if not files(reference,observed,name)['bitwise_equal']:
            raise ValueError('R8 instrumentation changed R7 '+name)
    a=json.loads((reference/'tdt/tdt_trace.json').read_text())
    b=json.loads((observed/'tdt/tdt_trace.json').read_text())
    if a!=b:raise ValueError('R8 instrumentation changed R7 TDT decisions')
    for i in range(len(a)):
        for name in ('g','h','c','enc','logits'):
            if not files(reference/'tdt'/f'{i:06}',observed/'tdt'/f'{i:06}',name)['bitwise_equal']:
                raise ValueError('R8 instrumentation changed R7 TDT bits')
    if (reference/'result.json').read_bytes()!=(observed/'result.json').read_bytes():
        raise ValueError('R8 instrumentation changed R7 returned result')
    return {'status':'PASS_BITWISE_R7_R8_ALL_OBSERVED_STAGES_AND_TDT','encoder_stages':len(names),
        'tdt_decisions':len(a),'same_backend':True}


def execute_clip(s, binaries, row, report, variant, source_manifest):
    wav=Path(row['wav']);reference=report/'nemo'
    subprocess.run([str(s['python']),'-m','hdb2c.c3_r8_nemo','--nemo',str(s['nemo']),
        '--model-dir',str(s['model']),'--wav',str(wav),'--output',str(reference)],check=True)
    ref=json.loads((reference/'manifest.json').read_text())
    if ref['pcm_sha256']!=row['pcm_sha256']:raise ValueError('R8 real reference PCM mismatch')
    cfg=json.loads((s['model']/'config.json').read_text())['preprocessor']
    if cfg['pad_to']!=ref['preprocessor_state']['pad_to']:raise ValueError('R8 actual padding config mismatch')
    baseline,observed=binaries
    native(baseline,s['model'],wav,report/'baseline',variant)
    native(observed,s['model'],wav,report/'native',variant,frontend_trace=True)
    control=instrumentation_control(report/'baseline',report/'native')
    native(observed,s['model'],wav,report/'shared_mel',variant,reference=reference)
    native(observed,s['model'],wav,report/'operators',variant,reference=reference,operators=True)
    native(observed,s['model'],wav,report/'tdt_operators',variant,reference=reference,tdt=True)
    result=comparisons(reference,report/'native',report/'operators')
    enc={mode:encoder_audit(reference/'encoder_trace',report/mode,ref['encoder_valid']) for mode in ('native','shared_mel')}
    tdt={mode:trajectory_comparison(reference,report/mode,ref) for mode in ('native','shared_mel')}
    common=operator_comparison(reference,report/'tdt_operators',ref)
    observed_result=json.loads((report/'native/result.json').read_text())
    if not files(reference,report/'native','pcm')['bitwise_equal']:
        raise ValueError('R8 native pipeline consumed different PCM bytes')
    if not files(reference,report/'shared_mel','mel')['bitwise_equal']:
        raise ValueError('R8 shared Mel diagnostic input changed')
    if variant=='cuda-cudnn' and not observed_result['device'].startswith('Cuda('):
        raise RuntimeError('R8 CUDA test silently fell back to CPU')
    all_numeric=(all(r['numeric_pass'] for r in result['frontend']+result['common_input_frontend']) and
        all(r['common_input_operator']['numeric_pass'] for r in result['common_input_subsampling']) and
        all(r['numeric_pass'] for mode in enc.values() for r in mode['stages']) and
        common['numeric_pass'] and common['argmax_exact'] and
        all(r['numeric_pass'] and r['tdt_discrete_exact'] for r in tdt.values()))
    bits=(all(r['bitwise_equal'] for r in result['frontend']) and enc['native']['bitwise_equal'] and
          tdt['native']['bitwise_equal'] and tdt['native']['tdt_discrete_exact'])
    # Keep captions/tokens/local paths in local evidence; publish metrics only.
    public_tdt={mode:{k:v for k,v in item.items() if k not in ('candidate_output','decisions')}
                for mode,item in tdt.items()}
    for mode,item in public_tdt.items():
        item['decision_counts']={k:v for k,v in tdt[mode]['decisions'].items() if k!='first_difference'}
    result.update({'signature':SIGNATURE,'revision':'HDB2-C3-A-R8','clip':{k:row[k] for k in ('id','split','emotion','pcm_sha256')},
        'execution_status':'REAL_FRONTEND_OPERATORS_AND_ASR_COMPLETE','variant':variant,'device':observed_result['device'],
        'checkpoint_sha256':ref['checkpoint_sha256'],'native_pipeline_internal_tensor_injection':'NONE',
        'qualification':'PASS_ALL_DECLARED_NUMERIC_GATES' if all_numeric else 'NUMERIC_PARITY_NOT_ESTABLISHED',
        'global_bitwise_status':'PASS_BITWISE' if bits else 'FAIL_DIFFERENT_F32_BITS',
        'instrumentation_control':control,'nemo_frontend_control':ref['frontend_control'],
        'nemo_encoder_control':ref['encoder_instrumentation_control'],'encoder':enc,'tdt':public_tdt,
        'common_input_tdt':{k:v for k,v in common.items() if k!='cases'},'source_manifest':source_manifest,
        'limits':'Batch 1 offline FP32; no streaming or microphone claim; numerical and strict bits are independent.'})
    return result


def main(corpus, variant, *, keep_raw=False, wav=None, require_numeric=False):
    s=settings(); s['reports'].mkdir(parents=True,exist_ok=True)
    root=take_run_dir(s['reports'])
    if wav is not None:
        from .audio import normalize,describe
        normalized=root/'input-16k.wav'; normalize(wav,normalized)
        rows=[{'id':'local-microphone','split':'local','emotion':'unspecified','wav':str(normalized),
               'pcm_sha256':describe(normalized)['pcm_sha256']}]
    else:
        rows=selected_corpus(json.loads(corpus.read_text()))
    published=Path(os.environ.get('GITHUB_WORKSPACE',str(root)))/'halfduplex-ci-published'
    if wav is not None:published=root/'local-summaries'
    published.mkdir(parents=True,exist_ok=True)
    binaries,manifest=prepare_sources(s,variant)
    summaries=[];errors=[]
    for index,row in enumerate(rows):
        report=root/f'clip-{index:02}';report.mkdir()
        try:
            result=execute_clip(s,binaries,row,report,variant,manifest)
            write_json(published/f'r8-{variant}-{index:02}.json',result)
            summaries.append({'clip':result['clip'],'qualification':result['qualification'],
                'global_bitwise_status':result['global_bitwise_status'],
                'first_frontend_bit_difference':result['first_frontend_bit_difference'],
                'first_frontend_numeric_mismatch':result['first_frontend_numeric_mismatch'],
                'final_encoder':result['encoder']['native']['valid_final_output'],
                'tdt_numeric_and_decision_parity':all(x['numeric_pass'] and x['tdt_discrete_exact'] for x in result['tdt'].values())})
            print(f'HDB2C3_R8_CLIP={index+1}/{len(rows)} '+result['qualification']+' bits='+result['global_bitwise_status'],flush=True)
        except Exception as exc:
            import traceback
            errors.append({'index':index,'id':row['id'],'error':type(exc).__name__+': '+str(exc)})
            traceback.print_exc()
            write_json(published/f'r8-{variant}-{index:02}-error.json',errors[-1])
        finally:
            if not keep_raw and wav is None:shutil.rmtree(report)
    summary={'signature':SIGNATURE,'variant':variant,'expected_clips':len(rows),'completed_clips':len(summaries),
        'scope':'LOCAL_MICROPHONE_PCM' if wav is not None else 'ALL_14_REAL_CAFE_OREAU_EXCERPTS',
        'execution_status':'COMPLETE' if not errors and len(summaries)==len(rows) else 'INCOMPLETE',
        'qualification':'PASS_ALL_DECLARED_NUMERIC_GATES' if not errors and len(summaries)==len(rows) and all(r['qualification']=='PASS_ALL_DECLARED_NUMERIC_GATES' for r in summaries) else 'NUMERIC_PARITY_NOT_ESTABLISHED',
        'results':summaries,'errors':errors,'streaming':'NOT_RUN','microphone':'LOCAL_WAV_ONLY' if wav is not None else 'NOT_RUN'}
    write_json(published/f'r8-{variant}-summary.json',summary)
    write_json(root/'summary.json',summary)
    print('HDB2C3_R8='+summary['execution_status']+' '+summary['qualification'],flush=True)
    print('HDB2C3_R8_REPORT='+str(root),flush=True)
    return 1 if errors else 2 if require_numeric and summary['qualification']!='PASS_ALL_DECLARED_NUMERIC_GATES' else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--corpus',type=Path);source.add_argument('--wav',type=Path)
    parser.add_argument('--variant',choices=('cpu','cuda-cudnn'),default='cpu')
    parser.add_argument('--keep-raw',action='store_true')
    parser.add_argument('--require-numeric',action='store_true',help='Fail the job when any declared numerical gate fails')
    a=parser.parse_args();sys.exit(main(a.corpus,a.variant,keep_raw=a.keep_raw,wav=a.wav,require_numeric=a.require_numeric))
