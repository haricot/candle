#!/usr/bin/env python3
"""HDB2-B2 evidence-only Pascal sm_61 benchmarking and 3-way comparison.

Never fabricates ASR output or substitutes an unseen baseline. Only a real
Candle runtime with the HDB2-B2 in-process profiling marker can qualify.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
import uuid

from hdb2b_driver import (VARIANTS, TARGET_CAP, digest, ensure_gpu, get_model_meta,
                          pcm_digest, report_folder, require_runtime, run_text)
from run_candle_hdb2a import MODEL_ID, PARAKEET_PIN, CANDLE_PIN, convert_audio

PASS='REAL_CANDLE_HDB2B2_QUALIFIED_ON_SINGLE_AUDIO_UNQUALIFIED_WER'
PHASES=('wav_decode_ms','logmel_ms','fastconformer_ms','tdt_decode_ms','text_alignment_ms','profile_wall_ms')

def out_root(runtime, variant):
    return runtime/'target'/f'hdb2b2-{variant}'

def manifest_path(runtime, variant):
    return out_root(runtime,variant)/'hdb2b2-build.json'

def binary_path(runtime, variant):
    return out_root(runtime,variant)/'release'/'parakeet-candle'

def require_patch(runtime):
    runtime=require_runtime(runtime)
    src=(runtime/'src/main.rs').read_text(encoding='utf-8')
    if src.count('// HDB2B2_PROFILE_V1')!=1:
        raise ValueError('HDB2-B2 runtime patch absent; execute ./run-hdb2b2.sh repair-runtime')
    return runtime

def build(args):
    runtime=require_patch(args.runtime_dir)
    gpu=ensure_gpu() if args.variant!='cpu' else None
    if gpu and not run_text(['nvcc','--version']):
        raise RuntimeError('nvcc unavailable; target sm_61 cannot be built')
    if gpu:
        codes=run_text(['nvcc','--list-gpu-code'])
        if codes and not ('sm_61' in codes or 'compute_61' in codes):
            raise RuntimeError('nvcc does not offer sm_61')
    env=os.environ.copy()
    env['CARGO_TARGET_DIR']=str(out_root(runtime,args.variant))
    if gpu:env['CUDA_COMPUTE_CAP']=TARGET_CAP
    subprocess.run([sys.executable,str(args.project_root/'scripts/hdb2a_cargo_lock.py'),
                    '--manifest-path',str(runtime/'Cargo.toml')],check=True)
    cmd=['cargo','build','--release','--manifest-path',str(runtime/'Cargo.toml')]
    if VARIANTS[args.variant]:cmd.extend(['--features',','.join(VARIANTS[args.variant])])
    print(f'HDB2B2=BUILDING_{args.variant.upper()}',flush=True)
    subprocess.run(cmd,cwd=runtime,env=env,check=True)
    b=binary_path(runtime,args.variant)
    if not b.is_file():raise RuntimeError(f'Built binary not found: {b}')
    manifest=dict(schema_version=1,variant=args.variant,features=list(VARIANTS[args.variant]),
        target_sm=TARGET_CAP if gpu else None,gpu=gpu,
        runtime_pin=PARAKEET_PIN,candle_pin=CANDLE_PIN,
        binary_sha256=digest(b),main_sha256=digest(runtime/'src/main.rs'),
        cargo_lock_sha256=digest(runtime/'Cargo.lock') if (runtime/'Cargo.lock').is_file() else None,
        rust_version=run_text(['rustc','--version']),nvcc_version=run_text(['nvcc','--version']) if gpu else None,
        built_at=datetime.now(timezone.utc).isoformat())
    path=manifest_path(runtime,args.variant)
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'HDB2B2=BUILT_NOT_RUNTIME_QUALIFIED\nHDB2B2_BINARY={b}\nHDB2B2_BUILD_MANIFEST={path}')

def check_build(runtime,variant):
    p=manifest_path(runtime,variant);b=binary_path(runtime,variant)
    if not p.is_file() or not b.is_file():
        raise ValueError(f'Missing independent HDB2-B2 build; run ./run-hdb2b2.sh build {variant}')
    m=json.loads(p.read_text(encoding='utf-8'))
    if m.get('variant')!=variant or m.get('runtime_pin')!=PARAKEET_PIN or m.get('candle_pin')!=CANDLE_PIN:
        raise ValueError('Build manifest revision or variant mismatch')
    if m.get('binary_sha256')!=digest(b) or m.get('main_sha256')!=digest(runtime/'src/main.rs'):
        raise ValueError('Built binary/source mismatch; rebuild')
    if variant!='cpu' and m.get('target_sm')!='61':
        raise ValueError('GPU binary lacks sm_61 evidence')
    return b,m

def finite_positive(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0

def nearest_rank(values,p):
    if not values or not 0<p<=1:raise ValueError('Invalid percentile request')
    return sorted(values)[max(0,math.ceil(p*len(values))-1)]

def describe(values):
    if not values or any(not finite_positive(x) for x in values):
        raise ValueError('Non-positive or non-finite timing')
    median=statistics.median(values)
    return dict(count=len(values),median_ms=median,mean_ms=statistics.mean(values),
        min_ms=min(values),max_ms=max(values),p95_nearest_rank_ms=nearest_rank(values,.95),
        sample_stdev_ms=statistics.stdev(values) if len(values)>1 else None,
        coefficient_of_variation=statistics.stdev(values)/statistics.mean(values) if len(values)>1 else None)

def mib(v):
    try:
        n=float(v.strip().replace('MiB','').replace('Mi',''))
        return n if math.isfinite(n) and n>=0 else None
    except (ValueError,AttributeError):return None

class VramObserver:
    """Global GPU memory and optional per-PID memory, not a CUDA allocator trace."""
    def __init__(self, active=False, period=.25):
        self.active=active;self.period=period
        self.pid=None;self.event=threading.Event();self.thread=None
        self.global_samples=[];self.process_samples=[]
        self.initial_global=None
        self.errors=[]
    def gpu_memory(self):
        out=run_text(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i','0'],timeout=5)
        return mib(out.splitlines()[0]) if out else None
    def process_memory(self):
        if self.pid is None:return None
        out=run_text(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                      '--format=csv,noheader,nounits','-i','0'],timeout=5)
        if not out:return None
        matching=[]
        for line in out.splitlines():
            fields=[p.strip() for p in line.split(',')]
            if len(fields)<2:continue
            try:pid=int(fields[0])
            except ValueError:continue
            if pid==self.pid:
                mem=mib(fields[1])
                if mem is not None:matching.append(mem)
        return sum(matching) if matching else None
    def __enter__(self):
        if self.active:self.initial_global=self.gpu_memory()
        return self
    def begin(self,pid):
        self.pid=pid
        if not self.active:return
        def job():
            while not self.event.is_set():
                g=self.gpu_memory()
                if g is not None:self.global_samples.append(g)
                p=self.process_memory()
                if p is not None:self.process_samples.append(p)
                self.event.wait(self.period)
        self.thread=threading.Thread(target=job,daemon=True);self.thread.start()
    def __exit__(self,*_):
        self.event.set()
        if self.thread:self.thread.join(timeout=5)
    def result(self):
        global_peak=max(self.global_samples) if self.global_samples else None
        return dict(source='nvidia-smi (sampled, not allocator high-water mark)',
                    sampling_period_s=self.period,
                    gpu_wide_scope='entire GPU including unrelated processes',
                    global_baseline_mib=self.initial_global,global_peak_mib=global_peak,
                    global_peak_delta_mib=max(0,global_peak-self.initial_global)
                    if global_peak is not None and self.initial_global is not None else None,
                    global_samples=len(self.global_samples),
                    process_scope='only reported compute application PID, if supported',
                    process_peak_mib=max(self.process_samples) if self.process_samples else None,
                    process_samples=len(self.process_samples),
                    process_pid=self.pid if self.active else None)

def run_with_vram(cmd, variant, timeout):
    with VramObserver(active=variant!='cpu') as monitor:
        start=time.perf_counter()
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        monitor.begin(proc.pid)
        try:
            stdout,stderr=proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill();stdout,stderr=proc.communicate()
            raise TimeoutError(f'Runtime timeout after {timeout}s; PID {proc.pid}')
        wall_ms=(time.perf_counter()-start)*1000
    return proc.returncode,stdout,stderr,wall_ms,monitor.result()

def bench(args):
    runtime=require_patch(args.runtime_dir)
    folder=report_folder(args.report_root,'hdb2b2-'+args.variant)
    result=dict(schema_version=1,stage='HDB2B2_SM61_FINAL_SINGLE_AUDIO',
        status='NOT_QUALIFIED',model_id=MODEL_ID,variant=args.variant,
        input=str(args.input.resolve()),runtime_pin=PARAKEET_PIN,candle_pin=CANDLE_PIN,
        report_dir=str(folder.resolve()),warmup_requested=args.warmup,
        measured_requested=args.runs,profile_requested=args.profile_runs)
    begun=time.perf_counter()
    try:
        if not args.input.is_file():raise ValueError(f'Audio not found: {args.input}')
        if not (args.model_dir/'model.safetensors').is_file():raise ValueError('Converted model.safetensors missing')
        model=get_model_meta(args.model_dir)
        result['checkpoint_sha256']=model['source_nemo_sha256']
        b,build_meta=check_build(runtime,args.variant)
        result['build_manifest']=build_meta
        result['binary_sha256']=build_meta['binary_sha256']
        if args.variant!='cpu':result['gpu']=ensure_gpu()
        pcm=folder/'input-16k.wav'
        info=convert_audio(args.input,pcm,folder/'ffmpeg.log')
        result['normalized_audio']=info;result['normalized_pcm_sha256']=pcm_digest(pcm)
        cmd=[str(b),'--model-id',MODEL_ID,'--model-dir',str(args.model_dir.resolve()),
             '--input',str(pcm.resolve()),'--format','json',
             '--hdb2b2-hot-runs',str(args.runs),
             '--hdb2b2-warmup',str(args.warmup),
             '--hdb2b2-profile-runs',str(args.profile_runs)]
        if args.variant=='cpu':cmd.append('--cpu')
        print('HDB2B2=RUNNING_REAL_CANDLE_'+args.variant.upper(),flush=True)
        code,stdout,stderr,process_wall,vram=run_with_vram(cmd,args.variant,args.timeout)
        (folder/'runtime.stdout').write_text(stdout,encoding='utf-8')
        (folder/'runtime.stderr').write_text(stderr,encoding='utf-8')
        result.update(runtime_exit_code=code,process_wall_ms=process_wall,vram=vram)
        if code:raise RuntimeError(f'Candle runtime returned {code}; see runtime.stderr')
        payload=json.loads(stdout)
        device=str(payload.get('device',''))
        if payload.get('model_id')!=MODEL_ID:raise RuntimeError('Wrong model identity in output')
        if args.variant=='cpu' and 'cpu' not in device.lower():raise RuntimeError('CPU run selected wrong device')
        if args.variant!='cpu' and 'cuda' not in device.lower():raise RuntimeError('CUDA fell back to CPU')
        result['measured_device']=device
        text=payload.get('text')
        if not isinstance(text,str) or not text.strip():raise RuntimeError('Empty transcript')
        lines=[line[len('HDB2B2_BENCH_JSON='):] for line in stderr.splitlines()
               if line.startswith('HDB2B2_BENCH_JSON=')]
        if len(lines)!=1:raise RuntimeError('Expected exactly one real HDB2-B2 metrics line')
        metrics=json.loads(lines[0]);result['metrics']=metrics
        if metrics.get('schema_version')!=1 or metrics.get('dtype')!='f32' or metrics.get('text_consistent') is not True:
            raise RuntimeError('Invalid FP32 metrics metadata')
        if metrics.get('measured_device')!=device:raise RuntimeError('Runtime device mismatch')
        if metrics.get('hot_scope')!='wav_decode+logmel+fastconformer+tdt+alignment':
            raise RuntimeError('Unknown measurement scope')
        warm=metrics.get('warmup_ms');hot=metrics.get('hot_samples_ms');phases=metrics.get('phase_profile_samples')
        if not isinstance(warm,list) or len(warm)!=args.warmup or any(not finite_positive(t) for t in warm):
            raise RuntimeError('Warmup evidence missing or corrupt')
        if not isinstance(hot,list) or len(hot)!=args.runs:
            raise RuntimeError('Hot sample count mismatch')
        if not isinstance(phases,list) or len(phases)!=args.profile_runs:
            raise RuntimeError('Stage profile sample count mismatch')
        for phase in phases:
            if not isinstance(phase,dict) or any(not finite_positive(phase.get(k)) for k in PHASES):
                raise RuntimeError('Invalid phase timing evidence')
            # Profile includes synchronization overhead. Phase totals can vary slightly from whole wall.
            if sum(phase[k] for k in PHASES if k!='profile_wall_ms')>phase['profile_wall_ms']*1.05:
                raise RuntimeError('Segmented timing exceeds profile wall time unexpectedly')
        summary=describe(hot)
        profiles={phase:describe([row[phase] for row in phases]) for phase in PHASES}
        result.update(status=PASS,transcript=text.strip(),transcript_characters=len(text.strip()),
            hot_samples=summary,phase_profile=profiles,
            hot_ms_median=summary['median_ms'],hot_ms_p95=summary['p95_nearest_rank_ms'],
            hot_rtf_median=summary['median_ms']/(info['duration_sec']*1000),
            measurement_note='wall-time full calls; phase profile separately synchronized, not additive',
            percentile_note='nearest-rank p95; with 10 samples this is the sample maximum',
            accuracy_note='text consistency only; WER/CER not yet established')
        (folder/'transcription.txt').write_text(text.strip()+'\n',encoding='utf-8')
        print(f'HDB2B2={PASS} VARIANT={args.variant} DEVICE={device} '
              f'HOT_P50_MS={summary["median_ms"]:.3f} HOT_P95_MS={summary["p95_nearest_rank_ms"]:.3f} '
              f'RTF={result["hot_rtf_median"]:.5f}')
        print('TRANSCRIPTION_BEGIN\n'+text.strip()+'\nTRANSCRIPTION_END')
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        print('HDB2B2=FAIL_NOT_QUALIFIED '+result['error'],file=sys.stderr)
    finally:
        result['end_to_end_ms']=(time.perf_counter()-begun)*1000
        path=folder/'report.json'
        path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print('HDB2B2_REPORT='+str(path),flush=True)
    if result['status']!=PASS:raise SystemExit(1)

def validate_report(r,label):
    if r.get('stage')!='HDB2B2_SM61_FINAL_SINGLE_AUDIO' or r.get('status')!=PASS:
        raise ValueError(f'{label}: not a successful HDB2-B2 report')
    v=r.get('variant')
    if v not in VARIANTS:raise ValueError(f'{label}: unknown variant')
    device=str(r.get('measured_device','')).lower()
    if v=='cpu' and 'cpu' not in device:raise ValueError(f'{label}: wrong CPU device')
    if v!='cpu' and 'cuda' not in device:raise ValueError(f'{label}: wrong CUDA device')
    m=r.get('metrics')
    if not isinstance(m,dict) or m.get('text_consistent') is not True or m.get('dtype')!='f32':
        raise ValueError(f'{label}: no FP32 consistency evidence')
    if m.get('measured_device')!=r['measured_device']:
        raise ValueError(f'{label}: inconsistent device identity')
    if len(m.get('hot_samples_ms',[]))<10 or len(m.get('warmup_ms',[]))<3 or len(m.get('phase_profile_samples',[]))<3:
        raise ValueError(f'{label}: insufficient repetitions for B2 qualification')
    if not finite_positive(r.get('hot_ms_median')) or not finite_positive(r.get('hot_ms_p95')):
        raise ValueError(f'{label}: invalid medians')
    if not isinstance(r.get('transcript'),str) or not r['transcript'].strip():
        raise ValueError(f'{label}: empty transcript')
    if not isinstance(r.get('build_manifest'),dict) or r['build_manifest'].get('variant')!=v:
        raise ValueError(f'{label}: missing build evidence')
    if r.get('binary_sha256')!=r['build_manifest'].get('binary_sha256'):
        raise ValueError(f'{label}: binary identity mismatch')
    return v

def compare3(args):
    folder=report_folder(args.report_root,'hdb2b2-compare3')
    out=dict(schema_version=1,stage='HDB2B2_3WAY_COMPARISON',status='NOT_QUALIFIED',
        report_paths=[str(x.resolve()) for x in (args.cpu,args.cuda,args.cuda_cudnn)])
    try:
        docs=[json.loads(p.read_text(encoding='utf-8')) for p in (args.cpu,args.cuda,args.cuda_cudnn)]
        variants=[validate_report(doc,label) for doc,label in zip(docs,('CPU','CUDA','CUDA_CUDNN'))]
        if variants!=['cpu','cuda','cuda-cudnn']:
            raise ValueError(f'Expected cpu, cuda, cuda-cudnn in that order; got {variants}')
        for field in ('checkpoint_sha256','normalized_pcm_sha256','model_id','runtime_pin','candle_pin'):
            ref=docs[0].get(field)
            if not ref or any(doc.get(field)!=ref for doc in docs):
                raise ValueError(f'Different or missing evidence: {field}')
        ref_text=unicodedata.normalize('NFC',docs[0]['transcript'])
        if any(unicodedata.normalize('NFC',doc['transcript'])!=ref_text for doc in docs):
            out['status']='FAIL_TEXT_DIVERGENCE'
            raise ValueError('Real CUDA/CPU transcription mismatch')
        t=[doc['hot_ms_median'] for doc in docs]
        detail={v:dict(hot_p50_ms=d['hot_ms_median'],hot_p95_ms=d['hot_ms_p95'],
                 rtf=d['hot_rtf_median'],device=d['measured_device'],
                 profile_p50_ms={phase:d['phase_profile'][phase]['median_ms'] for phase in PHASES},
                 vram=d.get('vram')) for v,d in zip(variants,docs)}
        out.update(status='PASS_IDENTICAL_TEXT_ALL_VARIANTS',text_identical=True,
            model_id=docs[0]['model_id'],checkpoint_sha256=docs[0]['checkpoint_sha256'],
            normalized_pcm_sha256=docs[0]['normalized_pcm_sha256'],
            variants=detail,fastest_variant=variants[min(range(3),key=lambda i:t[i])],
            speedup_cpu_over_cuda=t[0]/t[1],speedup_cpu_over_cuda_cudnn=t[0]/t[2],
            speedup_cuda_over_cuda_cudnn=t[1]/t[2],
            cuda_cudnn_latency_reduction_vs_cuda_pct=100*(1-t[2]/t[1]),
            reference_text=docs[0]['transcript'],
            percentile_note='p95 nearest rank over 10 samples is maximum; no CI or significance claim',
            limitation='single audio, FP32, no WER; avoid claiming global optimality')
        print('HDB2B2=PASS_IDENTICAL_TEXT_ALL_VARIANTS '
              f'cpu/cuda={t[0]/t[1]:.4f} cpu/cuda_cudnn={t[0]/t[2]:.4f} '
              f'cuda/cuda_cudnn={t[1]/t[2]:.4f}')
    except (OSError,KeyError,IndexError,json.JSONDecodeError,TypeError,ValueError,ZeroDivisionError) as exc:
        out['error']=f'{type(exc).__name__}: {exc}'
        print('HDB2B2=FAIL_COMPARISON '+out['error'],file=sys.stderr)
    p=folder/'comparison.json'
    p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('HDB2B2_COMPARISON='+str(p),flush=True)
    if out['status']!='PASS_IDENTICAL_TEXT_ALL_VARIANTS':raise SystemExit(1)

def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('build','bench','compare3'):
        sp=sub.add_parser(name)
        sp.add_argument('--report-root',type=Path,default=Path.home()/'tmp/haricot-half-duplex/reports')
        if name in ('build','bench'):
            sp.add_argument('--variant',required=True,choices=sorted(VARIANTS))
            sp.add_argument('--runtime-dir',type=Path,required=True)
        if name=='build':sp.add_argument('--project-root',required=True,type=Path)
        if name=='bench':
            sp.add_argument('--input',required=True,type=Path)
            sp.add_argument('--model-dir',required=True,type=Path)
            sp.add_argument('--runs',type=int,default=10)
            sp.add_argument('--warmup',type=int,default=3)
            sp.add_argument('--profile-runs',type=int,default=3)
            sp.add_argument('--timeout',type=int,default=2400)
        if name=='compare3':
            for k in ('cpu','cuda','cuda-cudnn'):
                sp.add_argument('--'+k,required=True,type=Path,dest=k.replace('-','_'))
    a=parser.parse_args()
    if a.command=='bench' and (not 10<=a.runs<=30 or not 3<=a.warmup<=10 or not 1<=a.profile_runs<=10 or not 30<=a.timeout<=7200):
        parser.error('B2 requires measured runs 10..30, warmup 3..10, profile 1..10, timeout 30..7200')
    {'build':build,'bench':bench,'compare3':compare3}[a.command](a)

if __name__=='__main__':main()
