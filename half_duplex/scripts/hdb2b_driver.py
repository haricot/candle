#!/usr/bin/env python3
"""HDB2-B: build isolated CPU/CUDA sm61 variants, measure hot ASR, compare.

No fake text, no mock GPU, no substitutions: failure is reported as a failure.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
import wave

from run_candle_hdb2a import MODEL_ID, PARAKEET_PIN, CANDLE_PIN, convert_audio, git_head
from hdb2a_model_config import validate_config

VARIANTS = {'cpu': (), 'cuda': ('cuda',), 'cuda-cudnn': ('cuda', 'cudnn')}
TARGET_CAP='61'

def run_text(cmd, timeout=12):
    try:
        p=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout,check=True)
        return p.stdout.strip()
    except (OSError, subprocess.SubprocessError): return None

def query_gpu():
    output=run_text(['nvidia-smi','--query-gpu=name,compute_cap,memory.total,driver_version','--format=csv,noheader,nounits','-i','0'])
    if not output: return None
    fields=[v.strip() for v in output.splitlines()[0].split(',')]
    if len(fields)!=4:return None
    return dict(name=fields[0],compute_cap=fields[1],total_mib=fields[2],driver=fields[3])

def ensure_gpu():
    gpu=query_gpu()
    if gpu is None: raise RuntimeError('No NVIDIA GPU found via nvidia-smi -i 0')
    if gpu['compute_cap'] not in ('6.1','61'):
        raise RuntimeError(f'HDB2B target is Pascal sm_61; detected compute_cap={gpu["compute_cap"]}; do not silently benchmark a different GPU')
    return gpu

def require_runtime(runtime):
    runtime=runtime.resolve()
    src=runtime/'src/main.rs'
    cargo=runtime/'Cargo.toml'
    if not src.exists() or not cargo.exists():
        raise ValueError(f'Missing local Candle runtime: {runtime}; run HDB2A bootstrap/repair-runtime first')
    if git_head(runtime)!=PARAKEET_PIN:
        raise ValueError('Runtime Git commit differs from HDB2A pinned revision')
    if 'HDB2B_IN_PROCESS_BENCH_V1' not in src.read_text(encoding='utf-8'):
        raise ValueError('HDB2B patch missing; execute repair-runtime first')
    manifest=cargo.read_text(encoding='utf-8')
    if CANDLE_PIN not in manifest and 'path = ' not in manifest:
        raise ValueError('Candle dependency is neither pinned nor explicitly local')
    return runtime

def target_root(runtime,variant): return runtime/'target'/f'hdb2b-{variant}'
def binary(runtime,variant):return target_root(runtime,variant)/'release'/'parakeet-candle'
def variant_sha(runtime,variant):return target_root(runtime,variant)/'hdb2b-build.json'

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def build(args):
    runtime=require_runtime(args.runtime_dir)
    variant=args.variant
    gpu=ensure_gpu() if variant!='cpu' else None
    nvcc=run_text(['nvcc','--version']) if gpu else None
    if gpu and not nvcc:raise RuntimeError('CUDA build requested but nvcc is missing')
    gpu_codes=run_text(['nvcc','--list-gpu-code']) if gpu else None
    if gpu_codes and not re.search(r'(?:sm_|compute_)61\b',gpu_codes):
        raise RuntimeError('nvcc does not advertise sm_61; CUDA toolkit does not support this target')
    env=os.environ.copy()
    if gpu: env['CUDA_COMPUTE_CAP']=TARGET_CAP
    env['CARGO_TARGET_DIR']=str(target_root(runtime,variant))
    subprocess.run(['python3',str(args.project_root/'scripts/hdb2a_cargo_lock.py'),
                   '--manifest-path',str(runtime/'Cargo.toml')],check=True)
    cmd=['cargo','build','--release','--manifest-path',str(runtime/'Cargo.toml')]
    if VARIANTS[variant]:cmd+=['--features',','.join(VARIANTS[variant])]
    print('HDB2B_BUILD='+variant,flush=True)
    subprocess.run(cmd,cwd=runtime,env=env,check=True)
    b=binary(runtime,variant)
    if not b.is_file():raise RuntimeError(f'Build succeeded but binary missing: {b}')
    metadata=dict(schema_version=1,variant=variant,features=list(VARIANTS[variant]),
      architecture=TARGET_CAP if gpu else None, gpu=gpu, cuda_toolchain=nvcc,
      rust_toolchain=run_text(['rustc','--version']),cargo_features=list(VARIANTS[variant]),
      binary_sha256=digest(b),runtime_commit=PARAKEET_PIN,candle_pin=CANDLE_PIN,
      source_main_sha256=digest(runtime/'src/main.rs'),built_at=datetime.now(timezone.utc).isoformat())
    m=variant_sha(runtime,variant)
    m.write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('HDB2B=BUILT_NOT_RUNTIME_QUALIFIED')
    print('HDB2B_BINARY='+str(b))
    print('HDB2B_BUILD_MANIFEST='+str(m))

def sample_used_mib():
    o=run_text(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i','0'],timeout=5)
    if not o:return None
    try:return float(o.splitlines()[0].strip())
    except ValueError:return None

class GlobalVramSampler:
    def __init__(self,active):
        self.active=active
        self.baseline=sample_used_mib() if active else None
        self.samples=[]
        self.stop=threading.Event()
        self.thread=None
    def __enter__(self):
        if self.active:
            def monitor():
                while not self.stop.is_set():
                    v=sample_used_mib()
                    if v is not None:self.samples.append(v)
                    self.stop.wait(0.6)
            self.thread=threading.Thread(target=monitor,daemon=True)
            self.thread.start()
        return self
    def __exit__(self,*_):
        self.stop.set()
        if self.thread:self.thread.join(timeout=2)
    def result(self):
        peak=max(self.samples) if self.samples else None
        return dict(scope='GPU-wide nvidia-smi memory.used (not process-attributed)',
                    baseline_mib=self.baseline,peak_mib=peak,
                    peak_delta_mib=max(0,peak-self.baseline) if peak is not None and self.baseline is not None else None,
                    observations=len(self.samples))

def nearest_p(samples,p):
    a=sorted(samples)
    return a[max(0, math.ceil(len(a)*p)-1)]

def check_build(runtime,variant):
    meta_path=variant_sha(runtime,variant)
    bin_path=binary(runtime,variant)
    if not meta_path.is_file() or not bin_path.is_file():
        raise ValueError(f'Missing {variant} build. Execute: ./run-hdb2b.sh build {variant}')
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    if meta['variant']!=variant or meta['binary_sha256']!=digest(bin_path):
        raise ValueError('Built binary changed or variant manifest is invalid; rebuild')
    if meta['source_main_sha256']!=digest(runtime/'src/main.rs'):
        raise ValueError('Source changed since build; rebuild before benchmarking')
    if meta['runtime_commit']!=PARAKEET_PIN or meta['candle_pin']!=CANDLE_PIN:
        raise ValueError('Pinned dependency changed; refusing mixed evidence')
    return bin_path,meta

def report_folder(root,stage):
    folder=root/f'{stage}-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}-{os.getpid()}-{uuid.uuid4().hex[:6]}'
    folder.mkdir(parents=True,exist_ok=False)
    return folder

def get_model_meta(folder):
    validate_config(json.loads((folder/'config.json').read_text(encoding='utf-8')))
    src=json.loads((folder/'conversion.json').read_text(encoding='utf-8'))
    if src.get('model')!=MODEL_ID:raise ValueError('Not converted QVAC EU-sub')
    return src

def pcm_digest(path):
    h=hashlib.sha256()
    with wave.open(str(path),'rb') as f:
        while True:
            data=f.readframes(65536)
            if not data:break
            h.update(data)
    return h.hexdigest()

def benchmark(args):
    runtime=require_runtime(args.runtime_dir)
    report_dir=report_folder(args.report_root,'hdb2b-'+args.variant)
    report=dict(schema_version=1,stage='HDB2B_SM61_LATENCY',status='NOT_QUALIFIED',
        variant=args.variant,input=str(args.input.resolve()),model_id=MODEL_ID,
        runtime_pin=PARAKEET_PIN,candle_pin=CANDLE_PIN,report_dir=str(report_dir))
    t0=time.perf_counter()
    try:
        if not args.input.is_file():raise ValueError(f'Input missing: {args.input}')
        if not (args.model_dir/'model.safetensors').is_file():raise ValueError('Missing model.safetensors')
        meta=get_model_meta(args.model_dir)
        report['checkpoint_sha256']=meta['source_nemo_sha256']
        b,build_meta=check_build(runtime,args.variant)
        report['binary_sha256']=build_meta['binary_sha256']
        report['build_manifest']=build_meta
        if args.variant!='cpu':report['gpu']=ensure_gpu()
        pcm=report_dir/'input-16k.wav'
        info=convert_audio(args.input,pcm,report_dir/'ffmpeg.log')
        report['normalized_audio']=info
        report['normalized_pcm_sha256']=pcm_digest(pcm)
        cmd=[str(b),'--model-id',MODEL_ID,'--model-dir',str(args.model_dir.resolve()),
             '--input',str(pcm.resolve()),'--format','json',
             '--hdb2b-bench-runs',str(args.runs),'--hdb2b-bench-warmup',str(args.warmup)]
        if args.variant=='cpu':cmd.append('--cpu')
        print(f'HDB2B=RUNNING_REAL_CANDLE_{args.variant.upper()}',flush=True)
        with GlobalVramSampler(args.variant!='cpu') as mem:
            process_start=time.perf_counter()
            try:
                proc=subprocess.run(cmd,capture_output=True,text=True,timeout=args.timeout)
            finally:
                report['process_wall_ms']=(time.perf_counter()-process_start)*1000
        report['vram']=mem.result()
        (report_dir/'runtime.stdout').write_text(proc.stdout,encoding='utf-8')
        (report_dir/'runtime.stderr').write_text(proc.stderr,encoding='utf-8')
        report['runtime_exit_code']=proc.returncode
        if proc.returncode:
            raise RuntimeError(f'Runtime exited {proc.returncode}; examine runtime.stderr')
        payload=json.loads(proc.stdout)
        if payload.get('model_id')!=MODEL_ID:raise RuntimeError('Runtime returned the wrong model identity')
        device=str(payload.get('device',''))
        report['measured_device']=device
        if args.variant=='cpu' and 'cpu' not in device.lower():raise RuntimeError('CPU variant did not run on CPU')
        if args.variant!='cpu' and 'cuda' not in device.lower():raise RuntimeError('GPU fell back to CPU; refusing CUDA qualification')
        text=payload.get('text')
        if not isinstance(text,str) or not text.strip():raise RuntimeError('No real nonempty transcript')
        metric_lines=[line[len('HDB2B_BENCH_JSON='):] for line in proc.stderr.splitlines()
                      if line.startswith('HDB2B_BENCH_JSON=')]
        if len(metric_lines)!=1:raise RuntimeError('Missing or duplicate in-process benchmark metrics')
        metrics=json.loads(metric_lines[0])
        samples=metrics.get('hot_samples_ms')
        if not isinstance(samples,list) or len(samples)!=args.runs or any(not isinstance(t,(int,float)) or not math.isfinite(t) or t<=0 for t in samples):
            raise RuntimeError('Invalid hot timing evidence')
        if len(metrics.get('warmup_ms',[]))!=args.warmup or metrics.get('dtype')!='f32' or metrics.get('text_consistent') is not True:
            raise RuntimeError('Warmup/dtype/transcript-consistency mismatch')
        if metrics.get('measured_device')!=device:
            raise RuntimeError('Device metadata mismatch')
        med=statistics.median(samples)
        report.update(status='REAL_CANDLE_HOT_BENCH_OBSERVED_UNQUALIFIED_WER',
            transcript=text.strip(),transcript_characters=len(text.strip()),
            metrics=metrics,hot_ms_median=med,hot_ms_min=min(samples),hot_ms_p95=nearest_p(samples,0.95),
            hot_rtf_median=med/(1000*info['duration_sec']))
        (report_dir/'transcription.txt').write_text(text.strip()+'\n',encoding='utf-8')
        print(f'HDB2B={report["status"]}')
        print(f'HDB2B_VARIANT={args.variant} DEVICE={device} HOT_MEDIAN_MS={med:.3f} HOT_RTF={report["hot_rtf_median"]:.5f}')
        print('TRANSCRIPTION_BEGIN')
        print(text.strip())
        print('TRANSCRIPTION_END')
    except Exception as e:
        report['error']=f'{type(e).__name__}: {e}'
        print('HDB2B=FAIL_NOT_QUALIFIED '+report['error'],file=sys.stderr)
    finally:
        report['end_to_end_ms']=round((time.perf_counter()-t0)*1000,3)
        (report_dir/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(f'HDB2B_REPORT={report_dir/"report.json"}',flush=True)
    if report['status']!='REAL_CANDLE_HOT_BENCH_OBSERVED_UNQUALIFIED_WER':sys.exit(1)

def validate_comparison_report(report, label):
    """Validate an observed runtime report, not an inferred label or mock result."""
    if report.get('status') != 'REAL_CANDLE_HOT_BENCH_OBSERVED_UNQUALIFIED_WER':
        raise ValueError(f'{label}: not a successful real hot ASR measurement')
    variant=report.get('variant')
    if variant not in VARIANTS:
        raise ValueError(f'{label}: unknown variant {variant!r}')
    device=str(report.get('measured_device',''))
    if variant == 'cpu' and 'cpu' not in device.lower():
        raise ValueError(f'{label}: CPU variant not measured on CPU: {device!r}')
    if variant != 'cpu' and 'cuda' not in device.lower():
        raise ValueError(f'{label}: GPU variant not measured on CUDA: {device!r}')
    metrics=report.get('metrics')
    if not isinstance(metrics,dict) or metrics.get('dtype')!='f32':
        raise ValueError(f'{label}: expected FP32 runtime metrics')
    if metrics.get('text_consistent') is not True:
        raise ValueError(f'{label}: missing in-process transcript consistency proof')
    if metrics.get('measured_device')!=device:
        raise ValueError(f'{label}: runtime device evidence mismatch')
    if not isinstance(report.get('transcript'),str) or not report['transcript'].strip():
        raise ValueError(f'{label}: nonempty transcript required')
    for field in ('hot_ms_median','hot_rtf_median'):
        number=report.get(field)
        if not isinstance(number,(int,float)) or not math.isfinite(number) or number<=0:
            raise ValueError(f'{label}: invalid positive metric: {field}')
    manifest=report.get('build_manifest')
    if manifest is not None:
        if not isinstance(manifest,dict) or manifest.get('variant')!=variant:
            raise ValueError(f'{label}: build manifest/variant mismatch')
        if manifest.get('binary_sha256')!=report.get('binary_sha256'):
            raise ValueError(f'{label}: built binary identity mismatch')
    return variant


def compare(args):
    """Compare any two distinct CPU/CUDA/cuDNN variants, with strict parity."""
    first_path=args.first
    second_path=args.second
    folder=report_folder(args.report_root,'hdb2b-compare')
    out=dict(schema_version=2,stage='HDB2B_VARIANT_COMPARISON',
             first_report=str(first_path.resolve()),second_report=str(second_path.resolve()),
             status='NOT_QUALIFIED',direction='speedup_first_over_second > 1 means second is faster')
    try:
        if first_path.resolve()==second_path.resolve():
            raise ValueError('Cannot compare a report to itself')
        first=json.loads(first_path.read_text(encoding='utf-8'))
        second=json.loads(second_path.read_text(encoding='utf-8'))
        a_variant=validate_comparison_report(first,'first')
        b_variant=validate_comparison_report(second,'second')
        if a_variant==b_variant:
            raise ValueError(f'Expected different variants, both reports are {a_variant!r}')
        for field in ('checkpoint_sha256','normalized_pcm_sha256','model_id','runtime_pin','candle_pin'):
            if not first.get(field) or first.get(field)!=second.get(field):
                raise ValueError(f'Nonmatching or missing evidence field: {field}')
        t_a=unicodedata.normalize('NFC',first['transcript']).strip()
        t_b=unicodedata.normalize('NFC',second['transcript']).strip()
        same=t_a==t_b
        ratio=first['hot_ms_median']/second['hot_ms_median']
        out.update(first_variant=a_variant,second_variant=b_variant,
                   first_device=first['measured_device'],second_device=second['measured_device'],
                   model_id=first['model_id'],checkpoint_sha256=first['checkpoint_sha256'],
                   normalized_pcm_sha256=first['normalized_pcm_sha256'],
                   first_hot_median_ms=first['hot_ms_median'],
                   second_hot_median_ms=second['hot_ms_median'],
                   first_hot_p95_ms=first.get('hot_ms_p95'),
                   second_hot_p95_ms=second.get('hot_ms_p95'),
                   first_hot_rtf=first['hot_rtf_median'],
                   second_hot_rtf=second['hot_rtf_median'],
                   speedup_first_over_second=ratio,
                   second_latency_reduction_pct=100*(1-second['hot_ms_median']/first['hot_ms_median']),
                   difference_first_minus_second_ms=first['hot_ms_median']-second['hot_ms_median'],
                   text_identical=same,normalized_text_nfc=True,
                   status='PASS_IDENTICAL_TEXT' if same else 'FAIL_TEXT_DIVERGENCE')
        # Legacy fields for existing downstream CPU -> GPU consumers.
        if a_variant=='cpu' and b_variant in ('cuda','cuda-cudnn'):
            out.update(cpu_report=out['first_report'],gpu_report=out['second_report'],
                       cpu_hot_median_ms=out['first_hot_median_ms'],
                       gpu_hot_median_ms=out['second_hot_median_ms'],
                       speedup_cpu_over_gpu=ratio,
                       rtf_cpu=out['first_hot_rtf'],rtf_gpu=out['second_hot_rtf'],
                       gpu_variant=b_variant)
        if same:
            print(f'HDB2B=PASS_IDENTICAL_TEXT first={a_variant} second={b_variant} speedup_first_over_second={ratio:.4f}')
        else:
            print(f'HDB2B=FAIL_TEXT_DIVERGENCE first={a_variant} second={b_variant}',file=sys.stderr)
    except (OSError,json.JSONDecodeError,KeyError,TypeError,ValueError,ZeroDivisionError) as exc:
        out['error']=f'{type(exc).__name__}: {exc}'
        print('HDB2B=FAIL_COMPARISON '+out['error'],file=sys.stderr)
    result_path=folder/'comparison.json'
    result_path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('HDB2B_COMPARISON='+str(result_path))
    if out['status']!='PASS_IDENTICAL_TEXT':sys.exit(1)

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    for cmd in ('build','benchmark','compare'):
        sp=sub.add_parser(cmd)
        sp.add_argument('--report-root',type=Path,default=Path.home()/'tmp/haricot-half-duplex/reports')
        if cmd in ('build','benchmark'):
            sp.add_argument('--variant',required=True,choices=sorted(VARIANTS))
            sp.add_argument('--runtime-dir',required=True,type=Path)
        if cmd=='build':sp.add_argument('--project-root',required=True,type=Path)
        if cmd=='benchmark':
            sp.add_argument('--input',type=Path,required=True)
            sp.add_argument('--model-dir',type=Path,required=True)
            sp.add_argument('--runs',type=int,default=3)
            sp.add_argument('--warmup',type=int,default=1)
            sp.add_argument('--timeout',type=int,default=900)
        if cmd=='compare':
            sp.add_argument('--first','--cpu',dest='first',required=True,type=Path)
            sp.add_argument('--second','--gpu',dest='second',required=True,type=Path)
    a=p.parse_args()
    if a.command=='benchmark' and (not 1<=a.runs<=30 or not 0<=a.warmup<=10 or not 30<=a.timeout<=3600):
        p.error('runs must be 1..30, warmup 0..10, timeout 30..3600')
    {'build':build,'benchmark':benchmark,'compare':compare}[a.command](a)

if __name__=='__main__':main()
