"""HDB2-C controlled same-WAV C1 and optional real tensor C2/C3 runner."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime,timezone

from .audio import normalize,describe,sha256,report
from .compare import compare_texts,compare_probes

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME=Path.home()/'tmp/haricot-half-duplex/integration/parakeet-candle/runtime'
DEFAULT_MODEL=Path.home()/'tmp/haricot-half-duplex/models/transcribepsy-eu-sub-candle'
DEFAULT_NEMO=Path.home()/'tmp/haricot-half-duplex/models/source/TranscribePsy-250M-EU-sub.nemo'
DEFAULT_REPORTS=Path.home()/'tmp/haricot-half-duplex/reports'
MODEL_ID='qvac/TranscribePsy-250M-EU-sub'


def resolved_nemo_python():
    # Same search order as run-hdb2c.sh; the source ZIP may be in another dir.
    if os.environ.get('HDB2C_NEMO_PYTHON'):
        return Path(os.environ['HDB2C_NEMO_PYTHON'])
    if os.environ.get('HDB2C_ENV_DIR'):
        return Path(os.environ['HDB2C_ENV_DIR'])/'bin/python'
    for candidate in (
        ROOT/'.venv-hdb2c-py313/bin/python',
        Path.home()/'tmp/haricot-half-duplex/.venv-hdb2c-py313/bin/python',
        ROOT.parent/'haricot-half-duplex-hdb2c-v0.6.1/.venv-hdb2c-py313/bin/python',
    ):
        if candidate.is_file():
            return candidate
    return ROOT/'.venv-hdb2c-py313/bin/python'


def settings():
    return dict(runtime=Path(os.getenv('HDB2A_RUNTIME_DIR',DEFAULT_RUNTIME)),
                model=Path(os.getenv('HDB2A_MODEL_DIR',DEFAULT_MODEL)),
                nemo=Path(os.getenv('HDB2C_NEMO',DEFAULT_NEMO)),
                reports=Path(os.getenv('HDB2A_REPORT_ROOT',DEFAULT_REPORTS)),
                python=resolved_nemo_python())


def take_run_dir(report_root):
    now=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    # Should never overwrite runs; create with exclusive mkdir
    for i in range(1000):
        p=Path(report_root)/f'hdb2c-{now}-{os.getpid()}-{i:03}'
        try:
            p.mkdir(parents=True,exist_ok=False);return p
        except FileExistsError:pass
    raise RuntimeError('Could not allocate HDB2C report dir')


def run_log(cmd,report_dir,name,env=None):
    p=subprocess.run([str(s) for s in cmd],capture_output=True,text=True,env=env)
    (report_dir/f'{name}.stdout').write_text(p.stdout,encoding='utf-8')
    (report_dir/f'{name}.stderr').write_text(p.stderr,encoding='utf-8')
    return p


def choose_binary(s,variant):
    if variant not in ('cpu','cuda','cuda-cudnn'):
        raise ValueError(f'Unknown variant {variant}')
    return s['runtime']/f'target/hdb2b-{variant}/release/parakeet-candle'


def parse_candle_json(stdout):
    # Main prints one JSON object; fail rather than heuristically parse unrelated logs.
    try:
        value=json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f'Candle stdout not JSON: {str(e)}; see candle.stdout') from e
    if not isinstance(value,dict):raise ValueError('Candle response must be a JSON object')
    return value


def candle_infer(s,variant,wav,output,pcm_sha):
    exe=choose_binary(s,variant)
    if not exe.is_file():raise FileNotFoundError(f'Missing existing Candle binary: {exe}')
    args=[exe,'--model-id',MODEL_ID,'--model-dir',s['model'],'--input',wav,'--format','json']
    if variant=='cpu':args.append('--cpu')
    p=run_log(args,output,'candle')
    if p.returncode:
        raise RuntimeError(f'Candle ASR exited {p.returncode}; check {output}/candle.stderr')
    obj=parse_candle_json(p.stdout)
    if obj.get('model_id')!=MODEL_ID:raise RuntimeError('Incorrect model id in Candle output')
    if not str(obj.get('text','')).strip():raise RuntimeError('Empty Candle transcript')
    dev=str(obj.get('device',''))
    if variant=='cpu' and not dev.startswith('Cpu'):
        raise RuntimeError(f'Expected CPU, got {dev}')
    if variant!='cpu' and not dev.startswith('Cuda('):
        raise RuntimeError(f'Expected real CUDA, got {dev}')
    conv=s['model']/'conversion.json'
    meta=json.loads(conv.read_text()) if conv.exists() else {}
    result={'stage':'HDB2C_CANDLE_ORIGINAL_PORT','status':'REAL_CANDLE_ASR','device':dev,
       'variant':variant,'model_id':obj.get('model_id'),'pcm_sha256':pcm_sha,
       'checkpoint_source_nemo_sha256':meta.get('source_nemo_sha256'),
       'text':obj['text'],'binary':str(exe)}
    report(output/'candle_report.json',result)
    return result


def do_c1(a):
    s=settings()
    r=take_run_dir(s['reports'])
    print('HDB2C_REPORT_DIR='+str(r),flush=True)
    try:
        wav=r/'input-16k.wav'
        desc=normalize(a.wav,wav)
        report(r/'audio.json',desc)
        c=candle_infer(s,a.variant,wav,r,desc['pcm_sha256'])
        if not s['python'].is_file():
            raise FileNotFoundError(f'NeMo isolated Python missing: {s["python"]}; run install-nemo')
        cmd=[s['python'],'-m','hdb2c.nemo_runner','--nemo',s['nemo'],'--wav',wav,
             '--output',r]
        if a.probes:cmd.append('--probe')
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1'
        env['PYTHONPATH']=str(ROOT)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        p=run_log(cmd,r,'nemo',env)
        if p.returncode:
            raise RuntimeError(f'NeMo ASR failed (code {p.returncode}); see {r}/nemo.stderr')
        compare=compare_texts(r/'nemo_report.json',r/'candle_report.json',r/'c1_comparison.json',a.reference_file)
        print('HDB2C_C1='+compare['status'],flush=True)
        print('HDB2C_NEMO_TEXT='+compare['nemo_text'],flush=True)
        print('HDB2C_CANDLE_TEXT='+compare['candle_text'],flush=True)
        if a.probes:print('HDB2C_C2C3=WAITING_FOR_RUST_PROBE',flush=True)
        if compare['status'] in ('CHECKPOINT_SOURCE_MISMATCH','CHECKPOINT_SOURCE_UNVERIFIED'):
            print('HDB2C_C1=SOURCE_CHECKPOINT_MISMATCH',file=sys.stderr)
            return 2
        return 0
    except Exception as e:
        report(r/'error.json',{'stage':'HDB2C_C1','status':'NOT_QUALIFIED','error':str(e),
                              'traceback':traceback.format_exc()[-7000:]})
        print('HDB2C_C1=NOT_QUALIFIED '+str(e),file=sys.stderr)
        return 1



def do_probe(a):
    s=settings();base=Path(a.report_dir).resolve()
    if not base.is_dir() or not (base/'nemo_report.json').is_file():
        raise FileNotFoundError('Requires a C1 report directory containing nemo_report.json')
    nr=json.loads((base/'nemo_report.json').read_text())
    if nr.get('probe_status')!='REAL_NEMO_MEL_AND_ENCODER':
        raise ValueError('C1 was not invoked with --probes, so there are no NeMo numeric tensors')
    from .probe import install
    install(ROOT/'rust/hdb2c_probe.rs',s['runtime'])
    if a.variant not in ('cpu','cuda-cudnn'):
        raise ValueError('Probe supports cpu or cuda-cudnn')
    target=s['runtime']/f'target/hdb2c-probe-{a.variant}'
    exe=target/'release/hdb2c_probe'
    if not a.skip_build:
        # Cargo locked package updates performed by included existing script
        cmd=[sys.executable,str(ROOT/'scripts/hdb2a_cargo_lock.py'),
             '--manifest-path',str(s['runtime']/'Cargo.toml')]
        subprocess.run(cmd,check=True)
        env=os.environ.copy();env['CARGO_TARGET_DIR']=str(target)
        env['CUDA_COMPUTE_CAP']='61'
        cargo=['cargo','build','--release','--manifest-path',str(s['runtime']/'Cargo.toml'),
               '--bin','hdb2c_probe']
        if a.variant=='cuda-cudnn':cargo+=['--features','cuda cudnn']
        subprocess.run(cargo,check=True,env=env)
    if not exe.is_file():raise FileNotFoundError(f'Probe binary not built: {exe}')
    output=base/f'candle_probe_{a.variant}'
    output.mkdir(exist_ok=True)
    cmd=[exe,'--model-dir',s['model'],'--input',base/'input-16k.wav','--output',output]
    if a.variant=='cpu':cmd.append('--cpu')
    p=run_log(cmd,output,'probe')
    if p.returncode:raise RuntimeError(f'Rust probe failed; see {output}/probe.stderr')
    rr=compare_probes(base,output,base/f'c2c3_{a.variant}.json')
    print('HDB2C_C2C3='+rr['status'])
    print('HDB2C_NUMERIC_REPORT='+str(base/f'c2c3_{a.variant}.json'))
    return 0


def main():
    p=argparse.ArgumentParser(description='HDB2-C real NeMo/Candle CPU numerical reference')
    sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('c1');s.add_argument('wav',type=Path)
    s.add_argument('--variant',choices=['cpu','cuda','cuda-cudnn'],default='cuda-cudnn')
    s.add_argument('--reference-file',type=Path);s.add_argument('--probes',action='store_true')
    s=sub.add_parser('probe');s.add_argument('report_dir',type=Path)
    s.add_argument('--variant',choices=['cpu','cuda-cudnn'],default='cpu')
    s.add_argument('--skip-build',action='store_true')
    s=sub.add_parser('preflight');s.add_argument('--wav',type=Path)
    a=p.parse_args()
    if a.cmd=='c1':sys.exit(do_c1(a))
    if a.cmd=='probe':sys.exit(do_probe(a))
    st=settings()
    for k,val in st.items():print(f'HDB2C_{k.upper()}={val} EXISTS={val.exists()}')
    for variant in ('cpu','cuda-cudnn'):
        binary=choose_binary(st,variant)
        print(f'HDB2C_CANDLE_{variant.upper().replace("-","_")}={binary} EXISTS={binary.is_file()}')
    if a.wav:print('HDB2C_AUDIO='+json.dumps(describe(a.wav),indent=2))
    if (st['model']/'config.json').is_file():
        cfg=json.loads((st['model']/'config.json').read_text())
        pre=cfg.get('preprocessor',{})
        from .static_audit import audit
        print('HDB2C_STATIC_RISKS='+json.dumps(audit(st['model']),ensure_ascii=False))
        print('HDB2C_PREPROCESSOR_CONFIG='+json.dumps({k:pre.get(k) for k in (
            'window','mag_power','log_zero_guard_type','log_zero_guard_value','mel_norm',
            'pad_to','normalize','n_fft','sample_rate','preemph','exact_pad')},ensure_ascii=False))

if __name__=='__main__':main()
