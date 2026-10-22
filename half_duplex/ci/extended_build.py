#!/usr/bin/env python3
"""Compile the native STT consumer against this exact Extended source tree.

Build-only configuration; no model weights, reference inference or GPU execution.
The generated R9 CUDA source is the source used by the measured GPU campaign.
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
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hdb2c.c2 import prepare
from hdb2c.c3 import build_candidate as r3
from hdb2c.c3_r3c import build_masked_candidate as r3c
from hdb2c.c3_r4 import build_r4_candidate as r4
from hdb2c.c3_r5 import build_r5_candidate as r5
from hdb2c.c3_r6 import build_candidate as r6
from hdb2c.c3_r7 import build_candidate as r7
from hdb2c.c3_r7_focused import verify_native_sources
from hdb2c.c3_r8_source import outputs as r8_outputs
from hdb2c.c3_r9_source import outputs as r9_outputs


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def prepare_build(core, work):
    core = core.resolve()
    if not (core/'candle-core/Cargo.toml').is_file():
        raise ValueError('Exact Candle source tree required')
    runtime = work/'runtime'
    run(sys.executable, ROOT/'scripts/prepare_candle_runtime.py',
        '--runtime-dir', runtime, '--candle-root', core)
    cargo = tomllib.loads((runtime/'Cargo.toml').read_text())
    for name, package in (('candle','candle-core'),('candle-nn','candle-nn')):
        dep = cargo['dependencies'][name]
        if Path(dep.get('path','')).resolve() != core/package or 'git' in dep:
            raise ValueError('STT consumer did not bind to the exact candidate: '+name)
    config = work/'buildonly-model'; config.mkdir()
    (config/'config.json').write_text(json.dumps({'preprocessor':
        {'normalize':'NA','n_fft':512,'features':128}}))
    c2 = work/'c2'; prepare(runtime,c2,ROOT,config)
    parent = c2
    for index, builder in enumerate((r3,r3c,r4,r5,r6,r7)):
        target = work/f'stage-{index}'
        builder(parent,target); parent = target
    evidence = json.loads((ROOT/'ci/r7_verified_native_sources.json').read_text())
    verify_native_sources(parent,evidence)
    target = work/'r8'
    shutil.copytree(parent,target,ignore=shutil.ignore_patterns('.git','target','__pycache__'))
    for name,content in r8_outputs(parent,ROOT).items():
        path=target/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(content)
    metadata=json.loads((ROOT/'ci/results/r13-ci-full-cpu-20261021.json').read_text())[
        'campaign_summary']['source_manifest']['tokenizer_metadata']
    final=work/'r9'
    shutil.copytree(target,final,ignore=shutil.ignore_patterns('.git','target','__pycache__'))
    for name,content in r9_outputs(target,ROOT,metadata).items():
        path=final/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(content)
    return final


def main(args):
    work=args.output.resolve(); work.mkdir(parents=True,exist_ok=False)
    core=args.candle_root.resolve()
    source=prepare_build(core,work)
    env={**os.environ,'CUDA_COMPUTE_CAP':'61','CARGO_TARGET_DIR':str(work/'cargo-target')}
    run(sys.executable,ROOT/'scripts/hdb2a_cargo_lock.py','--manifest-path',source/'Cargo.toml')
    command=['cargo','check','--release','--manifest-path',source/'Cargo.toml',
             '--bin','hdb2c_r9_probe']
    if args.backend == 'cuda-cudnn':
        command+=['--features','cuda cudnn']
    run(*command,env=env,cwd=ROOT)
    if args.backend == 'cpu':
        run('cargo','test','--release','--manifest-path',source/'Cargo.toml',
            '--lib','hdb2c',env=env,cwd=ROOT)
    data={'scope':'EXACT_EXTENDED_NATIVE_STT_CONSUMER_COMPILE_ONLY',
        'backend':args.backend,'gpu_executed':False,'asr_inference_executed':False,
        'candidate_sha':subprocess.check_output(['git','-C',str(core),'rev-parse','HEAD'],text=True).strip(),
        'candle_source':'LOCAL_EXACT_CANDIDATE_PATH','target':'sm_61' if args.backend == 'cuda-cudnn' else 'cpu',
        'status':'COMPILE_PASSED','cargo_lock_sha256':hashlib.sha256((source/'Cargo.lock').read_bytes()).hexdigest(),
        'generated_source_sha256':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((source/'src').rglob('*.rs'))}}
    (work/'report.json').write_text(json.dumps(data,indent=2)+'\n')
    print('HALF_DUPLEX_EXTENDED_BUILD=COMPILE_PASSED_NO_GPU_EXECUTION',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('cpu','cuda-cudnn'),required=True)
    p.add_argument('--candle-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
