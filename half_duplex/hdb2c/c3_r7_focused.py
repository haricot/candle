"""R7B config-only requalification, retaining real R6 and R7 executions.

Compile two isolated native variants instead of rebuilding historical ablations.
Every native source byte must match the successful full R7 run before this
profile is allowed. The normal full C2-R7 workflow remains available.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .cli import ROOT, settings
from .c2 import prepare
from .c3 import build_candidate as build_r3
from .c3_r3 import read_tensor
from .c3_r3c import build_masked_candidate
from .c3_r4 import build_r4_candidate
from .c3_r5 import build_r5_candidate, digest
from .c3_r6 import build_candidate as build_r6
from .c3_r7 import build_candidate as build_r7, execute
from .c3_r7_metrics import tensor_metrics


def verify_native_sources(candidate, evidence):
    for name, sha in evidence['native_source_sha256'].items():
        if digest(candidate/name) != sha:
            raise ValueError('R7 focused profile requires unchanged qualified native sources: '+name)


def main(corpus):
    s=settings()
    evidence=json.loads((ROOT/'ci/r7_verified_native_sources.json').read_text())
    if digest(s['nemo']) != evidence['checkpoint_sha256']:
        raise ValueError('R7 focused profile requires the exact fully qualified checkpoint')
    runtime=s['runtime']
    def path(suffix):
        return runtime.with_name(runtime.name+suffix)
    c2=path('-hdb2c-c2-v0.6.6'); r3=path('-hdb2c-c3-r3-v0.7.2')
    r3c=path('-hdb2c-c3-r3c-v0.7.3'); r4=path('-hdb2c-c3-r4-v0.7.4')
    r5=path('-hdb2c-c3-r5-v0.7.5'); r6=path('-hdb2c-c3-r6-v0.7.6')
    r7=path('-hdb2c-c3-r7-v0.7.7')
    prepare(runtime,c2,ROOT,s['model'])
    # Resolve the same isolated Cargo lock as the full workflow, without
    # building its historical native variants.
    subprocess.run([str(s['python']),str(ROOT/'scripts/hdb2a_cargo_lock.py'),
                    '--manifest-path',str(c2/'Cargo.toml')],check=True)
    build_r3(c2,r3); build_masked_candidate(r3,r3c); build_r4_candidate(r3c,r4)
    build_r5_candidate(r4,r5); build_r6(r5,r6); build_r7(r6,r7)
    verify_native_sources(r7,evidence)
    target=r6/'target/r6-cpu'
    env={**os.environ,'CARGO_TARGET_DIR':str(target)}
    subprocess.run(['cargo','build','--release','--manifest-path',str(r6/'Cargo.toml'),
                    '--bin','hdb2c_r6_probe'],check=True,env=env)
    subprocess.run(['cargo','test','--release','--manifest-path',str(r6/'Cargo.toml'),
                    '--lib','hdb2c6_tests'],check=True,env=env)
    rows=[row for row in json.loads(corpus.read_text())['rows']
          if row['split']=='cafe' and row['emotion'] in ('neutral','sadness')]
    if len(rows)!=2:
        raise ValueError('R7 focused profile requires the same two CaFE excerpts')
    published=Path(os.environ['GITHUB_WORKSPACE'])/'halfduplex-ci-published'
    published.mkdir(exist_ok=True)
    results=[]
    for i,row in enumerate(rows):
        report=s['reports']/('r7b-'+str(i)); report.mkdir(parents=True)
        shutil.copyfile(row['wav'],report/'input-16k.wav')
        subprocess.run([str(s['python']),'-m','hdb2c.nemo_runner','--nemo',str(s['nemo']),
                        '--wav',str(report/'input-16k.wav'),'--output',str(report),'--probe'],check=True)
        trace=report/'c3_cpu/nemo_trace'; trace.mkdir(parents=True)
        subprocess.run([str(s['python']),'-m','hdb2c.c3_nemo','--nemo',str(s['nemo']),
                        '--report',str(report),'--output',str(trace)],check=True)
        original=json.loads((report/'nemo_report.json').read_text())
        with np.load(report/'nemo_mel.npz') as data:
            values=data['values']; total=int(values.shape[2]); valid=int(data['valid_frames'][0])
            shared=values.transpose(0,2,1).astype('<f4').tobytes()
        mel=report/'c3_cpu/nemo_mel_injected_BTD.f32le'; mel.write_bytes(shared)
        control=report/'c3_cpu/candle_r6_both_trace'
        probe=report/'c3_cpu/r6_focused_probe'
        args=[str(target/'release/hdb2c_r6_probe'),'--model-dir',str(s['model']),
              '--input',str(report/'input-16k.wav'),'--output',str(probe),'--cpu']
        subprocess.run(args,check=True,env={**os.environ,'HDB2C6_MASK_MODE':'both',
            'HDB2C3_TRACE_DIR':str(control),'HDB2C3_NEMO_MEL_PATH':str(mel),
            'HDB2C3_NEMO_MEL_FRAMES':str(total),'HDB2C3_NEMO_MEL_VALID_FRAMES':str(valid)})
        meta=json.loads((probe/'candle_probe.json').read_text())
        with np.load(report/'nemo_encoder.npz') as data:
            encoded_valid=int(data['valid_frames'][0])
        if ((probe/'candle_mel.f32le').read_bytes()!=shared or meta['injected_mel_valid_frames']!=valid
                or meta['encoder_valid_frames']!=encoded_valid):
            raise ValueError('R7B real R6 input/length control differs')
        for name in ('encoder_pad_mask','encoder_att_mask'):
            if not tensor_metrics(read_tensor(trace/f'{name}.f32le'),read_tensor(control/f'{name}.f32le'))['bitwise_equal']:
                raise ValueError('R7B real R6 masks differ from actual NeMo masks')
        observed={'status':'REAL_R6_BOTH_MASKS_INPUT_AND_MASK_BITS_VERIFIED',
                  'scope':'One actual both-mask forward; historical 2x2 ablation not rerun',
                  'pcm_sha256':original['pcm_sha256'],'checkpoint_sha256':original['checkpoint_sha256']}
        (report/'c3_r6_focused_control.json').write_text(json.dumps(observed,indent=2)+'\n')
        result=execute(report,focused_control=True)
        shutil.copyfile(report/'c3_tdt_cpu.json',published/f'c3_tdt_{i}.json')
        shutil.copyfile(report/'c3_r6_focused_control.json',published/f'c3_r6_focused_{i}.json')
        # A config fix must close the observed shape mismatch. Strict f32
        # equality continues to be reported independently, without rounding.
        assert result['frontend_native_pcm']['mel']['status']!='SHAPE_MISMATCH'
        assert result['reference']['preprocessor_state']['pad_to']==16
        assert result['status']=='TDT_NUMERIC_AND_DECISION_PARITY'
        results.append(result['global_bitwise_status'])
    print('HDB2C3_R7B=PADDED_SHAPES_AND_TDT_VERIFIED strict_bits='+','.join(results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    main(p.parse_args().corpus)
