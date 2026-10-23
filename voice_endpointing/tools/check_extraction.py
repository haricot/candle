#!/usr/bin/env python3
"""Compare qualified and extracted Rust implementations on the same CPU.

No ONNX Runtime, PyTorch, Transformers or reference inference is imported.
Bit equality is a same-backend refactoring check, not a CPU/CUDA requirement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
from corpus import cases, wav

if any(name in sys.modules for name in ('onnxruntime','transformers','torch')):
    raise RuntimeError('Native comparison must not load an external inference runtime')

BASELINE='fced4317d64715174c32f48bc3bc9211f514c342'

def run(command):
    result=subprocess.run([str(x) for x in command],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    return result

def pcm_sha(path):
    with wave.open(str(path),'rb') as f:return hashlib.sha256(f.readframes(f.getnframes())).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-binary','candidate-binary','models','cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    rows=cases(a.cache);report={'scope':'SAME_CPU_MODEL_EXTRACTION_COMPATIBILITY',
        'baseline_commit':BASELINE,'python_onnx_runtime_loaded':False,'cases':[]}
    for name,pcm,kind in rows:
        folder=a.output/name;folder.mkdir();wav(folder/'input.wav',pcm)
        comparisons={}
        for mode in ('native','shared'):
            results={}
            for label,binary in [('baseline',a.baseline_binary),('candidate',a.candidate_binary)]:
                target=folder/(label+'-'+mode)
                command=[binary.resolve(),'--models',a.models.resolve(),'--input',folder/'input.wav','--output',target]
                if mode=='shared':command+=['--shared-vad',folder/'baseline-native/vad.features.f32le',
                    '--shared-smart',folder/'baseline-native/smart.features.f32le']
                run(command);results[label]=target
            old,new=results['baseline'],results['candidate']
            tensors={x.name:x.read_bytes()==(new/x.name).read_bytes() for x in old.glob('*.f32le')}
            finite=all(np.isfinite(np.fromfile(new/key,dtype='<f4')).all() for key in tensors)
            exact=json.loads((old/'result.json').read_text())==json.loads((new/'result.json').read_text())
            comparisons[mode]={'tensor_bits_equal':tensors,'result_and_decisions_exact':exact,
                'finite':bool(finite),'pass':bool(len(tensors)==6 and all(tensors.values()) and exact and finite)}
        report['cases'].append({'name':name,'kind':kind,'samples':len(pcm),
            'pcm_sha256':hashlib.sha256(pcm.tobytes()).hexdigest(),'paths':comparisons,
            'pass':all(v['pass'] for v in comparisons.values())})
        print(json.dumps({'case':name,'pass':report['cases'][-1]['pass']}),flush=True)
    # Replay four actual turns through both unchanged native controllers.
    blocks=[]
    for _,pcm,kind in rows:
        if kind=='PUBLIC_SPEECH':blocks.extend([np.zeros(16000,dtype=np.int16),pcm,np.zeros(40000,dtype=np.int16)])
    wav(a.output/'session.wav',np.concatenate(blocks))
    sessions={}
    for label,probe in [('baseline',a.baseline_binary),('candidate',a.candidate_binary)]:
        binary=probe.resolve().parent/'voice-endpointing'
        run([binary,'--models',a.models.resolve(),'--input',a.output/'session.wav',
            '--max-turns','8','--report-root',a.output/('session-'+label)])
        path,=list((a.output/('session-'+label)).glob('*/report.json'))
        data=json.loads(path.read_text())
        events=[]
        for line in (path.parent/'events.jsonl').read_text().splitlines():
            event=json.loads(line);event.pop('latency_ms',None);events.append(event)
        sessions[label]={'turns':[{'samples':t['samples'],'reason':t['reason'],'pcm_sha256':pcm_sha(t['wav'])} for t in data['turns']],
            'events':events,'python_runtime':data['python_runtime'],'onnx_runtime':data['onnx_runtime']}
    report['session']={'turn_count':len(sessions['candidate']['turns']),
        'turns_and_events_exact':sessions['baseline']==sessions['candidate'],
        'python_runtime':sessions['candidate']['python_runtime'],'onnx_runtime':sessions['candidate']['onnx_runtime']}
    report['case_count']=len(rows)
    report['pass']=(len(rows)==10 and all(x['pass'] for x in report['cases'])
        and report['session']['turn_count']==4 and report['session']['turns_and_events_exact']
        and not report['session']['python_runtime'] and not report['session']['onnx_runtime'])
    (a.output/'extraction-compatibility.json').write_text(json.dumps(report,indent=2)+'\n')
    print('MODEL_EXTRACTION='+('PASS' if report['pass'] else 'FAIL'),flush=True)
    if not report['pass']:raise SystemExit(1)

if __name__=='__main__':main()
