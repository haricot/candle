#!/usr/bin/env python3
"""Four-turn PCM/decision parity with the existing HDB1 controller, FP32 reference.

Uses actual FireRed ONNX + actual Smart Turn FP32. No ASR, microphone, dummy
model or private audio. The native subprocess has no Python/ONNX dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
ort.disable_telemetry_events()
from qualify import cases, wav


def digest_wav(path):
    import wave
    with wave.open(str(path),'rb') as f:
        return hashlib.sha256(f.readframes(f.getnframes())).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('half-root','models','cache','binary','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.half_root.resolve()))
    from hdb1.firered_onnx import FireRedOnnx
    from hdb1.smart_turn import SmartTurnOnnx
    from hdb1.pipeline import TurnPipeline
    from hdb1.decision import Config
    blocks=[]
    for name,pcm,kind in cases(a.cache):
        if kind=='PUBLIC_SPEECH':blocks.extend([np.zeros(16000,dtype=np.int16),pcm,np.zeros(40000,dtype=np.int16)])
    pcm=np.concatenate(blocks);wav(a.output/'session.wav',pcm)
    vad=FireRedOnnx(a.cache/'firered.onnx',a.cache/'cmvn.ark')
    smart=SmartTurnOnnx(a.cache/'smart-fp32.onnx')
    pipeline=TurnPipeline(vad,smart,None,Config(),a.output/'reference',do_asr=False)
    for start in range(0,len(pcm),1600):pipeline.feed(pcm[start:start+1600])
    pipeline.end_of_stream();reference=pipeline.close()
    command=[str(a.binary.resolve()),'--models',str(a.models.resolve()),'--input',str(a.output/'session.wav'),
        '--max-turns','8','--report-root',str(a.output/'native')]
    run=subprocess.run(command,capture_output=True,text=True)
    (a.output/'native.stdout').write_text(run.stdout);(a.output/'native.stderr').write_text(run.stderr)
    if run.returncode:raise RuntimeError(run.stderr)
    path,=list((a.output/'native').glob('*/report.json'));native=json.loads(path.read_text())
    reference_turns=[{'reason':r['reason'],'samples':r['samples'],'pcm_sha256':digest_wav(r['wav'])} for r in reference['turns']]
    native_turns=[{'reason':r['reason'],'samples':r['samples'],'pcm_sha256':digest_wav(r['wav'])} for r in native['turns']]
    def events(path):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        return [{k:r[k] for k in ('event','sample','complete','candidate','vad_frame_sample') if k in r}
            for r in rows if r['event'] in ('speech_start','speech_end','smart_turn','endpoint_candidate_cancelled','forced_endpoint')]
    reference_events=events(Path(reference['report_dir'])/'events.jsonl')
    native_events=events(path.parent/'events.jsonl')
    report={'scope':'FOUR_PUBLIC_TURNS_WITH_REFERENCE_FIRERED_AND_SMART_FP32_NO_ASR',
        'reference_smart_dtype':'FP32','turns':native_turns,'turn_pcm_and_reason_exact':reference_turns==native_turns,
        'controller_events_and_sample_clocks_exact':reference_events==native_events,
        'native_python_runtime':native['python_runtime'],'native_onnx_runtime':native['onnx_runtime']}
    report['pass']=(len(native_turns)==4 and report['turn_pcm_and_reason_exact'] and report['controller_events_and_sample_clocks_exact']
        and not native['python_runtime'] and not native['onnx_runtime'])
    (a.output/'session-qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['pass']:raise SystemExit(1)

if __name__=='__main__':main()
