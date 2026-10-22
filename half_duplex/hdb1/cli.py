"""Local HDB1 CLI. Capture: PipeWire; endpointing: ONNX; ASR: Rust Candle."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .decision import Config
from .bootstrap import SMART_SHA256
from .pipeline import CandleAsr, TurnPipeline, hash_file


def parse():
    p=argparse.ArgumentParser(description='HDB1 real ONNX streaming VAD + SmartTurn + Candle ASR')
    p.add_argument('mode', choices=['file', 'mic', 'preflight'])
    p.add_argument('--input', type=Path)
    p.add_argument('--model-root', type=Path, default=Path.home()/'tmp/haricot-half-duplex/models/hdb1')
    p.add_argument('--asr-model-dir', type=Path, default=Path(os.getenv('HDB2A_MODEL_DIR',str(Path.home()/'tmp/haricot-half-duplex/models/transcribepsy-eu-sub-candle'))))
    p.add_argument('--asr-runtime-dir', type=Path, default=Path(os.getenv('HDB2A_RUNTIME_DIR',str(Path.home()/'tmp/haricot-half-duplex/integration/parakeet-candle/runtime'))))
    p.add_argument('--variant', choices=['cpu','cuda','cuda-cudnn'], default='cuda-cudnn')
    p.add_argument('--report-root',type=Path,default=Path(os.getenv('HDB2A_REPORT_ROOT',str(Path.home()/'tmp/haricot-half-duplex/reports'))))
    p.add_argument('--max-turns',type=int,default=1,help='Mic mode: limit number of transcribed turns')
    p.add_argument('--no-asr',action='store_true',help='Endpoint-only diagnostic; never claims an ASR transcription')
    p.add_argument('--smart-threshold',type=float,default=.5)
    p.add_argument('--endpoint-hold-ms',type=int,default=1200,help='Uninterrupted post-VAD silence before Smart Turn check; resets on speech')
    p.add_argument('--max-hesitation-ms',type=int,default=3200)
    p.add_argument('--max-turn-ms',type=int,default=25000)
    p.add_argument('--pw-target',help='Optional PipeWire target device ID or name')
    return p.parse_args()


def preflight(a):
    needed={'FireRedVAD_ONNX':a.model_root/'fireredvad_stream_vad_with_cache.onnx',
            'FireRedVAD_CMVN':a.model_root/'cmvn.ark',
            'SmartTurn_v3.2_ONNX':a.model_root/'smart-turn-v3.2-cpu.onnx'}
    result={'stage':'HDB1_PREFLIGHT','models':{}, 'binaries':{}}
    for key,path in needed.items():
        result['models'][key]={'path':str(path),'exists':path.is_file(),
                               'sha256':hash_file(path) if path.is_file() else None}
    for tool in ('ffmpeg','pw-record'):
        result['binaries'][tool]=shutil.which(tool)
    result['asr_binary']=str(a.asr_runtime_dir/f'target/hdb2b-{a.variant}/release/parakeet-candle')
    result['asr_binary_exists']=Path(result['asr_binary']).is_file()
    result['asr_model_exists']=(a.asr_model_dir/'model.safetensors').is_file()
    try:
        import numpy, onnxruntime, kaldiio, kaldi_native_fbank, transformers
        result['onnx_python_imports']='AVAILABLE'
    except ImportError as e:
        result['onnx_python_imports']=f'MISSING: {e}'
    smart=result['models']['SmartTurn_v3.2_ONNX']
    result['smart_turn_sha256_verified']=smart['sha256']==SMART_SHA256
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return (all(x['exists'] for x in result['models'].values())
            and result['onnx_python_imports']=='AVAILABLE'
            and result['smart_turn_sha256_verified'])


def raw_chunks(process, chunk_bytes=3200):
    """Exactly 100ms chunks; prevent odd-byte PCM truncation."""
    buf=bytearray()
    while True:
        part=process.stdout.read(chunk_bytes-len(buf))
        if not part:
            if buf:
                if len(buf)%2: raise IOError('Incomplete PCM16 sample from capture')
                yield bytes(buf)
            break
        buf.extend(part)
        if len(buf) >= chunk_bytes:
            yield bytes(buf)
            buf.clear()


def launch_capture(a, log):
    if a.mode == 'file':
        cmd=['ffmpeg','-hide_banner','-nostdin','-loglevel','warning', '-i',str(a.input),
             '-map','0:a:0','-ar','16000','-ac','1','-f','s16le','-acodec','pcm_s16le','-']
    else:
        cmd=['pw-record','--raw','--format','s16','--rate','16000','--channels','1']
        if a.pw_target: cmd.extend(['--target',a.pw_target])
        cmd.append('-')
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, bufsize=0)


def main():
    import numpy as np
    a=parse()
    if a.mode=='file' and (a.input is None or not a.input.is_file()):
        raise SystemExit('HDB1=FAIL: file mode requires --input PATH')
    if a.max_turns < 1: raise SystemExit('HDB1=FAIL: max-turns must be >= 1')
    if a.mode=='preflight':
        raise SystemExit(0 if preflight(a) else 2)
    if not preflight(a): raise SystemExit('HDB1=MODEL_PREFLIGHT_FAILED: install models first')
    from .firered_onnx import FireRedOnnx
    from .smart_turn import SmartTurnOnnx
    vad=FireRedOnnx(a.model_root/'fireredvad_stream_vad_with_cache.onnx', a.model_root/'cmvn.ark')
    smart=SmartTurnOnnx(a.model_root/'smart-turn-v3.2-cpu.onnx')
    asr=None if a.no_asr else CandleAsr(a.asr_model_dir,a.asr_runtime_dir,a.variant)
    cfg=Config(threshold=a.smart_threshold,endpoint_hold_ms=a.endpoint_hold_ms,max_hesitation_ms=a.max_hesitation_ms,max_turn_ms=a.max_turn_ms)
    pipeline=TurnPipeline(vad,smart,asr,cfg,a.report_root,do_asr=not a.no_asr)
    print(f'HDB1=RUNNING_REAL_ONNX_16KHZ mode={a.mode} asr={"DISABLED" if a.no_asr else a.variant}',flush=True)
    failed=False
    try:
        with (pipeline.run_dir/'capture.stderr').open('wb') as log:
            while True:
                proc=launch_capture(a,log)
                try:
                    for data in raw_chunks(proc):
                        pcm=np.frombuffer(data,dtype='<i2')
                        if pipeline.feed(pcm) and a.mode=='mic':
                            # ASR runs synchronously. If another turn is allowed,
                            # ALWAYS stop this capture and restart it: otherwise
                            # queued PCM captured while ASR was loading is replayed
                            # later as if it were live microphone input.
                            if pipeline.turn_index < a.max_turns:
                                pipeline.log('capture_restart_required_after_asr',
                                             reason='DISCARD_PCM_QUEUED_DURING_COLD_ASR')
                            break
                    if a.mode=='file':
                        status=proc.wait(timeout=15)
                        if status != 0: raise RuntimeError(f'ffmpeg failed: exit={status}')
                        pipeline.end_of_stream()
                        break
                    if pipeline.turn_index >= a.max_turns: break
                    if proc.poll() is not None:
                        raise RuntimeError(f'pw-record terminated: exit={proc.returncode}')
                    # A completed ASR turn reset the VAD; restart PipeWire so
                    # queued microphone audio from the ASR pause is discarded.
                    pipeline.log('capture_restarted_after_asr')
                finally:
                    if proc.poll() is None: proc.terminate()
                    try: proc.wait(timeout=3)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                    if proc.stdout: proc.stdout.close()
    except KeyboardInterrupt:
        pipeline.log('interrupted')
        pipeline.end_of_stream()
    except Exception as exc:
        failed=True
        pipeline.log('error',stage='PIPELINE',message=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report=pipeline.close('FAILED' if failed else 'COMPLETE')
    if a.mode=='file' and not report['turns']:
        print('HDB1=NO_SPEECH_DETECTED (nothing transcribed)',flush=True)
    elif report['turns']:
        print(f'HDB1=REAL_ENDPOINTS_OBSERVED turns={len(report["turns"])}',flush=True)


if __name__=='__main__':main()
