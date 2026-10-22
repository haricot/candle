#!/usr/bin/env python3
"""HDB2-A: real (never simulated) FastConformer/TDT transcription using Candle.
All runs write auditable JSON+text, including failures."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import wave

MODEL_ID = 'qvac/TranscribePsy-250M-EU-sub'
PARAKEET_PIN = '53c0d3065f9dd8de0356b25391e60ed3a471192b'
CANDLE_PIN = 'f5ce7f6f8b7a68d14a55419f9f221e1ae3537fd3'


def wav_info(wav:Path):
    with wave.open(str(wav),'rb') as f:
        return dict(sample_rate=f.getframerate(),channels=f.getnchannels(),sample_width=f.getsampwidth(),frames=f.getnframes(),duration_sec=f.getnframes()/f.getframerate())


def convert_audio(input_path:Path, output_wav:Path, conversion_log:Path):
    output_wav.parent.mkdir(parents=True,exist_ok=True)
    cmd = ['ffmpeg','-hide_banner','-nostdin','-y','-i',str(input_path),'-map','0:a:0','-ar','16000','-ac','1','-c:a','pcm_s16le',str(output_wav)]
    with conversion_log.open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    info=wav_info(output_wav)
    if info['sample_rate']!=16000 or info['channels']!=1 or info['sample_width']!=2 or info['frames']<=0:
        raise ValueError('ffmpeg output is not nonempty PCM16 16k mono WAV')
    return info


def git_head(p:Path):
    try: return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--runtime-dir',type=Path,required=True)
    parser.add_argument('--report-root',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if not args.input.is_file(): parser.error(f'Missing input audio file {args.input}')
    if not (args.model_dir/'model.safetensors').is_file():parser.error('No converted model.safetensors; execute prepare-model first')
    if not (args.model_dir/'config.json').is_file():parser.error('No converted config.json; execute prepare-model first')
    runtime_bin=args.runtime_dir/'target/release/parakeet-candle'
    if not runtime_bin.is_file(): parser.error(f'Candle binary missing: {runtime_bin}; execute build {args.device}')
    if git_head(args.runtime_dir)!=PARAKEET_PIN:parser.error('Runtime pinned commit mismatch')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report_dir=args.report_root/f'hdb2a-{stamp}-{os.getpid()}-{uuid.uuid4().hex[:6]}'
    report_dir.mkdir(parents=True,exist_ok=False)
    report={'stage':'HDB2A_REAL_CANDLE_ASR','status':'NOT_QUALIFIED','model_id':MODEL_ID,
        'backend':'candle-fastconformer-tdt','device_requested':args.device,'input':str(args.input.resolve()),
        'runtime_commit':git_head(args.runtime_dir),'candle_pin':CANDLE_PIN,
        'asr_output_is_synthetic':False,'report_dir':str(report_dir.resolve())}
    start=time.perf_counter()
    try:
        from hdb2a_model_config import validate_config
        validate_config(json.loads((args.model_dir/'config.json').read_text(encoding='utf-8')))
        conversion_info=json.loads((args.model_dir/'conversion.json').read_text())
        if conversion_info.get('model')!=MODEL_ID:
            raise ValueError('Conversion provenance does not identify QVAC EU-sub')
        report['checkpoint_sha256']=conversion_info['source_nemo_sha256']
        info=convert_audio(args.input,report_dir/'input-16k.wav',report_dir/'ffmpeg.log')
        report['normalized_audio']=info
        cmd=[str(runtime_bin),'--model-id',MODEL_ID,'--model-dir',str(args.model_dir.resolve()),
             '--input',str((report_dir/'input-16k.wav').resolve()),'--format','json']
        if args.device=='cpu':cmd.append('--cpu')
        print('HDB2A=RUNNING_REAL_CANDLE_INFERENCE',flush=True)
        t_inf=time.perf_counter()
        run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        report['inference_wall_ms']=round((time.perf_counter()-t_inf)*1000,3)
        (report_dir/'runtime.stdout').write_text(run.stdout,encoding='utf-8')
        (report_dir/'runtime.stderr').write_text(run.stderr,encoding='utf-8')
        report['runtime_exit_code']=run.returncode
        if run.returncode!=0:
            raise RuntimeError('Candle runtime failed; inspect runtime.stderr')
        payload=json.loads(run.stdout)
        transcript=payload.get('text','')
        if not isinstance(transcript,str) or not transcript.strip():
            raise ValueError('Candle emitted no nonempty transcript; not qualified')
        if payload.get('model_id')!=MODEL_ID:
            raise ValueError('Runtime did not report requested model ID')
        device=str(payload.get('device','')).lower()
        report['runtime_device']=payload.get('device')
        if args.device=='cuda' and 'cuda' not in device:
            raise ValueError('GPU requested but backend fell back to CPU')
        if args.device=='cpu' and 'cpu' not in device:
            raise ValueError('CPU requested but GPU was selected')
        (report_dir/'transcription.txt').write_text(transcript.strip()+'\n',encoding='utf-8')
        report['text']=transcript.strip()
        report['num_characters']=len(transcript.strip())
        report['rtf'] = (report['inference_wall_ms']/1000)/info['duration_sec']
        report['status']='REAL_CANDLE_TRANSCRIPTION_OBSERVED_UNQUALIFIED_WER'
        print(f'HDB2A={report["status"]}')
        print('TRANSCRIPTION_BEGIN')
        print(transcript.strip())
        print('TRANSCRIPTION_END')
        print(f'TRANSCRIPTION_FILE={report_dir/"transcription.txt"}')
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        print(f'HDB2A=FAIL_NOT_QUALIFIED {report["error"]}',file=sys.stderr)
    finally:
        report['end_to_end_wall_ms']=round((time.perf_counter()-start)*1000,3)
        (report_dir/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(f'HDB2A_REPORT={report_dir/"report.json"}',flush=True)
    if report['status']!='REAL_CANDLE_TRANSCRIPTION_OBSERVED_UNQUALIFIED_WER':
        sys.exit(1)

if __name__=='__main__':main()
