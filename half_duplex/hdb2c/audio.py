"""Explicit, reproducible mono PCM16 16kHz audio contract."""
import hashlib
import json
from pathlib import Path
import subprocess
import wave


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def normalize(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
                        '-i',str(source),'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',
                        str(destination)],capture_output=True,text=True)
    if p.returncode:
        raise RuntimeError('ffmpeg normalization failed: '+p.stderr[-2000:])
    return describe(destination)


def describe(path):
    with wave.open(str(path),'rb') as w:
        params = dict(channels=w.getnchannels(),samplerate=w.getframerate(),
                      width=w.getsampwidth(),frames=w.getnframes(),
                      compression=w.getcomptype())
    if (params['channels'],params['samplerate'],params['width'],params['compression']) != (1,16000,2,'NONE'):
        raise ValueError('Expected PCM16 mono 16000 Hz')
    params['duration_sec'] = params['frames']/16000
    params['wav_sha256'] = sha256(path)
    # fingerprint only sample payload, independent of RIFF metadata
    with wave.open(str(path),'rb') as w:
        params['pcm_sha256'] = hashlib.sha256(w.readframes(w.getnframes())).hexdigest()
    return params


def read_pcm(path):
    import numpy as np
    describe(path)
    with wave.open(str(path),'rb') as w:
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768.0


def report(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
