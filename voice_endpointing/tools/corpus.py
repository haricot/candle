"""Pinned public PCM corpus; no inference engine or telemetry SDK is imported."""
import hashlib
import io
from pathlib import Path
import wave

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from convert import sha

CORPUS_REV='c7f2fc9f0538ca871d38e183ba8ad6c329120ecc'
CORPUS_SHA={
    'cafe':'cef4c83676038c80841c8b9712be62bbf3d03c8dc148310203b634aad7c4c9e1',
    'oreau':'949ad6d6ec74ce2a95e8cba857dcadc98f374366f3b4a0ad719bf067780734e1',
}
ANCHORS={
    ('cafe','anger'):'6794242ad35ae6125884457d25f481d520251656044e4ba1f7bfb1c8ae0b1a52',
    ('cafe','neutral'):'3863734a5ee478104a3152a293f5f5b6c57ba9edbbfe21bea84bda0c94f65028',
    ('oreau','fear'):'264f4f3727129cbdaed2d3bdf7182246649e2dfbba6c5b545fc03959e1b7a4d4',
    ('oreau','surprise'):'17c071bdf88e6f594151ca35896ad90e913aad80cdac9d3ac61ea6ee29c44c5c',
}
def wav(path,pcm):
    with wave.open(str(path),'wb') as out:
        out.setnchannels(1);out.setsampwidth(2);out.setframerate(16000);out.writeframes(pcm.astype('<i2').tobytes())

def cases(cache):
    import urllib.request
    found={}
    for split in ('cafe','oreau'):
        path=cache/(split+'.parquet')
        if not path.exists():
            url=f'https://huggingface.co/datasets/amu-cai/CAMEO/resolve/{CORPUS_REV}/data/{split}-00000-of-00001.parquet'
            with urllib.request.urlopen(url,timeout=120) as response:path.write_bytes(response.read())
        if sha(path)!=CORPUS_SHA[split]:raise ValueError('Corpus parquet digest mismatch')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                key=(split,str(row.get('emotion','')).casefold())
                if key not in ANCHORS or key in found:continue
                data,sr=sf.read(io.BytesIO(row['audio']['bytes']),dtype='float32')
                if sr!=16000 or data.ndim!=1:raise ValueError('Unexpected corpus audio format')
                pcm=np.clip(np.rint(data*32768),-32768,32767).astype('<i2')
                digest=hashlib.sha256(pcm.tobytes()).hexdigest()
                if digest==ANCHORS[key]:found[key]=pcm
            if all(k in found for k in ANCHORS if k[0]==split):break
    if set(found)!=set(ANCHORS):raise ValueError('Missing pinned public excerpts')
    rows=[]
    for key,pcm in sorted(found.items()):
        name='-'.join(key)
        rows.append((name,pcm,'PUBLIC_SPEECH'))
        rows.append((name+'-prefix',pcm[:max(4800,int(len(pcm)*0.55))],'TRUNCATED_PUBLIC_SPEECH'))
    rows.extend([('silence',np.zeros(32000,dtype='<i2'),'SYNTHETIC_SILENCE'),
        ('tone',(3000*np.sin(np.arange(32000)*2*np.pi*440/16000)).astype('<i2'),'SYNTHETIC_TONE')])
    return rows
