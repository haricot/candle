#!/usr/bin/env python3
"""Compare real native Candle inference with pinned upstream ONNX models on CPU.

ONNX/Transformers/Kaldi are isolated qualification dependencies, never runtime
dependencies. No NeMo/ASR rerun, no transcripts and no microphone audio uploaded.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import wave

import kaldi_native_fbank as knf
import kaldiio
import numpy as np
import onnx
import onnxruntime as ort
# Upstream reference only. Disable ORT's default telemetry BEFORE creating any
# inference session, including sessions constructed for intermediate outputs.
ort.disable_telemetry_events()
import pyarrow.parquet as pq
import soundfile as sf
from transformers import WhisperFeatureExtractor

from convert import source, SOURCES, sha

from corpus import ANCHORS, CORPUS_REV, CORPUS_SHA, cases, wav

# Fixed before measuring. Feature limits cover FP32 window/FFT/CMVN rounding;
# shared-input limits isolate model arithmetic. Absolute + relative limits avoid
# using unstable relative errors near zero. INT8 differences are separate data.
LIMITS={
    'vad.features':(0.001,0.0001),
    'smart.features':(0.00005,0.00001),
    'vad.probs':(0.00005,0.0005),
    'vad.cache':(0.001,0.0005),
    'smart.encoder':(0.002,0.0005),
    'smart.pooled':(0.001,0.0005),
    'shared.vad.probs':(0.00001,0.0001),
    'shared.vad.cache':(0.0001,0.0001),
    'shared.smart.encoder':(0.0003,0.0003),
    'shared.smart.pooled':(0.0003,0.0003),
    'probability':(0.0005,0.0005),
    'shared.probability':(0.00005,0.0001),
}

def metric(a,b,limits):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:raise ValueError(f'Shape mismatch: {a.shape} / {b.shape}')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite reference/candidate tensor')
    x=a.astype(np.float64);y=b.astype(np.float64)
    atol,rtol=limits
    return {'pass':bool(np.allclose(x,y,atol=atol,rtol=rtol)), 'atol':atol,'rtol':rtol,
        'elements':a.size,'max_abs':float(np.max(np.abs(x-y))) if a.size else 0,
        'relative_l2':float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-30)),
        'bitwise_equal_diagnostic':bool(a.dtype==b.dtype and a.tobytes()==b.tobytes())}

def session(model):
    ort.disable_telemetry_events()
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model,sess_options=opts,providers=['CPUExecutionProvider'])

def smart_session(path):
    model=onnx.load(str(path))
    for name,shape in [('layer_norm_8',[1,400,384]),('sum_1',[1,384])]:
        model.graph.output.append(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,shape))
    return session(model.SerializeToString())

def reference_vad(pcm,model,stats):
    opts=knf.FbankOptions();opts.frame_opts.samp_freq=16000
    opts.frame_opts.frame_length_ms=25;opts.frame_opts.frame_shift_ms=10
    opts.frame_opts.dither=0;opts.frame_opts.snip_edges=True;opts.mel_opts.num_bins=80
    mean=np.asarray(stats[0,:80]/stats[0,80],dtype=np.float32)
    invstd=np.asarray(1/np.sqrt(np.maximum(stats[1,:80]/stats[0,80]-mean**2,1e-20)),dtype=np.float32)
    pending=np.empty(0,dtype=np.int16);cache=np.zeros((8,1,128,19),dtype=np.float32)
    features=[];probabilities=[]
    for start in range(0,len(pcm),1600):
        samples=np.concatenate((pending,pcm[start:start+1600]))
        if len(samples)<400:pending=samples;continue
        fb=knf.OnlineFbank(opts);fb.accept_waveform(16000,samples.tolist())
        n=fb.num_frames_ready
        if not n:pending=samples;continue
        raw=np.stack([fb.get_frame(i) for i in range(n)]).astype(np.float32)
        pending=samples[n*160:]
        feat=((raw-mean)*invstd)[None]
        probability,cache=model.run(None,{'feat':feat,'caches_in':cache})
        features.extend(feat.reshape(-1));probabilities.extend(probability.reshape(-1))
    return {'vad.features':np.asarray(features,dtype=np.float32),'vad.probs':np.asarray(probabilities,dtype=np.float32),'vad.cache':cache.reshape(-1)}

def reference_events(probs):
    # Same published HDB1 smoothing/hysteresis contract, driven by ORT outputs.
    window=[];state='SILENCE';speech=0;silence=0;events=[]
    for frame,p in enumerate(probs,1):
        window.append(float(p));window=window[-5:];active=sum(window)/len(window)>=0.4
        if state=='SILENCE':
            if active:speech=1;state='POSSIBLE_SPEECH'
        elif state=='POSSIBLE_SPEECH':
            if active:
                speech+=1
                if speech>=8:state='SPEECH';silence=0;events.append({'kind':'speech_start','sample':max(0,frame-speech-5)*160})
            else:state='SILENCE';speech=0
        elif state=='SPEECH':
            speech+=1
            if active:silence=0
            else:silence=1;state='POSSIBLE_SILENCE'
        else:
            speech+=1
            if active:silence=0;state='SPEECH'
            else:
                silence+=1
                if silence>=20:events.append({'kind':'speech_end','sample':frame*160});state='SILENCE';speech=0
        if state in ('SPEECH','POSSIBLE_SILENCE') and speech>=2000:
            events.append({'kind':'speech_end','sample':frame*160});state='SILENCE';speech=0
    return events

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--only',help='Optional exact case name for development; report remains scoped to that case')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);a.cache.mkdir(parents=True,exist_ok=True)
    vad=session(str(source(a.cache,'firered.onnx')));smart=smart_session(source(a.cache,'smart-fp32.onnx'))
    int8=session(str(source(a.cache,'smart-int8.onnx')))
    stats=np.asarray(kaldiio.load_mat(str(source(a.cache,'cmvn.ark'))))
    extractor=WhisperFeatureExtractor(chunk_length=8)
    report={'scope':'NATIVE_ENDPOINT_MODELS_CPU_4_PUBLIC_4_PREFIX_2_SYNTHETIC','models':{k:v[1] for k,v in SOURCES.items()},
        'corpus_revision':CORPUS_REV,'asr_retested':False,'limits':LIMITS,'cases':[]}
    for name,pcm,kind in cases(a.cache):
        if a.only and name!=a.only:continue
        folder=a.output/name;folder.mkdir();wav(folder/'input.wav',pcm)
        ref=reference_vad(pcm,vad,stats)
        padded=pcm[-128000:].astype(np.float32)/32768.0
        if len(padded)<128000:padded=np.pad(padded,(128000-len(padded),0))
        feat=np.asarray(extractor(padded,sampling_rate=16000,return_tensors='np',padding='max_length',max_length=128000,truncation=True,do_normalize=True).input_features,dtype=np.float32)
        probability,encoder,pooled=smart.run(None,{'input_features':feat})
        ref.update({'smart.features':feat.reshape(-1),'smart.encoder':encoder.reshape(-1),'smart.pooled':pooled.reshape(-1)})
        probability=float(probability.reshape(-1)[0]);probability8=float(int8.run(None,{'input_features':feat})[0].reshape(-1)[0])
        for key,data in ref.items():np.asarray(data,dtype='<f4').tofile(folder/(key+'.f32le'))
        row={'name':name,'kind':kind,'pcm_sha256':hashlib.sha256(pcm.tobytes()).hexdigest(),'samples':len(pcm),
            'reference_probability_fp32':probability,'reference_probability_int8':probability8,
            'fp32_int8_decision_agree':(probability>=0.5)==(probability8>=0.5),'paths':{}}
        for mode in ('native','shared'):
            command=[str(a.binary.resolve()),'--models',str(a.models.resolve()),'--input',str(folder/'input.wav'),'--output',str(folder/mode)]
            if mode=='shared':command+=['--shared-vad',str(folder/'vad.features.f32le'),'--shared-smart',str(folder/'smart.features.f32le')]
            run=subprocess.run(command,capture_output=True,text=True)
            (folder/(mode+'.stderr')).write_text(run.stderr)
            if run.returncode:raise RuntimeError(run.stderr)
            out=json.loads((folder/mode/'result.json').read_text())
            metrics={}
            for key in ref:
                limits_key=('shared.'+key) if mode=='shared' and not key.endswith('.features') else key
                metrics[key]=metric(ref[key],np.fromfile(folder/mode/(key+'.f32le'),dtype='<f4'),LIMITS[limits_key])
            limits_key='shared.probability' if mode=='shared' else 'probability'
            metrics['probability']=metric(np.array([probability],dtype=np.float32),np.array([out['smart_probability']],dtype=np.float32),LIMITS[limits_key])
            exact={'vad_events':out['vad_events']==reference_events(ref['vad.probs']),
                'smart_decision':out['smart_complete']==(probability>=0.5),'device':out['device']=='Cpu',
                'vad_frame_count':out['vad_frames']==len(ref['vad.probs'])}
            row['paths'][mode]={'metrics':metrics,'exact':exact,'pass':all(x['pass'] for x in metrics.values()) and all(exact.values())}
        row['pass']=all(x['pass'] for x in row['paths'].values());report['cases'].append(row)
        print(json.dumps({'case':name,'pass':row['pass'],'fp32':probability,'int8':probability8}),flush=True)
    if not report['cases']:raise ValueError('No qualification cases executed')
    report['pass']=all(x['pass'] for x in report['cases'])
    report['case_count']=len(report['cases']);report['fp32_int8_decision_disagreements']=sum(not x['fp32_int8_decision_agree'] for x in report['cases'])
    (a.output/'qualification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('ENDPOINT_QUALIFICATION='+('PASS' if report['pass'] else 'FAIL'),flush=True)
    if not report['pass']:raise SystemExit(1)

if __name__=='__main__':main()
