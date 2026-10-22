"""Compare the standalone ordered norm on 14 actual NeMo common inputs."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import numpy as np
from safetensors import safe_open

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from hdb2c.c3_r3 import read_tensor
from hdb2c.c3_r8 import selected_corpus,metric


def main(a):
    rows=selected_corpus(json.loads(a.corpus.read_text()))
    lib=ctypes.CDLL(str(a.library.resolve()))
    array=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.hdb2c12_layernorm.argtypes=[ctypes.c_int]*3+[ctypes.c_float]+[array]*4
    with safe_open(str(a.model/'model.safetensors'),framework='np') as f:
        gamma=f.get_tensor('encoder.layers.0.norm_feed_forward1.weight').copy()
        beta=f.get_tensor('encoder.layers.0.norm_feed_forward1.bias').copy()
    conversion=json.loads((a.model/'conversion.json').read_text());results=[]
    for i,row in enumerate(rows):
        folder=a.references/f'clip-{i:02}/nemo';reference=json.loads((folder/'manifest.json').read_text())
        if reference['pcm_sha256']!=row['pcm_sha256'] or reference['checkpoint_sha256']!=conversion['source_nemo_sha256']:
            raise ValueError('Reference PCM/checkpoint identity differs')
        x=read_tensor(folder/'encoder_trace/block0_input.f32le').copy()
        expected=read_tensor(folder/'encoder_trace/block0_norm_ff1.f32le').copy()
        out=np.empty_like(expected)
        if lib.hdb2c12_layernorm(x.size//x.shape[-1],x.shape[-1],8,1e-5,x,gamma,beta,out):
            raise RuntimeError('Standalone norm failed')
        result=metric(expected,out);results.append({'id':row['id'],**result})
        print(row['id'],'bitwise='+str(result['bitwise_equal']),'max_abs='+str(result['max_abs']))
    a.output.write_text(json.dumps({'scope':'COMMON_INPUT_NORM_ONLY_NOT_FULL_PIPELINE','results':results},indent=2)+'\n')
    return 0 if all(r['bitwise_equal'] for r in results) else 2


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('model','corpus','references','library','output'):p.add_argument('--'+name,type=Path,required=True)
    sys.exit(main(p.parse_args()))
