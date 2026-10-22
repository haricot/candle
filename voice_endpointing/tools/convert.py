#!/usr/bin/env python3
"""One-time, hash-pinned conversion. No ONNX evaluation and no PyTorch needed.

The Rust runtime loads only safetensors/config/manifest. No converter import or
ONNX graph interpreter is linked into the application.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

FIRERED_PIN = 'faa757dafa0058d0b231ec82e084ec910a3a46d5'
SMART_PIN = 'f766f81d3cfdf7737ac64aad813d91bbfd56bf93'
SOURCES = {
    'firered.onnx': (f'https://raw.githubusercontent.com/eschmidbauer/fireredvad-onnx/{FIRERED_PIN}/onnx_models/fireredvad_stream_vad_with_cache.onnx',
        '9d67fa8cc3c0d54b3a3f83b2f6905d98a2740693635296bd2fde5679f3ab2f11'),
    'cmvn.ark': (f'https://raw.githubusercontent.com/eschmidbauer/fireredvad-onnx/{FIRERED_PIN}/onnx_models/cmvn.ark',
        'c87f6f13edf0f0ec7535ddfc9cc3387d9268cb234b70182d566c5e2edf3ca473'),
    'smart-fp32.onnx': (f'https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/{SMART_PIN}/smart-turn-v3.2-gpu.onnx',
        'ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be'),
    'smart-int8.onnx': (f'https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/{SMART_PIN}/smart-turn-v3.2-cpu.onnx',
        '2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(root, name):
    root.mkdir(parents=True, exist_ok=True)
    dest = root/name
    url, expected = SOURCES[name]
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Upstream asset digest mismatch: '+name)
        dest.write_bytes(data)
    if sha(dest) != expected:
        raise ValueError('Cached asset digest mismatch: '+name)
    return dest


def tensors(path):
    import onnx
    from onnx import numpy_helper
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return model, {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}


def write_package(root, model, source_name, weights, config):
    import numpy as np
    from safetensors.numpy import save_file
    root.mkdir(parents=True, exist_ok=False)
    weights = {k: np.ascontiguousarray(v, dtype=np.float32) for k,v in weights.items()}
    if not weights or not all(np.isfinite(v).all() for v in weights.values()):
        raise ValueError('Nonfinite or empty model tensors')
    save_file(weights, str(root/'model.safetensors'))
    (root/'config.json').write_text(json.dumps(config, indent=2, allow_nan=False)+'\n')
    manifest = {'schema_version': 1, 'model': model,
        'source_sha256': SOURCES[source_name][1],
        'source_url': SOURCES[source_name][0], 'weights_dtype': 'float32',
        'weight_count': len(weights), 'parameter_count': sum(v.size for v in weights.values()),
        'files': {name: sha(root/name) for name in ('model.safetensors','config.json')}}
    if model == 'fireredvad-stream':
        manifest['cmvn_source_sha256'] = SOURCES['cmvn.ark'][1]
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'package': str(root), **manifest}), flush=True)


def convert_firered(cache, output):
    import kaldiio
    import numpy as np
    _, raw = tensors(source(cache,'firered.onnx'))
    weights = {}
    # These folded MatMul operands belong to the exact SHA above. Restore the
    # Linear [out,in] layout; the model is reconstructed explicitly in Rust.
    first = [9,52,90,128,166,204,242,280]
    second = [11,54,92,130,168,206,244,282]
    for i in range(8):
        prefix = 'model.dfsmn.'
        fc1 = 'fc1.0' if i==0 else f'fsmns.{i-1}.fc1.0'
        filter_name = 'fsmn1' if i==0 else f'fsmns.{i-1}.fsmn'
        weights[f'blocks.{i}.fc1.weight'] = raw[f'val_{first[i]}'].T
        weights[f'blocks.{i}.fc1.bias'] = raw[prefix+fc1+'.bias']
        weights[f'blocks.{i}.fc2.weight'] = raw[f'val_{second[i]}'].T
        if i==0: weights['blocks.0.fc2.bias'] = raw['model.dfsmn.fc2.0.bias']
        weights[f'blocks.{i}.filter'] = raw[prefix+filter_name+'.lookback_filter.weight']
    for dst, val, bias in [('dnn',318,'model.dfsmn.dnns.0.bias'),('out',320,'model.out.bias')]:
        weights[dst+'.weight'] = raw[f'val_{val}'].T
        weights[dst+'.bias'] = raw[bias]
    stats = np.asarray(kaldiio.load_mat(str(source(cache,'cmvn.ark'))))
    if stats.shape != (2,81) or stats[0,80] <= 0: raise ValueError('Unsupported CMVN stats')
    # Retain the existing HDB1 rounding order exactly.
    mean = np.asarray(stats[0,:80]/stats[0,80], dtype=np.float32)
    variance = np.maximum(stats[1,:80]/stats[0,80]-mean**2,1e-20)
    invstd = np.asarray(1/np.sqrt(variance),dtype=np.float32)
    write_package(output/'firered', 'fireredvad-stream','firered.onnx',weights,
        {'architecture':'fireredvad-stream-8x256x128-k20', 'dtype':'float32',
         'sample_rate':16000,'mean':mean.tolist(),'invstd':invstd.tolist(),
         'frontend':'kaldi-povey-400-hop160-fft512-mel80-snip-edges-no-dither'})


def convert_smart(cache, output):
    model, raw = tensors(source(cache,'smart-fp32.onnx'))
    weights = {k.removeprefix('inner.'):v for k,v in raw.items() if k.startswith('inner.')}
    # Recover each folded Linear weight from its exporter module namespace.
    for node in model.graph.node:
        if node.op_type != 'MatMul' or node.input[1] not in raw: continue
        namespace = dict((p.key,p.value) for p in node.metadata_props).get('namespace','')
        parts = [p for p in namespace.split('/') if p.startswith('inner.')]
        if not parts: raise ValueError('Missing Linear module provenance')
        name = parts[-1].split(':')[0].removeprefix('inner.')
        if name not in ('pool_attention.0','pool_attention.2') and not (
                name.startswith('encoder.layers.') and name.endswith(('q_proj','k_proj','v_proj','out_proj','fc1','fc2'))):
            raise ValueError('Unrecognized Linear owner: '+name)
        key = name+'.weight'
        if key in weights: raise ValueError('Duplicate Linear owner: '+key)
        weights[key] = raw[node.input[1]].T
    if len(weights) != 79: raise ValueError(f'Unexpected Smart Turn tensor count: {len(weights)}')
    write_package(output/'smart-turn','smart-turn-v3.2-fp32','smart-fp32.onnx',weights,
        {'architecture':'smart-turn-v3.2-whisper-tiny-400','dtype':'float32',
         'sample_rate':16000,'num_mel_bins':80,'frames':800,'max_source_positions':400,
         'layers':4,'hidden':384,'heads':6,'ffn':1536,'layer_norm_epsilon':1e-5,
         'reference_int8_sha256':SOURCES['smart-int8.onnx'][1],
         'reference_int8_is_not_the_fp32_numeric_reference':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise SystemExit('Refusing to overwrite an existing model package directory')
    import tempfile
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.endpoint-conversion-',dir=a.output.parent) as temp:
        stage=Path(temp)/'models'
        stage.mkdir()
        convert_firered(a.cache,stage)
        convert_smart(a.cache,stage)
        stage.rename(a.output)

if __name__=='__main__': main()
