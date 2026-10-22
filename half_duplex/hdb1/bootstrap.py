"""Download externally hosted ONNX models to a user-owned local directory."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

FIRERED_REPO='https://github.com/eschmidbauer/fireredvad-onnx.git'
FIRERED_PIN='faa757dafa0058d0b231ec82e084ec910a3a46d5'
SMART_REPO='pipecat-ai/smart-turn-v3'
SMART_PIN='f766f81d3cfdf7737ac64aad813d91bbfd56bf93'
SMART_SHA256='2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1024*1024): h.update(b)
    return h.hexdigest()


def run(cmd):
    print('+ '+' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)), check=True)


def download(root):
    root.mkdir(parents=True,exist_ok=True)
    repo=root/'external/fireredvad-onnx'
    if not repo.exists(): run(['git','clone',FIRERED_REPO,repo])
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if head!=FIRERED_PIN:
        # Never rebase an existing checkout; perform controlled pinned checkout.
        if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
            raise RuntimeError('ONNX checkout dirty: refusing to change repository')
        run(['git','-C',repo,'fetch','origin',FIRERED_PIN])
        run(['git','-C',repo,'checkout','--detach',FIRERED_PIN])
    src=repo/'onnx_models'
    for name in ('fireredvad_stream_vad_with_cache.onnx','cmvn.ark'):
        srcfile=src/name
        if not srcfile.is_file() or srcfile.stat().st_size<1000:
            raise FileNotFoundError(f'FireRed ONNX source absent or Git LFS pointer: {srcfile}')
        dest=root/name
        shutil.copy2(srcfile,dest)
        print(f'HDB1_FIRERED={dest} sha256={sha(dest)}',flush=True)
    smart=root/'smart-turn-v3.2-cpu.onnx'
    if smart.exists() and sha(smart)==SMART_SHA256:
        print(f'HDB1_SMART=REUSED_SHA256_OK {smart}',flush=True)
    else:
        from huggingface_hub import hf_hub_download
        file=hf_hub_download(repo_id=SMART_REPO,filename='smart-turn-v3.2-cpu.onnx',revision=SMART_PIN)
        if sha(file)!=SMART_SHA256:
            raise ValueError('Smart Turn model sha256 mismatch, refusing unverified model')
        shutil.copy2(file,smart)
        print(f'HDB1_SMART=DOWNLOADED_SHA256_OK {smart}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model-root',type=Path,default=Path.home()/'tmp/haricot-half-duplex/models/hdb1')
    a=p.parse_args()
    download(a.model_root)
