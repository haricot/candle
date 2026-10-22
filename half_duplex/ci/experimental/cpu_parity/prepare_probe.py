"""Prepare an isolated local R11 probe from an unmodified, built R10 candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BUNDLE=Path(__file__).resolve().parent
sys.path.insert(0,str(BUNDLE.parents[2]))
from hdb2c.c3_r8_source import once

REVISION='03c022d3ffdcee958cfacbe720048e725fdf644c'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(parent,target,source,build):
    parent=parent.resolve();target=target.resolve();source=source.resolve();build=build.resolve()
    if target.exists() or parent in target.parents or target in parent.parents:
        raise ValueError('R11 requires a new directory outside the parent candidate')
    old=json.loads((parent/'R10-CPU-MANIFEST.json').read_text())
    if old['signature']!='hdb2c3-r10-native-cpu-mkl-v0.7.10':raise ValueError('R11 requires the measured R10 source profile')
    for name,digest in {**old['parent_source_sha256'],**old['target_sha256']}.items():
        if sha(parent/name)!=digest:raise ValueError('R10 source changed: '+name)
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=REVISION:
        raise ValueError('R11 requires the exact oneDNN source revision')
    subprocess.run(['git','-C',str(source),'diff','--exit-code'],check=True,stdout=subprocess.DEVNULL)
    cache=(build/'CMakeCache.txt').read_text()
    for flag in ('DNNL_GPU_RUNTIME:STRING=NONE','DNNL_CPU_RUNTIME:STRING=OMP'):
        if flag not in cache:raise ValueError('R11 unexpected SDK setting: '+flag)
    if 'REORDER' not in cache or 'CONVOLUTION' not in cache:raise ValueError('R11 needs convolution and reorder primitives')
    library=build/'src/libdnnl.so.3.11'
    if not library.is_file():raise ValueError('Build the standalone CPU SDK first')
    shutil.copytree(parent,target,ignore=shutil.ignore_patterns('target','.git','__pycache__'))
    shutil.copyfile(BUNDLE/'onednn_bridge.rs',target/'src/parakeet/hdb2c_r11_onednn.rs')
    shutil.copyfile(BUNDLE/'onednn_bridge.cpp',target/'src/parakeet/hdb2c_r11_onednn.cpp')
    p=target/'src/parakeet/mod.rs';p.write_text(p.read_text()+'\npub mod hdb2c_r11_onednn;\n')
    p=target/'src/parakeet/conformer.rs';text=p.read_text()
    text=once(text,'x = conv.forward(&x)?;','x = crate::parakeet::hdb2c_r11_onednn::conv2d(conv,&x)?;')
    text=once(text,'let y=pre.conv[index].forward(&x)?;','let y=crate::parakeet::hdb2c_r11_onednn::conv2d(&pre.conv[index],&x)?;')
    p.write_text(text)
    probe=(target/'src/bin/hdb2c_r10_probe.rs').read_text()
    probe=once(probe,'R10_NATIVE_CPU_MKL_CHECKPOINT_FRONTEND_FUSED_PROJECTION','R11_LOCAL_CPU_MKL_ONEDNN_SUBSAMPLING')
    (target/'src/bin/hdb2c_r11_probe.rs').write_text(probe)
    p=target/'build.rs';text=p.read_text()
    text=once(text,'let obj=out.join("hdb2c_cpu.o");let lib=out.join("libhdb2c_cpu.a");','''let obj=out.join("hdb2c_cpu.o");let lib=out.join("libhdb2c_cpu.a");
 let cpp=out.join("onednn_cpu.o");
 let dnnl=PathBuf::from(env::var_os("HDB2C_ONEDNN_BUILD").expect("oneDNN CPU build required"));
 let include=PathBuf::from(env::var_os("HDB2C_ONEDNN_INCLUDE").expect("oneDNN source include required"));
 assert!(Command::new("g++").args(["-O2","-fPIC","-fopenmp","-std=c++17","-c","src/parakeet/hdb2c_r11_onednn.cpp","-o"]).arg(&cpp).arg("-I").arg(include).arg("-I").arg(dnnl.join("include")).status().unwrap().success());
 println!("cargo:rustc-link-search=native={}/src",dnnl.display());
 println!("cargo:rustc-link-arg=-Wl,-rpath,{}/src",dnnl.display());
 println!("cargo:rustc-link-lib=dylib=dnnl");
 println!("cargo:rustc-link-lib=dylib=stdc++");
 println!("cargo:rustc-link-lib=dylib=gomp");''')
    text=once(text,'arg(&lib).arg(&obj).status()','arg(&lib).arg(&obj).arg(&cpp).status()')
    p.write_text(text)
    manifest={'scope':'LOCAL_R11_CPU_EXPERIMENT','parent':old,'onednn_commit':REVISION,
        'onednn_library_sha256':sha(library),'source_sha256':{str(p.relative_to(target)):sha(p)
            for p in target.rglob('*') if p.is_file() and 'target' not in p.parts and p.suffix in ('.rs','.c','.cpp','.toml','.lock')}}
    (target/'R11-LOCAL-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    env={**os.environ,'HDB2C_MKL_ROOT':old['sdk']['root'],'HDB2C_ONEDNN_BUILD':str(build),
        'HDB2C_ONEDNN_INCLUDE':str(source/'include'),'CARGO_TARGET_DIR':str(target/'target/r11-cpu')}
    subprocess.run(['cargo','build','--release','--manifest-path',str(target/'Cargo.toml'),
        '--bin','hdb2c_r11_probe'],env=env,check=True)
    binary=target/'target/r11-cpu/release/hdb2c_r11_probe'
    manifest['binary_sha256']=sha(binary)
    (target/'R11-LOCAL-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('R11_LOCAL_CPU_BINARY='+str(binary))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('parent','target','onednn-source','onednn-build'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.parent,a.target,a.onednn_source,a.onednn_build)
