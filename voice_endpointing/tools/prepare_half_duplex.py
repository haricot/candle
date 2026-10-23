#!/usr/bin/env python3
"""Build-time composition of qualified R9 and native endpointing. No runtime Python.

Reuses the existing R9 source generator from an explicit half_duplex checkout.
Only a NEW binary and a dependency are added; R9 library/probe bytes are checked
unchanged. All consumers bind to one explicitly selected Candle source tree.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1]
MODEL_FILES=('fireredvad.rs','fireredvad/audio.rs','smart_turn.rs','smart_turn/audio.rs','SMART-TURN-LICENSE')

def digest_sources(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root/'src').rglob('*.rs'))}

def copy_endpointing_crate(source, destination):
    """Copy build inputs, including when output lives under source/integration."""
    source=source.resolve();destination=destination.resolve()
    if destination==source or destination.is_relative_to(source/'src'):
        raise ValueError('Endpointing crate output must be outside its Rust sources')
    destination.mkdir(parents=True,exist_ok=False)
    # Never traverse integration, virtualenvs, weights or other local artifacts:
    # the installer creates its output inside this checkout's integration/.
    for name in ('Cargo.toml','Cargo.lock','NOTICE.md'):
        shutil.copy2(source/name,destination/name)
    shutil.copytree(source/'src',destination/'src')

def toml_value(value):
    """Serialize the scalar/inline values in Candle's workspace metadata."""
    if isinstance(value,str):return json.dumps(value,ensure_ascii=False)
    if isinstance(value,bool):return 'true' if value else 'false'
    if isinstance(value,(int,float)):return str(value)
    if isinstance(value,list):return '['+', '.join(toml_value(v) for v in value)+']'
    if isinstance(value,dict):return '{'+', '.join(json.dumps(k)+' = '+toml_value(v) for k,v in value.items())+'}'
    raise TypeError(f'Unsupported workspace metadata: {type(value)}')

def prepare_transformers(core, work):
    """Overlay only the models in an isolated crate, keeping one exact backend.

    An older promoted Extended checkout does not yet contain these models. Its
    backend and existing model sources remain unchanged. A small private
    workspace binds the copied transformer crate to that checkout's absolute
    dependency paths; no second candle-core is built from the feature branch.
    """
    core=core.resolve()
    source=ROOT.parent/'candle-transformers/src/models'
    for name in MODEL_FILES:
        if not (source/name).is_file():raise ValueError(f'Missing Candle model source: {source/name}')
    support=work/'candle-models';support.mkdir()
    target=support/'candle-transformers';target.mkdir()
    for name in ('Cargo.toml','README.md'):
        shutil.copy2(core/'candle-transformers'/name,target/name)
    shutil.copytree(core/'candle-transformers/src',target/'src')
    for name in MODEL_FILES:
        dst=target/'src/models'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/name,dst)
    manifest=target/'Cargo.toml';text=manifest.read_text()
    if 'rustfft' not in tomllib.loads(text)['dependencies']:
        text=text.replace('[dependencies]','[dependencies]\nrustfft = { workspace = true }',1)
    manifest.write_text(text)
    exports=target/'src/models/mod.rs';text=exports.read_text()
    for name in ('fireredvad','smart_turn'):
        if f'pub mod {name};' not in text:text+=f'\npub mod {name};\n'
    exports.write_text(text)
    workspace=tomllib.loads((core/'Cargo.toml').read_text())['workspace']
    dependencies=workspace['dependencies']
    dependencies.setdefault('rustfft',tomllib.loads((ROOT.parent/'Cargo.toml').read_text())['workspace']['dependencies']['rustfft'])
    for value in dependencies.values():
        if isinstance(value,dict) and 'path' in value:
            value['path']=str((core/value['path']).resolve())
    content='[workspace]\nmembers = ["candle-transformers"]\nresolver = "2"\n'
    for section,values in [('package',workspace['package']),('dependencies',dependencies)]:
        content+=f'\n[workspace.{section}]\n'
        content+=''.join(json.dumps(k)+' = '+toml_value(v)+'\n' for k,v in values.items())
    (support/'Cargo.toml').write_text(content)
    return target

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--half-root',type=Path,required=True)
    p.add_argument('--candle-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();half=a.half_root.resolve();core=a.candle_root.resolve();work=a.output.resolve()
    work.mkdir(parents=True,exist_ok=False)
    transformers=prepare_transformers(core,work)
    sys.path.insert(0,str(half))
    spec=importlib.util.spec_from_file_location('qualified_r9_builder',half/'ci/extended_build.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    r9=builder.prepare_build(core,work)
    original=digest_sources(r9)
    probe=(r9/'src/bin/hdb2c_r9_probe.rs').read_text()
    matches=re.findall(r'cfg\["hdb2c9_tokenizer"\]=serde_json::from_str\(r###"(.*?)"###\)',probe)
    if len(matches)!=1:raise ValueError('Qualified R9 tokenizer anchor changed')
    endpoint=work/'voice_endpointing'
    copy_endpointing_crate(ROOT,endpoint)
    manifest=(endpoint/'Cargo.toml').read_text()
    for name in ('candle-core','candle-nn'):
        manifest=manifest.replace(f'path = "../{name}"',f'path = {json.dumps(str(core/name))}')
    manifest=manifest.replace('path = "../candle-transformers"',f'path = {json.dumps(str(transformers))}')
    (endpoint/'Cargo.toml').write_text(manifest)
    manifest=(r9/'Cargo.toml').read_text()
    if manifest.count('\n[features]')!=1:raise ValueError('Qualified R9 manifest anchor changed')
    manifest=manifest.replace('\n[features]',f'\ncandle-voice-endpointing = {{ path = {json.dumps(str(endpoint))} }}\n\n[features]',1)
    # Only the old hub-downloading CLI uses this dependency. The native local
    # inference binaries do not need HTTP/OpenSSL at build time or runtime.
    manifest,count=re.subn(r'^hf-hub = "([^"]+)"$',r'hf-hub = { version = "\1", optional = true }',manifest,flags=re.MULTILINE)
    if count!=1:raise ValueError('Upstream hub dependency anchor changed')
    manifest=manifest.replace('[features]\n','[features]\nupstream-hub-cli = ["dep:hf-hub"]\n',1)
    manifest+='\n[[bin]]\nname = "parakeet-candle"\npath = "src/main.rs"\nrequired-features = ["upstream-hub-cli"]\n'
    (r9/'Cargo.toml').write_text(manifest)
    code=(ROOT/'tools/hdb1_native.rs').read_text().replace('@TOKENIZER@',matches[0])
    (r9/'src/bin/hdb1_native.rs').write_text(code)
    after=digest_sources(r9)
    if any(after.get(k)!=v for k,v in original.items()):raise ValueError('Qualified R9 source changed during composition')
    subprocess.run([sys.executable,str(half/'scripts/hdb2a_cargo_lock.py'),'--manifest-path',str(r9/'Cargo.toml')],check=True)
    report={'scope':'R9_RESIDENT_WITH_NATIVE_ENDPOINTING_BUILD_PREPARATION','asr_inference_executed':False,
        'candle_sha':subprocess.check_output(['git','-C',str(core),'rev-parse','HEAD'],text=True).strip(),
        'qualified_r9_sources_unchanged':True,'r9_source_sha256':original,
        'upstream_hub_cli_dependency_optional':True,
        'voice_models_provider':'candle-transformers::models::{fireredvad,smart_turn}',
        'voice_model_sha256':{name:hashlib.sha256((transformers/'src/models'/name).read_bytes()).hexdigest()
            for name in MODEL_FILES},
        'backend_source_paths':{name:str(core/name) for name in ('candle-core','candle-nn')},
        'new_binary_sha256':hashlib.sha256(code.encode()).hexdigest(),
        'endpointing_sources_sha256':digest_sources(endpoint)}
    (work/'native-build.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NATIVE_HALF_DUPLEX_PREPARED='+str(r9/'Cargo.toml'),flush=True)

if __name__=='__main__':main()
