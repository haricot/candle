#!/usr/bin/env python3
"""Materialize the qualified pre-extraction Rust adapter, without model runtimes."""
import argparse
import json
from pathlib import Path
import subprocess

BASELINE='fced4317d64715174c32f48bc3bc9211f514c342'
REPO=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candle-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    names=subprocess.check_output(['git','-C',str(REPO),'ls-tree','-r','--name-only',BASELINE,'voice_endpointing/src'],text=True).splitlines()
    names+=['voice_endpointing/'+n for n in ('Cargo.toml','Cargo.lock','NOTICE.md')]
    for name in names:
        target=a.output/Path(name).relative_to('voice_endpointing')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(subprocess.check_output(['git','-C',str(REPO),'show',BASELINE+':'+name]))
    manifest=a.output/'Cargo.toml';text=manifest.read_text()
    for name in ('candle-core','candle-nn'):
        text=text.replace(f'path = "../{name}"',f'path = {json.dumps(str(a.candle_root.resolve()/name))}')
    manifest.write_text(text)
    print('BASELINE_RUST_SOURCE='+BASELINE)

if __name__=='__main__':main()
