"""Install additional owned Rust Cargo bin; never patch production main.rs or core."""
import argparse
import hashlib
from pathlib import Path
import subprocess

PIN='53c0d3065f9dd8de0356b25391e60ed3a471192b'
# Original HDB2-C v0.6.0-v0.6.3 probe installed before this Rust E0596 fix.
LEGACY_PROBE_SHA256='de961e68a77e600fe44bc06aa08c9e8ea344011565e757b0fae9338f8babe9f2'

def install(source:Path,runtime:Path):
    if not (runtime/'Cargo.toml').is_file():
        raise FileNotFoundError(f'Expected existing HDB2A runtime checkout: {runtime}')
    sha=subprocess.run(['git','-C',str(runtime),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    if sha!=PIN:
        raise ValueError(f'Unexpected runtime Git HEAD {sha} != pinned {PIN}')
    if not (runtime/'src/main.rs').read_text().count('HDB2A_CANDLE_LOCAL_MODEL'):
        raise ValueError('Local HDB2A model loader missing: original runtime not bootstrapped')
    dst=runtime/'src/bin/hdb2c_probe.rs'
    new_bytes=source.read_bytes()
    old_digest=hashlib.sha256(dst.read_bytes()).hexdigest() if dst.exists() else None
    new_digest=hashlib.sha256(new_bytes).hexdigest()
    if old_digest is not None and old_digest not in (LEGACY_PROBE_SHA256,new_digest):
        raise RuntimeError(f'Existing {dst} differs from both known HDB2C probe versions; refuse to overwrite unreviewed code')
    dst.parent.mkdir(parents=True,exist_ok=True)
    if old_digest == new_digest:
        print('HDB2C_PROBE_SOURCE=INSTALLED_IDEMPOTENTLY')
    else:
        # Update the *owned additional probe source*, never the production main.rs.
        # Replace atomically after verifying that the old file is the known buggy version.
        temp=dst.with_name(dst.name+'.hdb2c-installing')
        if temp.exists():
            raise RuntimeError(f'Stale probe staging file, refusing overwrite: {temp}')
        try:
            temp.write_bytes(new_bytes)
            temp.replace(dst)
        finally:
            temp.unlink(missing_ok=True)
        print('HDB2C_PROBE_SOURCE=REPLACED_KNOWN_E0596_V0_6_3' if old_digest else 'HDB2C_PROBE_SOURCE=INSTALLED')
    return dst


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime-dir',type=Path,required=True)
    p.add_argument('--probe-source',type=Path,required=True)
    a=p.parse_args();install(a.probe_source,a.runtime_dir)
if __name__=='__main__':main()
