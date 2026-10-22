#!/usr/bin/env python3
"""Resolve only a stale pinned runtime libc Cargo.lock entry.

The source Candle revision remains unchanged. Cargo.lock belongs to the
standalone checkout of parakeet-candle, never to the user's Candle fork.
"""
import argparse
from pathlib import Path
import subprocess
import tomllib

MIN_LIBC = (0, 2, 186)
TARGET_LIBC = '0.2.191'


def libc_locked_version(path: Path):
    if not path.exists():
        return None
    data = tomllib.loads(path.read_text())
    versions = [p['version'] for p in data['package'] if p['name'] == 'libc']
    if len(versions) != 1:
        raise ValueError(f'Expected exactly one locked libc, found {versions}')
    return versions[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-path', type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest_path.resolve()
    lockfile = manifest.parent / 'Cargo.lock'
    observed = libc_locked_version(lockfile)
    if observed is None:
        print('HDB2A_LIBC_LOCK=ABSENT_CARGO_WILL_RESOLVE')
        return
    if tuple(int(x) for x in observed.split('.')[:3]) >= MIN_LIBC:
        print(f'HDB2A_LIBC_LOCK=COMPATIBLE version={observed}')
        return
    print(f'HDB2A_LIBC_LOCK=UPDATING version={observed} target={TARGET_LIBC}', flush=True)
    subprocess.run(['cargo', 'update', '--manifest-path', str(manifest),
                    '-p', 'libc', '--precise', TARGET_LIBC], check=True)
    new = libc_locked_version(lockfile)
    if not new or tuple(int(x) for x in new.split('.')[:3]) < MIN_LIBC:
        raise RuntimeError(f'libc Cargo.lock still incompatible: {new}')
    print(f'HDB2A_LIBC_LOCK=RESOLVED version={new}')


if __name__ == '__main__':
    main()
