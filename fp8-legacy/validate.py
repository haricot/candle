#!/usr/bin/env python3
"""Standalone source/CPU/hosted CUDA validation, with explicit operator-run GPU phase."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
META = json.loads((ROOT / 'fp8-legacy/sources.json').read_text())
GIT = ['git', '-c', f'safe.directory={ROOT}']

def run(args):
    print('+ ' + ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)

def sources():
    run(GIT + ['merge-base', '--is-ancestor', META['base'], 'HEAD'])
    changed = subprocess.check_output(GIT + ['diff', '--name-only', META['base']], cwd=ROOT, text=True).splitlines()
    assert set(changed) <= set(META['payload_sha256']) | {'fp8-legacy/sources.json'}, 'Unexpected source delta'
    for path, digest in META['payload_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    print('FP8_LEGACY_SOURCES=PASS', flush=True)

def compile_tests(args):
    """Use build-script receipts for this invocation, excluding stale PTX from other builds."""
    print('+ ' + ' '.join(args), flush=True)
    proc = subprocess.Popen(args + ['--message-format=json-render-diagnostics'], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    directories = set()
    for line in proc.stdout:
        msg = json.loads(line)
        if msg.get('reason') == 'build-script-executed' and 'candle-kernels' in msg.get('package_id', ''):
            directories.add(Path(msg['out_dir']))
        if msg.get('reason') == 'compiler-artifact' and msg.get('executable'):
            print('Executable: ' + msg['executable'], flush=True)
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, args)
    assert directories, 'No current candle-kernels build receipt'
    cc = os.environ['CUDA_COMPUTE_CAP']
    assert cc in META['compile_targets'], f'Unlisted CUDA compilation target {cc}'
    for directory in sorted(directories):
        ptx = (directory / 'quantized.ptx').read_text()
        ptx = re.sub(r'/\*.*?\*/|//[^\n]*', '', ptx, flags=re.S)
        assert re.search(r'\.target\s+sm_' + re.escape(cc) + r'\b', ptx), f'PTX target mismatch: {directory}'
        for symbol in META['required_ptx']:
            assert re.search(r'\.entry\s+' + re.escape(symbol) + r'\s*\(', ptx), f'Missing entry {symbol}: {directory}'
        print(f'FP8_LEGACY_PTX=PASS CUDA_CC={cc} OUT={directory}', flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['sources', 'cpu', 'cuda-compile', 'gpu'])
    phase = parser.parse_args().phase
    sources()
    if phase == 'sources': return
    shutil.copyfile(ROOT / 'fp8-legacy/Cargo.lock', ROOT / 'Cargo.lock')
    run(['rustc', '--version'])
    targets = [arg for name in META['tests'] for arg in ['--test', name]]
    cargo = ['cargo', 'test', '--locked', '-p', 'candle-core', '--no-default-features']
    if phase == 'cpu':
        run(['cargo', 'check', '--locked', '-p', 'candle-core', '-p', 'candle-nn', '--no-default-features'])
        run(cargo + ['--lib'] + targets)
    else:
        features = ['--features', 'cuda,cuda-legacy-fp8']
        if phase == 'cuda-compile':
            run(['nvcc', '--version'])
            run(['cargo', 'check', '--locked', '-p', 'candle-core', '-p', 'candle-nn', '--no-default-features'] + features)
            compile_tests(cargo + features + targets + ['--no-run'])
        else:
            run(['nvidia-smi', '--query-gpu=name,compute_cap,driver_version', '--format=csv'])
            run(cargo + ['--release'] + features + targets + ['--', '--nocapture', '--test-threads=1'])
            run(cargo + ['--release'] + features + ['--test', 'fp8_w8abf16', 'packed_fp8_f16_bf16_range_residency_layout_and_modes', '--', '--ignored', '--nocapture', '--test-threads=1'])
    print('FP8_LEGACY_' + phase.upper().replace('-', '_') + '=PASS', flush=True)

if __name__ == '__main__': main()
