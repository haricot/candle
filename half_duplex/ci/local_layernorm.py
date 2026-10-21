#!/usr/bin/env python3
"""Run the isolated R12 operator locally in the existing unified image, outside CI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys

IMAGE = 'localhost/haricot/sm61-unified:v6-cuda12.9.2-ubuntu26.04'
ROOT = Path(__file__).resolve().parents[2]


def command(image, source, work, stamp, backend):
    args = ['podman', 'run', '--rm', '--pull=never', '--no-hosts',
            '--userns=keep-id:uid=20000,gid=20000', '--user=20000:20000',
            '--group-add=keep-groups', '--cap-drop=all', '--security-opt=no-new-privileges',
            '--pids-limit=4096', '--tmpfs=/tmp:rw,exec,nosuid,nodev,size=2g',
            '--name', 'haricot-layernorm-'+stamp,
            '--volume', str(source)+':/source:ro', '--volume', str(work)+':/work:rw',
            '--workdir', '/source', '--entrypoint', '/bin/bash',
            '--env', 'RUNNER_TEMP=/work', '--env', 'XDG_CACHE_HOME=/work/cache',
            '--env', 'HDB2C_R12_RUN_DIR=/work/runs/'+stamp]
    if backend == 'cuda':
        args += ['--device=nvidia.com/gpu=all']
    args += [image, 'half_duplex/ci/r12_layernorm.sh', backend]
    return args


def run(backend, work, image):
    if subprocess.check_output(['podman', 'info', '--format', '{{.Host.Security.Rootless}}'], text=True).strip() != 'true':
        raise RuntimeError('Use the rootless Podman account with the existing unified image')
    subprocess.run(['podman', 'image', 'exists', image], check=True)
    work = work.expanduser().resolve(); work.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid())
    folder = work/'runs'/stamp; folder.mkdir(parents=True, exist_ok=False)
    state = Path(os.environ.get('XDG_STATE_HOME', str(Path.home()/'.local/state')))/'haricot-sm61-runner'
    state.mkdir(parents=True, exist_ok=True)
    lock = (state/'gpu.lock').open('a') if backend == 'cuda' else (work/'cpu.lock').open('a')
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('The GPU runner lock is held; finish the other GPU process first'
                               if backend == 'cuda' else 'A local CPU LayerNorm campaign is already running')
        # Fail before downloads on a missing physical GPU or toolchain. No registration.
        if backend == 'cuda':
            subprocess.run(['podman', 'run', '--rm', '--pull=never', '--network=none',
                            '--device=nvidia.com/gpu=all', '--group-add=keep-groups',
                            image, 'diagnose'], check=True)
        args = command(image, ROOT, work, stamp, backend)
        print('R12_SCOPE=FIRST_LAYERNORM_COMMON_INPUT_ONLY_NOT_FULL_STT', flush=True)
        print('R12_BACKEND='+backend, flush=True)
        with (folder/'console.log').open('w') as log:
            with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, pass_fds=(lock.fileno(),)) as process:
                for line in process.stdout:
                    sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
                status = process.wait()
        print('R12_REPORT='+str(folder/'result/report.json'), flush=True)
        print('R12_LOG='+str(folder/'console.log'), flush=True)
        return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--work-dir', type=Path, default=Path.home()/'.cache/haricot-stt-layernorm')
    parser.add_argument('--image', default=os.environ.get('HARICOT_RUNNER_IMAGE', IMAGE))
    a = parser.parse_args()
    try:
        sys.exit(run(a.device, a.work_dir, a.image))
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print('R12_LOCAL_ERROR='+str(exc), file=sys.stderr)
        sys.exit(1)
