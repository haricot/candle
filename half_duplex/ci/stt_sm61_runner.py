#!/usr/bin/env python3
"""Run one STT campaign using the existing V6.3 unified rootless GPU image.

No archive/image/profile rewrite. Reuse the unified runner's GPU lock. Resolve
the exact queued job and commit twice; keep the registration token on stdin.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import uuid

REPOSITORY = 'haricot/candle'
WORKFLOW = '.github/workflows/half-duplex-cuda-sm61.yml'
IMAGE = 'localhost/haricot/sm61-unified:v6-cuda12.9.2-ubuntu26.04'
ROOT = Path(__file__).resolve().parents[2]
GPU_CAMPAIGNS_ENABLED = False


def require_gpu_qualification():
    if not GPU_CAMPAIGNS_ENABLED:
        raise RuntimeError('GPU tests are suspended until the complete CPU parity campaign passes')


def validate(run, jobs, expected_candidate=None):
    if (run.get('path') != WORKFLOW or run.get('head_branch') != 'half_duplex'
            or run.get('event') not in ('push', 'workflow_dispatch')):
        raise ValueError('not a half_duplex STT CUDA campaign')
    if run.get('status') not in ('queued', 'in_progress') or run.get('conclusion') is not None:
        raise ValueError('campaign already finished')
    sha = run.get('head_sha', '')
    if not re.fullmatch('[0-9a-f]{40}', sha):
        raise ValueError('missing exact candidate SHA')
    if expected_candidate is not None and sha != expected_candidate:
        raise ValueError('candidate SHA changed')
    run_id, attempt = run.get('id'), run.get('run_attempt')
    if type(run_id) is not int or type(attempt) is not int or attempt < 1:
        raise ValueError('invalid run identity')
    matches = [j for j in jobs if j.get('name') == 'cuda-cudnn-sm61']
    if len(matches) != 1:
        raise ValueError('no unique STT GPU job')
    job = matches[0]
    if (job.get('status') != 'queued' or job.get('conclusion') is not None
            or job.get('run_id') != run_id or job.get('run_attempt') != attempt):
        raise ValueError('STT GPU job is not queued in the current attempt')
    labels = ['self-hosted', 'linux', 'x64', 'cuda', 'sm61', 'half-duplex-stt', f'half-duplex-stt-{run_id}']
    actual_labels = job.get('labels', [])
    normalized = [s.casefold() for s in actual_labels if isinstance(s,str)]
    if set(normalized) != set(labels) or len(actual_labels) != len(labels) or len(normalized) != len(labels):
        raise ValueError('STT job lacks its exact campaign labels; use a fresh dispatch')
    if type(job.get('id')) is not int:
        raise ValueError('missing GPU job ID')
    return {'run_id': run_id, 'run_attempt': attempt, 'candidate': sha,
            'gpu_job_id': job['id'], 'runner_labels': labels[3:]}


def resolve(run_id, expected_candidate=None):
    headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'haricot-stt-sm61'}
    token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = 'Bearer '+token
    def get(path):
        request = urllib.request.Request('https://api.github.com/repos/'+REPOSITORY+path, headers=headers)
        with urllib.request.urlopen(request, timeout=25) as response:
            return json.load(response)
    run = get(f'/actions/runs/{run_id}')
    jobs = get(f'/actions/runs/{run_id}/jobs?filter=latest&per_page=100')['jobs']
    return validate(run, jobs, expected_candidate)


def registration_token():
    # TextIO's r+ mode requests a seekable read/write buffer. /dev/tty is a
    # character device: use an output stream and let getpass open its read fd.
    with open('/dev/tty', 'w') as tty:
        if not tty.isatty():
            raise RuntimeError('interactive terminal required for registration')
        value = getpass.getpass('One-time runner registration token (hidden): ', stream=tty)
    if not value:
        raise ValueError('empty registration token')
    return value


def gh_api(path, *, payload=None):
    args = ['gh', 'api', '--hostname', 'github.com', 'repos/'+REPOSITORY+path,
            '--method', 'POST' if payload is not None else 'GET']
    if payload is not None:
        args += ['--input', '-']
    result = subprocess.run(args, input=json.dumps(payload) if payload is not None else None,
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('GitHub CLI request failed: '+result.stderr.strip())
    return json.loads(result.stdout) if result.stdout.strip() else None


def select_r12_run(runs, previous, candidate, request):
    matches = [r for r in runs if r.get('id') not in previous
        and r.get('head_sha') == candidate and r.get('head_branch') == 'half_duplex'
        and r.get('event') == 'workflow_dispatch'
        and r.get('display_title') == 'R12 LayerNorm sm61 '+request]
    if len(matches) > 1:
        raise ValueError('More than one run matches this exact R12 request; refusing automatic selection')
    return matches[0] if matches else None


def dispatch_r12(expected_candidate=None):
    """Explicit user command: queue just R12, resolve its own run, then start it."""
    require_gpu_qualification()
    local_sha = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if expected_candidate is not None and expected_candidate != local_sha:
        raise ValueError('Local checkout differs from the expected candidate')
    branch = gh_api('/git/ref/heads/half_duplex')
    if branch['object']['sha'] != local_sha:
        raise ValueError('Update the local half_duplex clone with git pull --ff-only before dispatching')
    endpoint = '/actions/workflows/half-duplex-cuda-sm61.yml'
    listing = endpoint+'/runs?branch=half_duplex&event=workflow_dispatch&per_page=30'
    previous = {r['id'] for r in gh_api(listing)['workflow_runs']}
    request = uuid.uuid4().hex
    print('R12_SCOPE=FIRST_LAYERNORM_COMMON_INPUT_ONLY_NOT_FULL_STT', flush=True)
    gh_api(endpoint+'/dispatches', payload={'ref': 'half_duplex', 'inputs': {
        'run_cuda': True, 'r8_corpus': False, 'r12_layernorm': True, 'r12_request': request}})
    print('R12_CI_DISPATCHED_CANDIDATE='+local_sha, flush=True)
    deadline = time.monotonic()+120
    while time.monotonic() < deadline:
        run = select_r12_run(gh_api(listing)['workflow_runs'], previous, local_sha, request)
        if run is not None:
            if run.get('status') == 'completed':
                raise RuntimeError('The requested GPU run already completed: '+run.get('html_url', str(run['id'])))
            jobs = gh_api(f'/actions/runs/{run["id"]}/jobs?filter=latest&per_page=100')['jobs']
            if jobs:
                selected = validate(run, jobs, local_sha)
                print('R12_CI_RUN=https://github.com/'+REPOSITORY+'/actions/runs/'+str(run['id']), flush=True)
                # The existing launcher rechecks this exact run/job and holds
                # the unified GPU lock for the full one-job container lifetime.
                start(selected['run_id'], selected['candidate'])
                return
        time.sleep(2)
    raise RuntimeError('R12 was dispatched but its job did not appear within 120 seconds; inspect Actions before dispatching again')


def start(run_id, expected_candidate):
    require_gpu_qualification()
    image = os.environ.get('HARICOT_RUNNER_IMAGE', IMAGE)
    if subprocess.check_output(['podman', 'info', '--format', '{{.Host.Security.Rootless}}'], text=True).strip() != 'true':
        raise RuntimeError('use the rootless Podman GPU user')
    subprocess.run(['podman', 'image', 'exists', image], check=True)
    state = Path(os.environ.get('XDG_STATE_HOME', str(Path.home()/'.local/state')))/'haricot-sm61-runner'
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state/'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = resolve(run_id, expected_candidate)
        common = ['--rm', '--no-hosts', '--pull=never', '--device=nvidia.com/gpu=all',
                  '--group-add=keep-groups', '--cap-drop=all', '--security-opt=no-new-privileges',
                  '--pids-limit=4096', '--tmpfs=/tmp:rw,exec,nosuid,nodev,size=2g']
        subprocess.run(['podman', 'run', *common, '--network=none', image, 'diagnose'], check=True)
        print(json.dumps(selected, indent=2), flush=True)
        print('Registration: https://github.com/'+REPOSITORY+'/settings/actions/runners', flush=True)
        registration = registration_token()
        if resolve(run_id, selected['candidate']) != selected:
            raise ValueError('campaign changed during registration')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')
        name = f'haricot-sm61-stt-{run_id}-{stamp}-{os.getpid()}'
        environment = {'RUNNER_NAME': name, 'RUNNER_REPO': REPOSITORY,
            'RUNNER_LABELS': ','.join(selected['runner_labels']), 'EXPECTED_RUN_ID': str(run_id),
            'EXPECTED_JOB_ID': str(selected['gpu_job_id']), 'EXPECTED_CANDIDATE_SHA': selected['candidate'],
            'ASD_REQUIRED': 'false', 'RUNNER_NO_DEFAULT_LABELS': 'false'}
        args = ['podman', 'run', '-i', *common, '--name', name, '--hostname', 'haricot-sm61']
        for key, value in environment.items(): args += ['--env', key+'='+value]
        args += [image, 'run']
        # The same lock remains held until this one-job container exits.
        with (state/(name+'.log')).open('w') as logfile:
            with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, pass_fds=(lock.fileno(),)) as proc:
                proc.stdin.write(registration+'\n'); proc.stdin.close(); del registration
                for line in proc.stdout:
                    sys.stdout.write(line); sys.stdout.flush(); logfile.write(line); logfile.flush()
                code = proc.wait()
        if code:
            raise RuntimeError(f'unified runner exited {code}; log: {state/(name+".log")}')


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'start', 'layernorm-ci'))
    parser.add_argument('--run-id', type=int)
    parser.add_argument('--candidate', help='Expected full commit SHA')
    a = parser.parse_args()
    if a.action == 'layernorm-ci' and a.run_id is not None:
        parser.error('layernorm-ci creates its own run; omit --run-id')
    if a.action != 'layernorm-ci' and a.run_id is None:
        parser.error('--run-id is required for check/start')
    try:
        if a.action == 'check': print(json.dumps(resolve(a.run_id, a.candidate), indent=2))
        elif a.action == 'layernorm-ci': dispatch_r12(a.candidate)
        else: start(a.run_id, a.candidate)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print('STT_RUNNER_ERROR='+str(exc), file=sys.stderr)
        sys.exit(1)
