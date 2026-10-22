#!/usr/bin/env python3
"""Choose short/full/audit from the immutable candidate's actual engine diff.

Uses the last promoted R15 composition, or the reviewed R14 functional baseline
for migration. A requested short profile cannot downgrade an engine change.
No network, model inference, NumPy or PyTorch is needed by this resolver.
"""
import argparse
import json
from pathlib import Path
import subprocess

R14_BASELINE = '50ed7c1636e93fa5c8bf0a16e90bcce6a5ad7cc1'
MARKER = 'half_duplex/ci/r15-policy.json'
ENGINE_PREFIXES = ('candle-core/src/', 'candle-nn/src/', 'candle-kernels/',
                   'candle-metal-kernels/', 'candle-ug/', 'half_duplex/rust/')
ENGINE_FILES = {
    'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'rust-toolchain',
    'candle-core/Cargo.toml', 'candle-core/build.rs', 'candle-nn/Cargo.toml',
    'half_duplex/scripts/prepare_candle_runtime.py',
    'half_duplex/scripts/hdb2a_conformer_patch.py',
    'half_duplex/scripts/convert_nemo_hdb2a.py',
    'half_duplex/scripts/hdb2a_model_config.py',
    'half_duplex/scripts/hdb2a_cargo_lock.py', 'half_duplex/ci/extended_build.py',
    'half_duplex/run-hdb2c.sh',
    'half_duplex/ci/r7_verified_native_sources.json',
    'half_duplex/ci/results/r13-ci-full-cpu-20261021.json',
    *('half_duplex/hdb2c/'+name+'.py' for name in (
        'c2', 'c3', 'c3_r3', 'c3_r3c', 'c3_r4', 'c3_r5', 'c3_r6', 'c3_r7',
        'c3_r7_focused', 'c3_r8_source', 'c3_r9_source')),
}


def affects_engine(path):
    return path in ENGINE_FILES or path.startswith(ENGINE_PREFIXES)


def choose(requested, changed, *, baseline_available=True):
    if requested not in ('auto', 'short', 'full', 'audit'):
        raise ValueError('Unknown Half Duplex CI profile: '+requested)
    affected = sorted(p for p in changed if affects_engine(p))
    if requested == 'audit':
        return 'audit', affected, 'EXPLICIT_FP64_AUDIT'
    if requested == 'full':
        return 'full', affected, 'EXPLICIT_COMPLETE_CAMPAIGN'
    if not baseline_available:
        return 'full', affected, 'NO_QUALIFIED_BASELINE'
    if affected:
        return 'full', affected, 'ENGINE_CHANGED_SHORT_CANNOT_DOWNGRADE'
    return 'short', affected, 'QUALIFIED_ENGINE_UNCHANGED'


def git(repo, *args, check=True):
    return subprocess.run(['git', '-C', str(repo), *args], check=check, capture_output=True)


def resolve(repo, candidate, baseline, requested):
    candidate = git(repo, 'rev-parse', '--verify', candidate+'^{commit}').stdout.decode().strip()
    selected = None
    if baseline:
        marker = git(repo, 'show', baseline+':'+MARKER, check=False)
        if marker.returncode == 0:
            policy = json.loads(marker.stdout)
            if policy.get('schema_version') != 1 or policy.get('policy') != 'r15-functional':
                raise ValueError('Unsupported promoted Half Duplex CI policy')
            selected = baseline
    if selected is None and git(repo, 'cat-file', '-e', R14_BASELINE+'^{commit}', check=False).returncode == 0:
        selected = R14_BASELINE
    if selected:
        selected = git(repo, 'rev-parse', '--verify', selected+'^{commit}').stdout.decode().strip()
        # --no-renames retains both sides of a move/delete in the path classifier.
        changed = [p.decode() for p in git(repo, 'diff', '--no-renames', '--name-only', '-z',
                                         selected, candidate).stdout.split(b'\0') if p]
    else:
        changed = []
    profile, affected, reason = choose(requested, changed, baseline_available=selected is not None)
    return {'schema_version': 1, 'policy': 'r15-functional', 'profile': profile,
            'requested': requested, 'reason': reason, 'baseline_sha': selected,
            'evaluated_candidate_sha': candidate, 'engine_changed_paths': affected,
            'short_clips': 4, 'full_clips': 42,
            'fp64_audit': profile == 'audit',
            'blocking': ['exact_outputs_and_decisions', 'finite_observed_tensors',
                         'device_and_lengths', 'original_final_encoder_limits'],
            'intermediate_numeric_and_bits': 'DIAGNOSTIC'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--candidate', default='HEAD')
    parser.add_argument('--baseline', default='')
    parser.add_argument('--requested', choices=('auto', 'short', 'full', 'audit'), default='auto')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = resolve(args.repository, args.candidate, args.baseline, args.requested)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('HALF_DUPLEX_CI_PROFILE='+result['profile']+' reason='+result['reason'])
