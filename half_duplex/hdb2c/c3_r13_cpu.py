"""Full 14-clip CPU pipeline qualification with integrated ordered LayerNorm.

Original thresholds, real autonomous PCM inference and independent R7/R8
controls are retained. No CUDA path or reference-activation substitution.
"""
from __future__ import annotations

import argparse
from contextlib import chdir
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .cli import ROOT, settings, take_run_dir
from .c3_r5 import digest
from .c3_r7 import encoder_audit, trajectory_comparison, operator_comparison
from .c3_r8 import prepare_sources, selected_corpus, native, execute_clip, comparisons, files, write_json
from .c3_r9_source import original_metadata
from .c3_r9_text import strict_numeric
from .c3_r13_source import outputs

SIGNATURE = 'hdb2c3-r13-full-native-cpu-ordered-norm-v0.7.13'
ONEDNN_REVISION = '03c022d3ffdcee958cfacbe720048e725fdf644c'
SLEEF_REVISION = '5a1d179df9cf652951b59010a2d2075372d67f68'


def prepare_candidate(parent, root, metadata, python, source, build, sleef_source, sleef_build):
    sdk = json.loads(subprocess.check_output([str(python), '-c',
        'import sys,json,importlib.metadata as m;print(json.dumps({"root":sys.prefix,"mkl":m.version("mkl"),"headers":m.version("mkl-include")}))'], text=True))
    if sdk['mkl'] != '2024.2.2' or sdk['headers'] != '2024.2.2':
        raise ValueError('R13 requires the pinned standalone MKL SDK')
    sdkroot = Path(sdk['root']); source = source.resolve(); build = build.resolve()
    libraries = sorted((sdkroot/'lib').glob('libmkl*.so.2'))
    if not libraries or not (sdkroot/'include/mkl_dfti.h').is_file():
        raise ValueError('R13 MKL headers/libraries missing')
    sdk['library_sha256'] = {p.name: digest(p) for p in libraries}
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != ONEDNN_REVISION:
        raise ValueError('R13 requires the exact CPU oneDNN revision')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code'], check=True, stdout=subprocess.DEVNULL)
    cache = (build/'CMakeCache.txt').read_text()
    if any(s not in cache for s in ('DNNL_GPU_RUNTIME:STRING=NONE', 'DNNL_CPU_RUNTIME:STRING=OMP', 'CONVOLUTION', 'REORDER')):
        raise ValueError('R13 requires CPU/OMP convolution and reorder primitives')
    sdk['onednn_revision'] = ONEDNN_REVISION
    sdk['onednn_library_sha256'] = digest(build/'src/libdnnl.so.3.11')
    sleef_source = sleef_source.resolve(); sleef_build = sleef_build.resolve()
    if subprocess.check_output(['git', '-C', str(sleef_source), 'rev-parse', 'HEAD'], text=True).strip() != SLEEF_REVISION:
        raise ValueError('R13 requires the pinned standalone SLEEF revision')
    subprocess.run(['git', '-C', str(sleef_source), 'diff', '--exit-code'], check=True, stdout=subprocess.DEVNULL)
    if 'SLEEF_ENABLE_CUDA:BOOL=OFF' not in (sleef_build/'CMakeCache.txt').read_text():
        raise ValueError('R13 requires CPU-only standalone SLEEF')
    sdk['sleef_revision'] = SLEEF_REVISION
    sdk['sleef_library_sha256'] = digest(sleef_build/'lib/libsleef.so.3.8.0')
    generated = outputs(parent, root, metadata)
    names = sorted(p.relative_to(parent).as_posix() for p in (parent/'src').rglob('*.rs'))+['Cargo.toml', 'Cargo.lock']
    config = root.parent/'.cargo/config.toml'
    toolchain = {name: subprocess.check_output([name, '--version'], text=True).splitlines()[0]
                 for name in ('rustc', 'gcc', 'g++')}
    toolchain['cargo_config_sha256'] = digest(config) if config.is_file() else None
    toolchain['rustflags'] = {name: os.environ[name] for name in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS')
                              if name in os.environ}
    manifest = {'signature': SIGNATURE, 'sdk': sdk, 'toolchain': toolchain, 'tokenizer_metadata': metadata,
                'parent_source_sha256': {n: digest(parent/n) for n in names},
                'target_sha256': {n: hashlib.sha256(c.encode()).hexdigest() for n, c in generated.items()}}
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    target = parent.with_name(parent.name+'-r13-cpu-'+identity)
    if target.exists():
        stored = json.loads((target/'R13-CPU-MANIFEST.json').read_text())
        if {k: stored[k] for k in manifest} != manifest:
            raise ValueError('R13 cached source/SDK manifest differs')
        for name, sha in {**manifest['parent_source_sha256'], **manifest['target_sha256']}.items():
            if digest(target/name) != sha:
                raise ValueError('R13 cached candidate changed: '+name)
    else:
        shutil.copytree(parent, target, ignore=shutil.ignore_patterns('target', '.git', '__pycache__', '.venv'))
        for name, content in generated.items():
            path = target/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
        write_json(target/'R13-CPU-MANIFEST.json', manifest)
    # Only R13 variants share this build cache. Baseline artifacts stay isolated.
    cargo_target = parent.parent/'r13-cargo-build'
    env = {**os.environ, 'CARGO_TARGET_DIR': str(cargo_target), 'HDB2C_MKL_ROOT': str(sdkroot),
           'HDB2C_ONEDNN_BUILD': str(build), 'HDB2C_ONEDNN_INCLUDE': str(source/'include'),
           'HDB2C_SLEEF_BUILD': str(sleef_build)}
    for command in (['build', '--bin', 'hdb2c_r13_probe'], ['test', '--lib', 'hdb2c']):
        subprocess.run(['cargo', command[0], '--release', '--manifest-path', str(target/'Cargo.toml'), *command[1:]],
                       cwd=root, env=env, check=True)
    binary = target/'bin/hdb2c_r13_probe'; binary.parent.mkdir(exist_ok=True)
    shutil.copy2(cargo_target/'release/hdb2c_r13_probe', binary)
    manifest['binary_sha256'] = digest(binary)
    write_json(target/'R13-CPU-MANIFEST.json', manifest)
    for name, sha in manifest['parent_source_sha256'].items():
        if digest(parent/name) != sha:
            raise ValueError('R13 build changed an independent baseline source')
    public = {**manifest, 'sdk': {k: v for k, v in sdk.items() if k != 'root'}}
    return binary, public


def execute_candidate(s, binary, row, report, baseline, manifest):
    reference = report/'nemo'; ref = json.loads((reference/'manifest.json').read_text())
    for mode in ('native', 'shared_mel', 'operators', 'tdt_operators'):
        native(binary, s['model'], Path(row['wav']), report/('r13_'+mode), 'cpu',
               reference=reference if mode != 'native' else None, frontend_trace=mode == 'native',
               operators=mode == 'operators', tdt=mode == 'tdt_operators')
    target = report/'r13_native'
    if not files(reference, target, 'pcm')['bitwise_equal']:
        raise ValueError('R13 autonomous PCM identity changed')
    if not files(reference, report/'r13_shared_mel', 'mel')['bitwise_equal']:
        raise ValueError('R13 diagnostic shared-Mel identity changed')
    result = comparisons(reference, target, report/'r13_operators')
    encoder = {mode: encoder_audit(reference/'encoder_trace', report/('r13_'+mode), ref['encoder_valid'])
               for mode in ('native', 'shared_mel')}
    tdt = {mode: trajectory_comparison(reference, report/('r13_'+mode), ref) for mode in ('native', 'shared_mel')}
    common = operator_comparison(reference, report/'r13_tdt_operators', ref)
    numeric = strict_numeric({**result, 'common_input_tdt': common}, encoder, tdt)
    bits = (all(t['bitwise_equal'] for t in result['frontend']) and encoder['native']['bitwise_equal']
            and tdt['native']['bitwise_equal'] and tdt['native']['tdt_discrete_exact'])
    public_tdt = {}
    for mode, item in tdt.items():
        public_tdt[mode] = {k: v for k, v in item.items() if k not in ('candidate_output', 'decisions')}
        public_tdt[mode]['decision_counts'] = {k: v for k, v in item['decisions'].items() if k != 'first_difference'}
    return {**result, 'signature': SIGNATURE, 'revision': 'HDB2-C3-A-R13-FULL-CPU', 'clip': baseline['clip'],
            'execution_status': 'REAL_FULL_CPU_PIPELINE_COMPLETE',
            'qualification': 'PASS_ALL_DECLARED_NUMERIC_GATES' if numeric else 'NUMERIC_PARITY_NOT_ESTABLISHED',
            'global_bitwise_status': 'PASS_BITWISE' if bits else 'FAIL_DIFFERENT_F32_BITS',
            'encoder': encoder, 'tdt': public_tdt, 'common_input_tdt': common,
            'cpu_backend': json.loads((target/'cpu_backend.json').read_text()),
            'checkpoint_sha256': baseline['checkpoint_sha256'], 'source_manifest': manifest,
            'independent_r7_r8b_control': baseline['instrumentation_control'],
            'native_pipeline_internal_tensor_injection': 'NONE', 'gpu': 'NOT_RUN',
            'limits': 'CPU FP32 offline batch 1; full declared stage gates, no threshold relaxation.'}


def campaign_summary(results, errors, manifest):
    pairs = [(r['clip']['split'], r['clip']['emotion']) for r in results]
    from .c3_r8 import EMOTIONS
    expected = {(split, emotion) for split in ('cafe', 'oreau') for emotion in EMOTIONS}
    complete = (len(results) == 14 and not errors and set(pairs) == expected
                and len({r['clip']['pcm_sha256'] for r in results}) == 14)
    qualified = complete and all(r['qualification'] == 'PASS_ALL_DECLARED_NUMERIC_GATES'
        and r['text_exact'] and r['tdt_exact_numeric'] and r['final_encoder']['numeric_pass']
        and not any(r['failed_encoder_stages'].values()) for r in results)
    return {'signature': SIGNATURE, 'scope': 'FULL_CPU_PIPELINE_14_REAL_CLIPS', 'expected_clips': 14,
        'completed_clips': len(results), 'execution_status': 'COMPLETE' if complete else 'INCOMPLETE',
        'qualification': 'PASS_ALL_DECLARED_NUMERIC_GATES' if qualified else 'NUMERIC_PARITY_NOT_ESTABLISHED',
        'global_bitwise_status': ('NOT_ESTABLISHED_INCOMPLETE' if not complete else
            'PASS_BITWISE' if all(r['global_bitwise_status'] == 'PASS_BITWISE' for r in results) else 'FAIL_DIFFERENT_F32_BITS'),
        'numeric_threshold_policy': 'ORIGINAL_R7_R8B_UNCHANGED',
        'native_pipeline_internal_tensor_injection': 'NONE',
        'results': results, 'errors': errors, 'gpu': 'NOT_RUN', 'microphone': 'NOT_RUN', 'source_manifest': manifest}


def main(corpus, source, build, sleef_source, sleef_build, require_numeric=False, keep_raw=False):
    s = settings(); root = take_run_dir(s['reports']); rows = selected_corpus(json.loads(corpus.read_text()))
    # Cargo reads configuration from the process cwd, even with --manifest-path.
    # Keep the independent baseline and candidate on the repository CPU profile.
    with chdir(ROOT):
        binaries, parent_manifest = prepare_sources(s, 'cpu')
    hashes = {str(p): digest(p) for p in binaries}
    binary, manifest = prepare_candidate(binaries[1].parents[3], ROOT,
        original_metadata(s['nemo'], s['model']), s['python'], source, build, sleef_source, sleef_build)
    if any(digest(Path(p)) != sha for p, sha in hashes.items()):
        raise ValueError('R13 build changed an independent baseline executable')
    published = Path(os.environ.get('GITHUB_WORKSPACE', str(root)))/'halfduplex-ci-published'
    published.mkdir(parents=True, exist_ok=True); results = []; errors = []
    print('HDB2C3_R13_REPORT='+str(root), flush=True)
    for index, row in enumerate(rows):
        report = root/f'clip-{index:02}'; report.mkdir()
        try:
            baseline = execute_clip(s, binaries, row, report, 'cpu', parent_manifest)
            write_json(published/f'r8b-r13-control-cpu-{index:02}.json', baseline)
            result = execute_candidate(s, binary, row, report, baseline, manifest)
            write_json(published/f'r13-cpu-{index:02}.json', result)
            results.append({'clip': result['clip'], 'qualification': result['qualification'],
                'global_bitwise_status': result['global_bitwise_status'],
                'failed_encoder_stages': {mode: [{'name': t['name'], 'max_abs': t['max_abs']}
                    for t in enc['stages'] if not t['numeric_pass']] for mode, enc in result['encoder'].items()},
                'text_exact': all(t['output_exact']['text'] for t in result['tdt'].values()),
                'tdt_exact_numeric': all(t['tdt_discrete_exact'] and t['numeric_pass'] for t in result['tdt'].values()),
                'final_encoder': result['encoder']['native']['valid_final_output']})
            print(f'HDB2C3_R13_CLIP={index+1}/14 '+result['qualification'], flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc(); errors.append({'index': index, 'id': row['id'], 'error': type(exc).__name__})
            write_json(published/f'r13-cpu-{index:02}-error.json', errors[-1])
        finally:
            if not keep_raw:
                shutil.rmtree(report)
    summary = campaign_summary(results, errors, manifest)
    write_json(published/'r13-cpu-summary.json', summary); write_json(root/'summary.json', summary)
    print('HDB2C3_R13='+summary['execution_status']+' '+summary['qualification'], flush=True)
    return 1 if errors else 2 if require_numeric and summary['qualification'] != 'PASS_ALL_DECLARED_NUMERIC_GATES' else 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--onednn-source', type=Path, required=True); p.add_argument('--onednn-build', type=Path, required=True)
    p.add_argument('--sleef-source', type=Path, required=True); p.add_argument('--sleef-build', type=Path, required=True)
    p.add_argument('--variant', choices=('cpu',), default='cpu')
    p.add_argument('--keep-raw', action='store_true'); p.add_argument('--require-numeric', action='store_true')
    a = p.parse_args()
    sys.exit(main(a.corpus, a.onednn_source, a.onednn_build, a.sleef_source, a.sleef_build, a.require_numeric, a.keep_raw))
