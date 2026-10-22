"""R14: 14 regression + 28 new clips, exact decoding and FP64 local audits.

The R9 CUDA arithmetic and original numerical gates stay unchanged. Relative
tolerances are surveyed element by element, not automatically approved/fitted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from .cli import ROOT, settings, take_run_dir
from .c3_r8 import prepare_sources, execute_clip, write_json
from .c3_r9_source import original_metadata
from .c3_r9_text import prepare_text_candidate, execute_text_clip
from .c3_r14_corpus import select_expanded, coverage
from .c3_r14_precision import Fp64Auditor, encoder_survey, SIGNATURE
from .nemo_runner import load_nemo, ensure_probe_eval


def discrete_controls(result):
    """Every field is mandatory; exact text alone cannot satisfy the gate."""
    fields = ('ids', 'frames', 'durations', 'text')
    if set(result['tdt']) != {'native', 'shared_mel'}:
        raise ValueError('R14 requires both complete decoding execution modes')
    return {mode: {'outputs': {key: item['output_exact'][key] for key in fields},
                   'decisions_exact': item['decision_counts']['exact'],
                   'all_exact': item['tdt_discrete_exact'] is True and item['decision_counts']['exact'] is True
                                and all(item['output_exact'][k] is True for k in fields),
                   'counts': item['decision_counts']}
            for mode, item in result['tdt'].items()}


def cohort_summary(results):
    out = {}
    for cohort in ('regression14', 'additional28'):
        rows = [r for r in results if r['cohort'] == cohort]
        out[cohort] = {'completed': len(rows),
            'exact_decoding': sum(r['discrete_exact'] for r in rows),
            'legacy_numeric_pass': sum(r['legacy_qualification'] == 'PASS_ALL_DECLARED_NUMERIC_GATES' for r in rows),
            'fp64_audits_complete': sum(r['fp64_complete'] for r in rows),
            'final_encoder_pass': sum(r['final_encoder']['numeric_pass'] for r in rows)}
    return out


def main(corpus, *, keep_raw=False, require_numeric=False):
    s = settings()
    s['reports'].mkdir(parents=True, exist_ok=True)
    run = take_run_dir(s['reports'])
    rows = select_expanded(json.loads(corpus.read_text()))
    published = Path(os.environ.get('GITHUB_WORKSPACE', str(run)))/'halfduplex-ci-published'
    published.mkdir(parents=True, exist_ok=True)
    corpus_coverage = coverage(rows)
    write_json(published/'r14-corpus-coverage.json', corpus_coverage)
    binaries, parent = prepare_sources(s, 'cuda-cudnn')
    metadata = original_metadata(s['nemo'], s['model'])
    binary, manifest = prepare_text_candidate(binaries[1].parents[3], ROOT, metadata, 'cuda-cudnn')
    original, _ = load_nemo(s['nemo'])
    ensure_probe_eval(original)
    auditor = Fp64Auditor(original)
    del original
    results, errors = [], []
    for index, row in enumerate(rows):
        report = run/f'clip-{index:02}'
        report.mkdir()
        try:
            baseline = execute_clip(s, binaries, row, report, 'cuda-cudnn', parent)
            result = execute_text_clip(s, binary, row, report, 'cuda-cudnn', baseline, manifest)
            reference = report/'nemo'
            ref = json.loads((reference/'manifest.json').read_text())
            discrete = discrete_controls(result)
            precision, surveys = {}, {}
            for mode in ('native', 'shared_mel'):
                target = report/('r9_'+mode)
                precision[mode] = auditor.evaluate(reference, target,
                    report/'operators' if mode == 'native' else None,
                    include_reference=mode == 'native')
                surveys[mode] = encoder_survey(reference, target, ref['encoder_valid'])
            complete = all(p['execution_status'] == 'COMPLETE' for p in precision.values())
            payload = {'signature': SIGNATURE, 'clip': result['clip'], 'cohort': row['cohort'],
                'selection_index': row['selection_index'], 'variant': 'cuda-cudnn',
                'checkpoint_sha256': result['checkpoint_sha256'],
                'execution_status': 'COMPLETE' if complete else 'FP64_AUDIT_INCOMPLETE',
                'discrete_controls': discrete, 'fp64_precision': precision, 'encoder_tolerance_survey': surveys,
                'legacy_qualification': result['qualification'],
                'cross_backend_bitwise': {'status': result['global_bitwise_status'], 'acceptance_gate': False},
                'tolerance_policy': 'ORIGINAL_GATES_RETAINED_RELATIVE_SURVEY_NOT_APPROVED',
                'legacy_r9': result, 'legacy_r8b': baseline,
                'production_tensor_injection': 'NONE',
                'scope': 'OFFLINE_BATCH_1_FP32_CUDA_WITH_CPU_NEMO_AND_SEPARATE_FP64_OPERATOR_REPLAYS'}
            write_json(published/f'r14-cuda-{index:02}.json', payload)
            results.append({'clip': result['clip'], 'cohort': row['cohort'],
                'discrete_exact': all(d['all_exact'] for d in discrete.values()),
                'fp64_complete': complete, 'legacy_qualification': result['qualification'],
                'final_encoder': result['encoder']['native']['valid_final_output']})
            print(f'HDB2C3_R14_CLIP={index+1}/{len(rows)} cohort={row["cohort"]} '
                  f'exact={results[-1]["discrete_exact"]} fp64_complete={complete}', flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            item = {'index': index, 'id': row['id'], 'cohort': row['cohort'],
                    'error': type(exc).__name__+': '+str(exc)}
            errors.append(item)
            write_json(published/f'r14-cuda-{index:02}-error.json', item)
        finally:
            if not keep_raw:
                shutil.rmtree(report)
    complete = not errors and len(results) == 42 and all(r['fp64_complete'] for r in results)
    exact = complete and all(r['discrete_exact'] for r in results)
    numeric = complete and exact and all(r['legacy_qualification'] == 'PASS_ALL_DECLARED_NUMERIC_GATES'
                                        for r in results)
    summary = {'signature': SIGNATURE, 'expected_clips': 42, 'completed_clips': len(results),
        'execution_status': 'COMPLETE' if complete else 'INCOMPLETE',
        'qualification': 'PASS_ALL_DECLARED_NUMERIC_GATES' if numeric else 'NUMERIC_PARITY_NOT_ESTABLISHED',
        'discrete_exact': exact, 'cohorts': cohort_summary(results), 'coverage': corpus_coverage,
        'relative_tolerances': 'REVIEW_PENDING_SURVEY_ONLY',
        'cross_backend_bits_block_promotion': False,
        'results': results, 'errors': errors,
        'limits': 'Same public datasets, additional utterances; no speaker-disjoint, microphone, streaming or broad WER claim.'}
    write_json(published/'r14-cuda-summary.json', summary)
    write_json(run/'summary.json', summary)
    print('HDB2C3_R14='+summary['execution_status']+' '+summary['qualification'], flush=True)
    print('HDB2C3_R14_REPORT='+str(run), flush=True)
    return 1 if not complete else 2 if require_numeric and not numeric else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--variant', choices=('cuda-cudnn',), default='cuda-cudnn')
    parser.add_argument('--keep-raw', action='store_true')
    parser.add_argument('--require-numeric', action='store_true')
    args = parser.parse_args()
    raise SystemExit(main(args.corpus, keep_raw=args.keep_raw, require_numeric=args.require_numeric))
