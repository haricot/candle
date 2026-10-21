"""Prevent the observed hosts-file failure and accidental full-model GPU routing."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    loaded = importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded)
    return loaded


local = module('local_layernorm_r12_ci', 'half_duplex/ci/local_layernorm.py')
runner = module('stt_runner_r12_ci', 'half_duplex/ci/stt_sm61_runner.py')
verifier = module('verify_r12_cpu', 'half_duplex/ci/verify_r12_cpu.py')


class R12CiTests(unittest.TestCase):
    def test_diagnose_and_execution_both_disable_generated_hosts(self):
        preflight = local.diagnose_command(local.IMAGE)
        execution = local.command(local.IMAGE, Path('/repo'), Path('/cache'), 'test', 'cuda')
        self.assertIn('--no-hosts', preflight)
        self.assertIn('--no-hosts', execution)
        self.assertIn('--network=none', preflight)
        self.assertEqual(preflight[-2:], [local.IMAGE, 'diagnose'])

    def test_gpu_profile_is_manual_and_excludes_full_model_steps(self):
        w = yaml.load((ROOT/'.github/workflows/half-duplex-cuda-sm61.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(set(w['on']), {'workflow_dispatch'})
        self.assertEqual(w['on']['workflow_dispatch']['inputs']['r12_layernorm']['default'], 'false')
        job = w['jobs']['cuda-cudnn-sm61']
        self.assertEqual(job['if'], "${{ github.event_name == 'workflow_dispatch' && inputs.run_cuda }}")
        steps = job['steps']
        isolated = next(s for s in steps if s.get('run') == 'bash half_duplex/ci/r12_layernorm.sh cuda')
        self.assertEqual(isolated['if'], '${{ inputs.r12_layernorm }}')
        full_model = [s for s in steps if 'full_cpu.sh' in s.get('run', '') or 'install_update.sh' in s.get('run', '')]
        self.assertEqual(len(full_model), 2)
        self.assertTrue(all('!inputs.r12_layernorm &&' in s['if'] for s in full_model))
        evidence = next(s for s in steps if s.get('uses') == 'actions/download-artifact@v4')
        self.assertEqual(evidence['with']['run-id'], '39649356383')
        verify_index = next(i for i, s in enumerate(steps) if 'verify_r12_cpu.py' in s.get('run', ''))
        self.assertLess(verify_index, steps.index(isolated))

    def test_recorded_cpu_report_passes_unchanged_core_gate(self):
        report = json.loads((ROOT/'half_duplex/ci/results/r12-local-cpu-20261021.json').read_text())
        self.assertEqual(verifier.verify(report)['status'], 'PASS_ACTUAL_CPU_14_AND_UNCHANGED_OPERATOR')
        for mutate in (
            lambda r: r.update(completed_clips=13),
            lambda r: r.update(backend_requested='cuda'),
            lambda r: r['metadata']['source_sha256'].update({'hdb2c/layernorm_operator.py': '0'*64}),
            lambda r: r['results'][0]['cpu_vs_nemo'].update(bitwise_equal=False),
            lambda r: r['results'][0]['instrumentation_control'].update(mismatching_f32_words=1),
            lambda r: r['results'][0]['clip'].update(pcm_sha256=r['results'][1]['clip']['pcm_sha256']),
        ):
            broken = copy.deepcopy(report); mutate(broken)
            with self.assertRaises(ValueError):
                verifier.verify(broken)

    def test_run_selection_correlates_exact_request_candidate_and_new_identity(self):
        run = {'id': 123, 'head_sha': 'a'*40, 'head_branch': 'half_duplex',
               'event': 'workflow_dispatch', 'display_title': 'R12 LayerNorm sm61 request'}
        other = {**run, 'id': 124, 'display_title': 'R12 LayerNorm sm61 concurrent'}
        self.assertEqual(runner.select_r12_run([other, run], set(), 'a'*40, 'request'), run)
        self.assertIsNone(runner.select_r12_run([run], {123}, 'a'*40, 'request'))
        self.assertIsNone(runner.select_r12_run([run], set(), 'b'*40, 'request'))
        with self.assertRaises(ValueError):
            runner.select_r12_run([run, {**run, 'id': 125}], set(), 'a'*40, 'request')

    def test_dispatch_payload_explicitly_requests_only_r12(self):
        calls = []
        def api(path, *, payload=None):
            calls.append((path, payload))
            if path == '/git/ref/heads/half_duplex':
                return {'object': {'sha': 'a'*40}}
            if path.endswith('/dispatches'):
                raise RuntimeError('STOP_AFTER_RECORDED_POST')
            return {'workflow_runs': []}
        with patch.object(runner, 'gh_api', side_effect=api), \
                patch.object(runner.subprocess, 'check_output', return_value='a'*40):
            with self.assertRaisesRegex(RuntimeError, 'STOP_AFTER_RECORDED_POST'):
                runner.dispatch_r12()
        endpoint, payload = calls[-1]
        self.assertTrue(endpoint.endswith('/dispatches'))
        self.assertEqual(payload['ref'], 'half_duplex')
        self.assertEqual({k: v for k, v in payload['inputs'].items() if k != 'r12_request'},
                         {'run_cuda': True, 'r8_corpus': False, 'r12_layernorm': True})
        self.assertEqual(len(payload['inputs']['r12_request']), 32)

    def test_stale_local_head_is_rejected_before_dispatch(self):
        with patch.object(runner, 'gh_api', return_value={'object': {'sha': 'b'*40}}) as api, \
                patch.object(runner.subprocess, 'check_output', return_value='a'*40):
            with self.assertRaisesRegex(ValueError, 'git pull --ff-only'):
                runner.dispatch_r12()
        self.assertEqual(api.call_count, 1)

    def test_existing_start_and_check_still_require_run_id(self):
        path = str(ROOT/'half_duplex/ci/stt_sm61_runner.py')
        for action in ('start', 'check'):
            p = subprocess.run([sys.executable, path, action], capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertIn('--run-id is required', p.stderr)


if __name__ == '__main__':
    unittest.main()
