"""A full CPU pass requires every original gate and all 14 unique real clips."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import unittest

from hdb2c.c3_r8 import EMOTIONS
from hdb2c.c3_r13_cpu import campaign_summary

ROOT = Path(__file__).resolve().parents[3]


class FullCpuQualificationTests(unittest.TestCase):
    def rows(self):
        return [{'clip': {'split': split, 'emotion': emotion, 'pcm_sha256': f'{i:064x}'},
            'qualification': 'PASS_ALL_DECLARED_NUMERIC_GATES',
            'global_bitwise_status': 'FAIL_DIFFERENT_F32_BITS',
            'text_exact': True, 'tdt_exact_numeric': True,
            'failed_encoder_stages': {'native': [], 'shared_mel': []},
            'final_encoder': {'numeric_pass': True}}
            for i, (split, emotion) in enumerate((s, e) for s in ('cafe', 'oreau') for e in EMOTIONS)]

    def test_numeric_and_bitwise_claims_remain_separate(self):
        report = campaign_summary(self.rows(), [], {})
        self.assertEqual(report['qualification'], 'PASS_ALL_DECLARED_NUMERIC_GATES')
        self.assertEqual(report['gpu'], 'NOT_RUN')
        self.assertTrue(all(r['global_bitwise_status'] == 'FAIL_DIFFERENT_F32_BITS' for r in report['results']))

    def test_any_missing_clip_or_failed_gate_blocks_full_pass(self):
        for failure in ('missing', 'duplicate_pair', 'duplicate_pcm', 'execution_error',
                        'frontend', 'text', 'tdt', 'encoder', 'final'):
            rows = copy.deepcopy(self.rows()); errors = []
            if failure == 'missing': rows.pop()
            elif failure == 'duplicate_pair': rows[-1]['clip'] = copy.deepcopy(rows[0]['clip'])
            elif failure == 'duplicate_pcm': rows[-1]['clip']['pcm_sha256'] = rows[0]['clip']['pcm_sha256']
            elif failure == 'execution_error': errors.append({'error': 'RuntimeError'})
            elif failure == 'frontend': rows[-1]['qualification'] = 'NUMERIC_PARITY_NOT_ESTABLISHED'
            elif failure == 'text': rows[-1]['text_exact'] = False
            elif failure == 'tdt': rows[-1]['tdt_exact_numeric'] = False
            elif failure == 'encoder': rows[-1]['failed_encoder_stages']['native'] = [{'name': 'block0_ff1'}]
            elif failure == 'final': rows[-1]['final_encoder']['numeric_pass'] = False
            with self.subTest(failure=failure):
                self.assertEqual(campaign_summary(rows, errors, {})['qualification'], 'NUMERIC_PARITY_NOT_ESTABLISHED')

    def test_gpu_variant_rejected_before_bootstrap_or_model_access(self):
        result = subprocess.run(['bash', str(ROOT/'half_duplex/ci/full_cpu.sh')],
            env={**os.environ, 'HDB2C_R13_CPU_FULL': '1', 'HDB2C_R8_VARIANT': 'cuda-cudnn'},
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stderr.strip(), 'R13_CPU_ONLY')
        result = subprocess.run([sys.executable, '-m', 'hdb2c.c3_r13_cpu',
            '--corpus', '/missing', '--onednn-source', '/missing', '--onednn-build', '/missing',
            '--sleef-source', '/missing', '--sleef-build', '/missing',
            '--variant', 'cuda-cudnn'], cwd=ROOT/'half_duplex', capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)
        self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
