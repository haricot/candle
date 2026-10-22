"""CPU-first campaign boundary: no implicit GPU queue or R10 CUDA route."""
from pathlib import Path
import subprocess
import sys
import unittest
import yaml

ROOT=Path(__file__).resolve().parents[3]


class CpuFirstTests(unittest.TestCase):
    def test_gpu_workflow_requires_manual_dispatch(self):
        workflow=yaml.load((ROOT/'.github/workflows/half-duplex-cuda-sm61.yml').read_text(),Loader=yaml.BaseLoader)
        self.assertEqual(set(workflow['on']),{'workflow_dispatch'})
        self.assertEqual(workflow['jobs']['cuda-cudnn-sm61']['if'],
            "${{ false }}")

    def test_r10_rejects_cuda_before_opening_corpus_or_sdk(self):
        p=subprocess.run([sys.executable,'-m','hdb2c.c3_r10_cpu',
            '--corpus','does-not-exist.json','--variant','cuda-cudnn'],
            cwd=ROOT/'half_duplex',capture_output=True,text=True)
        self.assertEqual(p.returncode,2)
        self.assertIn('invalid choice',p.stderr)
        self.assertNotIn('Traceback',p.stderr)
