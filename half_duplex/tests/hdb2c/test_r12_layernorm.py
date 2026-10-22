"""Ordered operator arithmetic, actual observation and local execution boundary."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from hdb2c.c3_r12_layernorm import summary
from hdb2c.layernorm_operator import CpuNorm, SOURCE, cuda_source, operands

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('local_layernorm', ROOT/'half_duplex/ci/local_layernorm.py')
local = importlib.util.module_from_spec(spec); spec.loader.exec_module(local)


class OrderedLayerNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.cpu = CpuNorm(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_cpu_matches_actual_torch_moments_on_large_offset_rows(self):
        rng = np.random.default_rng(17412)
        for width in (512, 640, 1024):
            for offset in (0., 90000.):
                with self.subTest(width=width, offset=offset):
                    x = (offset+rng.standard_normal((1, 9, width))*200).astype(np.float32)
                    gamma = rng.normal(size=width).astype(np.float32)
                    beta = rng.normal(size=width).astype(np.float32)
                    expected = torch.nn.functional.layer_norm(torch.from_numpy(x), (width,),
                        torch.from_numpy(gamma), torch.from_numpy(beta), 1e-5).numpy()
                    self.assertEqual(self.cpu(x, gamma, beta, 1e-5).tobytes(), expected.tobytes())

    def test_invalid_operands_are_rejected_before_native_code(self):
        for x, g, b, eps in ((np.ones((1, 8)), np.ones(7), np.zeros(8), 1e-5),
                              (np.zeros((0, 8)), np.ones(8), np.zeros(8), 1e-5),
                              (np.full((1, 8), np.nan), np.ones(8), np.zeros(8), 1e-5),
                              (np.ones((1, 8)), np.ones(8), np.zeros(8), 0.)):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                operands(x, g, b, eps)

    def test_cuda_uses_archived_cpu_algorithm_and_a_real_device_entry(self):
        source = cuda_source()
        self.assertIn('extern "C" __global__ void hdb2c12_layernorm_cuda', source)
        self.assertIn('hdb2c12_layernorm(1,n,8,epsilon', source)
        self.assertIn('__device__ void row_moments', source)
        self.assertEqual(source.count('fmaf('), SOURCE.read_text().count('std::fma('))

    def test_local_container_mounts_source_readonly_and_never_registers_runner(self):
        args = local.command(local.IMAGE, Path('/repo'), Path('/work-local'), 'test', 'cuda')
        self.assertIn('/repo:/source:ro', args)
        self.assertIn('/work-local:/work:rw', args)
        self.assertIn('--device=nvidia.com/gpu=all', args)
        self.assertEqual(args[-3:], [local.IMAGE, 'half_duplex/ci/r12_layernorm.sh', 'cuda'])
        self.assertEqual(args[args.index('--entrypoint')+1], '/bin/bash')
        self.assertFalse(any('TOKEN' in s or 'RUNNER_REPO' in s or 'EXPECTED_RUN_ID' in s for s in args))
        cpu = local.command(local.IMAGE, Path('/repo'), Path('/work-local'), 'test', 'cpu')
        self.assertFalse(any(s.startswith('--device') for s in cpu))

    def test_ci_operator_job_is_manual_cpu_and_publishes_only_metrics(self):
        workflow = yaml.load((ROOT/'.github/workflows/half-duplex-cpu.yml').read_text(), Loader=yaml.BaseLoader)
        job = workflow['jobs']['layernorm-cpu']
        self.assertEqual(job['if'], "${{ github.event_name == 'workflow_dispatch' && inputs.r12_layernorm }}")
        self.assertEqual(job['runs-on'], 'ubuntu-24.04')
        scripts = [step.get('run', '') for step in job['steps']]
        self.assertIn('bash half_duplex/ci/r12_layernorm.sh cpu', scripts)
        upload = next(s for s in job['steps'] if s.get('uses', '').startswith('actions/upload-artifact'))
        self.assertEqual(upload['with']['path'], 'halfduplex-ci-published/r12-layernorm-cpu.json')

    def test_missing_clips_or_cpu_failure_cannot_qualify_gpu(self):
        row = {'cpu_vs_nemo': {'bitwise_equal': True}, 'cuda_vs_nemo': {'bitwise_equal': True}}
        self.assertNotEqual(summary([row]*13, 'cuda', False)['qualification'], 'PASS_BITWISE_14_COMMON_INPUTS')
        self.assertNotEqual(summary([row]*13, 'cuda', True)['qualification'], 'PASS_BITWISE_14_COMMON_INPUTS')
        self.assertEqual(summary([row]*14, 'cuda', True)['qualification'], 'PASS_BITWISE_14_COMMON_INPUTS')
        bad = {**row, 'cpu_vs_nemo': {'bitwise_equal': False}}
        self.assertNotEqual(summary([row]*13+[bad], 'cuda', True)['qualification'], 'PASS_BITWISE_14_COMMON_INPUTS')
        self.assertEqual(summary([row]*14, 'cpu', True)['gpu_execution'], 'NOT_RUN')
        self.assertFalse(summary([row]*14, 'cpu', True)['model_operator_replaced'])


if __name__ == '__main__':
    unittest.main()
