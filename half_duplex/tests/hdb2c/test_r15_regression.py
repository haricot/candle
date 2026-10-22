"""Fault-injection tests for the actual CI acceptance and profile boundaries."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

from hdb2c.c3_r15_regression import (assess, finite_trace, select_rows, summarize,
                                    SHORT_STRATA, REQUIRED_STAGES)
from hdb2c.c3_r14_corpus import DATASET, REVISION

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('r15_policy', ROOT/'ci/resolve_profile.py')
policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)


def dump(folder, name, values):
    path = folder/(name+'.f32le'); path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(values, dtype='<f4')
    path.write_bytes(a.tobytes())
    path.with_suffix('.json').write_text(json.dumps({'shape': list(a.shape), 'dtype': 'f32le'}))


class RegressionTests(unittest.TestCase):
    def fixture(self, root):
        ref, native = root/'nemo', root/'native'
        ref.mkdir(); native.mkdir()
        decision = {'frame': 0, 'token': 1, 'duration_index': 1, 'duration': 1,
                    'advance': 1, 'forced_advance': False}
        original = {'ids': [1], 'frames': [0], 'durations': [1], 'text': 'bonjour',
                    'decisions': [decision], 'mel_valid': 8, 'encoder_valid': 1,
                    'blank': 2, 'durations_config': [0, 1], 'max_symbols': 10}
        result = {k: copy.deepcopy(original[k]) for k in ('ids', 'frames', 'durations', 'text',
                  'blank', 'durations_config', 'max_symbols')}
        result.update(scope='NATIVE_PCM_FRONTEND_ENCODER_TDT', device='Cuda(CudaDevice(DeviceId(1)))',
                      encoder_valid_frames=1, mel_valid_frames=8)
        (ref/'manifest.json').write_text(json.dumps(original))
        (native/'result.json').write_text(json.dumps(result))
        (native/'tdt').mkdir()
        (native/'tdt/tdt_trace.json').write_text(json.dumps([decision]))
        dump(ref, 'encoder', [[[1., 2.]]])
        dump(native, 'encoder', [[[1. + 2e-7, 2.]]])
        for name in ('pcm', 'mel', *('encoder_trace/'+s for s in REQUIRED_STAGES),
                     *('tdt/000000/'+s for s in ('enc', 'g', 'h', 'c', 'logits'))):
            dump(native, name, [[[1.]]])
        return ref, native

    def evaluate(self, ref, native):
        return assess(ref, native, variant='cuda-cudnn', mode='native')

    def test_cross_backend_bits_are_diagnostic_and_final_limits_still_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, native = self.fixture(Path(tmp))
            r = self.evaluate(ref, native)
            self.assertTrue(r['pass'])
            self.assertFalse(r['final_encoder']['bitwise_equal'])
            dump(native, 'encoder', [[[1.01, 2.]]])
            self.assertFalse(self.evaluate(ref, native)['pass'])

    def test_text_success_cannot_hide_duration_frame_token_or_decision_drift(self):
        for field in ('ids', 'frames', 'durations', 'text', 'encoder_valid_frames', 'mel_valid_frames'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                ref, native = self.fixture(Path(tmp))
                result = json.loads((native/'result.json').read_text())
                result[field] = 'changed' if field == 'text' else [9] if isinstance(result[field], list) else 9
                (native/'result.json').write_text(json.dumps(result))
                self.assertFalse(self.evaluate(ref, native)['pass'])
        for field in ('token', 'duration_index', 'advance', 'forced_advance'):
            with self.subTest(decision=field), tempfile.TemporaryDirectory() as tmp:
                ref, native = self.fixture(Path(tmp))
                rows = json.loads((native/'tdt/tdt_trace.json').read_text())
                rows[0][field] = True if field == 'forced_advance' else 7
                (native/'tdt/tdt_trace.json').write_text(json.dumps(rows))
                self.assertFalse(self.evaluate(ref, native)['pass'])

    def test_nan_inf_or_corrupt_intermediate_blocks_despite_correct_final_and_text(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                ref, native = self.fixture(Path(tmp))
                dump(native, 'encoder_trace/intermediate', [[[value]]])
                with self.assertRaises(ValueError): self.evaluate(ref, native)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with self.assertRaises(ValueError): finite_trace(folder)
        with tempfile.TemporaryDirectory() as tmp:
            ref, native = self.fixture(Path(tmp))
            (native/'tdt/000000/logits.f32le').unlink()
            with self.assertRaises(ValueError): self.evaluate(ref, native)
            dump(folder, 'a', [[[1.]]])
            (folder/'a.f32le').write_bytes(b'bad')
            with self.assertRaises(ValueError): finite_trace(folder)

    def test_silent_cpu_fallback_and_wrong_execution_scope_block(self):
        for field, value in (('device', 'Cpu'), ('scope', 'SHARED_NEMO_MEL_ENCODER_TDT')):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                ref, native = self.fixture(Path(tmp))
                result = json.loads((native/'result.json').read_text()); result[field] = value
                (native/'result.json').write_text(json.dumps(result))
                self.assertFalse(self.evaluate(ref, native)['pass'])

    def test_missing_clips_or_a_single_failed_clip_never_pass_campaign(self):
        self.assertFalse(summarize([{'pass': True}]*3, [], 4)['pass'])
        self.assertFalse(summarize([{'pass': True}]*3+[{'pass': False}], [], 4)['pass'])
        self.assertFalse(summarize([{'pass': True}]*4, [{'error': 'reference unavailable'}], 4)['pass'])
        self.assertTrue(summarize([{'pass': True}]*4, [], 4)['pass'])

    def test_short_selection_is_hash_pinned_and_cannot_shrink_or_accept_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for i, (split, emotion) in enumerate(SHORT_STRATA):
                wav = Path(tmp)/f'{i}.wav'; pcm = bytes([i, 0])*320
                with wave.open(str(wav), 'wb') as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm)
                rows.append({'split': split, 'emotion': emotion, 'selection_index': 0,
                    'id': str(i), 'dataset': DATASET, 'revision': REVISION,
                    'pcm_sha256': hashlib.sha256(pcm).hexdigest(), 'wav': str(wav)})
            manifest = {'dataset': DATASET, 'revision': REVISION, 'rows': rows}
            self.assertEqual(len(select_rows(manifest, 'short', anchors=rows)), 4)
            for bad in ({**manifest, 'rows': rows[:-1]}, {**manifest, 'rows': rows+[rows[0]]},
                        {**manifest, 'revision': 'moved'}):
                with self.assertRaises(ValueError): select_rows(bad, 'short', anchors=rows)
            (Path(tmp)/'0.wav').write_bytes(b'invalid')
            with self.assertRaises((ValueError, wave.Error, EOFError)):
                select_rows(manifest, 'short', anchors=rows)


class ProfileTests(unittest.TestCase):
    def test_git_resolver_uses_promoted_tree_and_detects_deleted_native_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            def git(*args):
                return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.DEVNULL).decode().strip()
            git('init', '-q'); git('config', 'user.name', 'CI test'); git('config', 'user.email', 'ci@example.invalid')
            marker = repo/policy.MARKER; marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({'schema_version': 1, 'policy': 'r15-functional'}))
            engine = repo/'half_duplex/rust/model.rs'; engine.parent.mkdir(); engine.write_text('qualified')
            git('add', '.'); git('commit', '-qm', 'qualified'); baseline = git('rev-parse', 'HEAD')
            (repo/'README.md').write_text('documentation')
            git('add', '.'); git('commit', '-qm', 'docs')
            engine.write_text('uncommitted changes do not change the immutable candidate')
            self.assertEqual(policy.resolve(repo, 'HEAD', baseline, 'auto')['profile'], 'short')
            engine.unlink(); git('add', '-u'); git('commit', '-qm', 'remove engine')
            result = policy.resolve(repo, 'HEAD', baseline, 'short')
            self.assertEqual(result['profile'], 'full')
            self.assertEqual(result['engine_changed_paths'], ['half_duplex/rust/model.rs'])

    def test_engine_changes_escalate_even_explicit_short_and_documentation_stays_short(self):
        for path in ('half_duplex/rust/hdb2c_r9_text.rs', 'half_duplex/scripts/convert_nemo_hdb2a.py',
                     'half_duplex/hdb2c/c3_r6.py', 'candle-core/src/cuda_backend/mod.rs',
                     'candle-nn/src/conv.rs', 'candle-kernels/src/conv.cu'):
            self.assertEqual(policy.choose('short', [path])[0], 'full', path)
            self.assertEqual(policy.choose('auto', [path])[0], 'full', path)
        self.assertEqual(policy.choose('auto', ['half_duplex/STATE-MEMORY.md',
            'half_duplex/ci/extended_gate.sh', 'docs/extended-half-duplex.md'])[0], 'short')
        self.assertEqual(policy.choose('short', [], baseline_available=False)[0], 'full')
        self.assertEqual(policy.choose('full', [])[0], 'full')
        self.assertEqual(policy.choose('audit', [])[0], 'audit')

    def test_invalid_and_implicit_fp64_profiles_fail_before_environment_installation(self):
        import os
        for env, expected in (({'HDB_CI_PROFILE': 'typo'}, 'INVALID_HDB_CI_PROFILE'),
                              ({'HDB_CI_PROFILE': 'short', 'HDB2C_R14_CUDA': '1'},
                               'R15_FP64_REQUIRES_EXPLICIT_AUDIT')):
            p = subprocess.run(['bash', str(ROOT/'ci/full_cpu.sh')],
                env={**os.environ, **env}, capture_output=True, text=True)
            self.assertEqual(p.returncode, 3)
            self.assertIn(expected, p.stderr)
            self.assertNotIn('install-nemo', p.stdout)


if __name__ == '__main__':
    unittest.main()
