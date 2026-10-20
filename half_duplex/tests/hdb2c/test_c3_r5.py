"""R5 numerical and source-integrity gates; no model inference is synthesized."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from hdb2c.c3_r5 import (SIGNATURE, MARKER, transform_r5_attention,
                          build_r5_candidate, compile_r5, compare_exact, summarize_r5)

FREQUENCIES = '''        for i in 0..(d_model / 2) {
            let exp = -((10000.0f64).ln() / d_model as f64) * (2 * i) as f64;
            div_term.push(exp.exp() as f32);
        }'''


class R5Tests(unittest.TestCase):
    def test_cargo_target_and_control_binary_are_isolated(self):
        with tempfile.TemporaryDirectory() as temp:
            reference, candidate = Path(temp)/'r4', Path(temp)/'r5'
            control = reference/'target/r4-cpu/release/hdb2c_probe'
            control.parent.mkdir(parents=True)
            control.write_bytes(b'immutable R4 executable')
            with patch('subprocess.run') as run:
                target = compile_r5(reference, candidate)
            self.assertEqual(target, candidate/'target/r5-cpu')
            env = run.call_args.kwargs['env']
            self.assertEqual(env['CARGO_TARGET_DIR'], str(target))
            self.assertNotEqual(target, reference/'target/r4-cpu')
            self.assertEqual(control.read_bytes(), b'immutable R4 executable')
            with patch('subprocess.run', side_effect=lambda *a, **kw: control.write_bytes(b'changed')):
                with self.assertRaisesRegex(RuntimeError, 'modified the R4 control'):
                    compile_r5(reference, candidate)

    def test_pinned_frequency_change_is_scoped_and_rejects_drift(self):
        source = 'unchanged_before\n'+FREQUENCIES+'\nunchanged_after'
        changed = transform_r5_attention(source)
        self.assertIn(MARKER, changed)
        self.assertTrue(changed.startswith('unchanged_before\n'))
        self.assertTrue(changed.endswith('\nunchanged_after'))
        self.assertIn('div_term.push(exponent.exp())', changed)
        for invalid in ('unknown', changed, source+FREQUENCIES):
            with self.assertRaises(ValueError):
                transform_r5_attention(invalid)

    def test_numeric_parity_cannot_claim_bitwise_with_signed_zero(self):
        a = np.ones((1, 2, 4), dtype=np.float32)
        b = a.copy()
        a[0, 0, 0], b[0, 0, 0] = 0., -0.
        result = compare_exact(a, b)
        self.assertEqual(result['status'], 'PASS_NUMERIC_PARITY')
        self.assertFalse(result['bitwise_equal'])
        self.assertEqual(result['mismatching_f32_bits'], 1)

    def test_one_ulp_is_not_bitwise_exact(self):
        a = np.ones((1, 2, 4), dtype=np.float32)
        b = a.copy()
        b[0, 0, 0] = np.nextafter(np.float32(1.), np.float32(2.))
        result = compare_exact(a, b)
        self.assertEqual(result['status'], 'PASS_NUMERIC_PARITY')
        self.assertFalse(result['bitwise_equal'])
        self.assertTrue(compare_exact(a, a)['bitwise_equal'])

    def test_missing_shape_or_nonfinite_is_not_a_pass(self):
        a = np.ones((1, 2, 4), dtype=np.float32)
        self.assertEqual(compare_exact(a, a[:, :1])['status'], 'SHAPE_MISMATCH')
        for bad in (float('nan'), float('inf')):
            b = a.copy(); b[0, 0, 0] = bad
            with self.assertRaises(ValueError):
                compare_exact(a, b)

    def test_candidate_preserves_control_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref, out = root/'r4', root/'r5'
            (ref/'src/parakeet').mkdir(parents=True)
            (ref/'src/bin').mkdir()
            conf = b'R4 conformer unchanged\n'
            (ref/'src/parakeet/conformer.rs').write_bytes(conf)
            (ref/'src/parakeet/attention.rs').write_text(FREQUENCIES)
            (ref/'src/bin/hdb2c_probe.rs').write_text('R4 probe unchanged\n')
            (ref/'HDB2C3-R4-MANIFEST.json').write_text(json.dumps({
                'generated_conformer_sha256': hashlib.sha256(conf).hexdigest()}))
            first = build_r5_candidate(ref, out)
            self.assertEqual(build_r5_candidate(ref, out), first)
            self.assertEqual((ref/'src/parakeet/attention.rs').read_text(), FREQUENCIES)
            self.assertEqual((out/'src/parakeet/conformer.rs').read_bytes(), conf)
            (out/'src/parakeet/attention.rs').write_text('unreviewed change')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                build_r5_candidate(ref, out)

    def test_incomplete_operator_coverage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nemo, control, candidate, op = [root/x for x in ('nemo','r4','r5','op')]
            for folder in (nemo/'positional_nemo', control, candidate, op):
                folder.mkdir(parents=True)
            for folder in (control, candidate):
                np.ones((1, 2, 4), dtype='<f4').tofile(folder/'pre_encode.f32le')
                (folder/'pre_encode.json').write_text(json.dumps({'shape': [1, 2, 4]}))
            reference = {'signature': SIGNATURE, 'cases': []}
            (nemo/'positional_nemo/cases.json').write_text(json.dumps(reference))
            (op/'manifest.json').write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError, 'operator cases'):
                summarize_r5(nemo, control, candidate, op, 2)


if __name__ == '__main__':
    unittest.main()
