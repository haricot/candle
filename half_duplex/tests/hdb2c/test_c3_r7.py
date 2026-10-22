"""R7 strict bit accounting, TDT edge rules and independent build controls."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hdb2c.c3_r7_metrics import tensor_metrics, compare_decisions, advance
from hdb2c.c3_r7_source import transform_model
from hdb2c.c3_r7 import compile_candidate


class R7Tests(unittest.TestCase):
    def test_counts_words_and_actual_bits_separately(self):
        a = np.array([0x3f800000, 0x3f800000], dtype=np.uint32).view(np.float32)
        b = np.array([0x3f800001, 0x3f800003], dtype=np.uint32).view(np.float32)
        r = tensor_metrics(a, b)
        self.assertEqual(r['mismatching_f32_words'], 2)
        self.assertEqual(r['mismatching_bits'], 3)
        self.assertEqual(r['max_ulp'], 3)
        self.assertTrue(r['numeric_pass'])
        self.assertFalse(r['bitwise_equal'])

    def test_signed_zero_is_not_bit_equal(self):
        r = tensor_metrics(np.array([0.]), np.array([-0.]))
        self.assertEqual(r['max_abs'], 0.)
        self.assertEqual(r['mismatching_bits'], 1)
        self.assertFalse(r['bitwise_equal'])

    def test_all_ranks_and_padded_frames_are_checked(self):
        a = np.zeros((1, 1, 3, 4), dtype=np.float32)
        b = a.copy(); b[0, 0, -1, -1] = 1.
        r = tensor_metrics(a, b)
        self.assertEqual(r['first_difference']['index'], [0, 0, 2, 3])
        self.assertFalse(r['bitwise_equal'])
        self.assertFalse(tensor_metrics(a, a.flatten())['numeric_pass'])
        with self.assertRaises(ValueError):
            tensor_metrics([np.nan], [np.nan])

    def test_blank_zero_always_advances_and_resets_symbols(self):
        for maximum in (None, 10):
            self.assertEqual(advance(3, 10, 10, 0, 4, maximum), (4, 0, False))

    def test_label_zero_and_positive_duration_limit(self):
        self.assertEqual(advance(3, 2, 10, 0, 8, 10), (3, 9, False))
        self.assertEqual(advance(3, 2, 10, 0, 9, 10), (4, 0, True))
        self.assertEqual(advance(3, 2, 10, 2, 9, 10), (5, 0, False))

    def test_sequence_equality_requires_all_decisions(self):
        row = dict(frame=0, token=2, duration_index=1, duration=1, advance=1, forced_advance=False)
        self.assertTrue(compare_decisions([row], [row])['exact'])
        self.assertFalse(compare_decisions([row], [row, row])['exact'])
        changed = {**row, 'duration_index': 2}
        self.assertFalse(compare_decisions([row], [changed])['exact'])

    def test_unverified_source_is_rejected(self):
        for source in ('unknown', 'HDB2C3_R7_TDT_ADVANCE_AND_TRACE_V1'):
            with self.assertRaises(ValueError):
                transform_model(source, 'helper')

    def test_build_keeps_r6_binary_and_cargo_target_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, candidate = Path(tmp)/'r6', Path(tmp)/'r7'
            binary = reference/'target/r6-cpu/release/hdb2c_r6_probe'
            binary.parent.mkdir(parents=True); binary.write_bytes(b'R6')
            with patch('subprocess.run') as run:
                target = compile_candidate(reference, candidate)
            self.assertEqual(target, candidate/'target/r7-cpu')
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                self.assertEqual(call.kwargs['env']['CARGO_TARGET_DIR'], str(target))
            with patch('subprocess.run', side_effect=lambda *a, **kw: binary.write_bytes(b'changed')):
                with self.assertRaisesRegex(RuntimeError, 'modified the R6'):
                    compile_candidate(reference, candidate)


if __name__ == '__main__':
    unittest.main()
