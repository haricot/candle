"""R6 mask semantics, full-frame controls and native build isolation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from hdb2c.c3_r6 import expected_masks, compile_candidate, exact_control
from hdb2c.c3_r6_source import transform_attention, transform_conformer


class R6Tests(unittest.TestCase):
    def test_single_extra_frame_blocks_keys_and_queries(self):
        pad, att = expected_masks([3], 4)
        np.testing.assert_array_equal(pad[:, :, 0], [[0, 0, 0, 1]])
        np.testing.assert_array_equal(att[0], [[0,0,0,1],[0,0,0,1],[0,0,0,1],[1,1,1,1]])
        self.assertEqual(int(att.sum()), 7)

    def test_mixed_batch_and_all_valid_lengths(self):
        pad, att = expected_masks([4, 3], 4)
        self.assertFalse(pad[0].any())
        self.assertFalse(att[0].any())
        self.assertEqual(int(att[1].sum()), 7)

    def test_invalid_lengths_fail_closed(self):
        for lengths, total in (([0],4), ([5],4), ([],4), ([3.],4), ([1],0)):
            with self.subTest(lengths=lengths, total=total), self.assertRaises(ValueError):
                expected_masks(lengths, total)

    def test_exact_control_includes_padded_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/'a', Path(temp)/'b'
            for folder in (a, b):
                folder.mkdir()
                np.ones((1,4,2),dtype='<f4').tofile(folder/'layer_00.f32le')
                (folder/'layer_00.json').write_text(json.dumps({'shape':[1,4,2]}))
            self.assertEqual(exact_control(a,b,['layer_00'],label='OFF')['status'], 'PASS_BITWISE_ALL_FRAMES')
            data=np.ones((1,4,2),dtype='<f4'); data[:,3:]=2.
            data.tofile(b/'layer_00.f32le')
            with self.assertRaisesRegex(RuntimeError, 'OFF differs at layer_00'):
                exact_control(a,b,['layer_00'],label='OFF')

    def test_native_build_and_tests_use_an_independent_target(self):
        with tempfile.TemporaryDirectory() as temp:
            ref, candidate = Path(temp)/'r5', Path(temp)/'r6'
            control = ref/'target/r5-cpu/release/hdb2c_r5_probe'
            control.parent.mkdir(parents=True); control.write_bytes(b'R5 control')
            with patch('subprocess.run') as run:
                target=compile_candidate(ref,candidate)
            self.assertEqual(target,candidate/'target/r6-cpu')
            self.assertEqual(run.call_count,2)
            for call in run.call_args_list:
                self.assertEqual(call.kwargs['env']['CARGO_TARGET_DIR'],str(target))
            with patch('subprocess.run',side_effect=lambda *a,**kw:control.write_bytes(b'mutated')):
                with self.assertRaisesRegex(RuntimeError,'modified the R5 control'):
                    compile_candidate(ref,candidate)

    def test_attention_transform_rejects_unknown_or_unverified_source(self):
        for invalid in ('unknown','HDB2C3_R6_ENCODER_PADDING_MASKS_V1'):
            with self.assertRaises(ValueError):
                transform_attention(invalid,'helper')
            with self.assertRaises(ValueError):
                transform_conformer(invalid)


if __name__=='__main__':
    unittest.main()
