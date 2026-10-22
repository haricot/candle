"""Missing and explicit checkpoint padding defaults must retain NeMo semantics."""
import sys
import hashlib
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from convert_nemo_hdb2a import normalize_nemo_padding
from hdb2c.c3_r7_focused import verify_native_sources


class R7PaddingTests(unittest.TestCase):
    def test_missing_pad_to_materializes_the_nemo_default(self):
        pre={}
        self.assertIsNone(normalize_nemo_padding(pre))
        self.assertEqual(pre['pad_to'],16)

    def test_explicit_zero_and_positive_values_are_preserved(self):
        for value in (0,1,16,32):
            pre={'pad_to':value}
            self.assertEqual(normalize_nemo_padding(pre),value)
            self.assertEqual(pre['pad_to'],value)

    def test_unsupported_values_do_not_silently_become_sixteen(self):
        for value in (None,-1,True,16.0,'max'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_nemo_padding({'pad_to':value})

    def test_focused_profile_rejects_any_changed_native_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'model.rs'; source.write_bytes(b'qualified')
            evidence={'native_source_sha256':{'model.rs':hashlib.sha256(b'qualified').hexdigest()}}
            verify_native_sources(root,evidence)
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'unchanged qualified native'):
                verify_native_sources(root,evidence)


if __name__=='__main__':
    unittest.main()
