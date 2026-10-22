import unittest
from hdb2c.na_ab import correct_na_code, OLD, NEW

class NANormalizationCandidate(unittest.TestCase):
    def test_only_global_else_changes(self):
        src='fn test(){\n    if args.normalize == "per_feature" {\n        do_per_feature();\n'+OLD+'\n        do_global();\n    }\n}'
        got=correct_na_code(src)
        self.assertEqual(got,src.replace(OLD,NEW))
        self.assertEqual(got.count('if args.normalize == "per_feature"'),1)
    def test_reject_unknown_source(self):
        with self.assertRaises(ValueError):correct_na_code('hello')
    def test_idempotence_refusal(self):
        src='if args.normalize == "per_feature" {\n'+OLD
        once=correct_na_code(src)
        with self.assertRaises(ValueError):correct_na_code(once)
