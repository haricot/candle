"""Numerical gate integrity and distinct corpus controls, not ASR fixtures."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from hdb2c.c3_r14_precision import errors, tolerance_survey, dot_bound, gamma, _numpy64
from hdb2c.c3_r14_corpus import select_expanded, coverage, DATASET, REVISION
from hdb2c.c3_r14_cuda import discrete_controls, cohort_summary
from hdb2c.c3_r8 import EMOTIONS


class PrecisionTests(unittest.TestCase):
    def test_fp64_reference_is_not_rounded_to_fp32(self):
        high = np.array([1.+2.**-30], dtype=np.float64)
        low = high.astype(np.float32)
        self.assertGreater(errors(high, low)['max_abs'], 0)
        self.assertEqual(errors(high, low)['max_abs'], 2.**-30)

    def test_near_zero_error_cannot_hide_in_large_tensor_scale(self):
        ref = np.array([0., 1e6], dtype=np.float32)
        candidate = np.array([.002, 1e6], dtype=np.float32)
        result = tolerance_survey(ref, candidate)
        self.assertEqual(result['zero_reference_violations'], 1)
        self.assertIsNone(result['minimum_required_rtol_diagnostic'])
        self.assertTrue(all(r['violations'] == 1 for r in result['rtol_survey']))

    def test_relative_allowance_is_checked_element_by_element(self):
        result = tolerance_survey(np.array([1000., 0.]), np.array([1000.005, 0.]))
        rows = result['rtol_survey']
        self.assertFalse(rows[0]['all_elements_within_tolerance'])
        self.assertTrue(rows[-1]['all_elements_within_tolerance'])
        self.assertEqual(result['policy_status'], 'SURVEY_ONLY_NO_THRESHOLD_APPROVED')

    def test_nonfinite_shapes_and_invalid_absolute_floor_rejected(self):
        for a, b in [([float('nan')], [1]), ([1], [float('inf')]), ([], []), ([1, 2], [1])]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                tolerance_survey(a, b)
        with self.assertRaises(ValueError):tolerance_survey([1], [1], atol=-1)

    def test_cancellation_bound_uses_absolute_products_not_only_output(self):
        result = dot_bound([0.], [1e-7], [2.], 2)
        self.assertEqual(result['violations'], 0)
        self.assertGreater(result['atol_elementwise_range'][0], 0)
        self.assertFalse(result['acceptance_gate'])
        self.assertGreater(gamma(100), gamma(3))
        for invalid in (0, -1, 2**25):
            with self.assertRaises(ValueError):gamma(invalid)

    def test_float32_oracle_cannot_be_labelled_fp64(self):
        import torch
        with self.assertRaises(ValueError):_numpy64(torch.ones(2, dtype=torch.float32))
        np.testing.assert_array_equal(_numpy64(torch.ones(2, dtype=torch.float64)), [1., 1.])


class CorpusTests(unittest.TestCase):
    def fixture(self):
        rows = [dict(id=f'{s}-{e}-{i}', split=s, emotion=e, selection_index=i,
                     pcm_sha256=f'{s}-{e}-{i}', wav=f'{s}-{e}-{i}.wav',
                     dataset=DATASET, revision=REVISION, speaker_id='public-fixture')
                for s in ('cafe', 'oreau') for e in EMOTIONS for i in range(3)]
        return dict(dataset=DATASET, revision=REVISION, rows=rows), [r for r in rows if r['selection_index'] == 0]

    def test_distinct_holdout_and_fixed_regression_coverage(self):
        manifest, anchors = self.fixture()
        with patch('hdb2c.audio.describe', side_effect=lambda p: {'pcm_sha256': p.stem}):
            rows = select_expanded(manifest, anchors=anchors)
        self.assertEqual(coverage(rows)['cohorts'], {'regression14': 14, 'additional28': 28})
        self.assertTrue(all(r['cohort'] == 'regression14' for r in rows[:14]))

    def test_duplicates_missing_strata_and_changed_regression_fail(self):
        manifest, anchors = self.fixture()
        variants = []
        x = copy.deepcopy(manifest);x['rows'].pop();variants.append(x)
        x = copy.deepcopy(manifest);x['rows'][1]['pcm_sha256'] = x['rows'][0]['pcm_sha256'];variants.append(x)
        x = copy.deepcopy(manifest);x['rows'][0]['selection_index'] = 1;variants.append(x)
        x = copy.deepcopy(manifest);x['revision'] = 'wrong';variants.append(x)
        with patch('hdb2c.audio.describe', side_effect=lambda p: {'pcm_sha256': p.stem}):
            for variant in variants:
                with self.subTest(variant=variants.index(variant)), self.assertRaises(ValueError):
                    select_expanded(variant, anchors=anchors)
            wrong = copy.deepcopy(anchors);wrong[0]['pcm_sha256'] = 'wrong-original'
            with self.assertRaisesRegex(ValueError, 'regression inputs changed'):
                select_expanded(manifest, anchors=wrong)
        with patch('hdb2c.audio.describe', return_value={'pcm_sha256': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'PCM changed'):
                select_expanded(manifest, anchors=anchors)


class DiscreteTests(unittest.TestCase):
    def test_any_token_duration_frame_or_decision_mismatch_remains_blocking(self):
        good = {'output_exact': dict(ids=True, frames=True, durations=True, text=True),
                'tdt_discrete_exact': True, 'decision_counts': {'exact': True}}
        for field in ('ids', 'frames', 'durations', 'text', 'decision'):
            bad = copy.deepcopy(good)
            if field == 'decision':bad['tdt_discrete_exact'] = False
            else:bad['output_exact'][field] = False
            controls = discrete_controls({'tdt': {'native': bad, 'shared_mel': good}})
            self.assertFalse(controls['native']['all_exact'])
            self.assertTrue(controls['shared_mel']['all_exact'])

    def test_missing_output_field_and_empty_cohort_do_not_pass(self):
        with self.assertRaises(ValueError):discrete_controls({'tdt': {}})
        with self.assertRaises(KeyError):discrete_controls({'tdt': {
            'native': {'output_exact': {}}, 'shared_mel': {'output_exact': {}}}})
        self.assertEqual(cohort_summary([])['additional28']['completed'], 0)


if __name__ == '__main__':unittest.main()
