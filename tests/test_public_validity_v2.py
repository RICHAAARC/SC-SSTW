"""Focused q-only interface/math tests, not the 20 frozen diagnostic inputs."""
import math
import unittest
from main.sc_sstw.aisb import make_default_templates
from main.sc_sstw.public_scan import scan_public_q
from main.sc_sstw.public_validity_v2 import ValidityConfig, geometry_decision, scan_public_q_v2, _spectrum


class PublicValidityV2Tests(unittest.TestCase):
    def setUp(self):
        self.templates = make_default_templates()
        self.q = [tuple(.5+.04*v for v in p) for p in self.templates[0].points]

    def run_scan(self, q, **kwargs):
        return scan_public_q_v2(q, templates=self.templates, max_evaluations=885,
                                max_retained=128, **kwargs)

    def test_public_original_pool_unchanged(self):
        old = scan_public_q(self.q, templates=self.templates, missing_sets=((),),
                            max_evaluations=885, max_retained=128)
        new = self.run_scan(self.q)
        self.assertEqual(old, new['original'])
        self.assertEqual(new['records'][0]['certification_status'], 'CONDITIONALLY_CERTIFIED_ENGINEERING')

    def test_constant_low_score_remains_rejected(self):
        new = self.run_scan([(0.5, 0.5)] * 6)
        self.assertEqual(new['original']['scored_count'], 3)
        self.assertEqual(new['certified_count'], 0)
        self.assertEqual(new['records'][0]['residual'], 0)
        self.assertIn('FULL_WINDOW_MARGIN', new['records'][0]['rejection_reasons'])

    def test_later_spread_does_not_fix_first_anchors(self):
        new = self.run_scan([(.4,.4),(.5,.4),(.6,.4),(.5,.5),(.4,.6),(.6,.6)])
        row = new['records'][0]
        self.assertNotIn('FULL_WINDOW_MARGIN', row['rejection_reasons'])
        self.assertIn('ANCHOR_MARGIN', row['rejection_reasons'])

    def test_nonfinite_and_invalid_keep_original_indices(self):
        q = self.q + [(float('nan'), .5), (.5, .5)]
        result = self.run_scan(q)
        self.assertEqual(len(result['input_rows']), 8)
        self.assertEqual(result['input_rows'][6]['reason'], 'NONFINITE_Q')
        self.assertEqual(result['evaluated_count'], 9)
        self.assertEqual(result['records'][1]['invalid_indices'], [6])
        invalid = self.run_scan(self.q, valid=[True,False,True,True,True,True])
        self.assertEqual(invalid['records'][0]['invalid_indices'], [1])

    def test_strict_margin_closed_condition_bound(self):
        cfg = ValidityConfig()
        e, delta = cfg.epsilon, 2*math.sqrt(2)*cfg.epsilon
        self.assertIn('FULL_WINDOW_MARGIN', geometry_decision(2*e, 3*delta, 8, cfg))
        self.assertIn('ANCHOR_MARGIN', geometry_decision(3*e, 2*delta, 8, cfg))
        self.assertEqual(geometry_decision(math.nextafter(2*e, math.inf),
                         math.nextafter(2*delta, math.inf), 8, cfg), [])
        self.assertIn('CONDITION_CERTIFICATE', geometry_decision(3*e, 3*delta,
                      math.nextafter(8, math.inf), cfg))

    def test_near_line_small_singular_not_lost(self):
        small, large, det = _spectrum([(1., 0.), (0., 1e-14)])
        self.assertEqual(small, 1e-14)
        self.assertEqual(large, 1.)

    def test_budget_incomplete_not_frozen(self):
        result = scan_public_q_v2(self.q, templates=self.templates,
                                 max_evaluations=1, max_retained=128)
        self.assertFalse(result['enumeration_complete'])
        self.assertIsNone(result['full_enumeration_count'])
        self.assertIsNone(result['frozen'])
        self.assertEqual(result['ranking_scope'], 'partial_enumeration_only')

    def test_interface_errors_do_not_relax_config(self):
        for bad in ([True]*5, [1]*6):
            with self.assertRaises(ValueError):
                self.run_scan(self.q, valid=bad)
        with self.assertRaises(ValueError):
            self.run_scan([(1,2,3)]*6)
        with self.assertRaises(ValueError):
            ValidityConfig(epsilon=0.1)


if __name__ == '__main__':
    unittest.main()
