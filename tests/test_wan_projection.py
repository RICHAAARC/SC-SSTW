"""CPU constructed-array checks, no model or saved-video evidence."""
import unittest
import pytest
pytestmark = pytest.mark.unit
import numpy as np
from main.tube_state import projection_margin as p

class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.book = p.codebook(b"test-protocol-key")
        cls.z = np.random.default_rng(83).normal(size=p.SHAPE).astype(np.float32)

    def test_partition_orientation_and_inverse(self):
        values = p.blocks(self.z)
        for temporal, y, x in ((0, 0, 0), (3, 5, 11), (10, 9, 15)):
            expected = self.z[0, :, 1+4*temporal:5+4*temporal, 4*y:4*y+4, 4*x:4*x+4].transpose(1,0,2,3).reshape(-1)
            np.testing.assert_array_equal(values[temporal, y*16+x], expected)
        target = self.z.copy()
        target[:, :, 1:45] = 0
        p.put_blocks(target, values)
        np.testing.assert_array_equal(target, self.z)

    def test_margin_roundtrip_and_fixed_message_competition(self):
        original = self.z.copy()
        for message in (0, 1):
            marked, record = p.write(self.z, self.book, message)
            self.assertGreater(record['minimum_signed_projection_after'], 1-2e-6)
            np.testing.assert_array_equal(marked[:, :, [0,45]], original[:, :, [0,45]])
            values, valid = p.projections(marked, list(range(44)), self.book)
            scores = p.score(values, valid, self.book)
            self.assertEqual(scores[message]['aligned_payload_agreement'], 1.)
            self.assertGreater(scores[message]['score'], scores[1-message]['score'] + .5)
        np.testing.assert_array_equal(self.z, original)

    def test_fixed_grid_and_missing_denominator(self):
        self.assertEqual(p.allocation(45,0,1,1,0), list(range(44)))
        self.assertEqual(p.allocation(44,0,1,1,4), [None]+list(range(43)))
        selected = list(range(44))
        selected[17] = None
        _, valid = p.projections(self.z, selected, self.book)
        self.assertEqual(valid.sum(), 1600)
        self.assertAlmostEqual(p.score(np.ones(1760),valid, {'codes': np.ones((2,1760))})[0]['score'], 10/11)
        missing = p.read({}, self.book)
        self.assertEqual(missing['candidate_count'], 204)
        self.assertIsNone(missing['best'])
        self.assertTrue(all(r['status']=='MISSING_OBSERVATION' for r in missing['candidates']))

    def test_receiver_uses_reference_not_observed_support(self):
        marked, _ = p.write(self.z, self.book, 1)
        # Remove four complete ordinary latent groups; no VAE claimed here.
        shifted = np.concatenate((marked[:,:,:1], marked[:,:,5:]), axis=2)
        selection = p.allocation(shifted.shape[2]-1, 0, 1, 1, 16)
        values, valid = p.projections(shifted, selection, self.book)
        scores = p.score(values,valid,self.book)
        self.assertEqual(scores[1]['aligned_payload_agreement'],1.)
        self.assertEqual(scores[1]['matched_supports'],1600)

    def test_search_and_effective_equivalence(self):
        marked, _ = p.write(self.z,self.book,0)
        result=p.read({0:marked},self.book)
        self.assertEqual(result['best']['message'],0)
        identity=result['identity_path']
        self.assertEqual(identity['scores'][0]['matched_supports'],1760)
        same=[r for r in result['candidates'] if r.get('class')==identity['class']]
        self.assertGreater(len(same),1)
        self.assertTrue(all(r['g']==0 and r['selected_ordinary_groups']==identity['selected_ordinary_groups'] for r in same))

if __name__ == '__main__':
    unittest.main()
