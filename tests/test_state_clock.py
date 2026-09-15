"""Constructed state/clock tests; no VAE, GPU or real-video claims."""
import unittest
from unittest.mock import patch
import numpy as np
from runtime.tstwv2 import state_clock as s
from runtime.tstwv2 import projection_margin as p

class StateClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key=b'WanProjection-first-validation-key-v1'
        cls.book=s.codebook(cls.key)

    def test_recurrence_and_message_mapping(self):
        for m in (0,1):
            states,steps,drive=s.trajectory(self.key,m)
            np.testing.assert_array_equal(states,self.book['states'][m])
            np.testing.assert_allclose(np.sum(states**2,axis=1),2)
            for n in range(10):
                np.testing.assert_array_equal(s.rotate(states[n],steps[n]),states[n+1])
            independent=np.stack([states[0]]+[s.rotate(states[0],step) for step in steps])
            self.assertGreater(np.sum(np.any(independent!=states,axis=1)),0)
        self.assertGreater(np.sum(self.book['states'][0]!=self.book['states'][1]),0)

    def test_actual_projection_to_state(self):
        z=np.zeros(p.SHAPE,np.float32)
        for m in (0,1):
            marked,record=p.write(z,self.book,m)
            self.assertGreater(record['minimum_signed_projection_after'],1-2e-6)
            qs=[]
            for n in range(11):
                row=s.emissions({0:marked},self.book,n,0,1,1,0)
                np.testing.assert_allclose(row['q'],self.book['states'][m,n],atol=1e-6)
                qs.append(row['q'])
            obs=s.observe(np.array(qs),[True]*11,self.book['states'][m],self.book['steps'][m])
            self.assertLess(obs['innovation_mean'],1e-10)

    def test_dynamics_order_and_state_update_are_active(self):
        states=self.book['states'][0];steps=self.book['steps'][0]
        good=s.observe(states,[True]*11,states,steps)
        wrong=s.observe(states,[True]*11,states,(steps+2)%4)
        reorder=s.observe(states[::-1],[True]*11,states,steps)
        self.assertLess(good['innovation_mean'],wrong['innovation_mean'])
        self.assertLess(good['innovation_mean'],reorder['innovation_mean'])
        perturbed=states.copy();perturbed[3]=s.rotate(perturbed[3],1)
        updated=s.observe(perturbed,[True]*11,states,steps)
        fixed=s.observe(perturbed,[True]*11,states,steps,update=False)
        self.assertFalse(np.allclose(updated['updated_states'][3],fixed['updated_states'][3]))
        self.assertFalse(np.allclose(updated['predictions'][4],fixed['predictions'][4]))

    def test_clock_search_compares_same_cached_observations(self):
        calls=[]
        def fake(obs,book,n,g,num,den,b):
            calls.append((n,g,num,den,b))
            correct=(num,den)==(1,1) and ((g,b)==((0,0) if n<9 else (3,1)))
            q=book['states'][0,n] if correct else -book['states'][0,n]
            values=[float(q@book['states'][m,n]/2) for m in (0,1)]
            # Different artificial emissions must have different observation identities.
            return {'selected':[10000*num+1000*den+100*b+4*n+j for j in range(4)],'valid':True,'q':q.tolist(),'scores':values,'agreement':[(v+1)/2 for v in values]}
        with patch.object(s,'emissions',fake):
            result=s.read({},self.book)
        self.assertEqual(result['candidate_count'],4284)
        self.assertEqual(len(calls),len(set(calls)))
        local=result['rankings']['local_state']['best']
        self.assertEqual((local['message'],local['g'],local['scale'],local['offset'],local['boundary'],local['delta']),(0,0,[1,1],0,9,1))
        self.assertGreater(local['score'],result['rankings']['global_state']['best']['score'])
        report=s.report(result,0,1)
        self.assertTrue(report['modes']['local_state']['all_checked_windows_match'])
        self.assertEqual(report['modes']['local_state']['nominal_time_denominator'],10)

    def test_equal_path_cost_ablations_and_new_event_report(self):
        def fake(obs,book,n,g,num,den,b):
            q=book['states'][0,n]
            return {'selected':[10000*num+1000*den+100*b+4*n+j for j in range(4)],'valid':True,'q':q.tolist(),
                    'scores':[float(q@book['states'][m,n]/2) for m in (0,1)],'agreement':[1.,.5]}
        with patch.object(s,'emissions',fake):
            result=s.read({},self.book)
        for mode,field in (('local_matched','matched_score'),('local_without_update','without_update_score'),('local_state','state_score')):
            scores=[clscore[field]-row['event_cost'] for row in result['candidates'] for clscore in result['classes'][row['class']]['scores']]
            self.assertAlmostEqual(result['rankings'][mode]['best']['score'],max(scores))
        report=s.report(result,0,1,90)
        self.assertEqual(report['nominal_clock_reference']['boundary'],6)
        self.assertEqual(report['event_window_excluded_from_time_metric_only'],5)
        self.assertEqual(report['time_checked_windows'],[0,1,2,3,4,6,7,8,9,10])

if __name__=='__main__':
    unittest.main()
