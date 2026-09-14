import unittest,json
from pathlib import Path
import numpy as np
from experiments.public_statistic.minimum_rgb_complete import roster,perturb,statistics,old_C_comparison,A
from experiments.public_statistic.minimum_rgb_yminus import candidate
from runtime.public_statistic.phase1 import aggregate_evaluation
from main.sc_sstw.public_luma_statistic import read_rgb

class Tests(unittest.TestCase):
 def test_fixed_roster(self):
  r=roster('/old','/single');self.assertEqual(sum(x['mode']=='NEW' for x in r),7);self.assertEqual(sum(x['mode']=='REUSED' for x in r),5);self.assertEqual(sum(x['frames'] for x in r),192);self.assertEqual(next(x['source'] for x in r if x['input_id']=='JumpingJack' and x['arm']=='Y_MINUS'),'/single')
 def test_formula_four_signs(self):
  S=np.full((16,6,8,3),.5)
  for arm in ('X_PLUS','X_MINUS','Y_PLUS','Y_MINUS'):
   p,clip=perturb(S,arm,np);q=read_rgb(p/255,np)-read_rgb(S,np);axis=0 if arm.startswith('X') else 1;sign=1 if arm.endswith('PLUS') else -1
   self.assertTrue((sign*q[:,axis]>0).all());self.assertEqual(len(clip),16)
  np.testing.assert_array_equal(perturb(S,'Y_MINUS',np)[0],candidate(S,np)[0])
 def test_statistics_decomposition(self):
  S=np.full((16,6,8,3),.5);P=perturb(S,'X_PLUS',np)[0]/255;D=P+.001;r=statistics(S,P,D,np);v=r['decomposition'];self.assertAlmostEqual(v['writer_mse']+v['codec_mse']+v['twice_cross'],v['total_mse']);self.assertEqual(len(r['mp4']['q']),16)
 def test_aggregate_and_old_diagnostic_separate(self):
  c=json.loads((Path(__file__).parents[1]/'configs/public_luma_phase1.json').read_text());S=np.full((16,6,8,3),.5);records={}
  for sid in ('P50','JumpingJack'):
   arms={}
   for arm in c['arms']:
    P=S if arm.startswith('OFF') else perturb(S,arm,np)[0]/255
    arms[arm]=statistics(S,P,P,np)
   records[sid]=dict(arms=arms)
  result=aggregate_evaluation(records,c,np);self.assertTrue(result['mp4']['meets_method_gate'])
  old=old_C_comparison(records,3*A*np.eye(2),np);self.assertFalse(any(old['P50']['X_PLUS']['old_C_common_compatibility_mask']));self.assertTrue(result['mp4']['meets_method_gate'])
  del records['P50']['arms']['X_PLUS'];self.assertEqual(aggregate_evaluation(records,c,np)['mp4']['status'],'INCOMPLETE')
if __name__=='__main__':unittest.main()
