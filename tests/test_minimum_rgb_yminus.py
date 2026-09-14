import unittest
import numpy as np
from main.sc_sstw.public_luma_statistic import basis,read_rgb
from experiments.public_statistic.minimum_rgb_yminus import candidate,evaluate,A,BUDGET

class Tests(unittest.TestCase):
 def setUp(self):
  self.S=np.full((16,6,8,3),.5);self.w=np.array([.2126,.7152,.0722]);self.C=A*np.eye(2)
  self.delta=-A*basis(6,np)[None,:,None,None]*self.w/(self.w@self.w)
  self.P=self.S+self.delta;self.offs={'OFF1':self.S.copy(),'OFF2':self.S.copy()}
 def test_ideal_statistic_and_energy(self):
  q=read_rgb(self.P,np)-read_rgb(self.S,np);np.testing.assert_allclose(q[:,0],0,atol=1e-15);np.testing.assert_allclose(q[:,1],-A,atol=1e-15)
  ratio=np.mean(self.delta**2)/(A*A);self.assertAlmostEqual(ratio,1/(3*(self.w@self.w)));self.assertLess(ratio,1)
 def test_clip_boundaries(self):
  S=np.zeros((16,6,8,3));P,c=candidate(S,np);self.assertEqual(P.dtype,np.uint8);self.assertTrue(all(x['clipped_channel_fraction']>0 for x in c));self.assertGreaterEqual(P.min(),0);self.assertLessEqual(P.max(),255)
 def test_joint_times(self):
  offs={k:v.copy() for k,v in self.offs.items()}
  # Each OFF has just one different bad time, but common coverage is only14.
  hx=basis(8,np)[None,:,None]
  offs['OFF1'][0]+=A*hx;offs['OFF2'][1]+=A*hx
  result=evaluate(self.S,self.P,self.P,offs,self.C,np);self.assertEqual(result['common_passing_frames'],14);self.assertFalse(result['single_arm_gate'])
 def test_total_quality_and(self):
  D=self.P.copy();D[-1]+=.1
  result=evaluate(self.S,self.P,D,self.offs,self.C,np);self.assertEqual(result['common_passing_frames'],15);self.assertFalse(result['total_quality_pass']);self.assertFalse(result['single_arm_gate'])
 def test_decomposition(self):
  D=self.P+.001;result=evaluate(self.S,self.P,D,self.offs,self.C,np);v=result['decomposition'];self.assertAlmostEqual(v['writer_mse']+v['codec_mse']+v['twice_cross'],v['total_mse']);self.assertEqual(len(result['rows']),16)
if __name__=='__main__':unittest.main()
