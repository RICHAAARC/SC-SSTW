import unittest,copy
import numpy as np
import torch
from main.sc_sstw.public_luma_statistic import basis,read_rgb,read_rgb_differentiable
from main.sc_sstw.terminal_feedback import terminal_feedback,receding_feedback
from runtime.public_statistic.phase1 import perturb,decode_command,encode_command

class Tests(unittest.TestCase):
 def test_basis(self):
  h=basis(8,np);self.assertAlmostEqual(h.mean(),0);self.assertAlmostEqual((h*h).mean(),1);self.assertGreater(h[0],0);self.assertLess(h[-1],0)
 def test_public_ideal(self):
  x=np.full((2,6,8,3),.5);y,_=perturb(x,'X_PLUS',2/255,np);q=read_rgb(y,np)-read_rgb(x,np);np.testing.assert_allclose(q[:,0],2/255,atol=1e-15);np.testing.assert_allclose(q[:,1],0,atol=1e-15)
 def test_torch_reader(self):
  x=torch.full((2,6,8,3),.5,dtype=torch.float64,requires_grad=True);q=read_rgb_differentiable(x,torch);np.testing.assert_allclose(q.detach(),read_rgb(x.detach().numpy(),np),atol=1e-15);q.sum().backward();self.assertGreater(float(x.grad.abs().sum()),0)
 def test_invalid(self):
  with self.assertRaises(ValueError):read_rgb_differentiable(torch.zeros((1,2,2,3),dtype=torch.uint8),torch)
  with self.assertRaises(ValueError):read_rgb(np.zeros((0,2,2,3)),np)
 def test_clip(self):
  y,c=perturb(np.ones((2,6,8,3)),'Y_PLUS',2/255,np);self.assertLessEqual(y.max(),1);self.assertTrue(c)
 def test_feedback(self):
  state={'index':0,'history':[torch.tensor(.1)]}
  def rollout(z,s):
   s['history'].append(z);return z+sum(s['history'][:-1])
  z=torch.tensor(.2);target=torch.tensor(.5);ref=torch.tensor(.3)
  out,rec=terminal_feedback(z,state,rollout,lambda x:x,target,ref,learning_rate=.2,step_rms=.1,quality_rms_budget=.2,torch=torch)
  self.assertTrue(rec['accepted']);self.assertEqual(len(state['history']),1);self.assertLess(rec['actual_update_rms'],.1)
  out,rec=terminal_feedback(z,state,rollout,lambda x:x,target,ref,learning_rate=.2,step_rms=.1,quality_rms_budget=.001,torch=torch)
  self.assertFalse(rec['accepted']);self.assertEqual(float(out),float(z))
 def test_zero_gradient(self):
  z=torch.tensor(.2);out,r=terminal_feedback(z,{},lambda x,s:x,lambda x:x,z,z,learning_rate=.2,step_rms=.1,quality_rms_budget=.2,torch=torch);self.assertEqual(r['reason'],'ZERO_GRADIENT')
 def test_receding_full_state(self):
  original={'index':0,'history':[torch.tensor(.01)]};calls=[]
  def advance(z,s):
   calls.append(s['index']);s['history'].append(z);s['index']+=1;return z+.01,s
  def rollout(z,s):
   while s['index']<3:
    z=z+.01+.01*s['history'][-1];s['history'].append(z);s['index']+=1
   return z
  z,state,records=receding_feedback(torch.tensor(.2),original,feedback_indices=(0,1),advance=advance,rollout=rollout,readout=lambda x:x,target=torch.tensor(.4),reference_rgb=torch.tensor(.23),learning_rate=.1,step_rms=.03,cumulative_rms_budget=.04,quality_rms_budget=.2,torch=torch)
  self.assertEqual(calls,[0,1]);self.assertEqual(state['index'],2);self.assertEqual(original['index'],0);self.assertEqual(len(records),2);self.assertLessEqual(records[-1]['cumulative_accepted_rms'],.04000001)
 def test_config_and_unknown_arm(self):
  import json
  from pathlib import Path
  from runtime.public_statistic.phase1 import validate_config
  c=json.loads((Path(__file__).parents[1]/'configs/public_luma_phase1.json').read_text());validate_config(c)
  c['arms']=c['arms'][:-1]
  with self.assertRaises(ValueError):validate_config(c)
  with self.assertRaises(ValueError):perturb(np.zeros((2,4,4,3)),'UNKNOWN',2/255,np)
 def test_total_quality_not_waived(self):
  import json
  from pathlib import Path
  from runtime.public_statistic.phase1 import causal_evaluation
  c=json.loads((Path(__file__).parents[1]/'configs/public_luma_phase1.json').read_text());arms={}
  for name in c['arms']:
   q=np.zeros((16,2));quality=np.zeros(16)
   if name.startswith('X'):q[:,0]=c['amplitude']*(-1 if name.endswith('MINUS') else 1)
   if name.startswith('Y'):q[:,1]=c['amplitude']*(-1 if name.endswith('MINUS') else 1)
   if name=='X_PLUS':quality[-1]=.1
   arms[name]={label:dict(q=q.tolist(),per_frame_rmse_source=quality.tolist(),total_rmse_source=float(np.sqrt(np.mean(quality**2)))) for label in ('preencode','mp4')}
  result=causal_evaluation(arms,c,np)['mp4']['arms']['X_PLUS']['by_off']['OFF1']
  self.assertEqual(result['passing_frames'],15);self.assertFalse(result['raw_diagonal_diagnostic_pass']);self.assertFalse(result['total_quality_within_budget'])
 def test_commands_only(self):
  import json
  from pathlib import Path
  c=json.loads((Path(__file__).parents[1]/'configs/public_luma_phase1.json').read_text())
  self.assertIn('rgb24',decode_command('input.mp4',8,16));self.assertIn('libx264',encode_command('out.mp4',512,320,c))


class AggregateTests(unittest.TestCase):
 def setUp(self):
  import json
  from pathlib import Path
  self.c=json.loads((Path(__file__).parents[1]/'configs/public_luma_phase1.json').read_text());self.a=self.c['amplitude']
 def fixture(self,mat=None):
  if mat is None:mat=np.eye(2)
  result={}
  for sid in ('P50','JumpingJack'):
   arms={}
   for arm in self.c['arms']:
    q=np.zeros((16,2));quality=np.zeros(16)
    if arm[0] in 'XY':q[:]=self.a*mat[:,0 if arm[0]=='X' else 1]*(1 if arm.endswith('PLUS') else -1)
    arms[arm]={st:dict(q=q.tolist(),per_frame_rmse_source=quality.tolist(),total_rmse_source=0.) for st in ('preencode','mp4')}
   result[sid]=dict(arms=arms)
  return result
 def evaluate(self,r):
  from runtime.public_statistic.phase1 import aggregate_evaluation
  return aggregate_evaluation(r,self.c,np)['mp4']
 def test_rotation_reflection_mix(self):
  for M in (np.array([[0.,-1],[1,0]]),np.diag([-1.,1]),np.array([[1.,.3],[.2,1]])):
   e=self.evaluate(self.fixture(M));self.assertTrue(e['meets_method_gate']);self.assertEqual(e['C_denominator'],32)
  e=self.evaluate(self.fixture(np.array([[0.,-1],[1,0]])))
  self.assertFalse(e['contents']['P50']['rows'][0]['raw_diagnostic_only']['X_PLUS__OFF1']['near_diagonal_diagnostic_pass'])
 def test_low_absolute_gain(self):self.assertFalse(self.evaluate(self.fixture(.1*np.eye(2)))['meets_method_gate'])
 def test_local_collapse(self):
  r=self.fixture()
  for arm in ('Y_PLUS','Y_MINUS'):
   for i in (0,1):r['P50']['arms'][arm]['mp4']['q'][i]=[0,0]
  e=self.evaluate(r);self.assertTrue(e['common_spectrum']['passes']);self.assertFalse(e['meets_method_gate'])
 def test_content_rotation(self):
  r=self.fixture();rot=self.fixture(np.array([[0.,-1],[1,0]]));r['JumpingJack']=rot['JumpingJack'];self.assertFalse(self.evaluate(r)['meets_method_gate'])
 def test_same_direction_bias(self):
  r=self.fixture()
  for sid in r:
   for arm in ('X_PLUS','X_MINUS'):
    r[sid]['arms'][arm]['mp4']['q']=(np.asarray(r[sid]['arms'][arm]['mp4']['q'])+[2*self.a,0]).tolist()
  self.assertFalse(self.evaluate(r)['meets_method_gate'])
 def test_disjoint_bad_frames(self):
  r=self.fixture()
  for i,arm in enumerate(('X_PLUS','X_MINUS','Y_PLUS','Y_MINUS')):r['P50']['arms'][arm]['mp4']['per_frame_rmse_source'][i]=.1
  e=self.evaluate(r);self.assertEqual(e['contents']['P50']['common_passing_frames'],12);self.assertFalse(e['meets_method_gate'])
 def test_off_total_quality(self):
  r=self.fixture();r['P50']['arms']['OFF2']['mp4']['total_rmse_source']=.1;self.assertFalse(self.evaluate(r)['meets_method_gate'])
 def test_total_bad_one_frame(self):
  r=self.fixture();x=r['P50']['arms']['X_PLUS']['mp4'];x['per_frame_rmse_source'][-1]=.1;x['total_rmse_source']=.025;e=self.evaluate(r);self.assertEqual(e['contents']['P50']['common_passing_frames'],15);self.assertFalse(e['meets_method_gate'])
 def test_missing_nonfinite(self):
  r=self.fixture();del r['JumpingJack']['arms']['X_MINUS'];e=self.evaluate(r);self.assertEqual(e['status'],'INCOMPLETE');self.assertIsNone(e['common_C'])
  r=self.fixture();r['P50']['arms']['X_PLUS']['mp4']['q'][0][0]=float('nan');self.assertEqual(self.evaluate(r)['status'],'INCOMPLETE')

if __name__=='__main__':unittest.main()
