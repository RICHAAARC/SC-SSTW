import unittest,types
import torch
from diffusers import UniPCMultistepScheduler
from main.sc_sstw.terminal_feedback import clone_graph_state,terminal_feedback
from runtime.public_statistic.phase2 import SolverState,WanTerminalAdapter,FeedbackConfig,controlled_run

class SmallTransformer(torch.nn.Module):
 def __init__(self):
  super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.08))
 def forward(self,hidden_states,timestep,encoder_hidden_states,attention_kwargs=None,return_dict=False):
  return (self.weight*hidden_states+encoder_hidden_states.mean()*.001,)
class SmallVAE(torch.nn.Module):
 def __init__(self):
  super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.8));self.config=types.SimpleNamespace(z_dim=3,latents_mean=[.01,.02,.03],latents_std=[.9,1.,1.1]);self.cleared=0
 def decode(self,z,return_dict=False):return (torch.tanh(self.weight*z),)
 def clear_cache(self):self.cleared+=1

def setup():
 torch.manual_seed(3);s=UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True,flow_shift=3,solver_order=2);s.set_timesteps(6)
 z=torch.randn((1,3,2,4,4))*.05
 adapter=WanTerminalAdapter(SmallTransformer(),SmallVAE(),torch.ones(1,2,3),torch.zeros(1,2,3),guidance_scale=5,torch=torch)
 return adapter,z,SolverState(s,0)
def config(quality=.1):return FeedbackConfig((2,3),10.,.03,.05,quality,.01,0,1)
class Tests(unittest.TestCase):
 def test_nonleaf_full_history_clone(self):
  a,z,s=setup();z.requires_grad_();z,s=a.advance(z,s);clone=clone_graph_state(s,torch)
  self.assertIsNot(s.scheduler.model_outputs[-1],clone.scheduler.model_outputs[-1]);self.assertIsNotNone(clone.scheduler.model_outputs[-1].grad_fn)
  self.assertEqual(vars(s.scheduler).keys(),vars(clone.scheduler).keys());clone.scheduler.model_outputs[-1].add_(1);self.assertFalse(torch.equal(clone.scheduler.model_outputs[-1],s.scheduler.model_outputs[-1]))
 def test_gradient_finite_difference_and_state(self):
  a,z,s=setup()
  for _ in range(2):z,s=a.advance(z,s)
  before=clone_graph_state(s,torch);x=z.detach().clone().requires_grad_();rgb=a.rollout(x,clone_graph_state(s,torch));loss=a.readout(rgb).square().sum();g=torch.autograd.grad(loss,x)[0];v=g/g.norm();h=.001
  with torch.no_grad():
   plus=a.readout(a.rollout(x+h*v,clone_graph_state(s,torch))).square().sum();minus=a.readout(a.rollout(x-h*v,clone_graph_state(s,torch))).square().sum()
  finite=(plus-minus)/(2*h);self.assertAlmostEqual(float(finite),float((g*v).sum()),delta=2e-5);self.assertGreater(float(g.norm()),0);self.assertEqual(s.next_index,before.next_index)
  for key in ('lower_order_nums','this_order','_step_index','_begin_index'):self.assertEqual(getattr(s.scheduler,key),getattr(before.scheduler,key))
  self.assertTrue(torch.equal(s.scheduler.last_sample,before.scheduler.last_sample));self.assertTrue(all(p.grad is None for p in a.transformer.parameters()))
 def test_off_equivalence_and_acceptance(self):
  a,z,s=setup()
  with torch.no_grad():off=a.rollout(z,clone_graph_state(s,torch))
  disabled=controlled_run(a,z,s,config(),off1_terminal_rgb=off,enabled=False);self.assertTrue(torch.equal(off,disabled['rgb']));self.assertEqual(disabled['feedback'],[])
  enabled=controlled_run(a,z,s,config(),off1_terminal_rgb=off,enabled=True);self.assertEqual(len(enabled['feedback']),2);self.assertTrue(all(x['accepted'] for x in enabled['feedback']));self.assertLessEqual(enabled['cumulative_accepted_rms'],.05000001);self.assertEqual(s.next_index,0)
  self.assertTrue(torch.equal(enabled['reference_rgb'],off));self.assertTrue(torch.equal(enabled['target'][:,1],a.readout(off)[:,1]))
 def test_rejection_normal_advance(self):
  a,z,s=setup()
  with torch.no_grad():off=a.rollout(z,clone_graph_state(s,torch))
  result=controlled_run(a,z,s,config(1e-12),off1_terminal_rgb=off,enabled=True);self.assertTrue(all(not x['accepted'] for x in result['feedback']));self.assertTrue(torch.equal(off,result['rgb']));self.assertEqual(result['state'].next_index,6)
 def test_detach_failure_and_cache_cleanup(self):
  a,z,s=setup()
  with torch.no_grad():ref=a.rollout(z,clone_graph_state(s,torch))
  target=a.readout(ref)+.01
  _,r=terminal_feedback(z,s,lambda x,state:a.rollout(x,state).detach(),a.readout,target,ref,learning_rate=1,step_rms=.1,quality_rms_budget=.1,torch=torch);self.assertEqual(r['reason'],'CALLBACK_OR_DIFFERENTIABILITY_FAILURE')
  def bad(z,return_dict=False):raise RuntimeError('decoder injected failure')
  a.vae.decode=bad;before=a.vae.cleared
  with self.assertRaises(RuntimeError):a.decode_float(z)
  self.assertEqual(a.vae.cleared,before+1)
 def test_index_guard(self):
  a,z,s=setup();s.next_index=1
  with self.assertRaises(ValueError):a.advance(z,s)
 def test_guidance_configuration_guard(self):
  a,z,s=setup();a.guidance_scale=1
  with torch.no_grad():off=a.rollout(z,clone_graph_state(s,torch))
  with self.assertRaises(ValueError):controlled_run(a,z,s,config(),off1_terminal_rgb=off,enabled=False)
 def test_unsupported_path(self):
  with self.assertRaises(ValueError):WanTerminalAdapter(SmallTransformer(),SmallVAE(),torch.ones(1),None,guidance_scale=1,torch=torch,expand_timesteps=True)
if __name__=='__main__':unittest.main()
