import unittest
import torch
from tests.test_phase2_checkpointing import tiny_adapter
from runtime.public_statistic.after47_split import diagnose_split
from runtime.public_statistic.phase2_execution import ResourceGuard
from main.sc_sstw.terminal_feedback import clone_graph_state

class SplitTests(unittest.TestCase):
    def test_split_matches_coupled_gradient_and_fixed_budget(self):
        a,z,s=tiny_adapter(True);a.checkpoint_ledger.limits.update(transformer_block=16,vae_chunk=9)
        with torch.no_grad():
            for _ in range(4):z,s=a.advance(z,s)
            target=a.readout(a.rollout(z,clone_graph_state(s,torch))).detach();target[:,1]-=2/255
        x=z.detach().clone().requires_grad_(True);loss=(a.readout(a.rollout(x,clone_graph_state(s,torch)))-target).square().mean();expected=torch.autograd.grad(loss,x)[0]
        del x,loss
        a.resource_guard=ResourceGuard(dict(transformer_calls=12,vae_calls=3,backward_calls=5,mp4=0,saved_frames=0,wall_seconds=60))
        report={};saved={}
        diagnose_split(a,dict(latent=z,target=target,delta=torch.ones_like(z)*1e-5,scheduler_state=s),report,lambda:None,lambda name,g:saved.update({name:g}))
        self.assertEqual(a.resource_guard.counts,a.resource_guard.completed)
        self.assertEqual(a.resource_guard.counts,dict(transformer_calls=12,vae_calls=3,backward_calls=5,mp4=0,saved_frames=0))
        torch.testing.assert_close(saved['solver_vjp_1'],expected,rtol=2e-5,atol=1e-9)
        self.assertTrue(report['vae_repeat']['exact']);self.assertTrue(report['solver_repeat']['exact']);self.assertTrue(report['state_unchanged'])
        self.assertEqual(set(saved),{'vae_vjp_1','vae_vjp_2','solver_vjp_1','solver_vjp_2'})
        self.assertEqual([x['stage'] for x in report['residency_events']],['VAE_ONLY','SOLVER_ONLY'])

if __name__=='__main__':unittest.main()
