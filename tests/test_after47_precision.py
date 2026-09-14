import unittest
import torch
from tests.test_phase2_checkpointing import tiny_adapter
from runtime.public_statistic.after47_precision import diagnose_precision
from runtime.public_statistic.phase2_execution import ResourceGuard
from main.sc_sstw.terminal_feedback import clone_graph_state

class PrecisionTests(unittest.TestCase):
    def test_fixed_snapshot_nine_evaluations_and_precision_promotion(self):
        a,z,s=tiny_adapter(True)
        a.checkpoint_ledger.limits.update(transformer_block=12,vae_chunk=9)
        with torch.no_grad():
            for _ in range(4):z,s=a.advance(z,s)
            target=a.readout(a.rollout(z,clone_graph_state(s,torch))).detach();target[:,1]-=2/255
        delta=torch.full_like(z,1e-5)
        snapshot=dict(latent=z.detach(),scheduler_state=s,target=target,delta=delta,opposite_delta=-delta,gradient=torch.ones_like(z))
        original_delta=delta.clone();weights=[v.detach().float().clone() for v in a.transformer.parameters()]
        guard=ResourceGuard(dict(transformer_calls=36,vae_calls=9,backward_calls=3,mp4=0,saved_frames=0,wall_seconds=60));a.resource_guard=guard
        r={};saved={};old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        diagnose_precision(a,snapshot,r,lambda:None,lambda label,g:saved.update({label:g}))
        self.assertEqual(guard.counts,guard.completed)
        self.assertEqual(guard.counts,dict(transformer_calls=36,vae_calls=9,backward_calls=3,mp4=0,saved_frames=0))
        self.assertEqual(len(r['evaluations']),9);self.assertEqual(len(saved),3)
        self.assertTrue(r['state_unchanged']);self.assertTrue(r['fixed_inputs_unchanged'])
        self.assertTrue(torch.equal(delta,original_delta))
        self.assertTrue(r['gradient_comparisons']['mixed_repeat']['exact'])
        for label,g in saved.items():
            self.assertEqual(g.device.type,'cpu')
            self.assertAlmostEqual(r['gradient_metrics'][label]['g_dot_saved_delta'],float((g.double()*original_delta.double()).sum()))
        for before,after in zip(weights,a.transformer.parameters()):self.assertTrue(torch.equal(before,after))
        self.assertEqual(a.input_dtype(),torch.float32)
        self.assertEqual(old,(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))
        self.assertNotIn('accepted',r)

if __name__=='__main__':unittest.main()
