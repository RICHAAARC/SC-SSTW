"""Real random tiny Wan modules, CPU only; no pretrained weights."""
import unittest
import torch
from diffusers import AutoencoderKLWan,WanTransformer3DModel,UniPCMultistepScheduler
from runtime.public_statistic.checkpointing import CheckpointLedger,checkpoint_decode
from runtime.public_statistic.phase2 import WanTerminalAdapter,SolverState
from runtime.public_statistic.phase2_execution import ResourceGuard,execute_arms,initial_result
from tests.test_phase2_executable import config


def tiny_vae(temporal_two=False):
    return AutoencoderKLWan(base_dim=4,decoder_base_dim=4,z_dim=2,dim_mult=[1,2,4] if temporal_two else [1,2],num_res_blocks=1,temperal_downsample=[True,True] if temporal_two else [True],latents_mean=[0.,0.],latents_std=[1.,1.],scale_factor_temporal=4 if temporal_two else 2,scale_factor_spatial=4 if temporal_two else 2).eval().requires_grad_(False)


def tiny_adapter(checkpointed=False):
    torch.set_num_threads(1);torch.manual_seed(11)
    t=WanTransformer3DModel(patch_size=(1,2,2),num_attention_heads=1,attention_head_dim=8,in_channels=2,out_channels=2,text_dim=8,freq_dim=8,ffn_dim=16,num_layers=1,rope_max_seq_len=16).to(torch.bfloat16)
    for name,p in t.named_parameters():
        if any(x in name.split('.') for x in t._keep_in_fp32_modules):p.data=p.data.float()
    v=tiny_vae()
    ledger=CheckpointLedger(dict(transformer_block=6,vae_chunk=6)) if checkpointed else None
    a=WanTerminalAdapter(t,v,torch.randn(1,3,8,dtype=torch.bfloat16),torch.zeros(1,3,8,dtype=torch.bfloat16),guidance_scale=5.,torch=torch,checkpoint_ledger=ledger)
    s=UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True,flow_shift=3.,solver_order=2);s.set_timesteps(6);s.set_begin_index(0)
    return a,torch.randn(1,2,3,4,4)*.1,SolverState(s,0)


class CheckpointTests(unittest.TestCase):
    def test_native_decode_13_chunks_to49_gradient_and_cache_restore(self):
        torch.set_num_threads(1);torch.manual_seed(13)
        v=tiny_vae(True);z=torch.randn(1,2,13,2,2)*.1
        original=v.decoder.forward;results=[];ledger=CheckpointLedger(dict(transformer_block=0,vae_chunk=13))
        for checkpointed in (False,True):
            x=z.clone().requires_grad_()
            rgb=checkpoint_decode(v,x,ledger) if checkpointed else v.decode(x,return_dict=False)[0]
            loss=rgb.square().mean();grad=torch.autograd.grad(loss,x)[0]
            results.append((rgb.detach(),loss.detach(),grad))
            self.assertEqual(rgb.shape,(1,3,49,8,8));self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)
            self.assertEqual(v.decoder.forward,original);self.assertTrue(all(y is None for y in v._feat_map))
        self.assertTrue(torch.equal(results[0][0],results[1][0]))
        torch.testing.assert_close(results[0][2],results[1][2],rtol=2e-5,atol=1e-7)
        self.assertEqual(ledger.counts['vae_chunk']['recompute'],dict(attempted=13,completed=13))
        self.assertEqual(len(ledger.boundaries[0]['forward']['chunks']),13)
        self.assertTrue(all(row['unique_storage_bytes']>0 for row in ledger.boundaries[0]['forward']['chunks']))
    def test_cache_gradient_crosses_chunk_boundary(self):
        torch.set_num_threads(1);torch.manual_seed(2);v=tiny_vae();z=torch.randn(1,2,3,2,2)*.1;gradients=[]
        for flag in (False,True):
            x=z.clone().requires_grad_();ledger=CheckpointLedger(dict(transformer_block=0,vae_chunk=3))
            out=checkpoint_decode(v,x,ledger) if flag else v.decode(x,return_dict=False)[0]
            gradient=torch.autograd.grad(out[:,:,1:3].square().mean(),x)[0];gradients.append(gradient)
        self.assertGreater(float(gradients[0][:,:,0].abs().sum()),0)
        torch.testing.assert_close(*gradients,rtol=2e-5,atol=1e-7)
    def test_recompute_budget_and_native_forward_restoration(self):
        torch.set_num_threads(1);v=tiny_vae();original=v.decoder.forward
        ledger=CheckpointLedger(dict(transformer_block=0,vae_chunk=0));x=torch.randn(1,2,3,2,2,requires_grad=True)
        out=checkpoint_decode(v,x,ledger)
        self.assertEqual(v.decoder.forward,original)
        with self.assertRaisesRegex(RuntimeError,'RECOMPUTE_BUDGET_vae_chunk'):torch.autograd.grad(out.sum(),x)
        self.assertEqual(v.decoder.forward,original);self.assertEqual(ledger.counts['vae_chunk']['recompute']['attempted'],0)
    def test_actual_unipc_three_arms_equivalent_with_separate_counts(self):
        results=[]
        for enabled in (False,True):
            a,z,s=tiny_adapter(enabled);c=config();c['generation'].update(frames=5,height=8,width=8);c['budget']['saved_frames']=15
            guard=ResourceGuard(c['budget']);a.resource_guard=guard;r=initial_result(c);images={}
            execute_arms(a,z,s,c,r,lambda:None,lambda arm,rgb,off:images.update({arm:rgb.detach().clone()}))
            self.assertEqual(guard.counts['transformer_calls'],36);self.assertEqual(guard.counts['vae_calls'],7);self.assertEqual(guard.counts['backward_calls'],2)
            self.assertEqual(guard.counts,guard.completed)
            if enabled:
                for name in ('transformer_block','vae_chunk'):self.assertEqual(a.checkpoint_ledger.counts[name]['recompute'],dict(attempted=6,completed=6))
            results.append((images,r))
        for arm in ('OFF1','OFF2','Y_MINUS'):torch.testing.assert_close(results[0][0][arm],results[1][0][arm],rtol=0,atol=0)
        for left,right in zip(results[0][1]['arms']['Y_MINUS']['feedback'],results[1][1]['arms']['Y_MINUS']['feedback']):
            self.assertEqual(left['accepted'],right['accepted']);self.assertAlmostEqual(left['gradient_rms'],right['gradient_rms'],places=7)

if __name__=='__main__':unittest.main()
