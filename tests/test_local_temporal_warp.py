import unittest,copy
from unittest.mock import patch
import torch
from main.sc_sstw.local_temporal_warp import warp_latent,ARMS,reference_rows
from runtime.stage2.local_temporal_warp import continue_five
from runtime.stage2.off_baseline import attach_forward_counter
class Scheduler:
    def __init__(self):
        self._step_index=45;self.model_outputs=[torch.ones(1)];self.last_sample=torch.ones(1);self.lower_order_nums=2;self.timestep_list=[44,43]
    def step(self,v,t,z,return_dict=False):
        self.model_outputs[0].add_(1);self.last_sample.add_(1);self.timestep_list.append(int(t));self._step_index+=1
        return (z+v,)
class Tests(unittest.TestCase):
    def setUp(self):
        self.z=torch.arange(64).float()[None,None,None,None].expand(2,3,13,40,64).clone()
    def test_off_and_outside_identity(self):
        for arm in ARMS:
            out,grid,mask=warp_latent(self.z,arm,torch)
            self.assertTrue(torch.equal(out[:,:,:,:,0],self.z[:,:,:,:,0]));self.assertTrue(torch.equal(out[:,:,6],self.z[:,:,6]))
            self.assertEqual(float(mask[0].max()),0);self.assertEqual(float(mask[:,0].max()),0);self.assertEqual(float(mask[12,20]),1)
            if arm.startswith('OFF'):self.assertTrue(torch.equal(out,self.z));self.assertNotEqual(out.data_ptr(),self.z.data_ptr())
    def test_signed_axis_and_no_fold(self):
        for arm in ARMS[2:]:
            out,grid,mask=warp_latent(self.z,arm,torch)
            coordinate=(grid[...,0]+1)*63/2 if arm.startswith('X') else (grid[...,1]+1)*39/2
            dim=-1 if arm.startswith('X') else -2
            self.assertGreater(float(torch.diff(coordinate,dim=dim).min()),.4763)
        plus,_,_=warp_latent(self.z,'X_PLUS',torch)
        self.assertAlmostEqual(float(plus[1,2,12,20,20]),19,places=4)
        self.assertAlmostEqual(float(plus[1,2,0,20,20]),21,places=4)
    def test_y_and_time_layout(self):
        z=torch.arange(40).float()[None,None,None,:,None].expand(2,3,13,40,64).clone()
        out,_,_=warp_latent(z,'Y_MINUS',torch)
        self.assertAlmostEqual(float(out[1,2,12,20,20]),21,places=4)
    def test_reject_nonfinite_shape(self):
        with self.assertRaises(ValueError):warp_latent(self.z[:,:,:12],'OFF1',torch)
        self.z[0,0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):warp_latent(self.z,'OFF1',torch)
    def test_six_forks_preserve_all_mock_history(self):
        snapshot=Scheduler();before=copy.deepcopy(snapshot);calls=[];outputs=[]
        def v(pipe,z,t,e,b):calls.append(b);return torch.ones_like(z)
        with patch('runtime.stage2.local_temporal_warp.velocity',v):
            for arm in ARMS:outputs.append(continue_five(None,torch.zeros(1),snapshot,list(range(50)),None,None,5))
        self.assertEqual(len(calls),60);self.assertEqual(snapshot._step_index,45);self.assertEqual(snapshot.timestep_list,before.timestep_list)
        self.assertTrue(torch.equal(snapshot.model_outputs[0],before.model_outputs[0]));self.assertTrue(torch.equal(snapshot.last_sample,before.last_sample))
        self.assertTrue(torch.equal(outputs[0],outputs[1]))
    def test_prehook_survives_forward_replacement(self):
        module=torch.nn.Identity();module.forward=lambda **kw:kw['hidden_states'];counts={'transformer_calls':0};trace=[]
        handle=attach_forward_counter(module,counts,trace,lambda:'test')
        module(hidden_states=torch.ones(1),timestep=torch.ones(1));module.forward=lambda **kw:kw['hidden_states']+1;module(hidden_states=torch.ones(1),timestep=torch.ones(1));handle.remove()
        self.assertEqual(counts['transformer_calls'],2)
    def test_missing_reference_retained(self):
        q=[dict(sample_index=i,q=None if i==0 else [.5,.5],valid=i>0,reason='INITIAL' if i==0 else 'VALID') for i in range(3)]
        p=[dict(sample_index=i,p_pixel=None,alignment_confirmed=True,time_seconds=i/5,status='REVIEWED',subjective_radius_px=2) for i in range(3)]
        rows=reference_rows(q,p);self.assertEqual(len(rows),3);self.assertFalse(any(r['eligible'] for r in rows))
    def test_signed_position_is_not_q_sign(self):
        q=[dict(sample_index=i,q=[.5,.5],valid=True,reason='VALID') for i in range(2)]
        p=[dict(sample_index=i,p_pixel=[100-i,100],alignment_confirmed=True,time_seconds=i/5,status='REVIEWED',subjective_radius_px=2) for i in range(2)]
        self.assertEqual(reference_rows(q,p)[1]['actual_displacement_pixel'],[-1,0])
    def test_pending_points_not_actual_displacement(self):
        q=[dict(sample_index=i,q=[.5,.5],valid=True,reason='VALID') for i in range(2)]
        p=[dict(sample_index=i,time_seconds=i/5,p_pixel=[100,100],alignment_confirmed=True,status='PENDING',subjective_radius_px=2) for i in range(2)]
        r=reference_rows(q,p)[1];self.assertIsNone(r['midpoint']);self.assertIsNone(r['actual_displacement_pixel'])
    def test_full_orchestration_mock_models_real_tensors(self):
        import tempfile,json,types,contextlib,sys
        from pathlib import Path
        from runtime.stage2.local_temporal_warp import run
        class Transformer(torch.nn.Module):
            dtype=torch.bfloat16
            config=types.SimpleNamespace(in_channels=16)
            def cache_context(self,key):return contextlib.nullcontext()
            def forward(self,**kw):return (torch.zeros_like(kw['hidden_states']),)
        class UniPCMultistepScheduler:
            config={}
            def set_timesteps(self,n,device):self.timesteps=torch.arange(n,device=device);self._step_index=0;self.model_outputs=[torch.ones(1)]
            def set_begin_index(self,i):pass
            def step(self,v,t,z,return_dict=False):self._step_index+=1;self.model_outputs[0].add_(1);return (z,)
        class Pipe:
            _execution_device='cpu'
            def __init__(self):self.transformer=Transformer();self.scheduler=UniPCMultistepScheduler()
            @classmethod
            def from_pretrained(cls,*a,**kw):return cls()
            def enable_model_cpu_offload(self):pass
            def maybe_free_model_hooks(self):pass
            def prepare_latents(self,*a):return torch.zeros(1,16,13,40,64)
            def encode_prompt(self,**kw):return torch.zeros(1),torch.zeros(1)
        class VAE:
            dtype=torch.float32
            def to(self,*a):return self
        fake=types.SimpleNamespace(__version__='0.35.2',WanPipeline=Pipe,AutoencoderKLWan=object)
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules,{'diffusers':fake}))
            for name,value in [('is_available',True),('is_bf16_supported',True),('get_device_name','CPU_MOCK'),('max_memory_allocated',0)]:stack.enter_context(patch('torch.cuda.'+name,return_value=value))
            for name in ['reset_peak_memory_stats','empty_cache']:stack.enter_context(patch('torch.cuda.'+name))
            stack.enter_context(patch('importlib.metadata.version',return_value='MOCK'))
            loader=stack.enter_context(patch('runtime.stage2.local_temporal_warp.load_fp32_vae',return_value=VAE()))
            decoder=stack.enter_context(patch('runtime.stage2.local_temporal_warp.decode_rgb',return_value=np.zeros((49,2,2,3),dtype=np.uint8)))
            encoder=stack.enter_context(patch('runtime.stage2.local_temporal_warp.encode'))
            stack.enter_context(patch('runtime.stage2.local_temporal_warp.read_public'))
            stack.enter_context(patch('runtime.stage2.local_temporal_warp.reference_package'))
            stack.enter_context(patch('runtime.stage2.local_temporal_warp.save_preview'))
            result=run(Path(__file__).parents[1]/'configs/stage2_local_temporal_warp.json',Path(tmp)/'run')
            self.assertEqual(result['counts'],dict(transformer_calls=150,vae_decodes=6,mp4=6))
            self.assertEqual(loader.call_count,1);self.assertEqual(decoder.call_count,6);self.assertEqual(encoder.call_count,12)
            self.assertTrue(all(s['status']=='COMPLETE_PENDING_EXTERNAL_REVIEW' for s in result['outputs'].values()))
            self.assertTrue(json.loads((Path(tmp)/'run/off_repeat_latent.json').read_text())['exact'])
if __name__=='__main__':unittest.main()
