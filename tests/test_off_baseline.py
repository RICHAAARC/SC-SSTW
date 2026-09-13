"""CPU wiring tests: mock model/scheduler, real CPU tensors; no GPU/model."""
import tempfile
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import numpy as np
from runtime.stage2.off_baseline import fresh_scheduler,stock_latent,manual_latent,load_fp32_vae,describe_rgb,attach_forward_counter

class UniPCMultistepScheduler:
    def __init__(self):self._step_index=0;self.history=[torch.tensor([4.])]
    @classmethod
    def from_config(cls,config):
        value=cls();value.config=config;return value
    def set_timesteps(self,n,device=None):self.timesteps=torch.arange(n,device=device);self._step_index=0
    def set_begin_index(self,i):self._step_index=i
    def step(self,velocity,t,z,return_dict=False):
        self._step_index+=1;self.history[0].add_(1);return (z+velocity,)

class FakePipe:
    def __init__(self):self.scheduler=UniPCMultistepScheduler();self.transformer=SimpleNamespace(dtype=torch.float32);self.calls=[]
    def prepare_latents(self,*args,**kwargs):return kwargs['latents'].clone()
    def encode_prompt(self,**kwargs):return torch.zeros(1),torch.zeros(1)
    def __call__(self,**kwargs):
        self.calls.append(kwargs);z=self.prepare_latents(latents=kwargs['latents']);return SimpleNamespace(frames=z+1)

G=dict(prompt='fixed',negative_prompt='fixed negative',height=320,width=512,frames=49,guidance=5.,seed=1275,max_sequence_length=512)

class OffWiringTests(unittest.TestCase):
    def test_fresh_scheduler_copies_config(self):
        original={'nested':[1]};a=fresh_scheduler(UniPCMultistepScheduler,original);b=fresh_scheduler(UniPCMultistepScheduler,original)
        a.config['nested'][0]=9;self.assertEqual(b.config['nested'],[1]);self.assertEqual(original['nested'],[1])
    def test_stock_receives_same_initial_and_restores_preparer(self):
        pipe=FakePipe();initial=torch.randn(1,1,2,3,4);before=initial.clone();method=pipe.prepare_latents.__func__
        with tempfile.TemporaryDirectory() as d:
            result=stock_latent(pipe,initial,G,50,Path(d),torch)
            saved=torch.load(Path(d)/'actual_initial_latent.pt',weights_only=True)
        self.assertTrue(torch.equal(initial,before));self.assertTrue(torch.equal(saved,initial));self.assertTrue(torch.equal(result,initial+1))
        self.assertEqual(pipe.calls[0]['num_inference_steps'],50);self.assertEqual(pipe.calls[0]['output_type'],'latent')
        self.assertIs(pipe.prepare_latents.__func__,method)
    def test_stock_mismatch_is_not_silently_corrected(self):
        class BadPipe(FakePipe):
            def prepare_latents(self,*args,**kwargs):return kwargs['latents']+1
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'differs'):stock_latent(BadPipe(),torch.zeros(1,1,2,3,4),G,8,Path(d),torch)
    def test_manual_is_real_fork_path_16_mock_forwards(self):
        pipe=FakePipe();z=torch.zeros(1,1,2,3,4);calls=[]
        def v(*args):calls.append(1);return torch.ones_like(z)*2
        with tempfile.TemporaryDirectory() as d,patch('runtime.stage2.off_baseline.velocity',side_effect=v),patch('runtime.stage2.wan_translation.velocity',side_effect=v):
            final=manual_latent(pipe,z,G,Path(d),torch)
        self.assertEqual(len(calls),16);self.assertEqual(pipe.scheduler._step_index,6)
        self.assertTrue(torch.equal(final,z+16));self.assertTrue(torch.equal(z,torch.zeros_like(z)))
    def test_fp32_uses_new_checkpoint_load(self):
        class Vae:
            record=None
            @classmethod
            def from_pretrained(cls,*args,**kwargs):cls.record=(args,kwargs);return cls()
            def eval(self):return self
        result=load_fp32_vae(Vae,dict(id='source',revision='fixed'),torch)
        self.assertIsInstance(result,Vae);self.assertEqual(Vae.record[1]['torch_dtype'],torch.float32)
        self.assertEqual(Vae.record[1]['revision'],'fixed');self.assertEqual(Vae.record[1]['subfolder'],'vae')
    def test_module_counter_survives_forward_replacement(self):
        class Transformer(torch.nn.Module):
            def forward(self,hidden_states,timestep):return hidden_states+1
        module=Transformer();counts={'transformer_calls':0};trace=[]
        handle=attach_forward_counter(module,counts,trace,lambda:'P8')
        x=torch.zeros(1,1,2,3,4);t=torch.tensor([999])
        self.assertTrue(torch.equal(module(hidden_states=x,timestep=t),x+1))
        original=module.forward
        module.forward=lambda **kwargs:original(**kwargs)*2
        self.assertTrue(torch.equal(module(hidden_states=x,timestep=t),2*(x+1)))
        module.forward=original
        module(hidden_states=x,timestep=t)
        self.assertEqual(counts['transformer_calls'],3);self.assertEqual(len(trace),3)
        handle.remove();module(hidden_states=x,timestep=t);self.assertEqual(counts['transformer_calls'],3)

    def test_cli_preserves_existing_exit_without_gpu_worker(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);out=base/'run01';cfg=base/'config.json';cfg.write_text('{}')
            prior=base/'run01.execution_exit.json';prior.write_text('prior evidence')
            result=subprocess.run([sys.executable,'-m','experiments.stage2.run_off_baseline','--config',str(cfg),'--output',str(out)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('retain prior execution records',result.stderr)
            self.assertEqual(prior.read_text(),'prior evidence');self.assertFalse(out.exists())
    def test_notebook_rejects_old_output_before_writing_commit(self):
        nb=json.loads((Path(__file__).resolve().parents[1]/'notebooks/stage2_off_baseline.ipynb').read_text())
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run01';out.mkdir();prior=out/'source_commit.txt';prior.write_text('old-version')
            with self.assertRaises(FileExistsError):exec(compile(''.join(nb['cells'][9]['source']),'GPU cell guard','exec'),{'OUTPUT':out})
            self.assertEqual(prior.read_text(),'old-version')

    def test_descriptors_do_not_manufacture_clarity_pass(self):
        a=np.full((3,4,5,3),42,dtype=np.uint8);r=describe_rgb(a,np)
        self.assertEqual(r['per_pair_temporal_max'],[0.,0.]);self.assertEqual(r['std'],0.)
        self.assertEqual(r['visual_clarity_and_motion'],'NOT_AUTOMATICALLY_ADJUDICATED')

if __name__=='__main__':unittest.main()
