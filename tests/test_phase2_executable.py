"""CPU surrogate engineering only; never download weights or touch CUDA."""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from tests.test_phase2_terminal_adapter import setup, SmallTransformer, SmallVAE
from runtime.public_statistic.phase2_execution import ResourceGuard, initial_result, validate_config, execute_arms, load_wan, write
from main.sc_sstw.terminal_feedback import terminal_feedback
from experiments.public_statistic.run_phase2 import finalize_stopped, supervise

ROOT=Path(__file__).resolve().parents[1]

def config():
    c=json.loads((ROOT/'configs/public_luma_phase2_gpu.json').read_text())
    c['generation'].update(steps=6,frames=2,height=4,width=4)
    c['feedback']['after_indices']=[3,4]
    c['budget'].update(transformer_calls=36,saved_frames=6)
    return c

class ExecutableTests(unittest.TestCase):
    def test_bounded_three_arms_actual_unipc(self):
        a,z,s=setup();c=config();validate_config(c)
        guard=ResourceGuard(c['budget']);a.resource_guard=guard
        result=initial_result(c);saved=[]
        execute_arms(a,z,s,c,result,lambda:None,lambda arm,rgb,off:saved.append(arm))
        self.assertEqual(saved,['OFF1','OFF2','Y_MINUS'])
        self.assertEqual(guard.counts['transformer_calls'],36)
        self.assertEqual(guard.counts['vae_calls'],7)
        self.assertEqual(guard.counts['backward_calls'],2)
        self.assertEqual(result['prefix']['next_index'],3)
        self.assertEqual(s.next_index,3)
        self.assertEqual(result['arms']['OFF1']['float_q'],result['arms']['OFF2']['float_q'])
        self.assertEqual(len(result['arms']['Y_MINUS']['feedback']),2)
    def test_strict_detach_zero_nonfinite_and_oom(self):
        a,z,s=setup();ref=a.rollout(z,s);target=a.readout(ref)+.01
        kwargs=dict(learning_rate=1.,step_rms=.1,quality_rms_budget=.1,torch=torch,strict=True)
        for callback in (lambda x,s:ref.detach(),lambda x,s:ref+x.sum()*0,lambda x,s:ref+x.sum()*float('nan')):
            with self.assertRaises((RuntimeError,ValueError)):
                terminal_feedback(z,s,callback,a.readout,target,ref,**kwargs)
        def oom(x,s):raise torch.OutOfMemoryError('injected OOM')
        with self.assertRaises(torch.OutOfMemoryError):terminal_feedback(z,s,oom,a.readout,target,ref,**kwargs)
    def test_call_and_deadline_budget(self):
        c=config();c['budget']['transformer_calls']=1;clock=[0.]
        guard=ResourceGuard(c['budget'],clock=lambda:clock[0]);guard.consume('transformer_calls')
        with self.assertRaisesRegex(RuntimeError,'CALL_BUDGET'):guard.consume('transformer_calls')
        self.assertEqual(guard.counts['transformer_calls'],1)
        clock[0]=1800.
        with self.assertRaises(TimeoutError):guard.consume('vae_calls')
        self.assertEqual(guard.counts['vae_calls'],0)
    def test_fatal_during_forecast_no_continuation(self):
        a,z,s=setup();c=config();c['budget']['transformer_calls']=20
        guard=ResourceGuard(c['budget']);a.resource_guard=guard;result=initial_result(c);saved=[]
        with self.assertRaisesRegex(RuntimeError,'CALL_BUDGET'):
            execute_arms(a,z,s,c,result,lambda:None,lambda arm,rgb,off:saved.append(arm))
        self.assertEqual(saved,['OFF1','OFF2']);self.assertEqual(guard.counts['transformer_calls'],20)
        self.assertEqual(result['arms']['Y_MINUS']['status'],'RUNNING')
    def test_fixed_rows_survive_hard_stop(self):
        c=json.loads((ROOT/'configs/public_luma_phase2_gpu.json').read_text());r=initial_result(c)
        r['arms']['OFF1']['status']='COMPLETE'
        for row in r['arms']['OFF1']['rows']:row.update(status='COMPLETE',q=[0.,0.])
        r['arms']['OFF2']['status']='RUNNING'
        with tempfile.TemporaryDirectory() as tmp:
            write(Path(tmp)/'result.json',r);finalize_stopped(tmp,c,'WALL_BUDGET');got=json.loads((Path(tmp)/'result.json').read_text())
        self.assertEqual(sum(len(x['rows']) for x in got['arms'].values()),147)
        self.assertEqual(got['arms']['OFF1']['status'],'COMPLETE')
        self.assertEqual(got['arms']['OFF2']['rows'][0]['status'],'FAILED_WALL_BUDGET')
        self.assertEqual(got['arms']['Y_MINUS']['rows'][0]['status'],'NOT_EXECUTED')
    def test_real_supervisor_timeout_without_gpu(self):
        import subprocess,sys
        original=subprocess.Popen;c=config();c['budget']['wall_seconds']=.05
        def sleeping_child(command,**kwargs):return original([sys.executable,'-c','import time;time.sleep(10)'],**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'config.json';write(cp,c);out=Path(tmp)/'run'
            with patch('experiments.public_statistic.run_phase2.subprocess.Popen',side_effect=sleeping_child):code=supervise(cp,out)
            self.assertNotEqual(code,0)
            r=json.loads((out/'result.json').read_text());self.assertEqual(r['supervisor_stop'],'WALL_BUDGET')
            self.assertTrue(all(x['status']=='NOT_EXECUTED' for x in r['arms'].values()))
    def test_host_memory_is_diagnostic_only(self):
        import subprocess,sys
        original=subprocess.Popen
        for diagnostic in (64*2**30, PermissionError('diagnostic unavailable')):
            with self.subTest(diagnostic=str(diagnostic)):
                c=config();self.assertNotIn('host_rss_gib',c['budget']);validate_config(c)
                def short_child(command,**kwargs):
                    return original([sys.executable,'-c','import time;time.sleep(.3)'],**kwargs)
                kwargs={'side_effect':diagnostic} if isinstance(diagnostic,Exception) else {'return_value':diagnostic}
                with tempfile.TemporaryDirectory() as tmp:
                    cp=Path(tmp)/'config.json';write(cp,c);out=Path(tmp)/'run'
                    with patch('experiments.public_statistic.run_phase2.subprocess.Popen',side_effect=short_child), patch('experiments.public_statistic.run_phase2.process_tree_rss',**kwargs):
                        code=supervise(cp,out)
                    record=json.loads((out/'execution_exit.json').read_text())
                    self.assertEqual(code,0);self.assertIsNone(record['stop'])
                    memory=record['host_memory_diagnostic']
                    self.assertEqual(memory['status'],'unavailable' if isinstance(diagnostic,Exception) else 'available')
                    if not isinstance(diagnostic,Exception):self.assertEqual(memory['peak_tree_rss_bytes'],64*2**30)
    def test_mixed_parameter_input_dtype_and_terminal_gradient(self):
        a,z,state=setup()
        class Mixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale_shift_table=torch.nn.Parameter(torch.zeros(1,dtype=torch.float32))
                self.patch_embedding=torch.nn.Conv3d(3,3,1).to(torch.bfloat16)
                with torch.no_grad():
                    self.patch_embedding.weight.fill_(.02);self.patch_embedding.bias.zero_()
                self.seen=[]
            def forward(self,hidden_states,timestep,encoder_hidden_states,**kwargs):
                self.seen.append(hidden_states.dtype)
                return (self.patch_embedding(hidden_states),)
        model=Mixed()
        self.assertEqual(next(model.parameters()).dtype,torch.float32)
        with self.assertRaises(RuntimeError):model(z,torch.tensor(1),None)
        model.seen.clear()
        adapter=type(a)(model,a.vae,a.prompt,a.negative,guidance_scale=5.,torch=torch)
        latent=z.detach().clone().requires_grad_(True)
        rgb=adapter.rollout(latent,state)
        gradient=torch.autograd.grad(adapter.readout(rgb).square().sum(),latent)[0]
        self.assertEqual(model.seen,[torch.bfloat16]*12)  # six steps, cond and uncond
        self.assertEqual(latent.dtype,torch.float32);self.assertEqual(rgb.dtype,torch.float32)
        self.assertEqual(next(model.parameters()).dtype,torch.float32)
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.abs().sum()),0.)
    def test_real_loader_flow_mocked_weights(self):
        events=[];a,z,s=setup()
        class DeviceModel(SmallTransformer):
            def __init__(self):super().__init__();self.config=types.SimpleNamespace(in_channels=3)
            def to(self,*args,**kwargs):events.append('transformer_to');return self
        class Encoder:
            def to(self,d):events.append('encoder_'+str(d));return self
        class VAE(SmallVAE):
            def __init__(self):super().__init__();self.config.temperal_downsample=[False];self.config.scale_factor_temporal=1;self.config.scale_factor_spatial=2
            def to(self,*args,**kwargs):events.append('vae_to');return self
            @classmethod
            def from_pretrained(cls,*args,**kwargs):events.append(('vae_load',kwargs));return cls()
        class Pipe:
            @classmethod
            def from_pretrained(cls,*args,**kwargs):
                events.append(('pipeline_load',kwargs));p=cls();p.transformer=DeviceModel();p.text_encoder=Encoder();p.config=types.SimpleNamespace();p.vae=None;p.scheduler=s.scheduler;return p
            def encode_prompt(self,**kwargs):events.append('encode');return torch.ones(1,2,3),torch.zeros(1,2,3)
            def prepare_latents(self,*args):events.append('prepare');return z
        fake=types.SimpleNamespace(bfloat16=torch.bfloat16,float32=torch.float32,no_grad=torch.no_grad,device=lambda _:torch.device('cpu'),Generator=lambda **kwargs:torch.Generator(),cuda=types.SimpleNamespace(empty_cache=lambda:None),nn=torch.nn)
        c=config();guard=ResourceGuard(c['budget']);records=[]
        adapter,latent,state=load_wan(c,guard,torch=fake,pipeline_class=Pipe,vae_class=VAE,record=records.append)
        self.assertTrue(torch.equal(z,latent));self.assertEqual(state.next_index,0)
        self.assertLess(events.index('encode'),events.index('transformer_to'))
        load=next(x for x in events if isinstance(x,tuple) and x[0]=='vae_load')
        self.assertEqual(load[1]['torch_dtype'],torch.float32)
        self.assertNotIn('revision',load[1]);self.assertEqual(records[0]['scheduler_class'],'UniPCMultistepScheduler')
    def test_notebook_syntax_and_first_cell(self):
        import ast
        nb=json.loads((ROOT/'notebooks/public_statistic_phase2_gpu.ipynb').read_text())
        self.assertEqual(''.join(nb['cells'][0]['source']).splitlines(),['from google.colab import drive',"drive.mount('/content/drive')"])
        for cell in nb['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        text=''.join(''.join(c['source']) for c in nb['cells'])
        self.assertNotIn('files.upload',text);self.assertNotIn('diffusers==',text)

if __name__=='__main__':unittest.main()
