import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch=None
from main.sc_sstw.latent_translation import ARMS,translate_latent,evaluate_response
from runtime.stage2.wan_translation import fork_scheduler,continue_two

BUDGET=dict(common_valid_fraction=.8,minimum_common_pairs=10,minimum_singular_to_floor=5,numeric_floor=32*2**-23,maximum_condition=10)

def data(columns=((.02,.01),(-.01,.02)),bias=(0.,0.)):
    rows={}
    for a in ARMS:
        axis=0 if a.startswith('X') else 1
        sign=1 if a.endswith('PLUS') else -1
        shift=(0.,0.) if a.startswith('OFF') else tuple(sign*x+bias[d] for d,x in enumerate(columns[axis]))
        rows[a]=[dict(sample_index=i,time_seconds=i/5,q=[.4+.001*i+shift[0],.5+shift[1]] if i else None,valid=bool(i),reason='VALID' if i else 'INITIAL_FRAME') for i in range(21)]
    return rows

class UniPCMultistepScheduler:
    def __init__(self):self._step_index=6;self.model_outputs=[torch.tensor([3.])]
    def step(self,v,t,z,return_dict=False):
        self.model_outputs[0].add_(1);self._step_index+=1
        return (z+v,)

class TranslationTests(unittest.TestCase):
    @unittest.skipIf(torch is None,"CPU torch download pending")
    def test_actual_cpu_grid_sign_border_and_time(self):
        z=torch.arange(5,dtype=torch.float32).reshape(1,1,1,1,5).expand(1,2,3,4,5).clone()
        for dx in (.25,-.25):
            out=translate_latent(z,dx,0,torch)
            expected=torch.tensor([max(0.,min(4.,i-dx)) for i in range(5)])
            self.assertTrue(torch.allclose(out[0,0,0,1],expected,atol=1e-6))
            self.assertTrue(torch.equal(out[:,:,0],out[:,:,2]))
        zy=z.transpose(-1,-2).contiguous()
        self.assertTrue(torch.allclose(translate_latent(zy,0,.25,torch),translate_latent(z,.25,0,torch).transpose(-1,-2),atol=1e-6))
    @unittest.skipIf(torch is None,"CPU torch download pending")
    def test_off_exact_clone(self):
        z=torch.randn(1,2,3,4,5);v=translate_latent(z,0,0,torch)
        self.assertTrue(torch.equal(z,v));self.assertNotEqual(z.data_ptr(),v.data_ptr())
    @unittest.skipIf(torch is None,"CPU torch download pending")
    def test_bf16_dtype_and_no_input_mutation(self):
        z=torch.arange(60).reshape(1,1,3,4,5).to(torch.bfloat16);old=z.clone()
        out=translate_latent(z,.25,0,torch)
        self.assertEqual(out.dtype,z.dtype);self.assertTrue(torch.equal(z,old))
    @unittest.skipIf(torch is None,"CPU torch download pending")
    def test_invalid_values(self):
        z=torch.zeros(1,1,1,4,4)
        with self.assertRaises(ValueError):translate_latent(z,float('nan'),0,torch)
        with self.assertRaises(ValueError):translate_latent(z,.5,0,torch)
        z[0,0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):translate_latent(z,0,0,torch)
    def test_mixed_axes_pass_not_diagonal_gate(self):
        result=evaluate_response(data(),.25,BUDGET)
        self.assertEqual(result['status'],'ENGINEERING_RESPONSE_READY')
    def test_collinear_rejected(self):
        result=evaluate_response(data(((.02,.01),(.04,.02))),.25,BUDGET)
        self.assertFalse(result['checks']['two_direction_effect'])
    def test_common_bias_same_direction_rejected(self):
        result=evaluate_response(data(bias=(.1,.1)),.25,BUDGET)
        self.assertTrue(result['checks']['two_direction_effect']);self.assertFalse(result['checks']['signed_response'])
    def test_per_off_and_missing_original_axis(self):
        rows=data()
        for i in range(1,8):rows['Y_MINUS'][i].update(valid=False,q=None,reason='NO_MOTION_SUPPORT')
        v=evaluate_response(rows,.25,BUDGET)
        self.assertEqual(v['common_indices'],list(range(8,21)));self.assertEqual(len(v['excluded']),8)
        self.assertEqual(v['status'],'OPERATIONAL_BLOCKED')
        self.assertIn('even_rows',v['axes']['X']['by_off']['OFF1'])
    @unittest.skipIf(torch is None,"CPU torch download pending")
    def test_independent_full_history_continuation(self):
        snapshot=UniPCMultistepScheduler();fork=fork_scheduler(snapshot)
        fork.model_outputs[0].add_(9)
        self.assertEqual(float(snapshot.model_outputs[0]),3.)
        calls=[]
        with patch('runtime.stage2.wan_translation.velocity',side_effect=lambda *a:torch.tensor([2.])):
            z=continue_two(None,torch.tensor([0.]),snapshot,list(range(8)),None,None,5,lambda:calls.append(1))
        self.assertEqual(float(z),4.);self.assertEqual(len(calls),4);self.assertEqual(snapshot._step_index,6)
        self.assertEqual(float(snapshot.model_outputs[0]),3.)

if __name__=='__main__':unittest.main()
