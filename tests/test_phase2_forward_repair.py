"""Targeted CPU saved-storage and actual process-lifetime regression checks."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
import torch
from runtime.public_statistic.phase2 import SelectiveSavedTensors
from runtime.public_statistic.phase2_execution import ResourceGuard
from tests.test_phase2_executable import config


class RepairTests(unittest.TestCase):
    def test_resident_storage_views_and_frozen_activation(self):
        module=torch.nn.Linear(4,4).requires_grad_(False)
        module.register_buffer('constant',torch.arange(16.).reshape(4,4))
        policy=SelectiveSavedTensors(torch,[module])
        for tensor in (module.weight,module.weight.t(),module.weight[:,1:],module.constant[1:]):
            saved=policy.pack(tensor);restored=policy.unpack(saved)
            self.assertEqual(policy.storage_key(restored),policy.storage_key(tensor))
            self.assertEqual(restored.storage_offset(),tensor.storage_offset())
            self.assertEqual(restored.stride(),tensor.stride())
            self.assertTrue(torch.equal(restored,tensor))
        self.assertEqual(policy.counts,dict(resident_saves=4,activation_saves=0))
        # Both tensors are frozen; only storage membership distinguishes them.
        policy.pack(torch.ones(4,requires_grad=False))
        self.assertEqual(policy.counts['activation_saves'],1)
    def test_weight_view_backward_and_context_exit(self):
        module=torch.nn.Linear(4,4,bias=False).requires_grad_(False)
        x=torch.randn(2,4,requires_grad=True)
        expected=torch.autograd.grad((x@module.weight.t()).square().sum(),x)[0]
        policy=SelectiveSavedTensors(torch,[module])
        with policy.context():loss=(x@module.weight.t()).square().sum()
        observed=torch.autograd.grad(loss,x)[0]
        self.assertTrue(torch.equal(expected,observed));self.assertGreater(policy.counts['resident_saves'],0)
        counts=policy.counts.copy()
        torch.autograd.grad((x.sin()).sum(),x)
        self.assertEqual(policy.counts,counts)
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with policy.context():raise RuntimeError('injected')
        torch.autograd.grad((x.cos()).sum(),x)
        self.assertEqual(policy.counts,counts)
    def test_attempt_completed_distinct(self):
        guard=ResourceGuard(config()['budget'])
        guard.consume('vae_calls')
        self.assertEqual(guard.counts['vae_calls'],1);self.assertEqual(guard.completed['vae_calls'],0)
        self.assertEqual(guard.events[-1]['status'],'START')
        guard.complete('vae_calls')
        self.assertEqual(guard.completed['vae_calls'],1);self.assertEqual(guard.events[-1]['status'],'COMPLETE')

    def test_worker_running_and_compact_progress_on_load_failure(self):
        from unittest.mock import patch
        from contextlib import ExitStack
        import types
        import copy
        import runtime.public_statistic.phase2_execution as execution
        writes=[];original=execution.write
        def recorded(path,value):
            writes.append((Path(path).name,copy.deepcopy(value)));original(path,value)
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            out=Path(tmp);cp=out/'input.json';cp.write_text(json.dumps(config()))
            for name,value in dict(is_available=True,is_bf16_supported=True,get_device_properties=types.SimpleNamespace(total_memory=24*2**30),set_per_process_memory_fraction=None,reset_peak_memory_stats=None,get_device_name='CPU_MOCK',memory_allocated=0).items():
                stack.enter_context(patch.object(torch.cuda,name,return_value=value))
            stack.enter_context(patch('importlib.metadata.version',return_value='CPU_MOCK'))
            stack.enter_context(patch.object(execution,'load_wan',side_effect=RuntimeError('INJECTED_LOAD_FAILURE')))
            stack.enter_context(patch.object(execution,'write',side_effect=recorded))
            with self.assertRaisesRegex(RuntimeError,'INJECTED_LOAD_FAILURE'):execution.worker(cp,out)
        self.assertEqual(next(value for name,value in writes if name=='result.json')['status'],'RUNNING')
        self.assertEqual(sum(name=='progress.json' for name,value in writes),1)
        for name,value in writes:
            if name=='counts.json':self.assertEqual(set(value),{'attempted','completed','current'})
        self.assertEqual([value for name,value in writes if name=='result.json'][-1]['status'],'FATAL_STOP')

    def test_real_worker_cancel_and_parent_death_with_grandchild(self):
        # Replace GPU work only; exercise the real CLI lifetime setup and cleanup.
        script = r"""
import json,os,subprocess,sys,time
from pathlib import Path
import experiments.public_statistic.run_phase2 as entry
from runtime.public_statistic.phase2_execution import initial_result,write
folder=Path(sys.argv[1]);cp=folder/'config.json'
def dummy(config_path,output):
    r=initial_result(json.loads(cp.read_text()));r['status']='RUNNING';r['arms']['OFF1']['status']='RUNNING';write(folder/'result.json',r)
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    (folder/'ready.json').write_text(json.dumps(dict(worker=os.getpid(),grandchild=child.pid)))
    try:time.sleep(30)
    finally:(folder/'cleanup.txt').write_text('worker cleanup')
entry.worker=dummy
sys.argv=['runner','--worker','--parent-pid',str(os.getppid()),'--config',str(cp),'--output',str(folder)]
entry.main()
"""
        parent_script = 'import subprocess,sys,time;subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2]],start_new_session=True);time.sleep(30)'
        def dead(pid):
            path=Path('/proc')/str(pid)/'stat'
            return not path.exists() or path.read_text().split()[2]=='Z'
        for mode in ('cancel','parent_death'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);(folder/'config.json').write_text(json.dumps(config()));sp=folder/'dummy.py';sp.write_text(script)
                parent=subprocess.Popen([sys.executable,'-c',parent_script,str(sp),tmp],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                ids=None
                try:
                    until=time.monotonic()+5
                    while not (folder/'ready.json').exists() and time.monotonic()<until:time.sleep(.02)
                    self.assertTrue((folder/'ready.json').exists())
                    ids=json.loads((folder/'ready.json').read_text())
                    if mode=='parent_death':parent.kill();parent.wait()
                    else:os.kill(ids['worker'],signal.SIGTERM)
                    until=time.monotonic()+5
                    while not all(dead(pid) for pid in ids.values()) and time.monotonic()<until:time.sleep(.02)
                    self.assertTrue(all(dead(pid) for pid in ids.values()),ids)
                    self.assertTrue((folder/'cleanup.txt').exists())
                    result=json.loads((folder/'result.json').read_text())
                    self.assertEqual(result['supervisor_stop'],'PARENT_LOST' if mode=='parent_death' else 'CANCELLED')
                    self.assertEqual(result['arms']['Y_MINUS']['status'],'NOT_EXECUTED')
                finally:
                    if parent.poll() is None:parent.kill();parent.wait()
                    if ids:
                        try:os.killpg(ids['worker'],signal.SIGKILL)
                        except ProcessLookupError:pass

if __name__=='__main__':unittest.main()
