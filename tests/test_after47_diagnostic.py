import tempfile
import unittest
import torch
from pathlib import Path
from runtime.public_statistic.after47_diagnostic import diagnose,state_digest
from runtime.public_statistic.phase2_execution import ResourceGuard
from main.sc_sstw.terminal_feedback import clone_graph_state
from tests.test_phase2_checkpointing import tiny_adapter

class DiagnosticTests(unittest.TestCase):
    def test_supervisor_worker_failure_and_deadline(self):
        import json,subprocess,sys,types
        from unittest.mock import patch
        import experiments.public_statistic.run_after47_diagnostic as entry
        original=subprocess.Popen
        for mode in ('failure','deadline'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'output'
                a=types.SimpleNamespace(output=str(out),config=str(Path(tmp)/'config.json'),reference=str(Path(tmp)/'reference.npy'))
                script='import sys;sys.exit(7)' if mode=='failure' else 'import time;time.sleep(10)'
                def child(command,**kwargs):return original([sys.executable,'-c',script],**kwargs)
                with patch.object(entry.subprocess,'Popen',side_effect=child),patch.dict(entry.BUDGET,wall_seconds=5 if mode=='failure' else .05):
                    code=entry.supervise(a)
                self.assertNotEqual(code,0)
                result=json.loads((out/'result.json').read_text())
                self.assertEqual(result['status'],'FATAL_STOP')
                self.assertEqual(result['error'],'WORKER_FAILURE' if mode=='failure' else 'WALL_BUDGET')
                self.assertEqual(json.loads((out/'execution_exit.json').read_text())['automatic_retries'],0)

    def test_real_tiny_remaining_unipc_fixed_five_evaluations(self):
        a,z,s=tiny_adapter(True)
        with torch.no_grad():
            for _ in range(4):z,s=a.advance(z,s)
            reference=a.rollout(z,clone_graph_state(s,torch))
        a.checkpoint_ledger.limits.update(transformer_block=4,vae_chunk=3)
        guard=ResourceGuard(dict(transformer_calls=20,vae_calls=5,backward_calls=1,mp4=0,saved_frames=0,wall_seconds=60));a.resource_guard=guard
        r={};snapshots=[];before=state_digest(s,torch)
        diagnose(a,z,s,reference,learning_rate=10,step_rms=.001,amplitude=2/255,report=r,persist=lambda:None,save=snapshots.append)
        self.assertEqual(r['status'],'DIAGNOSTIC_EXECUTED_REQUIRES_REVIEW')
        self.assertEqual(guard.counts,guard.completed)
        self.assertEqual(guard.counts,dict(transformer_calls=20,vae_calls=5,backward_calls=1,mp4=0,saved_frames=0))
        self.assertEqual(len(r['evaluations']),5);self.assertEqual(before,state_digest(s,torch))
        self.assertTrue(all(v['state_unchanged'] for v in r['evaluations'].values()))
        d=snapshots[0];expected=(d['latent']-r['update']['scale']*d['gradient'])-d['latent']
        torch.testing.assert_close(d['delta'],expected,rtol=0,atol=0)
        self.assertAlmostEqual(r['update']['g_dot_delta'],float((d['gradient'].double()*d['delta'].double()).sum()))
        self.assertLess(r['update']['g_dot_delta'],0)
        self.assertNotIn('accepted',r)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pt';torch.save(d,path);loaded=torch.load(path,weights_only=False)
            torch.testing.assert_close(loaded['delta'],d['delta'])

if __name__=='__main__':unittest.main()
