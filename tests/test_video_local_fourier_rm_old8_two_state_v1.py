"""CPU/NumPy validation for the OLD8 two-state candidate; no real model/media."""
import ast,copy,hashlib,json,subprocess,tempfile,unittest
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch

from main.tube_state import video_local_fourier_rm_state as old
from main.tube_state import video_local_fourier_rm_control as oldcontrol
from main.tube_state import video_local_fourier_rm_old8_two_state_v1 as state
from main.tube_state import video_local_fourier_rm_old8_two_state_v1_control as control
from experiments.wan_state_clock import video_local_fourier_rm_old8_two_state_v1_run as run
from scripts import build_video_local_fourier_rm_old8_two_state_v1_notebook as builder

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.unit
SMALL=(2,2,4,3)


class FakeScheduler:
    def __init__(self):
        self.config=SimpleNamespace(prediction_type='flow_prediction',thresholding=False,lower_order_final=True)
        self.predict_x0=True;self.timesteps=torch.arange(50,0,-1);self.sigmas=torch.linspace(1,0,51);self.step_index=None;self.model_outputs=[None,None]
    def step(self,velocity,timestep,z,return_dict=False):
        index=0 if self.step_index is None else self.step_index;self.model_outputs=[self.model_outputs[-1],velocity.clone()];self.step_index=index+1
        return (z+(self.sigmas[index+1]-self.sigmas[index])*velocity,)


class FakeTransformer:
    def __call__(self,hidden_states,encoder_hidden_states,**kwargs):return (hidden_states*.01+encoder_hidden_states.mean(),)


class TwoStateCandidateTest(unittest.TestCase):
    def test_public_selection_old8_geometry_and_transition(self):
        for key in ("watermark", "watermark-wrong"):
            words=state.selected_words(key);receipt=state.selection_receipt(key)
            np.testing.assert_array_equal(words["A"],old.state_code(key)[0]);np.testing.assert_array_equal(words["B"],old.state_code(key)[1])
            self.assertIn(receipt["hamming"],(16,32));self.assertFalse(receipt["response_selected"])
            np.testing.assert_allclose(state.bases(key),old.bases(key),atol=0,rtol=0)
            for name in state.SEQUENCES:
                target=state.synthesize(key,name);self.assertEqual(target.shape,(1,16,46,40,64))
                self.assertAlmostEqual(float(np.sum(target*target)),1.0,places=12)
                signs=state.composite_signs(key,name);self.assertEqual(int(np.count_nonzero(signs)),5568)
                labels=state.start_labels(name);self.assertEqual(len(labels),45)
                if name=="AB":self.assertEqual(labels[21:24],("A","B","B"))
                if name=="BA":self.assertEqual(labels[21:24],("B","A","A"))
            ab=state.composite_signs(key,"AB")
            for u in (23,24,25):
                self.assertGreater(len({ab[u-1,:,age,:].tobytes() for age in range(4)}),1)

    def test_ideal_absolute_joint_and_local_component_evidence(self):
        key="watermark";mask=np.ones((44,4),bool)
        for name in state.SEQUENCES:
            received=state.synthesize(key,name)[0,:,1:45].transpose(1,0,2,3)
            q=state.extract(received,key,mask);result=state.score(q,key,mask,state.MODES[0])
            self.assertEqual(result["summary"]["top"],[name]);self.assertEqual(result["summary"]["canonical"],name)
            self.assertFalse(result["summary"]["state_sequence_accepted"]);self.assertEqual(result["counts"],dict(sequence_candidates=4,sequence_scores=4,local_candidate_costs=176,component_state_costs=340))
            rows=[r for r in result["component_evidence"] if r["status"]=="SCORED"]
            self.assertEqual(len(rows),170);labels=state.start_labels(name)
            self.assertTrue(all(r["top"]==[labels[r["source_start"]-1]] for r in rows))
            self.assertEqual(sum(r["status"]=="STRUCTURALLY_EXCLUDED" for r in result["component_evidence"]),6)

    def test_fixed_mask_preserves_difference_ambiguity_and_failures(self):
        key="watermark";full=np.ones((44,4),bool);received=state.synthesize(key,"AA")[0,:,1:45].transpose(1,0,2,3);q=state.extract(received,key,full)
        mask=full.copy();mask[:4]=False;mask[18:27]=False
        result=state.score(q,key,mask,state.MODES[1]);self.assertEqual(result["summary"]["top"],list(state.SEQUENCES));self.assertIsNone(result["summary"]["canonical"]);self.assertFalse(result["summary"]["unique_model_hypothesis"])
        partial=full.copy();partial[7,2]=False;part=state.score(q,key,partial,state.MODES[0]);self.assertEqual(part["local_rows"][7]["available_dimensions"],96)
        bad=q.copy();bad[5,0,0,0]=np.nan;invalid=state.score(bad,key,full,state.MODES[0]);self.assertEqual(invalid["summary"]["reason"],"INVALID_OBSERVATION");self.assertEqual(set(invalid["sequence_scores"]),set(state.SEQUENCES));self.assertEqual(len(invalid["local_rows"]),44);self.assertEqual(len(invalid["component_evidence"]),176)

    def test_original_payload_and_shrink_only_old8_update(self):
        torch.set_num_threads(1);latent=torch.zeros((1,16,46,40,64));bits=control.message_bits("OKOK")
        original=oldcontrol.build_targets(latent,"watermark",bits)
        for arm,sequence in control.STATE_ARMS.items():
            targets=control.build_targets(latent,"watermark",bits,arm)
            self.assertTrue(torch.equal(targets["payload_target"],original["payload_target"]));self.assertTrue(torch.equal(targets["payload_mask"],original["payload_mask"]))
            self.assertEqual(targets["sequence"],sequence);self.assertEqual(int(targets["pilot_active"].sum()),5568)
            expected=torch.tensor(state.synthesize("watermark",sequence,dtype=np.float32))
            delta,receipt=oldcontrol.pilot_delta(latent,targets);torch.testing.assert_close(delta,expected/4,atol=2e-8,rtol=3e-6)
            self.assertAlmostEqual(receipt["raw_delta_l2"],.25,places=6);self.assertAlmostEqual(receipt["actual_delta_l2"],.25,places=6);self.assertEqual(receipt["cap_scale"],1.0);self.assertEqual(receipt["cap"],1.0)
        targets=control.build_targets(latent,"watermark",bits,"STATE_AB_MULTI");clean=torch.tensor(state.synthesize("watermark","AB",dtype=np.float32))*10
        _,receipt=oldcontrol.pilot_delta(clean,targets);self.assertLess(receipt["cap_scale"],1.0);self.assertAlmostEqual(receipt["actual_delta_l2"],1.0,places=6)

    def test_config_architecture_and_unpublished_notebook(self):
        cfg=json.loads((ROOT/"experiments/wan_state_clock/configs/video_local_fourier_rm_old8_two_state_v1.json").read_text())
        self.assertEqual(cfg["fixed_denominator"]["sequence_candidate_scores"],288);self.assertEqual(tuple(cfg["arms"]),("OFF","PAYLOAD_MULTI","STATE_A_MULTI","STATE_B_MULTI","STATE_AB_MULTI","STATE_BA_MULTI"));self.assertEqual(cfg["source_sha"],None)
        for relative in ("main/tube_state/video_local_fourier_rm_old8_two_state_v1.py","main/tube_state/video_local_fourier_rm_old8_two_state_v1_control.py","runtime/wan/video_local_fourier_rm_old8_two_state_v1.py"):
            tree=ast.parse((ROOT/relative).read_text());imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            self.assertFalse(any(name and name.startswith("experiments") for name in imports))
            if relative.startswith("main/"):self.assertFalse(any(name and name.startswith("runtime") for name in imports))
        with tempfile.TemporaryDirectory() as folder:
            path=builder.build(None,Path(folder)/"draft.ipynb");nb=json.loads(path.read_text());sources=[''.join(c['source']) for c in nb['cells']]
            self.assertEqual(sources[0],"from google.colab import drive\ndrive.mount('/content/drive')\n");self.assertIsNone(nb['metadata']['candidate_binding']['source_sha']);self.assertIn("UNPUBLISHED_DRAFT",sources[2])
            for cell,source in zip(nb['cells'],sources):
                if cell['cell_type']=='code':ast.parse(source);self.assertEqual(cell['outputs'],[]);self.assertIsNone(cell['execution_count'])
            self.assertNotIn("force_remount",'\n'.join(sources));self.assertNotIn("--mode",builder.RUN)

    def test_failure_roster_and_sealed_blind_boundary(self):
        cfg=run.load_config();self.assertEqual(cfg["fixed_denominator"],run.FIXED)
        self.assertEqual(tuple(inspect.signature(run.blind_receive).parameters),("store","oid","normalized","key_roster"))
        with tempfile.TemporaryDirectory() as folder:
            store=run.Store(Path(folder)/"output",create=True)
            run.settle(store,"generation","static no model");run.settle(store,"media","static no media")
            blind=json.loads((store.output/"blind_receiver_readouts.json").read_text())
            self.assertFalse(blind["truth_inputs"]);self.assertNotIn("observations",blind);self.assertEqual(set(blind["selection"]),{"K0","K1"})
            self.assertTrue(all("arm" not in row and "message" not in row for group in ("projections","sequence_reads","payload_reads") for row in blind[group].values()))
            run.evaluate(store,cfg);run.quality_diagnostics(store);self.assertFalse(run.finish(store,cfg))
            self.assertTrue(all(len(store.data[name])==size for name,size in run.SIZES.items()))
            self.assertEqual(store.data["counts"]["sequence_score_records"],0);self.assertEqual(len(store.data["failures"]),0)

    def test_cpu_fake_six_arm_runner_success_path(self):
        from runtime.wan import generation
        cfg=run.load_config();torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            store=run.Store(Path(folder)/"output",create=True);pipe=SimpleNamespace(transformer=FakeTransformer(),scheduler=FakeScheduler());initial=torch.zeros((1,16,46,40,64))
            def prepare(*args,**kwargs):
                self.assertFalse(kwargs['load_vae']);return pipe,initial,torch.tensor([.03]),torch.tensor([-.02]),torch.float32
            with patch.object(run.backend,'execution_device_dtype',return_value=('cpu',torch.float32)),patch.object(generation,'prepare_generation',side_effect=prepare):run.generation_worker(store,cfg)
            self.assertEqual(store.data['counts']['generated'],6);self.assertEqual(store.data['counts']['generation_steps'],300);self.assertEqual(store.data['counts']['controlled_steps'],125);self.assertEqual(store.data['counts']['writer_sidecars'],150)
            for arm,row in store.data['generation'].items():
                expected=[] if arm=='OFF' else list(range(25,50));self.assertEqual([s['index'] for s in row['steps'] if s['enabled']],expected);self.assertTrue(row['receipt']['writer_diagnostics']['last_z_post_matches_terminal'])
            observed=SimpleNamespace(decode=0,encodes=[])
            class FakeVAE(torch.nn.Module):
                def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()),requires_grad=False);self.config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16)
                def decode(self,value,return_dict=False):observed.decode+=1;return (torch.linspace(-1.2,1.2,int(np.prod(SMALL))).reshape(1,3,*SMALL[:3])+float(value.mean())/100,)
                def encode(self,video):
                    observed.encodes.append(video.clone());value=torch.tensor(state.synthesize('watermark','AB',dtype=np.float32))+float(video.mean())/1000
                    return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:value))
            fake=FakeVAE()
            def load(cfg,device):return fake
            def save_received(path,source,flip):
                output=source.clone();output.reshape(-1)[0]^=flip;Path(path).write_bytes(output.numpy().tobytes());return output,dict(status='SAVED',path=str(path),sha256=run.sha(path),bytes=int(np.prod(SMALL)))
            def raw(source,yuv_path,rgb_path,*,count,event):
                source_hash=hashlib.sha256(source.numpy().tobytes()).hexdigest();count('rgb_to_raw420',False);Path(yuv_path).write_bytes(b'fake YUV');count('rgb_to_raw420',True);event('yuv420',dict(status='SAVED',sha256=run.sha(yuv_path),path=str(yuv_path),input_raster_sha256=source_hash));count('raw420_to_rgb24',False);output,row=save_received(rgb_path,source,1);count('raw420_to_rgb24',True);event('rgb24',row);return output
            def mp4(raster_path,raster_sha,mp4_path,rgb_path,*,count,event):
                source=run.media.reopen_raster(raster_path,raster_sha);count('mp4_save',False);Path(mp4_path).write_bytes(b'fake MP4');count('mp4_save',True);event('mp4',dict(status='SAVED',path=str(mp4_path),sha256=run.sha(mp4_path),input_raster_sha256=raster_sha));count('mp4_probe',False);count('mp4_probe',True);event('probe',dict(status='COMPLETE'));count('mp4_readback',False);output,row=save_received(rgb_path,source,2);count('mp4_readback',True);event('rgb24',row);return output
            with patch.object(run.media,'SHAPE',SMALL),patch.object(run.media,'RGB_BYTES',int(np.prod(SMALL))),patch.object(torch.cuda,'is_available',return_value=False),patch.object(generation,'load_frozen_vae',side_effect=load),patch.object(run.raw420,'roundtrip',side_effect=raw),patch.object(run.media,'mp4_roundtrip',side_effect=mp4):
                run.media_worker(store,cfg);self.assertEqual(observed.decode,6);self.assertEqual(len(observed.encodes),18)
                blind=store.output/'blind_receiver_readouts.json';sealed=blind.read_bytes();store.data['workers']={name:dict(status='COMPLETE') for name in ('generation','media')};run.evaluate(store,cfg);run.quality_diagnostics(store);self.assertEqual(sealed,blind.read_bytes());self.assertTrue(run.finish(store,cfg))
            self.assertEqual(store.data['counts'],{k:v for k,v in run.FIXED.items() if k not in ('source_cases','arms','channels','underlying_observations','keys')})
            self.assertEqual(store.data['call_integrity']['status'],'MATCH');self.assertEqual(len(store.data['failures']),0)
            self.assertTrue(all(not row.get('state_sequence_accepted',False) and not row.get('accepted_payload',False) for group in ('sequence_reads','sequence_posthoc','payload_evaluations') for row in store.data[group].values()))

    def test_worker_exception_reaps_child_before_store_reload(self):
        class FailingStdout:
            def __init__(self,error):self.error=error;self.closed=False
            def __iter__(self):return self
            def __next__(self):raise self.error
            def close(self):self.closed=True
        class FakePopen:
            def __init__(self,error,timeout=False):
                self.stdout=FailingStdout(error);self.timeout=timeout;self.terminated=False;self.killed=False;self.wait_calls=[]
            def poll(self):return None
            def terminate(self):self.terminated=True
            def kill(self):self.killed=True
            def wait(self,timeout=None):
                self.wait_calls.append(timeout)
                if timeout is not None and self.timeout:
                    self.timeout=False;raise subprocess.TimeoutExpired('fake-worker',timeout)
                return -9 if self.killed else 143
        cfg=run.load_config()
        for error,timeout,expected_interrupt in ((KeyboardInterrupt('stop'),False,True),(BrokenPipeError('log broke'),True,False)):
            with tempfile.TemporaryDirectory() as folder:
                store=run.Store(Path(folder)/'output',create=True);child=FakePopen(error,timeout)
                with patch.object(run.subprocess,'Popen',return_value=child):reloaded,interrupted=run.run_worker_phase(store,'generation')
                self.assertEqual(interrupted,expected_interrupt);self.assertTrue(child.terminated);self.assertEqual(child.killed,timeout);self.assertTrue(child.stdout.closed)
                self.assertEqual(child.wait_calls,[10,10] if timeout else [10]);row=reloaded.data['workers']['generation']
                self.assertEqual(row['status'],'FAILED');self.assertIn(type(error).__name__,row['error']);self.assertIsNone(row['cleanup_error'])
                self.assertTrue(any(f['stage']=='WORKER_generation' and type(error).__name__ in f['error'] for f in reloaded.data['failures']))
                self.assertTrue(all(item['status']=='NOT_COMPLETED' for item in reloaded.data['generation'].values()))


if __name__=="__main__":unittest.main()
