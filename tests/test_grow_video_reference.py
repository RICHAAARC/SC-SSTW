"""CPU implementation checks; synthetic tensors/fixture VAE, no model loading."""
from __future__ import annotations
import copy
import dataclasses
import inspect
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from main.tube_state import grow_video_reference as method
from runtime.wan import grow_video_reference as backend,trajectory,vae as vae_adapter
from experiments.wan_state_clock import grow_video_reference_run as run

pytestmark=pytest.mark.unit


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_clean(dtype=torch.float32):
    base=torch.zeros(method.PUBLIC.latent_shape,dtype=dtype)
    target,mask=method.build_target(base,"watermark",method.message_bits("OKOK"))
    return torch.fft.ifft2(target,dim=(-2,-1),norm="ortho").real,target,mask


def scheduler():
    from diffusers import UniPCMultistepScheduler
    result=UniPCMultistepScheduler(prediction_type="flow_prediction",thresholding=False,
        predict_x0=True,lower_order_final=True,use_flow_sigmas=True,
        flow_shift=3.0,final_sigmas_type="zero")
    result.set_timesteps(50,device="cpu");result.set_begin_index(0)
    return result


def test_actual_mean_mask_full_fft_gradient_and_scaling():
    clean,target,mask=make_clean(torch.float64)
    clean=clean+torch.randn(clean.shape,generator=torch.Generator().manual_seed(42),dtype=clean.dtype)*.02
    delta,receipt=method.local_delta(clean,target,mask)
    assert receipt["mask_count"]==44160 and receipt["eta"]==5520
    before=torch.fft.fft2(clean,norm="ortho").real[mask]-target[mask]
    after=torch.fft.fft2(clean+delta,norm="ortho").real[mask]-target[mask]
    torch.testing.assert_close(after,.875*before,rtol=1e-11,atol=1e-12)
    assert torch.count_nonzero(delta[:,4:])==0
    assert not delta.requires_grad and not receipt["denoiser_backward"]
    with pytest.raises(ValueError,match="actual mask"):
        method.local_delta(clean,target,torch.zeros_like(mask))


def test_flow_mapping_before_fp32_cfg_without_hidden_compensation():
    clean,target,mask=make_clean()
    z=clean+.2;c=torch.full_like(z,.1234).bfloat16();u=torch.full_like(z,-.03125).bfloat16()
    sigma=.4
    delta,_=method.local_delta(z-sigma*c.float(),target,mask)
    result,row=method.guided_velocity(z,c,u,sigma,target,mask,enabled=True)
    baseline=u.float()+5*(c.float()-u.float())
    torch.testing.assert_close(result,baseline-5*delta/sigma,rtol=1e-5,atol=2e-6)
    assert result.dtype==torch.float32 and row["cfg_clean_delta_rms"]==5*row["conditional_clean_delta_rms"]
    assert sum(method.control_enabled("MULTI",i) for i in range(50))==25
    assert sum(method.control_enabled("LAST",i) for i in range(50))==1
    with pytest.raises(ValueError,match="positive"):
        method.guided_velocity(z,c,u,0,target,mask,enabled=True)


def test_zero_eta_full_native_state_and_history_equal_off():
    class Transformer:
        def __call__(self,hidden_states,encoder_hidden_states,**kwargs):
            return (hidden_states*.01+encoder_hidden_states.mean(),)
    pipe=SimpleNamespace(transformer=Transformer())
    initial=torch.randn(method.PUBLIC.latent_shape,generator=torch.Generator().manual_seed(4))*.05
    pristine=scheduler();before=trajectory.fingerprint(vars(pristine))
    results=[];counts=[]
    for arm in ("OFF","MULTI","LAST"):
        clone=copy.deepcopy(pristine);rows=[];ledger={}
        def count(kind,complete):
            entry=ledger.setdefault(kind,[0,0]);entry[int(complete)]+=1
        final,receipt=backend.run_trajectory(pipe,initial,clone,torch.tensor([.03]),torch.tensor([-.02]),
            torch.bfloat16,arm,"watermark",method.message_bits("OKOK"),count,rows.append,reference_eta=0)
        results.append((final,trajectory.fingerprint(vars(clone))))
        counts.append(ledger)
        assert [x["cursor_after"] for x in rows]==list(range(1,51))
        assert receipt["final_cursor"]==50 and rows[49]["sigma"]>0 and float(clone.sigmas[50])==0
        assert receipt["cfg_dtype"]=="torch.float32"
    for final,history in results[1:]:
        assert torch.equal(final,results[0][0]) and history==results[0][1]
    assert trajectory.fingerprint(vars(pristine))==before
    assert counts[0]["native_step"]==[50,50]
    assert counts[1]["local_gradient"]==[25,25] and counts[2]["local_gradient"]==[1,1]


def test_layout_and_truth_free_repeated_payload_reader():
    clean,_,_=make_clean()
    row=method.read_latent_bits(clean,"watermark")
    assert row["decoded_bits"]==method.message_bits("OKOK")
    assert len(row["votes"])==32 and all(v["count"]==1380 for v in row["votes"])
    assert "message" not in inspect.signature(method.read_latent_bits).parameters
    assert not hasattr(method.PUBLIC,"message")
    receipt=method.layout_receipt("watermark","watermark-wrong")
    assert receipt["mask_support_same"] and receipt["assignment_different"]
    with pytest.raises(ValueError,match="collision"):method.layout_receipt("watermark","kramretaw")


class FakeVAE:
    def __init__(self,normalized):
        self.parameter=torch.nn.Parameter(torch.zeros(()),requires_grad=False)
        self.config=SimpleNamespace(latents_mean=[.1*i for i in range(16)],latents_std=[.5+.01*i for i in range(16)])
        self.normalized=normalized;self.mode_calls=0;self.encode_calls=0;self.clear_calls=0
    def parameters(self):yield self.parameter
    def clear_cache(self):self.clear_calls+=1
    def encode(self,video):
        self.encode_calls+=1
        assert video.shape==(1,3,181,320,512) and video.dtype==torch.float32
        assert -1<=float(video.min())<=float(video.max())<=1
        mean=torch.tensor(self.config.latents_mean).reshape(1,16,1,1,1)
        std=torch.tensor(self.config.latents_std).reshape(1,16,1,1,1)
        def mode():self.mode_calls+=1;return self.normalized*std+mean
        def sample():raise AssertionError("posterior sampling is forbidden")
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=mode,sample=sample,mean=torch.full_like(self.normalized,float('nan'))))


@pytest.fixture
def actual_mp4(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):pytest.skip("FFmpeg executables unavailable")
    path=tmp_path/"fixture.mp4"
    subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","color=c=gray:s=512x320:r=8",
        "-frames:v","181","-c:v","libx264","-crf","18","-pix_fmt","yuv420p",str(path)],check=True,capture_output=True)
    return path


def test_primary_real_mp4_readback_mode_normalization_and_partial_key_failure(actual_mp4,monkeypatch):
    clean,_,_=make_clean();vae=FakeVAE(clean);ledger={}
    def count(kind,complete):ledger.setdefault(kind,[0,0])[int(complete)]+=1
    original=method.read_latent_bits
    def partial(latent,key,public):
        if key=="watermark-wrong":raise RuntimeError("injected wrong-key reader error")
        return original(latent,key,public)
    monkeypatch.setattr(method,"read_latent_bits",partial)
    rows=backend.read_mp4_payload(actual_mp4,dict(CORRECT="watermark",WRONG="watermark-wrong"),method.PUBLIC,vae,count=count)
    assert rows["CORRECT"]["decoded_bits"]==method.message_bits("OKOK")
    assert rows["CORRECT"]["status"]=="READ" and rows["WRONG"]["status"]=="FAILED"
    assert rows["CORRECT"]["mp4_sha256"]==rows["WRONG"]["mp4_sha256"]
    assert rows["CORRECT"]["feature_sha256"]==rows["WRONG"]["feature_sha256"]
    assert vae.mode_calls==vae.encode_calls==1 and vae.clear_calls==2
    assert ledger==dict(mp4_read=[1,1],vae_encode=[1,1],bit_read=[2,1])
    assert tuple(inspect.signature(backend.read_mp4_payload).parameters)==("path","keys","public","frozen_vae","count")


def test_primary_wrong_geometry_rejected_before_vae(tmp_path,monkeypatch):
    path=tmp_path/"bad.mp4";path.write_bytes(b"geometry-fixture")
    monkeypatch.setattr(backend.io,"read_mp4",lambda _:torch.zeros(1,1,1,3).expand(180,320,512,3))
    vae=FakeVAE(torch.zeros(method.PUBLIC.latent_shape))
    with pytest.raises(ValueError,match="geometry mismatch"):
        backend.read_mp4_payload(path,dict(CORRECT="watermark",WRONG="watermark-wrong"),method.PUBLIC,vae)
    assert vae.encode_calls==0


def test_fixed_rows_partial_failure_truth_join_and_hard_exit_recovery(tmp_path):
    store=run.Store(tmp_path/"run",create=True);cfg=run.load_config()
    rows=dict(CORRECT=dict(status="READ",decoded_bits=method.message_bits("OKOK"),truth_used=False),
              WRONG=dict(status="FAILED",decoded_bits=None,truth_used=False,error="fixture"))
    store.set_reads("OFF","mp4",rows)
    run.recover_unfinished(store,"media","worker hard-exit fixture")
    path=store.output/"blind_readouts.json";before=path.read_bytes()
    run.evaluate_saved_reads(store,cfg)
    assert path.read_bytes()==before and [len(store.data[x]) for x in ("videos","reads","evaluations")]==[3,24,48]
    assert store.data["counts"]["reads"]==1 and store.data["counts"]["evaluated"]==2
    assert store.data["evaluations"]["OFF/mp4/CORRECT/REGISTERED"]["exact_bits"] is True
    cfg["wrong_message"]="ABCD";run.evaluate_saved_reads(store,cfg)
    assert path.read_bytes()==before


def test_media_layer_failure_continues_to_primary(tmp_path,monkeypatch):
    from runtime.wan import generation,io
    store=run.Store(tmp_path/"media",create=True);cfg=run.load_config()
    for arm in method.ARMS:store.data["generation"][arm]["status"]="NOT_COMPLETED"
    path=Path(store.data["generation"]["OFF"]["terminal_path"]);path.parent.mkdir();torch.save(torch.zeros(method.PUBLIC.latent_shape),path)
    store.data["generation"]["OFF"].update(status="COMPLETE",terminal_file_sha256=backend.file_sha256(path))
    monkeypatch.setattr(generation,"load_frozen_vae",lambda *a,**k:FakeVAE(torch.zeros(method.PUBLIC.latent_shape)))
    monkeypatch.setattr(vae_adapter,"decode_normalized_latent",lambda *a:torch.zeros(1,1,1,3).expand(method.PUBLIC.video_shape))
    monkeypatch.setattr(vae_adapter,"quantize_rgb8_no_codec",lambda _:torch.zeros(1,1,1,3,dtype=torch.uint8).expand(method.PUBLIC.video_shape))
    monkeypatch.setattr(io,"encode_rgb",lambda rgb,path,*args:path.write_bytes(b"saved-synthetic-mp4"))
    bits=method.message_bits("OKOK")
    def rows():return {k:dict(status="READ",decoded_bits=bits,truth_used=False) for k in run.KEY_IDS}
    monkeypatch.setattr(backend,"read_diagnostic_latent",lambda *a,**k:rows())
    attempts=[]
    def diagnostic(*a,**k):
        attempts.append("diagnostic")
        if len(attempts)==1:raise RuntimeError("floatRGB encode fixture failure")
        return rows()
    monkeypatch.setattr(backend,"reencode_diagnostic_rgb",diagnostic)
    def primary(path,*a,**k):
        attempts.append("primary");return {name:dict(**row,mp4_sha256=backend.file_sha256(path)) for name,row in rows().items()}
    monkeypatch.setattr(backend,"read_mp4_payload",primary)
    run.media_worker(store,cfg)
    assert attempts==["diagnostic","diagnostic","primary"]
    assert store.data["reads"]["OFF/float_rgb/CORRECT"]["status"]=="MISSING_READOUT"
    assert store.data["reads"]["OFF/mp4/CORRECT"]["status"]=="READ"
    assert len(store.data["reads"])==24 and store.data["counts"]["saved_mp4"]==1


def test_quality_is_separate_chunked_diagnostic(tmp_path,monkeypatch):
    from runtime.wan import io
    store=run.Store(tmp_path/"quality",create=True)
    store.data["status"]="EXECUTION_COMPLETE";before=copy.deepcopy(store.data["reads"])
    public=dataclasses.replace(method.PUBLIC,video_shape=(2,4,5,3));monkeypatch.setattr(method,"PUBLIC",public)
    values={}
    for arm,value in zip(method.ARMS,(0.2,0.3,0.4)):
        path=Path(store.data["videos"][arm]["path"]);path.parent.mkdir();path.write_bytes(arm.encode())
        store.data["videos"][arm].update(status="SAVED",sha256=backend.file_sha256(path))
        values[str(path)]=torch.full(public.video_shape,value)
    def read(path):
        if Path(path).stem=="LAST":raise RuntimeError("quality-only failure")
        return values[str(path)]
    monkeypatch.setattr(io,"read_mp4",read)
    run.quality_diagnostics(store)
    assert store.data["status"]=="EXECUTION_COMPLETE" and store.data["reads"]==before
    assert store.data["quality"]["MULTI"]["rgb_rmse"]==pytest.approx(.1,abs=1e-7)
    assert store.data["quality"]["MULTI"]["psnr_db"]==pytest.approx(20,abs=1e-5)
    assert store.data["quality"]["LAST"]["status"]=="FAILED"
    assert store.data["calls"]["QUALITY_MP4_DIAGNOSTICS"]["quality_mp4_read"]==dict(attempted=3,completed=2)


def test_precision_selection_has_no_model_name_gate(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    assert backend.execution_device_dtype()==("cpu",torch.float32)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"is_bf16_supported",lambda:True)
    assert backend.execution_device_dtype()==("cuda",torch.bfloat16)
    monkeypatch.setattr(torch.cuda,"is_bf16_supported",lambda:False)
    assert backend.execution_device_dtype()==("cuda",torch.float16)


def test_notebook_draft_setup_denominator_and_no_environment_gate(tmp_path):
    import ast
    from scripts.build_grow_video_reference_notebook import build
    path=build(output=tmp_path/"draft.ipynb");nb=json.loads(path.read_text())
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    sources=[]
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            text=''.join(cell['source']);sources.append(text);ast.parse(text)
            assert cell['outputs']==[] and cell['execution_count'] is None
    text='\n'.join(sources)
    assert 'sys.version_info' not in text and "'-m','venv'" not in text and 'ensurepip' not in text
    assert 'sys.executable' in text and 'requirements-grow-video-reference.txt' in text
    setup=sources[1].replace("Path('/content/drive/MyDrive/Video-WM/GROW-Video-Reference-V1')",f"Path({str(tmp_path)!r})")
    with pytest.raises(RuntimeError,match='Unpublished draft'):exec(compile(setup,'setup','exec'),{})
    receipt=json.loads(next(tmp_path.glob('*/setup_receipt.json')).read_text())
    assert [len(receipt[x]) for x in ('videos','reads','evaluations')]==[3,24,48]


def test_shared_loader_optional_cpu_dtype_preserves_default_signature(monkeypatch):
    import diffusers
    from runtime.wan import generation
    captured={}
    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__();self.patch_embedding=torch.nn.Linear(1,1,bias=False)
            self.config=SimpleNamespace(in_channels=16)
    class Pipe:
        config=SimpleNamespace();vae_scale_factor_temporal=4;vae_scale_factor_spatial=8
        def __init__(self):
            self.text_encoder=torch.nn.Linear(1,1);self.transformer=Transformer();self.scheduler=scheduler()
        def encode_prompt(self,**kwargs):
            captured['prompt_device']=str(kwargs['device'])
            return torch.zeros(1,3,2,dtype=torch.float64),torch.zeros(1,3,2,dtype=torch.float64)
        def prepare_latents(self,b,c,h,w,t,dtype,device,generator,unused):
            captured['latent_device']=str(device);captured['seed']=generator.initial_seed()
            return torch.zeros((b,c,(t-1)//4+1,h//8,w//8),dtype=dtype,device=device)
    def load(*args,**kwargs):captured.update(kwargs);return Pipe()
    monkeypatch.setattr(diffusers.WanPipeline,'from_pretrained',load)
    pipe,z,prompt,negative,dtype=generation.prepare_generation(run.load_config(),load_vae=False,device='cpu',model_dtype=torch.float32)
    assert captured['torch_dtype']==torch.float32 and captured['vae'] is None
    assert captured['prompt_device']==captured['latent_device']=='cpu'
    assert captured['seed']==2026092501 and z.shape==method.PUBLIC.latent_shape
    assert prompt.dtype==negative.dtype==dtype==torch.float32 and pipe.scheduler.step_index is None
    assert inspect.signature(generation.prepare_generation).parameters['device'].default is None
    assert inspect.signature(generation.prepare_generation).parameters['model_dtype'].default is None


def main_fixture(monkeypatch,tmp_path):
    output=tmp_path/'orchestrated'
    monkeypatch.setattr(run.sys,'argv',['review','--output',str(output)])
    monkeypatch.setattr(run.subprocess,'check_output',lambda *a,**k:'92583b6bdd58733427434ad54943c82dc7a9dcc2\n')
    return output


def assert_final_fixed_rows(output):
    result=json.loads((output/'result.json').read_text())
    assert [len(result[x]) for x in ('generation','videos','reads','evaluations')]==[3,3,24,48]
    assert all(row['status']!='PENDING' for name in ('generation','videos','reads','evaluations') for row in result[name].values())
    raw=json.loads((output/'blind_readouts.json').read_text())
    assert len(raw['reads'])==24 and raw['truth_inputs'] is False
    return result


def test_parent_main_worker_spawn_failure_finalizes_fixed_rows(tmp_path,monkeypatch):
    output=main_fixture(monkeypatch,tmp_path);attempts=[]
    def spawn(command,**kwargs):
        attempts.append(command[-1]);raise OSError('injected worker spawn failure')
    monkeypatch.setattr(run.subprocess,'Popen',spawn)
    with pytest.raises(SystemExit) as caught:run.main()
    result=assert_final_fixed_rows(output)
    assert caught.value.code==1 and attempts==['generation','media']
    assert result['status']=='INCOMPLETE' and result['stage']=='FINISHED'
    assert result['counts']==dict(generated=0,saved_mp4=0,reads=0,evaluated=0)
    assert all(row['status']=='START_FAILED' for row in result['workers'].values())
    assert len(result['failures'])==2 and result['actual_model_calls'] is False


def test_parent_main_monitor_failure_reaps_before_reload_and_preserves_child_data(tmp_path,monkeypatch):
    output=main_fixture(monkeypatch,tmp_path);events=[]
    class Stream:
        def __init__(self,phase):self.phase=phase
        def __iter__(self):
            if self.phase=='media':raise OSError('injected stdout monitor failure')
            return iter(())
        def close(self):events.append('close_'+self.phase)
    class Child:
        def __init__(self,phase):self.phase=phase;self.stdout=Stream(phase)
        def poll(self):return None
        def terminate(self):events.append('terminate_'+self.phase)
        def wait(self,timeout=None):
            events.append('wait_'+self.phase)
            child_store=run.Store(output)
            if self.phase=='generation':
                for row in child_store.data['generation'].values():row.update(status='COMPLETE')
                child_store.save();return 0
            # A final media checkpoint produced during shutdown must win over
            # the parent's pre-launch Store, including successful raw bits.
            child_store.data['videos']['OFF'].update(status='SAVED',sha256='fixture-sha')
            child_store.set_reads('OFF','mp4',{
                'CORRECT':dict(status='READ',decoded_bits=method.message_bits('OKOK'),truth_used=False),
                'WRONG':dict(status='FAILED',decoded_bits=None,truth_used=False,error='fixture key failure')})
            child_store.data['stage']='CHILD_FINAL_CHECKPOINT';child_store.save()
            return -15
    def spawn(command,**kwargs):
        events.append('spawn_'+command[-1]);return Child(command[-1])
    monkeypatch.setattr(run.subprocess,'Popen',spawn)
    with pytest.raises(SystemExit) as caught:run.main()
    result=assert_final_fixed_rows(output)
    assert caught.value.code==1 and events==['spawn_generation','wait_generation','close_generation',
        'spawn_media','terminate_media','wait_media','close_media']
    assert result['counts']==dict(generated=3,saved_mp4=1,reads=1,evaluated=2)
    assert result['reads']['OFF/mp4/CORRECT']['status']=='READ'
    assert result['reads']['OFF/mp4/WRONG']['status']=='FAILED'
    assert result['evaluations']['OFF/mp4/CORRECT/REGISTERED']['exact_bits'] is True
    assert result['workers']['generation']['status']=='COMPLETE'
    assert result['workers']['media']['status']=='MONITOR_FAILED'
    assert result['workers']['media']['last_child_stage']=='CHILD_FINAL_CHECKPOINT'
    assert result['status']=='INCOMPLETE' and len(result['failures'])==1


def test_parent_main_keyboard_interrupt_reaps_and_does_not_start_media(tmp_path,monkeypatch):
    output=main_fixture(monkeypatch,tmp_path);events=[]
    class Stream:
        def __iter__(self):raise KeyboardInterrupt('injected user interruption')
        def close(self):events.append('close')
    class Child:
        stdout=Stream()
        def poll(self):return None
        def terminate(self):events.append('terminate')
        def wait(self,timeout=None):events.append('wait');return -15
    def spawn(command,**kwargs):events.append('spawn_'+command[-1]);return Child()
    monkeypatch.setattr(run.subprocess,'Popen',spawn)
    monkeypatch.setattr(run,'quality_diagnostics',lambda *a:pytest.fail('interrupted run must not start diagnostics'))
    with pytest.raises(SystemExit) as caught:run.main()
    result=assert_final_fixed_rows(output)
    assert caught.value.code==130 and events==['spawn_generation','terminate','wait','close']
    assert result['status']=='INTERRUPTED' and result['stage']=='FINISHED'
    assert result['workers']['media']['status']=='NOT_STARTED_INTERRUPTED'
    assert all(row['status']=='NOT_RUN_INTERRUPTED' for row in result['quality'].values())
    assert result['counts']==dict(generated=0,saved_mp4=0,reads=0,evaluated=0)
