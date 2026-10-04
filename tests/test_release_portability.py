"""Portable release provenance and CPU-only complete-flow fixtures."""
from pathlib import Path
import json
import shutil
from types import SimpleNamespace
import pytest
import torch
from runtime.wan import provenance
from experiments.wan_state_clock import grow_video_reference_run as run
from main.tube_state import grow_video_reference as method

pytestmark=pytest.mark.unit
ROOT=Path(__file__).resolve().parents[1]

def copy_runtime(destination):
    for name in (*provenance.SOURCE_FILES,"release_manifest.json"):
        p=destination/name;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,p)
    return destination

def test_archive_initialization_without_git_or_parent_repository(tmp_path,monkeypatch):
    root=copy_runtime(tmp_path/"archive")
    (tmp_path/".git").mkdir()  # Must not inherit parent metadata.
    monkeypatch.setattr(provenance.subprocess,"check_output",lambda *a,**k:pytest.fail("archive must not invoke git"))
    monkeypatch.setattr(run,"ROOT",root)
    store=run.Store(tmp_path/"output",create=True)
    p=store.data["source_provenance"]
    assert store.data["source_sha"] is None
    assert p["kind"]=="release_manifest" and p["manifest_status"]=="MATCH"
    assert p["content_sha256"]==p["manifest_content_sha256"]
    assert [len(store.data[x]) for x in ("generation","videos","reads","evaluations")]==[3,3,24,48]

def test_modified_archive_is_recorded_without_fabricated_sha(tmp_path):
    root=copy_runtime(tmp_path/"archive")
    changed=root/"main/tube_state/grow_video_reference.py"
    changed.write_text(changed.read_text()+"\n# local mechanical edit\n")
    p=provenance.source_identity(root)
    assert p["git_commit"] is None and p["manifest_status"]=="MODIFIED"
    assert p["changed_files"]==["main/tube_state/grow_video_reference.py"]
    assert p["content_sha256"]!=p["manifest_content_sha256"]

def test_missing_or_invalid_manifest_retains_observed_content(tmp_path):
    root=copy_runtime(tmp_path/"archive")
    path=root/"release_manifest.json";path.unlink()
    p=provenance.source_identity(root)
    assert p["kind"]=="unversioned_directory" and p["manifest_status"]=="ABSENT"
    assert p["git_commit"] is None and len(p["files"])==len(provenance.SOURCE_FILES)
    path.write_text('{"schema": 1, "files": {"../outside": "bad"}}')
    p=provenance.source_identity(root)
    assert p["manifest_status"]=="INVALID" and p["git_commit"] is None

def test_git_unavailable_keeps_manifest_identity(tmp_path,monkeypatch):
    root=copy_runtime(tmp_path/"archive");(root/".git").write_text("fixture")
    def missing(*a,**k):raise FileNotFoundError("git not installed")
    monkeypatch.setattr(provenance.subprocess,"check_output",missing)
    p=provenance.source_identity(root)
    assert p["manifest_status"]=="MATCH" and p["git_commit"] is None and "git_error" in p

def test_complete_fixed_chain_with_native_scheduler_and_fixture_models(tmp_path,monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg/ffprobe needed for complete CPU media fixture")
    from diffusers import UniPCMultistepScheduler
    from runtime.wan import generation
    threads=torch.get_num_threads();torch.set_num_threads(1)
    class Transformer:
        def __call__(self,hidden_states,encoder_hidden_states,**kwargs):
            return (hidden_states*.01+encoder_hidden_states.mean(),)
    class FixtureVAE:
        config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16)
        parameter=torch.nn.Parameter(torch.zeros(()),requires_grad=False)
        def parameters(self):yield self.parameter
        def clear_cache(self):pass
        def decode(self,value,return_dict=False):
            return (torch.zeros(1,3,1,1,1).expand(1,3,181,320,512),)
        def encode(self,video):
            assert video.shape==(1,3,181,320,512)
            # Geometry fixture; no knowledge of message, terminal or original video.
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:torch.zeros(method.PUBLIC.latent_shape)))
    def prepare(*a,**k):
        scheduler=UniPCMultistepScheduler(prediction_type="flow_prediction",thresholding=False,
            predict_x0=True,lower_order_final=True,use_flow_sigmas=True,flow_shift=3.,final_sigmas_type="zero")
        scheduler.set_timesteps(50,device="cpu");scheduler.set_begin_index(0)
        pipe=SimpleNamespace(transformer=Transformer(),scheduler=scheduler)
        return pipe,torch.zeros(method.PUBLIC.latent_shape),torch.tensor([.03]),torch.tensor([-.02]),torch.float32
    monkeypatch.setattr(generation,"prepare_generation",prepare)
    monkeypatch.setattr(generation,"load_frozen_vae",lambda *a,**k:FixtureVAE())
    monkeypatch.setattr(run.backend,"execution_device_dtype",lambda:("cpu",torch.float32))
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    output=tmp_path/"fixed"
    def inline_phase(path,phase):
        store=run.Store(path)
        store.data["execution_kind"]="CPU_FIXTURE_NOT_REAL_MODEL"
        (run.generation_worker if phase=="generation" else run.media_worker)(store,run.load_config())
        store.data["workers"][phase]={"status":"COMPLETE","fixture_inline":True};store.save()
        return store,None
    monkeypatch.setattr(run,"run_worker_phase",inline_phase)
    monkeypatch.setattr(run.sys,"argv",["fixture","--output",str(output)])
    try:run.main()
    finally:torch.set_num_threads(threads)
    r=json.loads((output/"result.json").read_text())
    assert r["status"]=="EXECUTION_COMPLETE" and r["counts"]==dict(generated=3,saved_mp4=3,reads=24,evaluated=48)
    assert r["failures"]==[] and r["execution_kind"]=="CPU_FIXTURE_NOT_REAL_MODEL"
    assert all(len(row["steps"])==50 for row in r["generation"].values())
    assert sum(s["enabled"] for row in r["generation"].values() for s in row["steps"])==26
    assert all(Path(row["path"]).is_file() for row in r["videos"].values())
    assert all(row["status"]=="MEASURED" for row in r["quality"].values())
    raw=json.loads((output/"blind_readouts.json").read_text())
    assert raw["truth_inputs"] is False and len(raw["reads"])==24
