import ast,gzip,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.wan_state_clock import zero_mean_step7_mp4_run as runmod
from main.tube_state import video_overlap_zero_mean_state as state
pytestmark=pytest.mark.unit


def fixture_channel(monkeypatch,run,fail=None):
    target=torch.tensor(state.synthesize('watermark'),dtype=torch.float32)
    q8=torch.full((1,2,2,3),100,dtype=torch.uint8);rgb=q8.float()/255.
    trace=dict(encoded=[],decoded=[],encodes=0)
    def inputs(*args):
        run.data['parent_reference']=dict(q8_raster_sha256=hashlib.sha256(q8.numpy().tobytes()).hexdigest(),rgb_sha256=hashlib.sha256(q8.numpy().tobytes()).hexdigest())
        return target.clone(),target.clone(),rgb.clone()
    monkeypatch.setattr(runmod,'load_inputs',inputs)
    import runtime.wan.generation as generation
    dummy=SimpleNamespace(parameters=lambda:iter([torch.zeros(1)]),use_tiling=False,config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16))
    monkeypatch.setattr(generation,'load_frozen_vae',lambda *a,**k:dummy)
    def decode(vae,z):trace['decoded'].append(z.clone());return rgb.clone(),dict(fixture=True)
    monkeypatch.setattr(runmod.media,'decode_with_clamp_receipt',decode)
    def roundtrip(q,yuv,rgb_path,*,count,event):
        if fail=='RAW444_CHANNEL':raise RuntimeError('injected raw444 failure')
        for name,path,kind in [('yuv444',yuv,'rgb_to_raw444'),('rgb24',rgb_path,'raw444_to_rgb24')]:
            count(kind,False);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(q.numpy().tobytes())
            event(name,dict(status='SAVED',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),path=str(path)))
            count(kind,True)
        return q.clone()
    monkeypatch.setattr(runmod.color,'roundtrip',roundtrip)
    def encode_raster(q,path):
        trace['encoded'].append(q.clone())
        if fail=='MP4_SAVE':raise RuntimeError('injected MP4 encode failure')
        Path(path).write_bytes(b'fake MP4 fixture')
    monkeypatch.setattr(runmod.media,'encode_raster',encode_raster)
    monkeypatch.setattr(runmod.media,'read_full_mp4',lambda p:(q8+1).float()/255.)
    def reopen(path,sha):
        raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
        return torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy().reshape(q8.shape))
    monkeypatch.setattr(runmod.color,'reopen_rgb',reopen)
    def encode(vae,received):
        trace['encodes']+=1
        if fail=='RAW444_ENCODE' and trace['encodes']==1:raise RuntimeError('injected raw encode failure')
        return target.clone() if torch.equal(received,rgb) else torch.zeros_like(target)
    monkeypatch.setattr(runmod.media,'encode_normalized',encode)
    actual_run=runmod.subprocess.run
    def subprocess_run(command,*a,**kw):
        if command[0]=='ffprobe':return SimpleNamespace(returncode=0,stdout='{"streams":[{"codec_name":"h264","pix_fmt":"yuv420p"}]}',stderr='')
        return actual_run(command,*a,**kw)
    monkeypatch.setattr(runmod.subprocess,'run',subprocess_run)
    return trace,target,q8


def make_run(tmp_path):return runmod.Run(tmp_path/'output',json.loads(runmod.CONFIG.read_text()))


def test_matched_branches_fixed_mp4_endpoint_and_real_readback_persistence(tmp_path,monkeypatch):
    run=make_run(tmp_path);trace,target,q8=fixture_channel(monkeypatch,run)
    runmod.execute(run,tmp_path,tmp_path,tmp_path);run.finalize();r=run.data
    assert r['status']=='EXECUTION_COMPLETE'
    assert r['counts']==dict(observations=2,mp4_saved=1,mp4_read=1,path_reads=4,message_evaluations=8,valid_costs=696)
    assert len(trace['decoded'])==1 and torch.equal(trace['decoded'][0],target)
    assert len(trace['encoded'])==1 and torch.equal(trace['encoded'][0],q8)
    assert r['decoded_raster_matches_parent'] and r['baseline_replay']==dict(normalized_max_error=0.,rgb_sha_matches_parent=True)
    assert Path(r['mp4']['read']['path']).read_bytes()==(q8+1).numpy().tobytes()
    for name,n in run.cfg['planned_calls'].items():assert r['calls'][name]==dict(attempted=n,completed=n)
    for row in r['reads'].values():
        raw=json.loads(gzip.decompress(Path(row['path']).read_bytes()))
        assert not raw['truth_used'] and not raw['writer_inputs'] and raw['inference']['counts']['scored']==174
    assert r['comparisons']['RAW444']['true_unique_top']
    assert not r['comparisons']['MP4']['true_unique_top']
    assert r['fixed_endpoint']['observation']=='MP4' and not r['fixed_endpoint']['true_unique_top']
    assert r['observations']['MP4']['quality_vs_current_raw444']['rgb_rmse']>0


@pytest.mark.parametrize('failure',['RAW444_CHANNEL','RAW444_ENCODE','MP4_SAVE'])
def test_branch_failure_retains_other_branch_and_fixed_denominator(tmp_path,monkeypatch,failure):
    run=make_run(tmp_path);fixture_channel(monkeypatch,run,fail=failure)
    runmod.execute(run,tmp_path,tmp_path,tmp_path);run.finalize();r=run.data
    assert r['status']=='INCOMPLETE' and len(r['failures'])==1
    good='RAW444' if failure=='MP4_SAVE' else 'MP4';bad='MP4' if good=='RAW444' else 'RAW444'
    assert r['observations'][good]['status']=='SAVED' and r['observations'][bad]['status']=='FAILED'
    assert r['counts']['observations']==1 and r['counts']['path_reads']==2 and r['counts']['valid_costs']==348
    assert len(r['reads'])==4 and len(r['message_evaluations'])==8
    assert r['reads'][bad+'/CORRECT']['status']=='NOT_COMPLETED'
    if failure=='RAW444_CHANNEL':assert r['observations']['MP4']['quality_vs_current_raw444']==dict(status='REFERENCE_UNAVAILABLE')
    if failure=='MP4_SAVE':assert r['mp4']['save']['status']=='FAILED' and 'fixed_endpoint' not in r


def test_shared_input_failure_retains_full_roster(tmp_path):
    run=make_run(tmp_path)
    try:raise ValueError('injected identity failure')
    except ValueError as exc:run.failure(exc)
    run.finalize()
    assert run.data['counts']['observations']==0 and run.data['status']=='INCOMPLETE'
    assert all(v['status']=='NOT_COMPLETED' for v in run.data['observations'].values())
    assert len(run.data['reads'])==4 and len(run.data['message_evaluations'])==8


def test_notebook_frozen_step7_and_no_update_schedule(tmp_path):
    from scripts.build_zero_mean_step7_mp4_notebook import build
    for sha in [None,'a'*40]:
        nb=json.loads(build(sha,tmp_path/'mp4.ipynb').read_text())
        assert nb['metadata']['candidate_binding']['source_sha']==sha
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['outputs']==[]
        setup=''.join(nb['cells'][2]['source']);run=''.join(nb['cells'][4]['source'])
        assert "INPUT_ROOT=Path('/content/drive/MyDrive/Video-WM/Zero-Mean-Channel-Margin-Continue-V1/20261001T170749052584Z/fixed_reference')" in setup
        assert "OUTPUT=Path('/content/drive/MyDrive/Video-WM/Zero-Mean-Step7-MP4-V1')" in setup
        assert 'zero_mean_step7_mp4_run' in run
        assert 'POINT1' not in setup and 'new_updates' not in setup
        assert "('RAW444','MP4')" in setup and '696' in setup
