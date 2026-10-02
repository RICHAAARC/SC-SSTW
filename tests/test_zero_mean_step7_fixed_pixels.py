import ast,gzip,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.wan_state_clock import zero_mean_step7_fixed_pixels_run as runmod
from main.tube_state import video_overlap_zero_mean_state as state
pytestmark=pytest.mark.unit


def fixture_channel(monkeypatch,run,fail=None):
    target=torch.tensor(state.synthesize('watermark'),dtype=torch.float32)
    q8=torch.full((1,2,2,3),100,dtype=torch.uint8);rgb=q8.float()/255.
    trace=dict(encoded=[],decoded=[],encodes=0)
    def inputs(*args):
        from main.tube_state import video_overlap_zero_mean_control as payload
        inf=state.infer(target[0,:,1:45].numpy().transpose(1,0,2,3),'watermark',np.ones((44,4),bool))
        rawpath=tmp_input=run.output/'fixture_parent.raw.json.gz';rawpath.write_bytes(gzip.compress(json.dumps(dict(inference=inf)).encode()))
        run.data['parent_reference']=dict(q8=dict(raster_sha256=hashlib.sha256(q8.numpy().tobytes()).hexdigest()),rgb=dict(path='fixture_parent.rgb',sha256=hashlib.sha256(q8.numpy().tobytes()).hexdigest()),reads=dict(CORRECT=dict(path=str(rawpath))))
        return q8.clone(),target.clone(),rgb.clone()
    monkeypatch.setattr(runmod,'load_inputs',inputs)
    import runtime.wan.generation as generation
    dummy=SimpleNamespace(parameters=lambda:iter([torch.zeros(1)]),use_tiling=False,config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16))
    monkeypatch.setattr(generation,'load_frozen_vae',lambda *a,**k:dummy)
    def decode(vae,z):raise AssertionError('VAE decoder must never run')
    monkeypatch.setattr(runmod.media,'decode_with_clamp_receipt',decode)
    def roundtrip(q,yuv,rgb_path,*,count,event):
        if fail=='RAW420_CHANNEL':raise RuntimeError('injected raw444 failure')
        for name,path,kind in [('yuv420',yuv,'rgb_to_raw420'),('rgb24',rgb_path,'raw420_to_rgb24')]:
            count(kind,False);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(q.numpy().tobytes())
            event(name,dict(status='SAVED',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),path=str(path)))
            count(kind,True)
        return q.clone()+1
    monkeypatch.setattr(runmod.raw420,'roundtrip',roundtrip)
    def encode_raster(q,path):
        trace['encoded'].append(q.clone())
        if fail=='MP4_SAVE':raise RuntimeError('injected MP4 encode failure')
        Path(path).write_bytes(b'fake MP4 fixture')
    monkeypatch.setattr(runmod.media,'encode_raster',encode_raster)
    monkeypatch.setattr(runmod.media,'read_full_mp4',lambda p:(q8+2).float()/255.)
    def reopen(path,sha):
        raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
        return torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy().reshape(q8.shape))
    monkeypatch.setattr(runmod.color,'reopen_rgb',reopen)
    def encode(vae,received):
        trace['encodes']+=1
        if fail=='SAVED_RAW444_ENCODE' and trace['encodes']==1:raise RuntimeError('injected raw encode failure')
        return target.clone() if torch.equal(received,rgb) else torch.zeros_like(target)
    monkeypatch.setattr(runmod.media,'encode_normalized',encode)
    actual_run=runmod.subprocess.run
    def subprocess_run(command,*a,**kw):
        if command[0]=='ffprobe':return SimpleNamespace(returncode=0,stdout='{"streams":[{"codec_name":"h264","pix_fmt":"yuv420p"}]}',stderr='')
        return actual_run(command,*a,**kw)
    monkeypatch.setattr(runmod.subprocess,'run',subprocess_run)
    return trace,target,q8


def make_run(tmp_path):return runmod.Run(tmp_path/'output',json.loads(runmod.CONFIG.read_text()))


def test_fixed_pixels_three_branches_without_decode_or_updates(tmp_path,monkeypatch):
    run=make_run(tmp_path);trace,target,q8=fixture_channel(monkeypatch,run)
    runmod.execute(run,tmp_path,tmp_path);run.finalize();r=run.data
    assert r['status']=='EXECUTION_COMPLETE'
    assert r['counts']==dict(observations=3,raw420_files=1,raw420_rgb_readbacks=1,mp4_saved=1,mp4_read=1,path_reads=6,message_evaluations=12,valid_costs=1044)
    assert trace['decoded']==[] and trace['encodes']==3
    assert len(trace['encoded'])==1 and torch.equal(trace['encoded'][0],q8)
    assert r['saved_raster_matches_parent']
    assert r['baseline_replay']==dict(normalized_max_error=0.,normalized_rmse=0.,rgb_sha_matches_parent=True,decoder_calls=0,max_correct_key_path_cost_error=0.)
    assert Path(r['mp4']['read']['path']).read_bytes()==(q8+2).numpy().tobytes()
    for name,n in run.cfg['planned_calls'].items():assert r['calls'][name]==dict(attempted=n,completed=n)
    for row in r['reads'].values():
        raw=json.loads(gzip.decompress(Path(row['path']).read_bytes()))
        assert not raw['truth_used'] and not raw['writer_inputs'] and raw['inference']['counts']['scored']==174
    assert r['comparisons']['SAVED_RAW444']['true_unique_top']
    assert not r['comparisons']['MP4']['true_unique_top']
    assert r['fixed_endpoint']['observation']=='MP4'
    assert set(k for k in r['fixed_endpoint'] if k.startswith('delta_change'))=={'delta_change_vs_saved_raw444','delta_change_vs_raw420'}
    assert r['observations']['MP4']['quality_vs_current_raw420']['rgb_rmse']>0


@pytest.mark.parametrize('failure,bad',[('RAW420_CHANNEL','RAW420'),('SAVED_RAW444_ENCODE','SAVED_RAW444'),('MP4_SAVE','MP4')])
def test_branch_failure_retains_two_other_branches(tmp_path,monkeypatch,failure,bad):
    run=make_run(tmp_path);fixture_channel(monkeypatch,run,fail=failure)
    runmod.execute(run,tmp_path,tmp_path);run.finalize();r=run.data
    assert r['status']=='INCOMPLETE' and len(r['failures'])==1
    assert r['observations'][bad]['status']=='FAILED'
    assert all(r['observations'][p]['status']=='SAVED' for p in runmod.OBS if p!=bad)
    assert r['counts']['observations']==2 and r['counts']['path_reads']==4 and r['counts']['valid_costs']==696
    assert len(r['reads'])==6 and len(r['message_evaluations'])==12
    assert r['reads'][bad+'/CORRECT']['status']=='NOT_COMPLETED'
    if failure=='RAW420_CHANNEL':assert r['observations']['MP4']['quality_vs_current_raw420']==dict(status='REFERENCE_UNAVAILABLE')
    if failure=='MP4_SAVE':assert r['mp4']['save']['status']=='FAILED' and 'fixed_endpoint' not in r


def test_shared_input_failure_retains_full_roster(tmp_path):
    run=make_run(tmp_path)
    try:raise ValueError('injected identity failure')
    except ValueError as exc:run.failure(exc)
    run.finalize()
    assert run.data['counts']['observations']==0 and run.data['status']=='INCOMPLETE'
    assert all(v['status']=='NOT_COMPLETED' for v in run.data['observations'].values())
    assert len(run.data['reads'])==6 and len(run.data['message_evaluations'])==12


def test_notebook_frozen_step7_and_no_update_schedule(tmp_path):
    from scripts.build_zero_mean_step7_fixed_pixels_notebook import build
    for sha in [None,'a'*40]:
        nb=json.loads(build(sha,tmp_path/'mp4.ipynb').read_text())
        assert nb['metadata']['candidate_binding']['source_sha']==sha
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['outputs']==[]
        setup=''.join(nb['cells'][2]['source']);run=''.join(nb['cells'][4]['source'])
        assert "INPUT_ROOT=Path('/content/drive/MyDrive/Video-WM/Zero-Mean-Channel-Margin-Continue-V1/20261001T170749052584Z/fixed_reference')" in setup
        assert "OUTPUT=Path('/content/drive/MyDrive/Video-WM/Zero-Mean-Step7-Fixed-Pixels-V1')" in setup
        assert 'zero_mean_step7_fixed_pixels_run' in run
        assert 'POINT1' not in setup and 'new_updates' not in setup
        assert "('SAVED_RAW444','RAW420','MP4')" in setup and '1044' in setup
