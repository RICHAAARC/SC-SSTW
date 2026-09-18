"""Real tiny lossless codec, CPU oracle geometry, fake start-free receiver chain."""
import inspect
import shutil
from types import SimpleNamespace
import pytest
import torch
from main.tube_state import inversion_state as state
from experiments.wan_state_clock import inversion_crop_observation as method
from experiments.wan_state_clock import inversion_crop_observation_run as runner
from test_video_inversion import Model,scheduler
from runtime.wan.flow_inversion import public_schedule
from runtime.wan.io import dump

pytestmark=pytest.mark.unit
torch.set_num_threads(1)


def test_oracle_geometry_boundaries_and_no_best_phase():
    book=state.codebook(b'crop-test');full=state.write(torch.ones(state.SHAPE),book,0)
    for start,shift in [(0,0),(16,4)]:
        z=full[:,:,shift:shift+33].clone()
        raw=method.raw_observations(z);assert len(raw['per_time'])==33
        result=method.oracle(z,book,start,0)[0]
        assert result['complete_windows']==8 and result['missing_windows']==3 and result['partial_windows']==0
        assert result['complete_window_report']['exact_windows']==8
        assert result['complete_window_report']['component_errors']==0
        assert not result['per_time'][0]['core_eligible'] and result['per_time'][-1]['core_eligible']
        assert result['reporting_only']['raw_state']['component_errors']==6 # explicitly full-grid missing
    maps=method.oracle(full[:,:,4:37].clone(),book,17,None)
    assert [m['nominal_shift'] for m in maps]==[4,5]
    assert maps[0]['complete_windows']==8
    assert (maps[1]['complete_windows'],maps[1]['partial_windows'],maps[1]['missing_windows'])==(7,2,2)
    assert maps[1]['per_time'][1]['source_frame_range']==[18,21]
    assert maps[1]['per_time'][1]['phase_ambiguous']
    assert all(m['reporting_only']['truth'] is None and 'exact_windows' not in m['complete_window_report'] for m in maps)
    assert list(inspect.signature(method.raw_observations).parameters)==['z']
    assert 'start' not in inspect.signature(runner.invert_received_clips).parameters


def test_actual_rgb_lossless_crop_codec(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):pytest.skip('FFmpeg unavailable')
    source=torch.randint(0,256,(181,8,8,3),generator=torch.Generator().manual_seed(9)).float()/255
    for start in runner.STARTS:
        path=tmp_path/f'{start}.mp4';clip=source[start:start+129]
        runner.encode_lossless(clip,path)
        assert torch.equal(clip,runner.read_mp4(path))


def test_fake_full_case_receiver_no_source_or_start(tmp_path,monkeypatch):
    source=tmp_path/runner.SOURCE_RUN;case=runner.CASES[0];case_root=source/case
    fixed=runner.holdout.validate();info=fixed['holdout'][0]
    config=__import__('copy').deepcopy(fixed['base_config']);config['generation'].update(prompt=info['prompt'],seed=info['seed'])
    dump(case_root/'config.json',config);dump(case_root/'public_schedule.json',public_schedule(scheduler()))
    dump(case_root/'result.json',dict(source_commit=runner.SOURCE_COMMIT,source_dirty=False,status='EXECUTION_COMPLETE'))
    stored={};pixels=torch.arange(181).float().reshape(181,1,1,1).expand(181,2,2,3)/255
    for arm in runner.ARMS:
        path=case_root/'videos'/f'{arm}.mp4';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fake original');stored[path]=pixels.clone()
    def read(path):assert path.exists();return stored[path].clone()
    def encode(rgb,path):path.write_bytes(b'fake lossless crop');stored[path]=rgb.clone()
    encoded=[];inverse_inputs=[]
    def vae_encode(vae,rgb):
        assert rgb.shape==(129,2,2,3)
        z=torch.full(method.SHAPE,float(rgb.mean()));encoded.append(z.clone());return z
    def prepare(public,**kwargs):
        assert set(public)=={'model','generation'} and public['generation']['frames']==129
        assert public['generation']['seed']==0 and 'start' not in public['generation']
        return SimpleNamespace(transformer=Model(),scheduler=scheduler()),torch.full(method.SHAPE,999.),torch.tensor(1.),torch.tensor(-1.),torch.float32
    def inverse(transformer,z,prompt,negative,dtype,cfg,sigmas,times,count,record):
        inverse_inputs.append(z.clone());assert float(z.mean())<1
        for i in range(50):
            for _ in range(2):count('transformer',False);count('transformer',True)
            count('inverse_update',False);count('inverse_update',True);record({'index':i})
        return z
    monkeypatch.setattr(runner,'RGB_SHAPE',(129,2,2,3));monkeypatch.setattr(runner,'read_mp4',read)
    monkeypatch.setattr(runner,'encode_lossless',encode);monkeypatch.setattr(runner,'load_frozen_vae',lambda *a:Model())
    monkeypatch.setattr(runner,'reencode_rgb24_readback',vae_encode);monkeypatch.setattr(runner,'_clear_cache',lambda *a:None)
    monkeypatch.setattr(runner,'prepare_generation',prepare);monkeypatch.setattr(runner,'invert_received_latent',inverse)
    result=runner.run_case(case,source,tmp_path/'out')
    assert result['status']=='EXECUTION_COMPLETE' and len(result['clips'])==9
    for key,value in runner.PLAN.items():assert result['actual_calls'][key+'_completed']==value
    assert len(encoded)==len(inverse_inputs)==9
    for a,b in zip(encoded,inverse_inputs):assert torch.equal(a,b)
    assert sum(len(v['oracle']) for v in result['clips'].values())==12
    assert all(v['receiver']['receiver_contract']['source_start_used'] is False for v in result['clips'].values())


def test_missing_fixed_18_clip_24_map_denominators(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1))
    result=runner.run_all(tmp_path/runner.SOURCE_RUN,tmp_path/'out')
    assert result['clip_denominator']==18 and result['oracle_map_denominator']==24
    assert sum(len(c['clips']) for c in result['cases'].values())==18
    assert sum(len(v['oracle']) for c in result['cases'].values() for v in c['clips'].values())==24
    assert all(g['measured_maps']==0 and g['unobserved_expected_complete_windows']==g['expected_complete_window_denominator'] for g in result['oracle_summary']['groups'])


def test_schedule_default_metadata_order_and_output_guard(tmp_path):
    import copy
    a=public_schedule(scheduler());b=copy.deepcopy(a)
    a['config']['_use_default_values']=['foo','bar'];b['config']['_use_default_values']=['bar','foo']
    assert runner.same_schedule(a,b)
    b['sigmas'][0]-=.01
    assert not runner.same_schedule(a,b)
    source=tmp_path/runner.SOURCE_RUN
    for fn in (lambda:runner.run_all(source,source/'nested'),lambda:runner.run_case(runner.CASES[0],source,source/'nested')):
        with pytest.raises(ValueError,match='independent'):fn()
    assert not source.exists()
