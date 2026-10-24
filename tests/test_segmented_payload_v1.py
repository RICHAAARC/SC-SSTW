"""Public payload checks: arithmetic, support, voting and native-loop wiring.

Synthetic CPU inputs and fake runtime only; no real models or media work.
"""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest
import torch

from main.tube_state import segmented_payload_v1 as payload
from main.tube_state.given_time_mapping_v1 import plan_given_time_mapping
from runtime.wan import segmented_payload_v1 as runtime

pytestmark=pytest.mark.unit


@pytest.fixture(scope='module',autouse=True)
def small_cpu_pool():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_four_bytes_mask_and_guards():
    z=torch.zeros(1,16,46,40,64)
    bits=[int(b) for b in f'{int("A6D39C5E",16):032b}']
    target,mask=payload.build_target(z,'public-key',bits)
    assert int(mask.sum())==44160
    assert not mask[:,4:].any()
    coords=payload.original.coordinates('public-key')
    assert len(coords)==240
    assert {h for h,w in coords}==set(range(8,20))
    assert {w for h,w in coords}==set(range(12,32))
    for t in range(46):
        g=0 if t==0 else 3 if t==45 else (t-1)//11
        values=torch.tensor([(2*bits[g*8+j%8]-1)*.5 for j in range(240)])
        for c in range(4):
            assert torch.equal(target[0,c,t,[h for h,w in coords],[w for h,w in coords]],values)
    assert torch.equal(target[:,:,0],target[:,:,1])
    assert torch.equal(target[:,:,44],target[:,:,45])
    with pytest.raises(ValueError):payload.build_target(z,'key',bits[:8])


def test_complete_cell_support_excludes_synthetic_and_guards():
    clean=payload.segment_plan(list(range(181)))
    assert [s['observed_cells'] for s in clean['segments']]==[11]*4
    assert [c['source_wan_time'] for s in clean['segments'] for c in s['cells']]==list(range(1,45))
    assert clean['blind_sync_performed'] is False
    assert clean['operation']['mapping_origin']=='CALLER_SUPPLIED'
    # Delete frames 45..87: t12..22 have incomplete or absent support.
    deleted=payload.segment_plan(list(range(45))+list(range(88,181)))
    assert [s['observed_cells'] for s in deleted['segments']]==[11,0,11,11]
    assert len(deleted['operation']['synthetic_output_indices'])==43
    assert deleted['segments'][1]['status']=='MISSING'
    partial_cell=deleted['cell_support'][21]
    assert partial_cell['support_status']=='PARTIAL' and not partial_cell['eligible_for_votes']
    assert partial_cell['present_frames']==[88]
    partial=payload.segment_plan(list(range(2,45)))
    assert partial['operation']['front_boundary_copies']==2
    assert [s['observed_cells'] for s in partial['segments']]==[10,0,0,0]
    assert partial['segments'][0]['cells'][0]['source_wan_time']==2
    assert payload.segment_plan(list(range(60,105)))['segments'][0]['status']=='MISSING'


@pytest.mark.parametrize('bad',[[],[True],[1,0],[-1],[181],[0,1.0]])
def test_invalid_maps_are_real_format_errors(bad):
    with pytest.raises(ValueError):plan_given_time_mapping(bad)


def test_nearest_tie_and_short_span():
    op=plan_given_time_mapping([0,2,3,4])
    assert op['received_index_map']==[0,0,1,2,3]
    assert op['synthetic_output_indices']==[1]
    assert plan_given_time_mapping([180])['status']=='UNSUPPORTED'


def test_vote_order_zero_and_tie_remain_original():
    plan=payload.segment_plan(list(range(45)))
    shape=(4,11,240)
    signed=-np.ones(shape,dtype=np.int8)
    zeros=np.ones(shape,dtype=np.int8)
    # Two channels positive vs two negative: tied and first encounter is 1.
    signed[:2,:,0::8]=1;zeros[:2,:,0::8]=0
    result=payload.decode_signed_votes(plan,signed,zeros)
    first=result['segments'][0]['bits'][0]
    assert first['tie'] and first['decoded']==1 and first['count']==1320
    assert result['decoded_bits'][:8]==[1,0,0,0,0,0,0,0]
    assert result['decoded_bits'][8:]==[None]*24
    assert result['segments'][0]['bits'][1]['zero_coefficients']==1320
    broken=zeros.copy();broken[0,0,0]=1
    with pytest.raises(ValueError):payload.decode_signed_votes(plan,signed,broken)


def test_real_fft_sign_extraction_is_same_reducer():
    z=torch.randn(1,16,12,40,64,generator=torch.Generator().manual_seed(13))
    path=list(range(45));key='test-reader'
    got=payload.read_segments(z,key,path)
    q=torch.fft.fft2(z.float(),dim=(-2,-1),norm='ortho').real
    coords=payload.original.coordinates(key)
    expected=[]
    for slot in range(8):
        raw=[]
        for ch in range(4):
            for t in range(1,12):
                for h,w in coords[slot::8]:raw.append(int(q[0,ch,t,h,w]>0))
        expected.append(Counter(raw).most_common(1)[0][0])
    assert got['decoded_bits']==expected+[None]*24
    with pytest.raises(ValueError):payload.read_segments(z.to(torch.int32),key,path)


def test_control_math_before_and_after_25(monkeypatch):
    calls=[]
    def delta(clean,target,mask):
        assert torch.equal(clean,torch.tensor([1.75]))
        return torch.tensor([.25]),{'eta':5520.0}
    monkeypatch.setattr(payload.original,'local_delta',delta)
    z=torch.tensor([2.]);c=torch.tensor([.5]);u=torch.tensor([-.5])
    for index in (24,25,49):
        v,receipt=payload.guided_velocity(z,c,u,.5,None,None,index,lambda k,r:calls.append((k,r)))
        assert torch.equal(v,torch.tensor([4.5 if index==24 else 2.]))
        assert receipt['enabled']==(index>=25)
    assert calls==[('payload_gradient',False),('payload_gradient',True)]*2


def test_one_real_local_gradient_preserves_eta_and_full_mask():
    z=torch.zeros(1,16,46,40,64)
    target,mask=payload.build_target(z,'gradient-key',[0,1]*16)
    delta,receipt=payload.original.local_delta(z,target,mask)
    assert receipt['mask_count']==44160 and receipt['eta']==5520.0
    assert receipt['gradient_kind']=='full_local_FFT_real_mean_MSE_wrt_conditional_x0'
    assert float(delta.square().sum())>0 and torch.isfinite(delta).all()


def test_native_50_step_loop_keeps_one_scheduler_and_target(monkeypatch):
    seen=[];counts=[];records=[]
    class Scheduler:
        config=SimpleNamespace(prediction_type='flow_prediction',thresholding=False,lower_order_final=True)
        predict_x0=True
        step_index=None
        timesteps=torch.arange(50,0,-1)
        sigmas=torch.linspace(1,0,51)
        def step(self,v,t,z,return_dict=False):
            i=0 if self.step_index is None else self.step_index
            assert int(t)==50-i
            self.step_index=i+1
            return (z-v*.01,)
    class Transformer:
        def __call__(self,hidden_states,encoder_hidden_states,**kw):
            seen.append(encoder_hidden_states)
            return (torch.zeros_like(hidden_states),)
    def delta(clean,target,mask):
        assert target.shape==clean.shape and int(mask.sum())==44160
        return torch.zeros_like(clean),{'mask_count':44160,'eta':5520.0}
    monkeypatch.setattr(payload.original,'local_delta',delta)
    scheduler=Scheduler()
    terminal,receipt=runtime.run_trajectory(SimpleNamespace(transformer=Transformer()),
        torch.zeros(1,16,46,40,64),scheduler,'c','u',torch.float32,'key',[0,1]*16,
        lambda kind,returned:counts.append((kind,returned)),records.append)
    assert scheduler.step_index==50 and seen==['c','u']*50
    assert len(records)==50 and [r['index'] for r in records if r['enabled']]==list(range(25,50))
    assert counts.count(('payload_gradient',False))==counts.count(('payload_gradient',True))==25
    assert counts.count(('native_step',True))==50
    assert receipt['net_bits']==32 and receipt['actual_eta']==5520.0
    assert torch.count_nonzero(terminal)==0


def test_rgb_mapping_precedes_vae_and_no_missing_votes(monkeypatch):
    from runtime.wan import vae as adapter
    path=list(range(2,45))
    rgb=torch.arange(len(path),dtype=torch.uint8)[:,None,None,None].expand(-1,320,512,3)
    def encode(vae,mapped):
        expected=rgb[torch.tensor(plan_given_time_mapping(path)['received_index_map'])].float()/255
        assert torch.equal(mapped,expected)
        return torch.zeros(1,16,12,40,64)
    monkeypatch.setattr(adapter,'reencode_rgb24_readback',encode)
    got=runtime.read_received_frames(object(),rgb,'key',path)
    assert got['segments'][0]['observed_cells']==10
    assert got['segments'][0]['bits'][0]['count']==1200
    assert got['decoded_bit_count']==8


def test_cli_plan_and_saved_votes_need_no_truth_or_git(tmp_path):
    from experiments.wan_state_clock.segmented_payload_v1_run import main
    source=tmp_path/'map.json';source.write_text(json.dumps(list(range(45))))
    main(['plan','--source-map',str(source),'--output',str(tmp_path/'plan')])
    plan=json.loads((tmp_path/'plan/plan.json').read_text())
    assert not plan['operation']['blind_sync_performed']
    votes=tmp_path/'input.npz'
    np.savez(votes,signed_votes=-np.ones((4,11,240),dtype=np.int8),zero_coefficient_mask=np.zeros((4,11,240),dtype=np.int8))
    main(['read-votes','--source-map',str(source),'--input',str(votes),'--output',str(tmp_path/'read')])
    result=json.loads((tmp_path/'read/read.json').read_text())
    assert result['decoded_bits']==[0]*8+[None]*24
    assert result['conditional_on_given_mapping'] and not result['blind_sync_performed']


def test_generation_config_fixes_method_geometry():
    from experiments.wan_state_clock.segmented_payload_v1_run import validate_generation_config
    config=json.loads((Path(__file__).resolve().parents[1]/'experiments/wan_state_clock/configs/segmented_payload_v1.example.json').read_text())
    assert validate_generation_config(config)['bits'][:8]==[1,0,1,0,0,1,1,0]
    config['generation']['steps']=49
    with pytest.raises(ValueError):validate_generation_config(config)



def test_direct_entry_from_ordinary_source_copy(tmp_path):
    import shutil
    root=Path(__file__).resolve().parents[1]
    copy=tmp_path/'source'
    files=['main/tube_state/grow_video_reference.py',
        'main/tube_state/given_time_mapping_v1.py','main/tube_state/segmented_payload_v1.py',
        'experiments/wan_state_clock/segmented_payload_v1_run.py']
    for rel in files:
        dest=copy/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/rel,dest)
    source=tmp_path/'map.json';source.write_text('[0,1,2,3,4]')
    output=tmp_path/'ordinary-output'
    completed=subprocess.run([sys.executable,str(copy/files[-1]),'plan',
        '--source-map',str(source),'--output',str(output)],cwd=tmp_path,
        capture_output=True,text=True,check=True)
    assert str(output) in completed.stdout
    got=json.loads((output/'plan.json').read_text())
    assert got['segments'][0]['observed_cells']==1
    assert got['conditional_on_given_mapping'] and not got['blind_sync_performed']
