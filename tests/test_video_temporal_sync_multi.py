"""CPU/static MULTI and public observation checks; no model or VAE weights."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import video_temporal_sync_bridge as bridge
from main.tube_state import video_temporal_sync_multi as m
from runtime.wan import video_temporal_sync_multi as b,trajectory,io,vae
from experiments.wan_state_clock import video_temporal_sync_multi_run as run

pytestmark=pytest.mark.unit


@pytest.fixture(autouse=True)
def cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def latent(T=46):return torch.zeros(1,16,T,40,64)


def test_multi_native25_updates_pre25_identity_and_later_divergence(monkeypatch):
    from diffusers import UniPCMultistepScheduler
    class Transformer:
        def __call__(self,hidden_states,encoder_hidden_states,**kwargs):return (hidden_states*.01+encoder_hidden_states.mean(),)
    scheduler=UniPCMultistepScheduler(prediction_type='flow_prediction',thresholding=False,predict_x0=True,
        lower_order_final=True,use_flow_sigmas=True,flow_shift=3.,final_sigmas_type='zero')
    scheduler.set_timesteps(50,device='cpu');scheduler.set_begin_index(0)
    initial=latent();pristine=trajectory.fingerprint(vars(scheduler));receipts=[];terminals=[];ledger={};cleans=[]
    original=bridge.local_delta
    def fresh(clean,target,mask):
        cleans.append(trajectory.fingerprint(clean));return original(clean,target,mask)
    monkeypatch.setattr(bridge,'local_delta',fresh)
    def count(kind,completed):ledger.setdefault(kind,[0,0])[int(completed)]+=1
    for arm in m.ARMS:
        rows=[]
        terminal,receipt=b.run_trajectory(SimpleNamespace(transformer=Transformer()),initial,copy.deepcopy(scheduler),
            torch.tensor([.03]),torch.tensor([-.02]),torch.bfloat16,arm,'watermark',m.message_bits('OKOK'),count,rows.append)
        receipts.append(receipt);terminals.append(terminal)
        assert [row['index'] for row in rows if row['enabled']]==([] if arm=='OFF' else list(range(25,50)))
        assert [row['cursor_after'] for row in rows]==list(range(1,51))
        assert all(row['sigma']>0 for row in rows)
        for row in rows[25:] if arm!='OFF' else []:
            assert row['mask_count']==(47104 if arm=='PILOT_MULTI' else 44160)
            assert row['eta']==(5888 if arm=='PILOT_MULTI' else 5520)
            assert all(k in row for k in ('loss','payload_delta_squared_l2','pilot_delta_squared_l2','total_delta_squared_l2'))
    assert receipts[0]['before_step25']==receipts[1]['before_step25']==receipts[2]['before_step25']
    assert trajectory.fingerprint(vars(scheduler))==pristine and torch.count_nonzero(initial)==0
    assert ledger['local_gradient']==[50,50] and ledger['native_step']==[150,150]
    assert ledger['transformer_conditional']==ledger['transformer_unconditional']==[150,150]
    assert len(cleans)==50 and len(set(cleans[:25]))>1 and len(set(cleans[25:]))>1
    assert not torch.equal(terminals[0],terminals[1]) and not torch.equal(terminals[1],terminals[2])


def test_roster_and_posthoc_four_truth_maps(tmp_path):
    cfg=run.load_config();store=run.Store(tmp_path/'roster',create=True)
    assert {name:len(store.data[name]) for name in run.TABLE_SIZES}==run.TABLE_SIZES
    assert sum(r['expected_row_count'] for r in store.data['windows'].values())==3270
    assert len(store.data['views'])==15 and len(store.data['normalized'])==51 and len(store.data['candidates'])==1278
    run.recover_unfinished(store,'generation','fixture');run.recover_unfinished(store,'media','fixture')
    before=(store.output/'blind_readouts.json').read_bytes();run.evaluate_saved_reads(store,cfg)
    assert (store.output/'blind_readouts.json').read_bytes()==before
    for start,(g,a) in {4:(0,1),5:(3,2),6:(2,2),7:(1,2)}.items():
        row=store.data['evaluations'][f'PILOT_MULTI/CROP{start}_129/CORRECT/REGISTERED']
        assert row['registered_b']==start and m.candidates(129)[start]==dict(b=start,g=g,a=a,R=31)
    registered=[v for k,v in store.data['evaluations'].items() if k.endswith('/REGISTERED')]
    assert sum(r['expected_pilot_present'] for r in registered)==5
    assert {k:sum(r['pilot_negative_category']==k for r in registered) for k in ('OFF_UNWATERMARKED','PAYLOAD_ONLY','PILOT_WRONG_KEY')}==dict(OFF_UNWATERMARKED=10,PAYLOAD_ONLY=10,PILOT_WRONG_KEY=5)


@pytest.mark.parametrize('length,g',[(181,0),(129,0),(129,3)])
def test_window_stats_independent_and_primary_equals_bridge(length,g):
    T=m.window_row_count(length,g)+1
    z=torch.randn((1,16,T,40,64),generator=torch.Generator().manual_seed(62))*.2
    primary,artifact=m.phase_observations(z,'watermark',length,g)
    baseline=bridge.phase_features(z,'watermark',45 if length==181 else 31)
    assert primary==baseline
    R=primary['R'];ordered=m.aggregate_ordered_windows(artifact['windows'][:R])
    assert ordered['decoded_bits']==baseline['decoded_bits'] and ordered['votes']==baseline['votes']
    assert all(v['count']==(1350 if length==181 else 930) for v in ordered['votes'])
    spectrum=np.fft.fft2(z.numpy().astype(np.float64),axes=(-2,-1),norm='ortho').real
    coords=bridge.payload_coordinates('watermark')
    for j,row in enumerate(artifact['windows'],1):
        assert row['observed_regular_j']==j and row['nominal_received_new_frame_interval']==[g+4*j-3,g+4*j+1]
        for bit,stat in enumerate(row['payload_stats']):
            coeff=np.array([spectrum[0,bit//8,j,h,w] for h,w in coords[bit%8::8]])
            assert stat['sum']==pytest.approx(float(coeff.sum()),abs=2e-6,rel=2e-5)
            assert stat['sumsq']==pytest.approx(float(np.sum(coeff**2)),abs=2e-6,rel=2e-5)
            assert stat['positive_count']==int((coeff>0).sum()) and stat['negative_count']==int((coeff<0).sum())
            assert stat['zero_count']==int((coeff==0).sum()) and stat['support_count']==30
            assert stat['first_bit']==int(coeff[0]>0)
    assert artifact['window_count']==T-1 and artifact['first_latent_excluded']
    assert artifact['used_received_frame_interval']==list(m.phase_slice(length,g))
    forbidden={'source_b','tau','arm','view','attack_origin','path','key_id','CORRECT','WRONG'}
    def check_keys(value):
        if isinstance(value,dict):
            assert not (set(value)&forbidden)
            for item in value.values():check_keys(item)
        elif isinstance(value,list):
            for item in value:check_keys(item)
    check_keys(artifact)
    if length==129 and g==0:
        modified=z.clone();modified[:,:,32]=100
        next_primary,next_artifact=m.phase_observations(modified,'watermark',length,g)
        assert next_primary==primary and next_artifact['windows'][31]!=artifact['windows'][31]


def test_ordered_repeated_skipped_windows_and_first_coefficient_tie():
    def row(values,j):
        a=np.asarray(values,dtype=np.float64)
        stat=dict(sum=float(a.sum()),sumsq=float(np.sum(a*a)),positive_count=int((a>0).sum()),negative_count=int((a<0).sum()),
            zero_count=int((a==0).sum()),support_count=30,first_bit=int(a[0]>0))
        return dict(observed_regular_j=j,payload_stats=[dict(stat) for _ in range(32)])
    A=[0]+[1]*20+[-1]*9;B=[1]*10+[-1]*20;C=[0]*30
    rows=[row(A,1),row(B,2),row(C,3)]
    for order in ([0,1],[1,0],[0,0,2,1],[2,1,2],[1]):
        raw=np.concatenate([np.asarray([A,B,C][i]) for i in order]);out=m.aggregate_ordered_windows([rows[i] for i in order])
        ones=int((raw>0).sum());count=len(raw);expected=int(raw[0]>0) if 2*ones==count else int(2*ones>count)
        assert out['decoded_bits']==[expected]*32 and out['votes'][0]['ones']==ones
        assert out['payload_stats'][0]['sum']==float(raw.sum()) and out['payload_stats'][0]['sumsq']==float(np.sum(raw**2))
        assert out['votes'][0]['zeros']==int((raw<=0).sum())
    assert m.aggregate_ordered_windows(rows[:2])['decoded_bits']==[0]*32  # first window majority is1.
    assert m.aggregate_ordered_windows(rows[1::-1])['decoded_bits']==[1]*32
    with pytest.raises(ValueError):m.aggregate_ordered_windows([])


def test_independent_saved_tensor_golden():
    location=os.environ.get('VTSM_GOLDEN_FIXTURE')
    if not location:pytest.skip('Set VTSM_GOLDEN_FIXTURE for the optional independent saved-tensor receipt')
    fixture=json.loads(Path(location).read_text());source=Path(fixture['source_path'])
    assert b.file_sha256(source)==fixture['source_sha256']
    z=torch.load(source,map_location='cpu',weights_only=True)
    primary,artifact=m.phase_observations(z,'watermark',129,3)
    for observed,expected in zip(artifact['windows'],fixture['rows']):
        for actual,want in zip(observed['payload_stats'],expected):
            assert actual['sum']==pytest.approx(want['sum'],abs=2e-5,rel=2e-5)
            assert actual['sumsq']==pytest.approx(want['sumsq'],abs=2e-5,rel=2e-5)
            assert actual['positive_count']==want['positive'] and actual['negative_count']+actual['zero_count']==want['negative_or_zero']
            assert actual['first_bit']==want['first_bit']
    for expected in fixture['aggregations'].values():
        rows=[artifact['windows'][j-1] for j in expected['observed_regular_indices']]
        aggregate=m.aggregate_ordered_windows(rows)
        assert aggregate['decoded_bits']==expected['bits']
        assert [[r['ones'],r['count']] for r in aggregate['votes']]==expected['positive_and_count']
    assert primary['decoded_bits']==fixture['aggregations']['ordered']['bits']


def test_receiver_rename_length_driven_and_window_export(tmp_path,monkeypatch):
    original=tmp_path/'PILOT_MULTI_CROP4.mp4';renamed=tmp_path/'OFF_FULL_RESAVED181.mp4'
    original.write_bytes(b'129');renamed.write_bytes(original.read_bytes());ledger={}
    monkeypatch.setattr(io,'read_mp4',lambda p:torch.zeros(1,1,1,3).expand(int(Path(p).read_bytes()),320,512,3))
    monkeypatch.setattr(vae,'reencode_rgb24_readback',lambda _,rgb:latent((len(rgb)-1)//4+1))
    def count(kind,complete):ledger.setdefault(kind,[0,0])[int(complete)]+=1
    keys=dict(CORRECT='watermark',WRONG='watermark-wrong')
    a=b.read_mp4_search(original,keys,m.PUBLIC,object(),count=count)
    bresult=b.read_mp4_search(renamed,keys,m.PUBLIC,object())
    assert a['features']==bresult['features'] and a['searches']==bresult['searches'] and a['candidates']==bresult['candidates']
    assert len(a['window_exports'])==8 and sum(x['window_count'] for x in a['window_exports'].values())==250
    assert ledger['vae_encode']==[4,4] and ledger['phase_payload_read']==[8,8]
    assert all(r['status']=='NO_PILOT' for r in a['searches'].values())


def test_partial_view_phase_window_failure_keeps_all_slots_and_valid_primary(tmp_path,monkeypatch):
    from runtime.wan import generation
    store=run.Store(tmp_path/'run',create=True);cfg=run.load_config()
    for arm,row in store.data['generation'].items():row['status']='NOT_COMPLETED'
    row=store.data['generation']['OFF'];p=Path(row['terminal_path']);p.parent.mkdir();torch.save(latent(),p)
    row.update(status='COMPLETE',sha256=b.file_sha256(p))
    frozen=SimpleNamespace(parameters=lambda:iter([torch.zeros(())]))
    monkeypatch.setattr(generation,'load_frozen_vae',lambda *a,**k:frozen)
    monkeypatch.setattr(vae,'decode_normalized_latent',lambda *a:torch.zeros(1,1,1,3).expand(181,320,512,3))
    def encode(rgb,path,*args):
        if path.stem=='CROP4_129':raise RuntimeError('partial derived MP4 failure')
        path.write_bytes(b'fixture')
    monkeypatch.setattr(io,'encode_rgb',encode)
    monkeypatch.setattr(io,'read_mp4',lambda path:torch.zeros(1,1,1,3).expand(129 if path.stem.startswith('CROP') else 181,320,512,3))
    calls=[]
    def reencode(_,rgb):
        calls.append(len(rgb))
        if len(calls)==4:raise RuntimeError('CROP5 phase2 failure')
        return latent((len(rgb)-1)//4+1)
    monkeypatch.setattr(vae,'reencode_rgb24_readback',reencode)
    original_dump=run.dump
    def dump(path,value):
        if Path(path).name=='CROP6_129.phase1.WRONG.windows.json':raise OSError('window sidecar failure')
        original_dump(path,value)
    monkeypatch.setattr(run,'dump',dump)
    run.media_worker(store,cfg);run.recover_unfinished(store,'generation','fixture')
    before=(store.output/'blind_readouts.json').read_bytes();run.evaluate_saved_reads(store,cfg)
    assert store.data['counts']==dict(generated=1,source_mp4=1,derived_mp4=4,normalized=12,phase_reads=24,candidates=294,searches=6,evaluated=12,window_files=23,window_rows=747)
    assert store.data['window_interface_status']=='INCOMPLETE'
    assert store.data['windows']['OFF/CROP6_129/1/WRONG']['status']=='FAILED'
    assert store.data['phase_reads']['OFF/CROP6_129/1/WRONG']['status']=='READ'
    assert store.data['searches']['OFF/CROP6_129/WRONG']['status']=='NO_PILOT'
    assert store.data['searches']['OFF/CROP5_129/CORRECT']['status']=='SEARCH_INCOMPLETE'
    assert (store.output/'blind_readouts.json').read_bytes()==before
    assert all(r['status']!='PENDING' for table in run.TABLE_SIZES for r in store.data[table].values())
    saved=next(r for r in store.data['windows'].values() if r['status']=='SAVED')
    assert b.file_sha256(saved['path'])==saved['sha256'] and len(json.loads(Path(saved['path']).read_text())['windows'])==saved['row_count']


@pytest.mark.parametrize('kind',('phase','windows'))
@pytest.mark.parametrize('boundary',('before_canonical','after_canonical'))
def test_canonical_checkpoint_boundaries(tmp_path,monkeypatch,kind,boundary):
    store=run.Store(tmp_path/'run',create=True);slot='OFF/FULL_RESAVED181/0/CORRECT'
    table='phase_reads' if kind=='phase' else 'windows';done='READ' if kind=='phase' else 'SAVED'
    old_save,old_blind=store.save,store.blind
    def save():
        if boundary=='before_canonical' and store.data[table][slot]['status']==done:raise SystemExit('before canonical')
        old_save()
    def blind():
        if boundary=='after_canonical' and store.data[table][slot]['status']==done:raise SystemExit('after canonical')
        old_blind()
    monkeypatch.setattr(store,'save',save);monkeypatch.setattr(store,'blind',blind)
    p=tmp_path/'input.mp4';p.write_bytes(b'fixture')
    monkeypatch.setattr(io,'read_mp4',lambda _:torch.zeros(1,1,1,3).expand(181,320,512,3))
    monkeypatch.setattr(vae,'reencode_rgb24_readback',lambda *a:latent())
    with pytest.raises(SystemExit):
        b.read_mp4_search(p,dict(CORRECT='watermark',WRONG='watermark-wrong'),m.PUBLIC,object(),
            persist=lambda event,where,value:store.event('OFF/FULL_RESAVED181',event,where,value))
    result=json.loads((store.output/'result.json').read_text());projection=json.loads((store.output/'blind_readouts.json').read_text())
    assert result[table][slot]['status']==(done if boundary=='after_canonical' else 'PENDING')
    assert projection[table][slot]['status']=='PENDING'
    recovered=run.Store(store.output);run.recover_unfinished(recovered,'media','fixture hard exit')
    row=recovered.data[table][slot]
    assert row['status']==(done if boundary=='after_canonical' else 'NOT_COMPLETED')
    if kind=='windows' and boundary=='after_canonical':assert b.file_sha256(row['path'])==row['sha256'] and row['row_count']==45
    assert {table:len(recovered.data[table]) for table in run.TABLE_SIZES}==run.TABLE_SIZES


def test_normal_rejections_complete_only_with_window_interface(tmp_path,monkeypatch):
    output=tmp_path/'run';monkeypatch.setattr(run.sys,'argv',['review','--output',str(output)])
    def phase(output,name):
        s=run.Store(output)
        if name=='generation':
            for row in s.data['generation'].values():row['status']='COMPLETE'
        else:
            for table in ('sources','views','normalized'):
                for row in s.data[table].values():row['status']='SAVED'
            for row in s.data['windows'].values():row.update(status='SAVED',row_count=row['expected_row_count'])
            for row in s.data['phase_reads'].values():row.update(status='READ',decoded_bits=[0]*32)
            for row in s.data['candidates'].values():row.update(status='NO_ENERGY',score=0.)
            for row in s.data['searches'].values():row.update(status='NO_PILOT',decoded_bits=None)
        s.data['workers'][name]=dict(status='COMPLETE');s.save();s.blind();return s,None
    monkeypatch.setattr(run,'run_worker_phase',phase);monkeypatch.setattr(run,'quality_diagnostics',lambda *a:None)
    run.main();s=run.Store(output)
    assert s.data['status']=='EXECUTION_COMPLETE' and s.data['counts']['evaluated']==60
    assert all(not row['accepted_payload'] and row['ber'] is None for row in s.data['evaluations'].values())
    s.data['windows']['OFF/FULL_RESAVED181/0/CORRECT']['status']='FAILED';s.save()
    assert s.data['window_interface_status']=='INCOMPLETE' and s.data['counts']['window_files']==101


def test_notebook_static_fixed_roster_and_existing_requirements(tmp_path):
    from scripts.build_video_temporal_sync_multi_notebook import build
    nb=json.loads(build(output=tmp_path/'draft.ipynb').read_text())
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    sources=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'];requirements=[]
    for source in sources:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.List):
                for i,item in enumerate(node.elts):
                    if isinstance(item,ast.Constant) and item.value=='-r':requirements.append(node.elts[i+1].args[0].right.value)
    assert requirements and all((run.ROOT/path).is_file() for path in requirements)
    joined='\n'.join(sources)
    assert 'sys.executable' in joined and 'Video(str(' in joined and 'embed=True' in joined
    assert 'ensurepip' not in joined and 'sys.version_info' not in joined and 'CUDA_VISIBLE_DEVICES' not in joined
    assert nb['metadata']['candidate_binding']['source_sha'] is None
    source=sources[1].replace("Path('/content/drive/MyDrive/Video-WM/Video-Temporal-Sync-Multi-V1')",f'Path({str(tmp_path)!r})')
    with pytest.raises(RuntimeError,match='Unpublished draft'):exec(compile(source,'setup','exec'),{})
    receipt=json.loads(next(tmp_path.glob('*/setup_receipt.json')).read_text())
    assert {name:len(receipt[name]) for name in run.TABLE_SIZES}==run.TABLE_SIZES
    assert sum(row['expected_row_count'] for row in receipt['windows'].values())==3270
