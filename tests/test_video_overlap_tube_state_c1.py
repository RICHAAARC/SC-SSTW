"""CPU fixtures/static only. No model, VAE weights, codec or real media execution."""
import ast,copy,json,re,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import video_overlap_tube_state as s,video_overlap_tube_control as m
from runtime.wan import video_overlap_tube_state as b,trajectory,io,vae
from experiments.wan_state_clock import video_overlap_tube_state_c1_run as run
pytestmark=pytest.mark.unit

@pytest.fixture(autouse=True)
def threads():
    n=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)

def latent(T=46):return torch.zeros(1,16,T,40,64)


def test_pilot_actual_gradient_cap_and_orthogonal_payload():
    z=torch.randn((1,16,46,40,64),generator=torch.Generator().manual_seed(17))*10
    targets=m.build_targets(z,'watermark',m.message_bits('OKOK'))
    dp,detail=m.pilot_delta(z,targets)
    assert detail['raw_delta_l2']>1 and detail['cap_scale']<1
    assert float(dp.double().square().sum().sqrt())==pytest.approx(1,abs=1e-7)
    assert torch.count_nonzero(dp[:,[i for i in range(16) if i!=4]])==0
    # Orthonormal projection gradient is -1/4 times residual reconstructed on the actual tensor.
    expected=torch.zeros_like(z)
    for i,(h0,h1,w0,w1) in enumerate(s.PUBLIC.blocks):
        patch=z[0,4,1:,h0:h1,w0:w1].reshape(45,16)
        residual=(patch@targets['basis'][i]).reshape(45,4,2)-targets['pilot_target'][:,i]
        residual=torch.where(targets['pilot_active'][:,i],residual,0.)
        expected[0,4,1:,h0:h1,w0:w1]=(-.25*(residual.reshape(45,8)@targets['basis'][i].T)).reshape(45,4,4)
    expected*=min(1.,1./float(expected.double().norm()))
    torch.testing.assert_close(dp,expected,atol=2e-8,rtol=2e-6)
    v,row=m.guided_velocity(z,torch.zeros_like(z),torch.zeros_like(z),.2,targets,'OVERLAP_MULTI',25)
    assert row['cross_inner_product']==0 and abs(row['norm_decomposition_error'])<1e-7
    baseline,pinfo=m.payload.local_delta(z,targets['payload_target'],targets['payload_mask'])
    torch.testing.assert_close(v,-5*(baseline+dp)/.2,atol=2e-5,rtol=2e-6)
    assert row['payload']['eta']==5520 and row['pilot']['eta']==174


def test_native_full_histories_fresh_25_updates():
    from diffusers import UniPCMultistepScheduler
    class Transformer:
        def __call__(self,hidden_states,encoder_hidden_states,**kw):return (hidden_states*.01+encoder_hidden_states.mean(),)
    scheduler=UniPCMultistepScheduler(prediction_type='flow_prediction',thresholding=False,predict_x0=True,lower_order_final=True,use_flow_sigmas=True,flow_shift=3.,final_sigmas_type='zero')
    scheduler.set_timesteps(50,device='cpu');scheduler.set_begin_index(0)
    fingerprint=trajectory.fingerprint(vars(scheduler));initial=latent();receipts=[];terminals=[];ledger={}
    def count(kind,done):ledger.setdefault(kind,[0,0])[int(done)]+=1
    for arm in m.ARMS:
        rows=[];terminal,receipt=b.run_trajectory(SimpleNamespace(transformer=Transformer()),initial,copy.deepcopy(scheduler),torch.tensor([.03]),torch.tensor([-.02]),torch.bfloat16,arm,'watermark',m.message_bits('OKOK'),count,rows.append)
        receipts.append(receipt);terminals.append(terminal)
        assert [r['index'] for r in rows if r['enabled']]==([] if arm=='OFF' else list(range(25,50)))
        assert [r['cursor_after'] for r in rows]==list(range(1,51))
        if arm=='OVERLAP_MULTI':assert all(r['pilot_delta_l2']<=1+1e-7 for r in rows[25:])
    assert receipts[0]['before_step25']==receipts[1]['before_step25']==receipts[2]['before_step25']
    assert trajectory.fingerprint(vars(scheduler))==fingerprint and torch.count_nonzero(initial)==0
    assert ledger['native_step']==[150,150] and ledger['local_control']==[50,50]
    assert ledger['payload_gradient']==[50,50] and ledger['pilot_gradient']==[25,25]
    assert not torch.equal(terminals[0],terminals[1]) and not torch.equal(terminals[1],terminals[2])


def test_common_support_and_receiver_rename_partialphase(tmp_path,monkeypatch):
    a=tmp_path/'OVERLAP_CROP5.mp4';z=tmp_path/'OFF_FULL.mp4';a.write_bytes(b'129');z.write_bytes(a.read_bytes())
    monkeypatch.setattr(io,'read_mp4',lambda p:torch.zeros(1,1,1,3).expand(int(p.read_bytes()),320,512,3))
    lengths=[]
    def encode(_,rgb):lengths.append(len(rgb));return latent((len(rgb)-1)//4+1)
    monkeypatch.setattr(vae,'reencode_rgb24_readback',encode)
    keys=dict(CORRECT='watermark',WRONG='watermark-wrong')
    one=b.read_mp4_search(a,keys,m.PUBLIC,object());two=b.read_mp4_search(z,keys,m.PUBLIC,object())
    assert one==two and lengths==[129,125,125,125]*2
    assert all(r['status']=='NO_ENERGY' and r['canonical'] is None for r in one['searches'].values())
    assert all(f['phase']['R']==31 for f in one['features'].values())
    assert [m.phase_spec(181,g)['R'] for g in range(4)]==[44]*4
    assert [m.phase_spec(181,g)['regular_count'] for g in range(4)]==[45,44,44,44]
    calls=[]
    def fail(_,rgb):
        calls.append(0)
        if len(calls)==3:raise RuntimeError('phase fixture failure')
        return latent((len(rgb)-1)//4+1)
    monkeypatch.setattr(vae,'reencode_rgb24_readback',fail)
    partial=b.read_mp4_search(a,keys,m.PUBLIC,object())
    assert len(partial['features'])==8 and sum(r['status']=='NO_ENERGY' for r in partial['features'].values())==6
    assert all(r['status']=='SEARCH_INCOMPLETE' for r in partial['searches'].values())


def test_roster_counts_and_truth_after_saved(tmp_path):
    cfg=run.load_config();store=run.Store(tmp_path/'run',create=True)
    assert {k:len(store.data[k]) for k in run.TABLE_SIZES}==run.TABLE_SIZES
    assert sum(r['shared']['catalog_count'] for r in store.data['phase_reads'].values())==357480
    assert sum(r['shared']['valid_count'] for r in store.data['phase_reads'].values())==92016
    run.recover_unfinished(store,'generation','fixture');run.recover_unfinished(store,'media','fixture')
    before=(store.output/'blind_readouts.json').read_bytes();run.evaluate_saved_reads(store,cfg)
    assert (store.output/'blind_readouts.json').read_bytes()==before
    for start,g,tau in ((4,0,2),(5,3,3),(6,2,3),(7,1,3)):
        row=store.data['evaluations'][f'OVERLAP_MULTI/CROP{start}_129/CORRECT/REGISTERED']
        assert row['registered_g']==g and row['registered_tau'][0]==tau
        assert not row['accepted_payload'] and row['bit_errors'] is None
    assert all(r['status']!='PENDING' for name in run.TABLE_SIZES for r in store.data[name].values())


@pytest.mark.parametrize('boundary',['before','after'])
def test_canonical_exit_boundary(tmp_path,monkeypatch,boundary):
    store=run.Store(tmp_path/'run',create=True);slot='OFF/FULL_RESAVED181/0/CORRECT'
    infer=s.infer(np.zeros((44,16,40,64)),'watermark',np.ones((44,4),bool))
    row=dict(status='NO_ENERGY',inference=infer,payload=dict(status='READ',decoded_bits=[0]*32))
    save,blind=store.save,store.blind
    def interrupted_save():
        if boundary=='before' and store.data['phase_reads'][slot]['status']=='NO_ENERGY':raise SystemExit('before canonical')
        save()
    def interrupted_blind():
        if boundary=='after' and store.data['phase_reads'][slot]['status']=='NO_ENERGY':raise SystemExit('after canonical')
        blind()
    monkeypatch.setattr(store,'save',interrupted_save);monkeypatch.setattr(store,'blind',interrupted_blind)
    with pytest.raises(SystemExit):store.event('OFF/FULL_RESAVED181','phase',(0,'CORRECT'),row)
    reopened=run.Store(store.output);run.recover_unfinished(reopened,'media','interrupted')
    actual=reopened.data['phase_reads'][slot]
    assert actual['status']==('NO_ENERGY' if boundary=='after' else 'NOT_COMPLETED')
    if boundary=='after':assert actual['payload']['decoded_bits']==[0]*32


def test_parent_spawn_failure_finishes_all_slots(tmp_path,monkeypatch):
    output=tmp_path/'run';monkeypatch.setattr(sys,'argv',['test','--output',str(output)])
    monkeypatch.setattr(run.subprocess,'check_output',lambda *a,**kw:'47405e5b45e9065103dbb1dac447617ddaed4f2e\n')
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**kw:(_ for _ in ()).throw(OSError('spawn fixture')))
    with pytest.raises(SystemExit) as error:run.main()
    assert error.value.code==1
    saved=json.loads((output/'result.json').read_text())
    assert saved['status']=='INCOMPLETE' and saved['stage']=='FINISHED'
    assert len(saved['failures'])==2 and len(saved['workers'])==2
    assert all(r['status']!='PENDING' for name in run.TABLE_SIZES for r in saved[name].values())
    assert (output/'blind_readouts.json').is_file()


def test_reserved_real_rgb_edit_wrapper_fixture(tmp_path,monkeypatch):
    crop=torch.arange(129.).reshape(129,1,1,1).expand(129,320,512,3);donor=crop+1000
    monkeypatch.setattr(io,'read_mp4',lambda p:donor if p.name=='donor.mp4' else crop)
    calls=[]
    def save(rgb,path,fps,crf):calls.append((rgb[:,0,0,0].clone(),fps,crf));path.write_bytes(b'saved fixture')
    monkeypatch.setattr(io,'encode_rgb',save)
    monkeypatch.setattr(b,'read_mp4_search',lambda path,keys,public,frozen,**kw:dict(bytes=path.read_bytes(),keys=keys))
    result=b.prepare_reserved_edit_mp4(tmp_path/'crop.mp4',tmp_path/'edited.mp4','OFF_DONOR_INSERT4',{'X':'watermark'},m.PUBLIC,object(),off_donor_path=tmp_path/'donor.mp4')
    assert result['bytes']==b'saved fixture' and calls[0][1:]==(8,18)
    assert calls[0][0].shape==(133,) and calls[0][0][68:72].tolist()==[1064,1065,1066,1067]
    assert len(b.prepare_reserved_edit(crop,'DELETE4'))==125 and len(b.prepare_reserved_edit(crop,'REPEAT4'))==133


def test_notebook_static_and_builder(tmp_path):
    from scripts import build_video_overlap_tube_state_c1_notebook as builder
    path=builder.build(output=tmp_path/'candidate.ipynb');official=run.ROOT/'notebooks/video_overlap_tube_state_c1_v1_colab.ipynb'
    assert path.read_bytes()==official.read_bytes()
    cells=json.loads(path.read_text())['cells'];assert ''.join(cells[0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    code='\n'.join(''.join(c['source']) for c in cells if c['cell_type']=='code')
    for c in cells:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    assert 'SOURCE_SHA = None' in code and 'PYTHON=sys.executable' in code
    assert 'Video(str(row[' in code and 'venv' not in code and 'ensurepip' not in code
    required=re.findall(r"REPO/'([^']+requirements[^']+)'",code)
    assert required and all((run.ROOT/p).is_file() for p in required)
    assert 'prepare_reserved_edit' not in code
