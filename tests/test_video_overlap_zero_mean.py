"""Necessary CPU synthetic/static checks; no real model, VAE, codec or saved-run ranking."""
import ast,copy,hashlib,json,re,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import video_overlap_zero_mean_state as s,video_overlap_zero_mean_control as m
from main.tube_state import video_overlap_tube_state as old
from runtime.wan import video_overlap_zero_mean as b,trajectory,io,vae
from experiments.wan_state_clock import video_overlap_zero_mean_c1_run as run
pytestmark=pytest.mark.unit

@pytest.fixture(autouse=True)
def threads():
    n=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)

def latent(T=46):return torch.zeros(1,16,T,40,64)

def scheduler():
    from diffusers import UniPCMultistepScheduler
    value=UniPCMultistepScheduler(prediction_type='flow_prediction',thresholding=False,predict_x0=True,
        lower_order_final=True,use_flow_sigmas=True,flow_shift=3.,final_sigmas_type='zero')
    value.set_timesteps(50,device='cpu');value.set_begin_index(0)
    return value

class Transformer:
    def __call__(self,hidden_states,encoder_hidden_states,**kw):return (hidden_states*.01+encoder_hidden_states.mean(),)

def fixture_trajectory(arm,callback=None):
    hist=scheduler();ledger={};rows=[];initial=latent()
    before=trajectory.fingerprint(vars(hist));rng=torch.random.get_rng_state().clone()
    def count(kind,done):ledger.setdefault(kind,[0,0])[int(done)]+=1
    terminal,receipt=b.run_trajectory(SimpleNamespace(transformer=Transformer()),initial,hist,
        torch.tensor([.03]),torch.tensor([-.02]),torch.bfloat16,arm,'watermark',m.message_bits('OKOK'),
        count,rows.append,diagnostic=callback)
    assert torch.equal(rng,torch.random.get_rng_state()) and torch.count_nonzero(initial)==0
    assert hist.step_index==50 and before!=trajectory.fingerprint(vars(hist))
    return terminal,receipt,rows,ledger

def test_exact_replacement_fixed_keys_and_no_cache_alias():
    changed=unchanged=0
    for key in ('watermark','watermark-wrong','', 'representative-2','representative-17'):
        layout=s.basis_layout(key);U=s.bases(key);previous=old.bases(key)
        assert layout==s.basis_layout(key)
        np.testing.assert_array_equal(U,s.bases(key))
        np.testing.assert_array_equal(U.sum(axis=1),np.zeros((4,8)))
        for i,row in enumerate(layout['blocks']):
            order=sorted(range(16),key=lambda c:(hashlib.sha256(f'VTOS1/basis/order\0{key}\0{i}\0{c}'.encode()).digest(),c))
            signs=[2*(hashlib.sha256(f'VTOS1/basis/sign\0{key}\0{i}\0{q}'.encode()).digest()[0]&1)-1 for q in range(8)]
            assert row['full_original_order']==order and row['signs']==signs
            np.testing.assert_array_equal(U[i].T@U[i],np.eye(8))
            if 0 not in order[:8]:
                unchanged+=1;np.testing.assert_array_equal(U[i],previous[i]);continue
            changed+=1;q=order[:8].index(0);replacement=next(c for c in order if c and c not in order[:8])
            assert row['replaced_slot']==q and row['selected'][q]==replacement
            for slot in range(8):
                if slot!=q:np.testing.assert_array_equal(U[i,:,slot],previous[i,:,slot])
                else:np.testing.assert_array_equal(U[i,:,slot],[signs[slot]*(-1)**((r&replacement).bit_count())/4 for r in range(16)])
        U[:]=999;layout['blocks'].reverse()
        assert np.max(np.abs(s.bases(key)))==.25
    assert changed and unchanged
    assert s.PUBLIC!=old.PUBLIC and s.PUBLIC.method_version=='video-overlap-zero-mean-v1'

def test_actual_full_tensor_write_read_budget_and_inherited_sse():
    for key in ('watermark','watermark-wrong'):
        tensor=s.synthesize(key,dtype=np.float32)
        received=tensor[0,:,1:].transpose(1,0,2,3)
        projection=s.extract(received,key,np.ones((45,4),bool))
        expected=s.composite_signs(key)*s.PUBLIC.alpha
        np.testing.assert_allclose(projection,expected,atol=5e-9,rtol=0)
        assert np.count_nonzero(expected)==1392 and expected.size==1440
        assert np.linalg.norm(tensor.astype(np.float64))==pytest.approx(1,abs=1e-7)
        assert np.count_nonzero(tensor[:,:,0])==0 and np.count_nonzero(tensor[:,[i for i in range(16) if i!=4]])==0
        result=s.infer(received,key,np.ones((45,4),bool))
        reference=old.infer(old.synthesize(key)[0,:,1:].transpose(1,0,2,3),key,np.ones((45,4),bool))
        assert result['summary']['top_catalog_indices']==reference['summary']['top_catalog_indices']
        np.testing.assert_allclose(result['path_costs'],reference['path_costs'],atol=1e-10,rtol=0)
        assert s.catalog(45)==old.catalog(45)
        assert s.family_receipt(key,45,np.ones((45,4),bool))==old.family_receipt(key,45,np.ones((45,4),bool))

def test_small_path_cases_equivalence_and_unsupported_label_counterexample():
    clean=s.synthesize('watermark')[0,:,1:].transpose(1,0,2,3)
    for taus in (list(range(1,32)),list(range(7,38)),list(range(1,17))+list(range(16,31)),list(range(1,17))+list(range(18,33))):
        value=clean[np.asarray(taus)-1]
        out=s.infer(value,'watermark',np.ones((31,4),bool))
        assert out['summary']['unique_model_hypothesis']
        assert s.catalog(31)[out['summary']['canonical_catalog_index']]['taus']==taus
        assert not out['summary']['state_path_accepted']
    mask=np.ones((31,4),bool);mask[15]=False
    family=s.family_receipt('watermark',31,mask)
    group=next(c for c in family['classes'] if len(c['member_catalog_indices'])>1)
    rows=s.catalog(31);path=rows[group['member_catalog_indices'][0]]['taus']
    out=s.infer(clean[np.asarray(path)-1],'watermark',mask)
    assert set(group['member_catalog_indices'])<=set(out['summary']['top_catalog_indices'])
    assert not out['summary']['unique_model_hypothesis'] and out['summary']['structural_ambiguity']
    # An inserted exact copy and a repeat are the same physical observation: no label recovery claim.
    base=clean[:30];repeat=np.concatenate([base[:15],base[14:15],base[15:]])
    copy_insert=np.insert(base,15,base[14],axis=0)
    np.testing.assert_array_equal(repeat,copy_insert)
    assert s.infer(repeat,'watermark',np.ones((31,4),bool))==s.infer(copy_insert,'watermark',np.ones((31,4),bool))
    assert s.infer(np.zeros_like(repeat),'watermark',np.ones((31,4),bool))['summary']['status']=='NO_ENERGY'
    assert s.infer(repeat,'watermark',np.zeros((31,4),bool))['summary']['reason']=='NO_OBSERVATIONS'
    repeat[0,4,8,12]=np.nan
    assert s.infer(repeat,'watermark',np.ones((31,4),bool))['summary']['status']=='INCOMPLETE'

def test_gradient_actual_tensor_closure_and_fp64_projection():
    z=torch.randn((1,16,46,40,64),generator=torch.Generator().manual_seed(17))*10
    targets=m.build_targets(z,'watermark',m.message_bits('OKOK'));ds,detail=m.pilot_delta(z,targets)
    assert detail['raw_delta_l2']>1
    expected=torch.zeros_like(z)
    for i,(h0,h1,w0,w1) in enumerate(s.PUBLIC.blocks):
        patch=z[0,4,1:,h0:h1,w0:w1].reshape(45,16)
        residual=(patch@targets['basis'][i]).reshape(45,4,2)-targets['pilot_target'][:,i]
        residual=torch.where(targets['pilot_active'][:,i],residual,0.)
        expected[0,4,1:,h0:h1,w0:w1]=(-.25*(residual.reshape(45,8)@targets['basis'][i].T)).reshape(45,4,4)
    expected*=min(1.,1./float(expected.double().norm()))
    torch.testing.assert_close(ds,expected,atol=2e-8,rtol=2e-6)
    projected=m.project_tensor(ds,targets['basis'])
    oracle=s.extract(ds[0,:,1:].numpy().transpose(1,0,2,3),'watermark',np.ones((45,4),bool))
    np.testing.assert_allclose(projected.numpy(),oracle,atol=2e-8,rtol=2e-6)
    closure=m.delta_closure(ds,projected,targets)
    assert closure['actual_spatial_l2']==pytest.approx(1,abs=1e-7)
    assert abs(closure['parseval_error'])<2e-7 and abs(closure['outside_ROI_energy'])<1e-12
    assert closure['boundary_projection_energy']<1e-15 and closure['spatial_U_complement_l2']<1e-7
    leaky=ds.clone();leaky[0,7,0,0,0]=.25
    assert m.delta_closure(leaky,projected,targets)['outside_ROI_energy']==pytest.approx(.0625)
    capture={};v,row=m.guided_velocity(z,torch.zeros_like(z),torch.zeros_like(z),.2,targets,'OVERLAP_MULTI',25,diagnostics=capture)
    dp,_=m.payload.local_delta(z,targets['payload_target'],targets['payload_mask'])
    torch.testing.assert_close(v,-5*(dp+ds)/.2,atol=2e-5,rtol=2e-6)
    assert row['cross_inner_product']==0 and row['payload']['eta']==5520 and row['pilot']['eta']==174

def test_native_diagnostics_noninvasive_all_arms(tmp_path,monkeypatch):
    all_results=[];velocities=[];real_step=trajectory.native_step
    def step(scheduler,z,velocity,index,*a,**kw):
        velocities.append(trajectory.fingerprint(velocity))
        return real_step(scheduler,z,velocity,index,*a,**kw)
    monkeypatch.setattr(trajectory,'native_step',step)
    for arm in m.ARMS:
        captured={}
        def callback(i,arrays,metadata):captured[i]=(copy.deepcopy(arrays),copy.deepcopy(metadata))
        outcome=fixture_trajectory(arm,callback);all_results.append(outcome)
        terminal,receipt,rows,ledger=outcome
        assert [r['index'] for r in rows if r['enabled']]==([] if arm=='OFF' else list(range(25,50)))
        assert len(captured)==25 and receipt['writer_diagnostics']['last_z_post_matches_terminal'] is True
        assert receipt['writer_diagnostics']['completed']==25
        np.testing.assert_array_equal(captured[49][0]['z_post'],m.project_tensor(terminal,torch.tensor(s.bases('watermark'),dtype=torch.float32)).numpy())
        assert captured[49][1]['sigma_post']==0
        assert ledger['native_step']==[50,50] and ledger['transformer_conditional']==ledger['transformer_unconditional']==[50,50]
        if arm!='OVERLAP_MULTI':assert all(np.count_nonzero(v[0]['pilot_delta'])==0 for v in captured.values())
    assert all_results[0][1]['before_step25']==all_results[1][1]['before_step25']==all_results[2][1]['before_step25']
    assert not torch.equal(all_results[1][0],all_results[2][0])
    captured_velocity=velocities[-50:];velocities.clear()
    without=fixture_trajectory('OVERLAP_MULTI')
    with_=all_results[-1]
    torch.testing.assert_close(without[0],with_[0],atol=0,rtol=0)
    assert without[1]['final_history_sha256']==with_[1]['final_history_sha256']
    assert without[3]==with_[3] and velocities==captured_velocity
    assert with_[3]['payload_gradient']==with_[3]['pilot_gradient']==[25,25]

def test_terminal_diagnostic_failure_preserves_native_terminal(monkeypatch):
    original=m.project_tensor;n=0
    def project(*args):
        nonlocal n
        n+=1
        if n==126:raise RuntimeError('terminal projection fixture')
        return original(*args)
    monkeypatch.setattr(m,'project_tensor',project)
    terminal,receipt,rows,ledger=fixture_trajectory('OFF',lambda *a:None)
    assert n==126 and len(rows)==50 and ledger['native_step']==[50,50]
    assert torch.isfinite(terminal).all()
    assert receipt['writer_diagnostics']['status']=='FAILED' and receipt['writer_diagnostics']['last_z_post_matches_terminal'] is None

def test_callback_failure_does_not_abort_trajectory():
    observed=[]
    def callback(i,*rest):
        observed.append(i)
        if i==31:raise OSError('sidecar fixture')
    _,receipt,rows,_=fixture_trajectory('OFF',callback)
    assert observed==list(range(25,50)) and len(rows)==50
    assert receipt['writer_diagnostics']['completed']==24 and rows[31]['writer_diagnostic']['status']=='FAILED'

def test_receiver_uses_new_basis_rename_and_partialphase(tmp_path,monkeypatch):
    a=tmp_path/'OVERLAP_CROP5.mp4';z=tmp_path/'OFF_FULL.mp4';a.write_bytes(b'129');z.write_bytes(a.read_bytes())
    monkeypatch.setattr(io,'read_mp4',lambda p:torch.zeros(1,1,1,3).expand(int(p.read_bytes()),320,512,3))
    full=torch.from_numpy(s.synthesize('watermark',dtype=np.float32))
    lengths=[]
    def encode(_,rgb):lengths.append(len(rgb));return full[:,:,:((len(rgb)-1)//4+1)].clone()
    monkeypatch.setattr(vae,'reencode_rgb24_readback',encode)
    keys=dict(CORRECT='watermark',WRONG='watermark-wrong')
    one=b.read_mp4_search(a,keys,m.PUBLIC,object());two=b.read_mp4_search(z,keys,m.PUBLIC,object())
    assert one==two and lengths==[129,125,125,125]*2
    for name,key in keys.items():
        expected=s.extract(full[0,:,1:32].numpy().transpose(1,0,2,3),key,np.ones((31,4),bool))
        np.testing.assert_allclose(one['features'][0,name]['inference']['projection'],expected,atol=0,rtol=0)
    calls=[]
    def fail(_,rgb):
        calls.append(0)
        if len(calls)==3:raise RuntimeError('phase fixture failure')
        return full[:,:,:((len(rgb)-1)//4+1)].clone()
    monkeypatch.setattr(vae,'reencode_rgb24_readback',fail)
    partial=b.read_mp4_search(a,keys,m.PUBLIC,object())
    assert len(partial['features'])==8 and sum(r['status']=='FAILED' for r in partial['features'].values())==2
    assert all(r['status']=='SEARCH_INCOMPLETE' for r in partial['searches'].values())
    with pytest.raises(ValueError):b.read_mp4_search(a,keys,old.PUBLIC,object())

def test_writer_sidecar_fixed_slots_blind_separation_and_recovery(tmp_path,monkeypatch):
    store=run.Store(tmp_path/'run',create=True);cfg=run.load_config()
    assert len(store.data['writer_diagnostics'])==75
    arrays={k:np.zeros((45,4,4,2),np.float32) for k in b.DIAGNOSTIC_ARRAYS}
    store.blind();before=(store.output/'blind_readouts.json').read_bytes()
    store.writer_event('OFF',25,arrays,dict(status='COMPLETE',index=25))
    row=store.data['writer_diagnostics']['OFF/25']
    with np.load(row['path']) as saved:assert set(saved.files)==set(b.DIAGNOSTIC_ARRAYS)
    assert row['status']=='SAVED' and row['sha256']==b.file_sha256(row['path'])
    monkeypatch.setattr(run.np,'savez_compressed',lambda *a,**kw:(_ for _ in ()).throw(OSError('export fixture')))
    with pytest.raises(OSError):store.writer_event('OFF',26,arrays,dict(status='COMPLETE',index=26))
    reopened=run.Store(store.output);run.recover_unfinished(reopened,'generation','fixture')
    assert reopened.data['writer_diagnostics']['OFF/25']['status']=='SAVED'
    assert reopened.data['writer_diagnostics']['OFF/26']['status']=='FAILED'
    assert reopened.data['writer_diagnostic_counts']==dict(expected=75,saved=1,failed=1,not_completed=73,pending=0)
    # Only writer/generation state changed: blind tables unchanged.
    assert (store.output/'blind_readouts.json').read_bytes()==before
    assert 'writer_diagnostics' not in json.loads(before)
    run.recover_unfinished(reopened,'media','fixture');raw=(store.output/'blind_readouts.json').read_bytes()
    run.evaluate_saved_reads(reopened,cfg)
    assert (store.output/'blind_readouts.json').read_bytes()==raw
    assert len(reopened.data['evaluations'])==60 and all(not v['accepted_payload'] for v in reopened.data['evaluations'].values())


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



def test_notebook_static_source_binding_and_builder(tmp_path):
    from scripts import build_video_overlap_zero_mean_c1_notebook as builder
    official=run.ROOT/'notebooks/video_overlap_zero_mean_c1_v1_colab.ipynb'
    bound=json.loads(official.read_text())['metadata']['candidate_binding']['source_sha']
    assert bound is None or re.fullmatch('[0-9a-f]{40}',bound)
    path=builder.build(source_sha=bound,output=tmp_path/'candidate.ipynb')
    assert path.read_bytes()==official.read_bytes()
    value=json.loads(path.read_text());cells=value['cells']
    assert value['metadata']['candidate_binding']['source_sha']==bound
    assert ''.join(cells[0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    code='\n'.join(''.join(c['source']) for c in cells if c['cell_type']=='code')
    for c in cells:
        if c['cell_type']=='code':
            ast.parse(''.join(c['source']));assert c['outputs']==[] and c['execution_count'] is None
    assert f'SOURCE_SHA = {bound!r}' in code and 'PYTHON=sys.executable' in code
    draft=builder.build(output=tmp_path/'draft.ipynb')
    assert json.loads(draft.read_text())['metadata']['candidate_binding']['source_sha'] is None
    if bound:assert 'Published source: '+chr(96)+bound+chr(96) in ''.join(cells[1]['source'])
    assert 'Video(str(row[' in code and 'venv' not in code and 'ensurepip' not in code
    required=re.findall(r"REPO/'([^']+requirements[^']+)'",code)
    assert required and all((run.ROOT/p).is_file() for p in required)
    assert 'prepare_reserved_edit' not in code and 'writer_diagnostics=' in code
    assert 'Video-Overlap-Zero-Mean-C1-V1' in code and 'video_overlap_zero_mean_c1_run' in code
    assert run.TABLE_SIZES==dict(generation=3,sources=3,views=15,normalized=60,phase_reads=120,searches=30,evaluations=60)
