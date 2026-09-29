"""CPU fixtures only: no model weights, GPU or real generated-video claims."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import video_temporal_sync_bridge as m
from main.tube_state import grow_video_reference as old
from runtime.wan import video_temporal_sync_bridge as b,trajectory
from experiments.wan_state_clock import video_temporal_sync_bridge_run as run

pytestmark=pytest.mark.unit


@pytest.fixture(autouse=True)
def cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def latent(T=46):return torch.zeros(1,16,T,40,64)


def test_fp32_payload_delta_preserved_and_orthogonal_energy():
    z=torch.randn((1,16,46,40,64),generator=torch.Generator().manual_seed(42))*.1;bits=m.message_bits('OKOK')
    target0,mask0=old.build_target(z,'watermark',bits)
    target1,mask1=m.build_target(z,'watermark',bits,'PAYLOAD_LAST')
    target2,mask2=m.build_target(z,'watermark',bits,'PILOT_LAST')
    assert torch.equal(target0,target1) and torch.equal(mask0,mask1)
    d0,r0=old.local_delta(z,target0,mask0);d1,r1=m.local_delta(z,target1,mask1);d2,r2=m.local_delta(z,target2,mask2)
    assert torch.equal(d0,d1)
    torch.testing.assert_close(d2[:,:4],d1[:,:4],rtol=2e-6,atol=1e-7)
    assert float((d2[:,:4]-d1[:,:4]).abs().max())<1e-7
    assert int(mask2.sum())==47104 and r2['eta']==5888 and r1['eta']==5520
    assert torch.count_nonzero(d2[:,5:])==0
    assert r2['total_delta_squared_l2']==pytest.approx(r2['payload_delta_squared_l2']+r2['pilot_delta_squared_l2'],rel=1e-14)


def test_layout_lfsr_mapping_and_fixed_counts(tmp_path):
    assert m.LFSR_BITS=='111100010011010' and len(m.chips())==15
    coords,pn=m.pilot_layout('watermark')
    independent=lambda domain,h,w:hashlib.sha256((domain+'\0watermark\0'+str(h)+'\0'+str(w)).encode()).digest()
    expected=sorted(m.band(),key=lambda c:(independent('VTSB1/pilot/coords',*c),*c))[:64]
    assert coords==expected and pn==[2*(independent('VTSB1/pilot/sign',*c)[0]&1)-1 for c in expected]
    receipt=m.layout_receipt('watermark','watermark-wrong')
    assert receipt['correct']['pilot_support_sha256']!=receipt['wrong']['pilot_support_sha256']
    assert m.temporal_sign(0)==m.temporal_sign(1)==m.temporal_sign(3)
    with pytest.raises(ValueError):m.temporal_sign(46)
    rows=m.candidates(129);assert len(rows)==53 and rows[5]==dict(b=5,g=3,a=2,R=31)
    for row in rows:
        assert (row['b']+row['g'])%4==0 and 1<=1+row['a']<=31+row['a']<=45
    assert [m.phase_slice(129,g)[1]-g for g in range(4)]==[129,125,125,125]
    store=run.Store(tmp_path/'fixed',create=True)
    assert {name:len(store.data[name]) for name in run.TABLE_SIZES}==run.TABLE_SIZES
    assert len(store.data['candidates'])==3*2*(1+53)==324


def template(key,row):
    _,pn=m.pilot_layout(key)
    return np.asarray([m.temporal_sign(j+row['a']) for j in range(1,row['R']+1)])[:,None]*np.asarray(pn)[None,:]


def test_soft_cosine_zero_nonfinite_wrongkey_threshold_and_ties():
    row=m.candidates(129)[5];X=template('watermark',row)*.5
    assert m.score_candidate({'pilot_fft':X.tolist()},'watermark',row)['score']==pytest.approx(1)
    wrong=m.score_candidate({'pilot_fft':X.tolist()},'watermark-wrong',row)
    assert abs(wrong['score'])<.5
    zero=m.score_candidate({'pilot_fft':np.zeros_like(X).tolist()},'watermark',row)
    assert zero['status']=='NO_ENERGY' and zero['score']==0
    bad=X.copy();bad[0,0]=float('nan')
    assert m.score_candidate({'pilot_fft':bad.tolist()},'watermark',row)['status']=='FAILED'
    huge=np.full_like(X,1e152)
    assert m.score_candidate({'pilot_fft':huge.tolist()},'watermark',row)['status']=='FAILED'
    rows=[dict(**c,status='SCORED',score=0.) for c in m.candidates(129)]
    rows[5]['score']=.5;result=m.decide(rows,129)
    assert result['status']=='LOCATED' and result['selected_b']==5 and result['selected_g']==3
    rows[6]['score']=.5-5e-13;assert m.decide(rows,129)['status']=='AMBIGUOUS'
    rows[6]['score']=0;rows[5]['score']=np.nextafter(.5,0);assert m.decide(rows,129)['status']=='NO_PILOT'
    rows[1].update(status='FAILED',score=None);assert m.decide(rows,129)['status']=='SEARCH_INCOMPLETE'


def test_native_complete_history_same_before_step49_and_cfg():
    from diffusers import UniPCMultistepScheduler
    class Transformer:
        def __call__(self,hidden_states,encoder_hidden_states,**kwargs):return (hidden_states*.01+encoder_hidden_states.mean(),)
    pristine=UniPCMultistepScheduler(prediction_type='flow_prediction',thresholding=False,predict_x0=True,
        lower_order_final=True,use_flow_sigmas=True,flow_shift=3.,final_sigmas_type='zero')
    pristine.set_timesteps(50,device='cpu');pristine.set_begin_index(0)
    original=trajectory.fingerprint(vars(pristine));receipts=[];ledger={}
    def count(kind,complete):ledger.setdefault(kind,[0,0])[int(complete)]+=1
    for arm in m.ARMS:
        rows=[]
        z,receipt=b.run_trajectory(SimpleNamespace(transformer=Transformer()),latent(),copy.deepcopy(pristine),
            torch.tensor([.03]),torch.tensor([-.02]),torch.bfloat16,arm,'watermark',m.message_bits('OKOK'),count,rows.append)
        receipts.append(receipt);assert [r['cursor_after'] for r in rows]==list(range(1,51))
    assert receipts[0]['before_step49']==receipts[1]['before_step49']==receipts[2]['before_step49']
    assert trajectory.fingerprint(vars(pristine))==original
    assert ledger['native_step']==[150,150] and ledger['local_gradient']==[2,2]
    assert ledger['transformer_conditional']==ledger['transformer_unconditional']==[150,150]


class FakeVAE:
    def __init__(self,fail_call=None):
        self.p=torch.nn.Parameter(torch.zeros(()),requires_grad=False);self.calls=0;self.mode_calls=0;self.clears=0;self.fail_call=fail_call
        self.config=SimpleNamespace(latents_mean=[.1*i for i in range(16)],latents_std=[.5+.01*i for i in range(16)])
    def parameters(self):yield self.p
    def clear_cache(self):self.clears+=1
    def encode(self,video):
        self.calls+=1
        if self.calls==self.fail_call:raise RuntimeError('injected phase VAE failure')
        T=(video.shape[2]-1)//4+1
        mean=torch.tensor(self.config.latents_mean).reshape(1,16,1,1,1)
        def mode():self.mode_calls+=1;return latent(T)+mean
        def sample():raise AssertionError('posterior sample forbidden')
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=mode,sample=sample))


@pytest.fixture
def actual_mp4(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):pytest.skip('FFmpeg unavailable')
    path=tmp_path/'unknown.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=gray:s=512x320:r=8','-frames:v','129',
        '-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(path)],check=True,capture_output=True)
    return path


def test_actual_mp4_rename_blindness_and_partial_phase(actual_mp4,tmp_path):
    keys=dict(CORRECT='watermark',WRONG='watermark-wrong')
    first=b.read_mp4_search(actual_mp4,keys,m.PUBLIC,FakeVAE())
    renamed=tmp_path/'OFF_FULL_FALSE_NAME.mp4';shutil.copyfile(actual_mp4,renamed)
    second=b.read_mp4_search(renamed,keys,m.PUBLIC,FakeVAE())
    assert first['searches']==second['searches'] and first['candidates']==second['candidates']
    assert all(row['status']=='NO_PILOT' for row in first['searches'].values())
    assert all(v['count']==930 for row in first['features'].values() for v in row['votes'])
    store=run.Store(tmp_path/'partial',create=True);vae=FakeVAE(fail_call=2)
    result=b.read_mp4_search(actual_mp4,keys,m.PUBLIC,vae,persist=lambda kind,slot,row:store.event('OFF/CROP5_129',kind,slot,row))
    assert vae.calls==4 and vae.mode_calls==3 and vae.clears==8
    assert all(row['status']=='SEARCH_INCOMPLETE' for row in result['searches'].values())
    assert all(len(rows)==53 for rows in result['candidates'].values())
    assert len([r for r in result['features'].values() if r['status']=='READ'])==6
    run.recover_unfinished(store,'generation','fixture');run.recover_unfinished(store,'media','fixture')
    before=(store.output/'blind_readouts.json').read_bytes();run.evaluate_saved_reads(store,run.load_config())
    assert (store.output/'blind_readouts.json').read_bytes()==before
    assert all(r['status']!='PENDING' for group in run.TABLE_SIZES for r in store.data[group].values())
    assert tuple(inspect.signature(b.read_mp4_search).parameters)==('path','keys','public','frozen_vae','count','persist')


def test_media_partial_save_failure_keeps_roster_and_normal_rejections(tmp_path,monkeypatch):
    from runtime.wan import generation,io,vae
    store=run.Store(tmp_path/'media',create=True);cfg=run.load_config()
    for row in store.data['generation'].values():
        path=Path(row['terminal_path']);path.parent.mkdir();torch.save(latent(),path)
        row.update(status='COMPLETE',sha256=b.file_sha256(path))
    monkeypatch.setattr(generation,'load_frozen_vae',lambda *a,**k:FakeVAE())
    monkeypatch.setattr(vae,'decode_normalized_latent',lambda *a:torch.zeros(1,1,1,3).expand(181,320,512,3))
    def encode(rgb,path,*args):
        if path.parent.name=='PAYLOAD_LAST' and path.stem=='CROP5_129':raise RuntimeError('partial MP4 save failure')
        path.write_bytes(b'fixture MP4')
    monkeypatch.setattr(io,'encode_rgb',encode)
    monkeypatch.setattr(io,'read_mp4',lambda path:torch.zeros(1,1,1,3).expand(129 if path.stem=='CROP5_129' else 181,320,512,3))
    monkeypatch.setattr(vae,'reencode_rgb24_readback',lambda vae,rgb:latent((len(rgb)-1)//4+1))
    run.media_worker(store,cfg);before=(store.output/'blind_readouts.json').read_bytes()
    run.evaluate_saved_reads(store,cfg)
    assert store.data['counts']['source_mp4']==3 and store.data['counts']['derived_mp4']==5
    assert store.data['counts']['normalized']==11 and store.data['counts']['phase_reads']==22
    assert store.data['counts']['searches']==10 and store.data['counts']['evaluated']==20
    assert store.data['views']['PAYLOAD_LAST/CROP5_129']['status']=='FAILED'
    for row in store.data['evaluations'].values():
        if row['status']=='EVALUATED':
            assert row['normal_rejection'] and row['accepted_payload'] is False and row['bit_errors'] is None
    assert {r['pilot_negative_category'] for r in store.data['evaluations'].values()}=={None,'OFF_UNWATERMARKED','PAYLOAD_ONLY','PILOT_WRONG_KEY'}
    assert (store.output/'blind_readouts.json').read_bytes()==before
    assert all(r['status']!='PENDING' for group in run.TABLE_SIZES for r in store.data[group].values())


def test_parent_spawn_failure_finalizes_all_rows(tmp_path,monkeypatch):
    output=tmp_path/'run';monkeypatch.setattr(run.sys,'argv',['review','--output',str(output)])
    monkeypatch.setattr(run.subprocess,'check_output',lambda *a,**k:'91afbc71aee9f999ff6f7c1738c5ac0be1249a52\n')
    def spawn(*a,**k):raise OSError('spawn failure fixture')
    monkeypatch.setattr(run.subprocess,'Popen',spawn)
    with pytest.raises(SystemExit) as error:run.main()
    assert error.value.code==1
    result=json.loads((output/'result.json').read_text())
    assert result['status']=='INCOMPLETE' and len(result['failures'])==2
    assert all(r['status']!='PENDING' for group in run.TABLE_SIZES for r in result[group].values())
    assert len(json.loads((output/'blind_readouts.json').read_text())['candidates'])==324


def test_parent_monitor_cleanup_preserves_child_checkpoint(tmp_path,monkeypatch):
    output=tmp_path/'run';monkeypatch.setattr(run.sys,'argv',['review','--output',str(output)])
    monkeypatch.setattr(run.subprocess,'check_output',lambda *a,**k:'91afbc71aee9f999ff6f7c1738c5ac0be1249a52\n')
    events=[]
    class Stream:
        def __iter__(self):raise OSError('monitor failure')
        def close(self):events.append('close')
    class Child:
        stdout=Stream()
        def poll(self):return None
        def terminate(self):events.append('terminate')
        def wait(self,timeout=None):
            events.append('wait');s=run.Store(output);s.data['generation']['OFF'].update(status='COMPLETE',fixture='during_shutdown');s.save();return -15
    def spawn(command,**kwargs):
        if command[-1]=='generation':return Child()
        assert events==['terminate','wait','close'];raise OSError('media spawn failure')
    monkeypatch.setattr(run.subprocess,'Popen',spawn)
    with pytest.raises(SystemExit):run.main()
    result=json.loads((output/'result.json').read_text())
    assert result['generation']['OFF']['fixture']=='during_shutdown' and result['counts']['generated']==1


def test_normal_rejections_can_finish_execution(tmp_path,monkeypatch):
    output=tmp_path/'run';monkeypatch.setattr(run.sys,'argv',['review','--output',str(output)])
    def phase(output,name):
        s=run.Store(output)
        if name=='generation':
            for r in s.data['generation'].values():r['status']='COMPLETE'
        else:
            for group in ('sources','views','normalized'):
                for r in s.data[group].values():r['status']='SAVED'
            for r in s.data['phase_reads'].values():r.update(status='READ',decoded_bits=[0]*32)
            for r in s.data['candidates'].values():r.update(status='NO_ENERGY',score=0.)
            for r in s.data['searches'].values():r.update(status='NO_PILOT',decoded_bits=None)
        s.data['workers'][name]=dict(status='COMPLETE');s.save();s.blind();return s,None
    monkeypatch.setattr(run,'run_worker_phase',phase)
    monkeypatch.setattr(run,'quality_diagnostics',lambda *a:None)
    run.main();result=json.loads((output/'result.json').read_text())
    assert result['status']=='EXECUTION_COMPLETE' and result['counts']['evaluated']==24
    assert all(not r['accepted_payload'] and r['bit_errors'] is None for r in result['evaluations'].values())


def test_notebook_static_fixed_rows_and_unpublished_receipt(tmp_path):
    import ast
    from scripts.build_video_temporal_sync_bridge_notebook import build
    path=build(output=tmp_path/'draft.ipynb');nb=json.loads(path.read_text())
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    sources=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    requirements=[]
    for source in sources:
        tree=ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node,ast.List):continue
            for i,item in enumerate(node.elts):
                if isinstance(item,ast.Constant) and item.value=='-r':
                    arg=node.elts[i+1]
                    assert isinstance(arg,ast.Call) and isinstance(arg.func,ast.Name) and arg.func.id=='str'
                    path=arg.args[0]
                    assert isinstance(path,ast.BinOp) and isinstance(path.op,ast.Div)
                    assert isinstance(path.left,ast.Name) and path.left.id=='REPO' and isinstance(path.right,ast.Constant)
                    requirements.append(path.right.value)
    assert requirements and all((run.ROOT/relative).is_file() for relative in requirements)
    joined='\n'.join(sources)
    assert 'ensurepip' not in joined and 'sys.version_info' not in joined and 'sys.executable' in joined
    assert 'Video(str(' in joined and 'embed=True' in joined
    setup=sources[1].replace("Path('/content/drive/MyDrive/Video-WM/Video-Temporal-Sync-Bridge-V1')",f'Path({str(tmp_path)!r})')
    with pytest.raises(RuntimeError,match='Unpublished draft'):exec(compile(setup,'setup','exec'),{})
    receipt=json.loads(next(tmp_path.glob('*/setup_receipt.json')).read_text())
    assert {name:len(receipt[name]) for name in run.TABLE_SIZES}==run.TABLE_SIZES


def test_terminal_diagnostics_and_batched_search_hard_exit(tmp_path,monkeypatch):
    from runtime.wan import io,vae
    store=run.Store(tmp_path/'run',create=True);cfg=run.load_config()
    z=latent();target,_=m.build_target(z,cfg['key'],m.message_bits('OKOK'),'PILOT_LAST')
    controlled=torch.fft.ifft2(target,norm='ortho').real
    for arm in m.ARMS:run.terminal_diagnostics(store,arm,controlled if arm=='PILOT_LAST' else z,cfg)
    assert store.data['diagnostic_counts']['terminal_reads']==6
    assert store.data['terminal_diagnostics']['PILOT_LAST/CORRECT']['decoded_bits']==m.message_bits('OKOK')
    assert store.data['terminal_diagnostics']['PILOT_LAST/CORRECT']['pilot']['score']==pytest.approx(1)
    assert all(v['count']==1350 for v in store.data['terminal_diagnostics']['PILOT_LAST/CORRECT']['votes'])
    assert store.data['counts']['phase_reads']==0
    path=tmp_path/'input.mp4';path.write_bytes(b'fixture')
    monkeypatch.setattr(io,'read_mp4',lambda _:torch.zeros(1,1,1,3).expand(129,320,512,3))
    monkeypatch.setattr(vae,'reencode_rgb24_readback',lambda _,rgb:latent((len(rgb)-1)//4+1))
    def persist(kind,slot,row):
        store.event('PILOT_LAST/CROP5_129',kind,slot,row)
        if kind=='candidate' and slot==('CORRECT',3):raise SystemExit('hard exit inside uncommitted score batch')
    with pytest.raises(SystemExit):b.read_mp4_search(path,dict(CORRECT=cfg['key'],WRONG=cfg['wrong_key']),m.PUBLIC,object(),persist=persist)
    reloaded=run.Store(store.output)
    assert all(r['status']=='PENDING' for name,r in reloaded.data['candidates'].items() if name.startswith('PILOT_LAST/CROP5_129'))
    assert sum(r['status']=='READ' for r in reloaded.data['phase_reads'].values())==8
    run.recover_unfinished(reloaded,'media','hard exit')
    assert reloaded.data['searches']['PILOT_LAST/CROP5_129/CORRECT']['status']=='SEARCH_INCOMPLETE'
    assert reloaded.data['diagnostic_counts']['terminal_reads']==6


def test_crop_soft_template_localizes_finite_offset_without_payload():
    known=m.candidates(129)[5];features={g:{'pilot_fft':np.zeros((31,64)).tolist()} for g in range(4)}
    features[3]['pilot_fft']=template('watermark',known).tolist()
    rows=[m.score_candidate(features[row['g']],'watermark',row) for row in m.candidates(129)]
    result=m.decide(rows,129)
    assert result['status']=='LOCATED' and result['selected_b']==5 and result['selected_g']==3
    assert result['ties']==[5] and result['top_gap']>0


@pytest.mark.parametrize('boundary',('before_canonical','after_canonical'))
def test_phase_checkpoint_boundary_uses_canonical_result(tmp_path,monkeypatch,boundary):
    from runtime.wan import io,vae
    store=run.Store(tmp_path/'run',create=True);slot='OFF/FULL_RESAVED181/0/CORRECT'
    original_save=store.save;original_blind=store.blind
    def interrupted_save():
        if boundary=='before_canonical' and store.data['phase_reads'][slot]['status']=='READ':
            raise SystemExit('before canonical atomic commit')
        original_save()
    def interrupted_blind():
        if boundary=='after_canonical' and store.data['phase_reads'][slot]['status']=='READ':
            raise SystemExit('after canonical commit before blind projection')
        original_blind()
    monkeypatch.setattr(store,'save',interrupted_save);monkeypatch.setattr(store,'blind',interrupted_blind)
    path=tmp_path/'observation.mp4';path.write_bytes(b'fixture MP4')
    monkeypatch.setattr(io,'read_mp4',lambda _:torch.zeros(1,1,1,3).expand(181,320,512,3))
    monkeypatch.setattr(vae,'reencode_rgb24_readback',lambda *a:latent())
    with pytest.raises(SystemExit):
        b.read_mp4_search(path,dict(CORRECT='watermark',WRONG='watermark-wrong'),m.PUBLIC,object(),
            persist=lambda kind,sid,row:store.event('OFF/FULL_RESAVED181',kind,sid,row))
    canonical=json.loads((store.output/'result.json').read_text())
    projection=json.loads((store.output/'blind_readouts.json').read_text())
    assert projection['phase_reads'][slot]['status']=='PENDING'
    expected='READ' if boundary=='after_canonical' else 'PENDING'
    assert canonical['phase_reads'][slot]['status']==expected
    if boundary=='after_canonical':assert canonical['phase_reads'][slot]['decoded_bits']==[0]*32
    else:assert canonical['phase_reads'][slot]['decoded_bits'] is None
    recovered=run.Store(store.output);run.recover_unfinished(recovered,'media','checkpoint interruption')
    canonical=json.loads((store.output/'result.json').read_text())
    projection=json.loads((store.output/'blind_readouts.json').read_text())
    assert canonical['phase_reads']==projection['phase_reads']
    row=canonical['phase_reads'][slot]
    assert row['status']==('READ' if boundary=='after_canonical' else 'NOT_COMPLETED')
    assert row['decoded_bits']==([0]*32 if boundary=='after_canonical' else None)
    assert len(canonical['phase_reads'])==30 and len(canonical['candidates'])==324 and len(canonical['searches'])==12
    assert all(x['status']!='PENDING' for group in ('phase_reads','candidates','searches') for x in canonical[group].values())
