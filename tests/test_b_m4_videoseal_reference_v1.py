"""CPU/synthetic contract checks only: never call a real model or media tool."""
import ast,json,math,os,shutil,subprocess,sys,types,zipfile
from pathlib import Path
from collections import Counter
import pytest
from runtime.wan import b_m4_videoseal_reference_v1 as run
from runtime.wan import b_m4_videoseal_reference_storage_v1 as storage
from scripts import build_b_m4_videoseal_reference_v1_notebook as builder

class Pixels:
    def __init__(self,n,stage):self.n=n;self.stage=stage
    def __len__(self):return self.n

class Fake:
    def __init__(self,out,fail=0):self.out=out;self.calls=Counter();self.fail=fail;self.owner=[]
    def count(self,k):self.calls[k]+=1
    def input_info(self,source,cfg):return {'status':'FAKE_INPUT'}
    def load(self,cfg,event):self.count('load');self.event=event;return {'status':'FAKE_LOADED'}
    def raw(self,source,host):return Pixels(181,'raw')
    def embed(self,pixels,bits):
        assert pixels.stage=='raw';self.count('embed');self.owner.append(bits)
        for n in (32,14):self.event('embedder',False,n);self.event('embedder',True,n)
        for n in (128,53):self.event('JND',False,n);self.event('JND',True,n)
        return Pixels(181,'float')
    def quantize(self,pixels):assert pixels.stage=='float';self.count('quantize');return Pixels(181,'quantized')
    def encode(self,pixels,path):
        assert pixels.stage in ('quantized','edited');self.count('encode');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fake-media')
        return {'status':'FAKE_ENCODED'}
    def probe(self,path,n):self.count('probe');return {'frames':n}
    def decode(self,path,probe,n):self.count('decode');assert probe['frames']==n;return Pixels(n,'readback')
    def edit(self,pixels,indices):
        assert pixels.stage=='readback' and len(pixels)==181;self.count('edit');assert len(indices)==151;return Pixels(151,'edited')
    def extract(self,pixels):
        assert pixels.stage=='readback';self.count('extract')
        assert not (self.out/'post_truth_access.json').exists()
        if self.calls['extract']==self.fail:raise RuntimeError('controlled extract failure')
        for n in range(0,len(pixels),32):self.event('extractor',False,min(32,len(pixels)-n));self.event('extractor',True,min(32,len(pixels)-n))
        bits=[0]*256
        return dict(run.empty_extract('RETURNED'),hard_bits=bits,native_logits=[[0]*257],mean_bit_logits=[0]*256,slots=[dict(bit=i,value=0,status='READ') for i in range(256)])
    def quality(self,a,b):
        self.count('quality');assert len(a)==len(b)
        return dict(status='RETURNED',frames=len(a),adjacent=len(a)-1,unread_frames=0,unread_adjacent=0,frame_rows=[{}]*len(a),adjacent_rows=[{}]*(len(a)-1))
    def release(self):self.count('release');return {'status':'RELEASED'}


def saved_b_fixture():
    readers=('CDC_M','RANK3','RANK3R');tasks={};totals={};detection={}
    fields=('emitted','literal_errors','actual_supported_correct','false_attribution','null_outputs','technical_unread','conflict_slots')
    for h in run.HOSTS:
        for arm in ('OFF','P0','PM'):
            for fixture in run.FIXTURES:
                for key in ('key00','key01'):
                    positive=arm=='PM' and key=='key00'
                    tasks[f'{h}/{arm}/{fixture}/{key}']=dict(positive=positive,readers={reader:dict(slots=[dict(slot=i) for i in range(32)],counts=dict(planned=32,**{f:0 for f in fields}),detection_status='REJECT_GEOMETRY') for reader in readers})
    for reader in readers:
        for kind,n in [('positive',20),('null',100),('wrong_key_PM',20)]:
            totals[reader+'/'+kind]=dict(planned=n*32,**{f:0 for f in fields})
            detection[reader+'/'+kind]=dict(planned=n,statuses={'REJECT_GEOMETRY':n})
    return dict(status='POST_RECORDED',method_status='METHOD_NOT_SUPPORTED',contribution_status='CONTRIBUTION_NOT_MET',tasks=tasks,totals=totals,detection=detection,
        denominators=dict(sources=4,readers=3,tasks_per_reader=120,slots_per_reader=3840,positive_tasks=20,positive_slots=640,null_tasks=100,null_slots=3200),
        quality={f'{h}/quality/full/OFF_to_PM/readback':dict(status='SCORED',frame_denominator=181,adjacent_denominator=180,frames=[{}]*181,adjacent=[{}]*180,whole={'rms':.01},adjacent_residual={'rms':.01}) for h in run.HOSTS})


def source(tmp_path):
    path=tmp_path/'input';path.mkdir()
    run.dump(path/'protocol.json',{'config':{'model_id':'Wan', 'generation':{'frames':181,'height':320,'width':512,'fps':8},'sources':[dict(id=h,a=a,role='eval') for h,a in zip(run.HOSTS,run.STARTS)],'writer':{'key':'watermark','gamma':1,'bits':[1]*32}}})
    run.dump(path/'evaluation.json',saved_b_fixture())
    return path


def test_fixed_config_and_real_public_protocol_interface(tmp_path):
    cfg=run.load_config();assert len(run.tasks())==48 and len(run.qualities())==28
    assert sum(q['frames'] for q in run.qualities())==4468
    assert sum(q['adjacent'] for q in run.qualities())==4440
    assert len(set(tuple(m['canonical_bits']) for m in cfg['messages']))==1
    assert sum(cfg['messages'][0]['canonical_bits'])==132
    value=run.VideoSealBackend().input_info(source(tmp_path),cfg)
    assert value['original_protocol_payload_read'] is False and len(value['sources'])==4
    assert len(storage.MEDIA)==24 and all('/VideoSeal/' in p for p in storage.MEDIA)


def test_native_kwargs_and_float32_mean_only():
    torch=pytest.importorskip('torch')
    seen=[]
    class Model:
        device='cpu'
        def embed(self,x,**kw):seen.append(('embed',kw));return {'imgs_w':x}
        def detect(self,x,**kw):seen.append(('detect',kw));return {'preds':logits}
        def extract_message(self,*args,**kwargs):raise AssertionError('second inference forbidden')
    logits=torch.zeros((3,257,1,1),dtype=torch.float32);logits[:,0]=99
    logits[:,1,0,0]=torch.tensor([1e8,1,-1e8],dtype=torch.float32)
    logits[:,2,0,0]=torch.tensor([1e-8,1e-8,1e-8]);logits[:,3]=-1
    b=run.VideoSealBackend();b.model=Model();b.tensor=lambda rgb:torch.zeros((len(rgb),3,1,1),dtype=torch.float32)
    b.embed([0]*3,[1]*256);value=b.extract([0]*3)
    assert seen[0][1]['interpolation']==run.EMBED_INTERPOLATION
    assert seen[0][1]['lowres_attenuation'] is False and seen[0][1]['is_video'] is True
    assert seen[1][1]['interpolation']==run.EXTRACT_INTERPOLATION
    expected=logits[:,1:].mean(dim=0).reshape(256)
    assert value['mean_bit_logits']==expected.tolist() and value['hard_bits']==(expected>0).long().tolist()
    assert value['hard_bits'][:3]==[0,1,0] and value['mask_column_used_for_message'] is False
    with pytest.raises(ValueError):run.native_prediction(logits.double(),3)


def test_rint_quantization_and_quality():
    torch=pytest.importorskip('torch');np=pytest.importorskip('numpy')
    # Broadcast creates a single small storage; the implementation produces the real fixed-size uint8 result.
    values=torch.tensor([0.5/255,1.5/255,2.5/255],dtype=torch.float32).reshape(1,3,1,1).expand(181,3,320,512)
    result=run.quantize(values)
    assert result.dtype==np.uint8 and result.shape==(181,320,512,3) and result[0,0,0].tolist()==[0,2,2]
    left=np.zeros((3,1,1,3),dtype=np.uint8);right=left.copy();right[1:]=1
    q=run.quality_pair(left,right)
    assert q['frame_rows'][0]['psnr_status']=='POSITIVE_INFINITY' and q['frame_rows'][0]['psnr'] is None
    assert math.isclose(q['whole']['rms'],math.sqrt(2/3)/255)
    assert math.isclose(q['temporal_residual_rms'],math.sqrt(1/2)/255)


def test_full_fake_budget_and_blind_order(tmp_path,monkeypatch):
    src=source(tmp_path);out=tmp_path/'output';fake=Fake(out)
    original=run.evaluate
    def post(output,source):
        assert run.read(output/'blind_complete.json')['tasks']==48
        assert all(run.read(output/'extractions'/(t['id']+'.json'))['status']=='RETURNED' for t in run.tasks())
        return original(output,source)
    monkeypatch.setattr(run,'evaluate',post)
    result=run.run(src,out,backend=fake)
    assert result['status']=='COMPLETED_FIXED_ATTEMPT',result
    assert fake.calls==dict(load=1,embed=4,quantize=4,encode=24,probe=48,decode=48,edit=20,extract=48,quality=28,release=1)
    assert all(r['planned']==r['started']==r['returned'] and r['failed']==0 for r in result['physical_counts'].values())
    assert {k:v['returned'] for k,v in result['observed_internal'].items()}==run.INTERNAL_BUDGET
    assert result['observed_internal']['embedder']['frames_returned']==184
    assert result['observed_internal']['JND']['frames_returned']==724
    assert result['observed_internal']['extractor']['frames_returned']==7488
    assert all(v==fake.owner[0] for v in fake.owner)
    ev=run.read(out/'evaluation.json');assert len(ev['tasks'])==48 and sum(v['planned_bits'] for v in ev['tasks'].values())==12288
    assert ev['B_reference']['summary']['method_status']=='METHOD_NOT_SUPPORTED'
    assert len(ev['B_reference']['quality_full_OFF_to_PM_readback'])==4
    states={name:storage._core_state(name,run.read(out/name)) for name in storage.CORE_JSON}
    assert storage._final_state_issues(states,states)==[]
    old=dict(fake.calls);assert run.run(src,out,backend=fake)['status']=='READ_ONLY_EXISTING_ATTEMPT';assert fake.calls==old


def test_first_failure_keeps_all_rows_and_no_retry(tmp_path):
    src=source(tmp_path);out=tmp_path/'failed';fake=Fake(out,fail=3)
    result=run.run(src,out,backend=fake)
    assert result['status']=='TECHNICAL_UNRESOLVED' and fake.calls['extract']==3 and fake.calls['embed']==1
    ev=run.read(out/'evaluation.json');counts=Counter(t['status'] for t in ev['tasks'].values())
    assert counts==dict(RETURNED=2,FAILED=1,NOT_ENTERED=45)
    assert sum(t['unread'] for t in ev['tasks'].values())==46*256
    quality=run.read(out/'planned_outputs.json')['quality']
    assert sum(q['frames'] for q in quality.values())==4468
    assert sum(q['unread_frames'] for q in quality.values())==4468-362
    old=dict(fake.calls);run.run(src,out,backend=fake);assert fake.calls==old


def test_existing_receipt_and_reserved_handoff(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(run,'_execute_once',lambda *a,**k:calls.append(a) or {'status':'FRESH'})
    out=tmp_path/'out';out.mkdir();run.dump(out/'repeat_start.json',{'status':'STARTING_FIXED_REPEAT'})
    assert run.run('input',out)['status']=='READ_ONLY_EXISTING_ATTEMPT' and calls==[]
    assert run._start_reserved('input',out)['status']=='FRESH' and len(calls)==1
    for name in ('physical_events.json','progress.json.tmp'):
        run.dump(out/name,{})
        assert run._start_reserved('input',out)['status']=='READ_ONLY_EXISTING_ATTEMPT' and len(calls)==1
    (out/'receipts').mkdir();unit=next(iter(run.plan_units()))
    run.dump(out/'receipts/0.json',dict(slot=unit,kind=run.plan_units()[unit],status='STARTED',attempts=1))
    (out/'receipts/1.json').write_text('{broken')
    recovered=run.recover_ledger(out)
    assert recovered['ledger'][unit]['status']=='STARTED' and len(recovered['ledger'])==len(run.plan_units())


def test_notebook_builder_sync_and_readonly_skip(tmp_path,monkeypatch):
    result=builder.build();nb=run.read(result['notebook']);before=Path(result['notebook']).read_bytes();builder.build();assert before==Path(result['notebook']).read_bytes()
    nbformat=pytest.importorskip('nbformat');nbformat.validate(nb)
    codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for code in codes:ast.parse(code)
    assert all(c['execution_count'] is None and c['outputs']==[] for c in nb['cells'] if c['cell_type']=='code')
    namespace={'CAN_START':False}
    exec(codes[2],namespace);exec(codes[3],namespace)
    assert 'worker' not in namespace and 'VENDOR' not in namespace
    assert "import videoseal.utils.cfg" in codes[3] and 'extract_message(' not in codes[3]
    assert "cwd=VENDOR" in codes[3] and "'/content/video-wm-m4/setup.log'" in codes[3]
    # Preflight/helpers work with no project runtime import.
    prefix=codes[1].split('CLOUD_CLIENT=None')[0];scope={};exec(prefix,scope)
    assert len(scope['plan_units']())==len(run.plan_units())
    out=tmp_path/'old';out.mkdir();run.dump(out/'attempt.json',{'status':'STARTED'})
    units=run.plan_units();slot=next(iter(units))
    run.dump(out/'receipts/000.json',dict(slot=slot,kind=units[slot],status='RETURNED',attempts=1))
    recovered=scope['recover_ledger'](out)
    assert recovered['physical_reentry'] is False and recovered['ledger'][slot]['status']=='RETURNED'


def test_notebook_nested_import_probe_runs_from_no_git_source(tmp_path):
    nb=run.read(builder.build()['notebook'])
    code=next(''.join(c['source']) for c in nb['cells'] if c.get('id')=='m4-code-4')
    tree=ast.parse(code)
    probe=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='probe' for t in n.targets))
    # Execute the actual nested -c argument; parsing only the cell missed this failure.
    root=tmp_path/'source';vendor=tmp_path/'vendor';vendor.mkdir()
    for relative in builder.SOURCE_FILES:
        destination=root/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(builder.ROOT/relative,destination)
    environment=os.environ.copy();environment['PYTHONPATH']=str(root)
    result=subprocess.run([sys.executable,'-c',probe,str(root)],cwd=vendor,env=environment,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    paths=[Path(line).resolve() for line in result.stdout.splitlines()]
    assert len(paths)==3 and all(p.is_file() and p.is_relative_to(root) for p in paths)
    assert not (root/'.git').exists() and list(vendor.iterdir())==[]


def test_cloud_start_and_failed_local_media_are_not_complete(tmp_path,monkeypatch):
    class Client:
        def quota(self):return {'usage':'1'}
        def children(self,parents):return []
    state=storage.preflight(Client(),tmp_path/'new');assert state['safe_to_start']
    out=tmp_path/'old';out.mkdir();run.dump(out/'repeat_start.json',{})
    assert not storage.preflight(Client(),out)['safe_to_start']
    media=out/storage.MEDIA[0];media.parent.mkdir(parents=True);media.write_bytes(b'')
    report=storage.confirm_and_backup(None,out,tmp_path/'backups')
    assert report['backup']['status']=='INCOMPLETE_BACKUP' and report['physical_calls']==0
    assert any('empty mounted MP4' in str(e) for e in report['backup']['errors'])


def test_model_load_failure_is_spent_without_any_media(tmp_path):
    class LoadFailure(Fake):
        def load(self,cfg,event):self.count('load');raise RuntimeError('controlled model load failure')
    src=source(tmp_path);out=tmp_path/'load-failed';fake=LoadFailure(out)
    result=run.run(src,out,backend=fake)
    assert result['status']=='TECHNICAL_UNRESOLVED' and fake.calls==dict(load=1,release=1)
    assert result['physical_counts']['model_load']['failed']==1
    outputs=run.read(out/'planned_outputs.json')
    assert all(t['status']=='NOT_ENTERED' and len(t['slots'])==256 for t in outputs['tasks'].values())
    assert sum(q['unread_frames'] for q in outputs['quality'].values())==4468
    assert run.read(out/'blind_complete.json')['tasks']==48
    assert run.read(out/'evaluation.json')['summary']['VideoSeal/all']['unread']==6144
    run.run(src,out,backend=fake);assert fake.calls==dict(load=1,release=1)


def test_load_hooks_observe_network_and_separate_jnd(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');import sys
    class Attenuation:
        def heatmaps(self,x):return x
    class Model(torch.nn.Module):
        device=torch.device('cpu')
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.embedder=torch.nn.Identity();self.detector=torch.nn.Identity()
            self.attenuation=Attenuation();self.blender=types.SimpleNamespace(scaling_w=.2,scaling_i=1);self.chunk_size=32;self.step_size=4;self.video_mode='repeat';self.clamp=True
    vendor=tmp_path/'vendor';(vendor/'videoseal/utils').mkdir(parents=True);(vendor/'videoseal/utils/cfg.py').write_text('# synthetic cfg')
    monkeypatch.setattr(run,'VENDOR',vendor);monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    calls=[];model=Model()
    monkeypatch.setitem(sys.modules,'videoseal',types.SimpleNamespace(load=lambda card:calls.append(card) or model))
    events=[];backend=run.VideoSealBackend();original=Path.cwd()
    try:
        record=backend.load({},lambda *args:events.append(args))
        model.embedder(torch.zeros(32,1));model.detector(torch.zeros(21,1));model.attenuation.heatmaps(torch.zeros(128,1))
        assert calls==['videoseal_1.0'] and record['dtype']=='torch.float32' and Path.cwd()==vendor
        assert events==[('embedder',False,32),('embedder',True,32),('extractor',False,21),('extractor',True,21),('JND',False,128),('JND',True,128)]
    finally:backend.release()
    assert Path.cwd()==original


def test_recovery_pending_numeric_receipt_and_exact_attempt_pair(tmp_path):
    units=run.plan_units();names=list(units);out=tmp_path/'pending';out.mkdir()
    ledger={slot:dict(slot=slot,kind=kind,status='NOT_ENTERED',attempts=0) for slot,kind in units.items()}
    ledger[names[0]]['attempts']=1;ledger[names[1]].update(status='RETURNED',attempts=0)
    run.dump(out/'ledger.json',ledger)
    run.dump(out/'receipts/002.json.tmp',dict(slot=names[2],kind=units[names[2]],status='STARTED',attempts=1))
    (out/'receipts/003.json.tmp').write_text('{broken')
    run.dump(out/'receipts/004.json',dict(slot=names[4],kind=units[names[4]],status='STARTED',attempts=0))
    recovered=run.recover_ledger(out)
    assert [recovered['ledger'][n]['status'] for n in names[:5]]==['UNRESOLVED_LEDGER']*2+['UNRESOLVED_RECEIPT']*3
    assert all(recovered['ledger'][n]['attempts']==1 for n in names[:5])
    assert recovered['ledger'][names[5]]['status']=='NOT_ENTERED'
    assert run.run('unused',out)['status']=='READ_ONLY_EXISTING_ATTEMPT'


def test_config_actual_method_fields_not_provenance(tmp_path):
    import copy
    cfg=run.load_config();changed=copy.deepcopy(cfg)
    changed['model'].update(checkpoint_location='/different/usable/location',aggregation='different descriptive text',message_scope='optional record')
    path=tmp_path/'config.json';run.dump(path,changed);assert run.load_config(path)['model']['checkpoint_location']=='/different/usable/location'
    for key,value in [('scaling_w',.3),('chunk_size',16),('inference_mode',False),('autocast',True),('attenuation',None)]:
        changed=copy.deepcopy(cfg);changed['model'][key]=value;run.dump(path,changed)
        with pytest.raises(ValueError):run.load_config(path)


def test_actual_loaded_parameter_mismatch_is_recorded(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');import sys
    class Model(torch.nn.Module):
        device=torch.device('cpu')
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.blender=types.SimpleNamespace(scaling_w=.3,scaling_i=1)
            self.chunk_size=32;self.step_size=4;self.video_mode='repeat';self.clamp=True;self.attenuation=object()
    vendor=tmp_path/'vendor';(vendor/'videoseal/utils').mkdir(parents=True);(vendor/'videoseal/utils/cfg.py').write_text('# stub')
    monkeypatch.setattr(run,'VENDOR',vendor);monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setitem(sys.modules,'videoseal',types.SimpleNamespace(load=lambda card:Model()))
    out=tmp_path/'mismatch';result=run.run(source(tmp_path),out)
    assert result['status']=='TECHNICAL_UNRESOLVED'
    observed=run.read(out/'model.json');assert observed['observed']['mismatches']['scaling_w']==dict(expected=.2,actual=.3)
    assert result['physical_counts']['model_load']['failed']==1 and result['physical_counts']['embed']['started']==0


def test_B_reference_requires_real_structure_not_expected_negative_strings(tmp_path):
    src=tmp_path/'B';src.mkdir();run.dump(src/'evaluation.json',{})
    assert run.B_reference(src)['status']=='B_REFERENCE_UNRESOLVED'
    value=saved_b_fixture();value['method_status']='OTHER_RECORDED_OUTCOME';value['contribution_status']='OTHER_RECORDED_CONTRIBUTION'
    run.dump(src/'evaluation.json',value);record=run.B_reference(src)
    assert record['status']=='SAVED_ORIGINAL_M3' and record['summary']['method_status']=='OTHER_RECORDED_OUTCOME'
    assert 'negative_results_preserved' not in record
    out=tmp_path/'evaluation-only';run.dump(out/'protocol.json',{'config':run.load_config()})
    run.dump(out/'blind_complete.json',dict(status='READ_BACK',tasks=48))
    for task in run.tasks():run.dump(out/'extractions'/(task['id']+'.json'),run.empty_extract())
    for name,row in run.planned_outputs()['quality'].items():run.dump(out/'quality'/(name+'.json'),row)
    evaluated=run.evaluate(out,src)
    assert evaluated['original_M3_status_fields_preserved'] is True
    assert evaluated['B_reference']['summary']['method_status']=='OTHER_RECORDED_OUTCOME'
    assert 'original_M3_negative_conclusions_unchanged' not in evaluated
    value['tasks'][next(iter(value['tasks']))]['readers']['CDC_M']['slots'].pop();run.dump(src/'evaluation.json',value)
    assert run.B_reference(src)['status']=='B_REFERENCE_UNRESOLVED'
    assert run.evaluate(out,src)['original_M3_status_fields_preserved'] is False
    value=saved_b_fixture();value['quality'].pop(next(iter(value['quality'])));run.dump(src/'evaluation.json',value)
    assert run.B_reference(src)['status']=='B_REFERENCE_UNRESOLVED'
