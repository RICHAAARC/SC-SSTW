"""M3 storage/entry adaptation checks; API and media are stubs only."""
from copy import deepcopy
import ast,json,subprocess,sys,urllib.request,zipfile
from pathlib import Path
import pytest
from runtime.wan import b_rank3r_new_source_confirmation_storage_v1 as storage
from runtime.wan import b_rank3r_new_source_confirmation_v1 as run
from scripts import build_b_rank3r_new_source_confirmation_v1_notebook as builder
pytestmark=pytest.mark.unit

class FakeDrive:
    def __init__(self):
        self.nodes={};self.bodies={};self.reads=[];self.creates=0;self.fail=set();self.unknown_create=False
        self.parent='root'
        for name in storage.DRIVE_COMPONENTS:self.parent=self.add(self.parent,name,folder=True)
    def add(self,parent,name,body=b'',folder=False):
        ident='id'+str(len(self.nodes))
        self.nodes[ident]=dict(id=ident,name=name,parents=[parent],mimeType=storage.FOLDER_MIME if folder else 'application/octet-stream')
        if not folder:self.nodes[ident]['size']=str(len(body));self.bodies[ident]=body
        return ident
    def put(self,name,value):
        parent=self.parent;parts=name.split('/')
        for part in parts[:-1]:
            matches=[v for v in self.nodes.values() if v['parents']==[parent] and v['name']==part]
            parent=matches[0]['id'] if matches else self.add(parent,part,folder=True)
        body=value if isinstance(value,bytes) else json.dumps(value).encode()
        return self.add(parent,parts[-1],body)
    def children(self,parents):return [deepcopy(v) for v in self.nodes.values() if v['parents'][0] in parents]
    def quota(self):return dict(limit='10',usage='100')  # Even negative remaining is record-only, never an invented gate.
    def read_to(self,meta,path):
        self.reads.append(meta['id'])
        if meta['id'] in self.fail:raise IOError('synthetic cloud read interruption')
        Path(path).write_bytes(self.bodies[meta['id']]);return len(self.bodies[meta['id']])
    def create_start(self,parent,payload):
        self.creates+=1;ident=self.add(parent,'repeat_start.json',json.dumps(payload).encode())
        if self.unknown_create:raise IOError('response lost after create')
        return self.nodes[ident]

def fixed_core():
    counts=dict(run.BUDGET)
    values={n:{} for n in storage.CORE_JSON}
    values['result.json']=dict(status='COMPLETED_FIXED_ATTEMPT')
    values['evaluation.json']=dict(status='POST_RECORDED',method_status='M3_METHOD_CRITERIA_MET',contribution_status='DEMONSTRATED_FIXED_SCOPE',tasks={str(i):{} for i in range(120)},quality={str(i):{} for i in range(108)})
    values['plan.json']=dict(units={'input':'input','post':'post'})
    values['ledger.json']={k:dict(status='RETURNED',attempts=1) for k in ('input','post')}
    values['planned_outputs.json']=dict(tasks={str(i):dict(status='RETURNED',readers={r:dict(slots=[{}]*32) for r in ('RANK3R','RANK3','CDC_M')}) for i in range(120)},quality={str(i):dict(status='RETURNED') for i in range(108)})
    values['physical_counts.json']={k:dict(planned=n,started=n,returned=n,failed=0) for k,n in counts.items()}
    values['plans_complete.json']=dict(status='READ_BACK',task_plans=120,reader_plans=360)
    values['blind_complete.json']=dict(status='READ_BACK',task_decisions=120,reader_decisions=360)
    values['attempt.json']=dict(status='STARTED',attempts=1)
    values['progress.json']=dict(active='post',counts={'RETURNED':3313,'STARTED':1})
    return values

def populate(client,output):
    output.mkdir()
    for name,value in fixed_core().items():
        client.put(name,value);(output/name).write_text(json.dumps(value))
    for name in storage.MEDIA:
        data=b'fake-full-mp4-data';client.put(name,data)
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    path=output/'receipts/test.json';path.parent.mkdir();path.write_text('{"status":"RETURNED"}');client.put('receipts/test.json',dict(status='RETURNED'))

def test_cloud_guard_hidden_attempt_and_one_uncertain_create(tmp_path):
    client=FakeDrive();out=tmp_path/'new';out.mkdir();(out/'setup.log').write_text('setup only')
    state=storage.preflight(client,out)
    assert state['safe_to_start'] and state['quota']['usage']=='100'
    client.unknown_create=True
    result=storage.reserve_start(client,out,sleep=lambda _:None)
    assert result['status']=='START_UNKNOWN_NO_WORKER' and client.creates==1
    assert (out/'repeat_start_unknown.json').is_file() and not (out/'attempt.json').exists()
    hidden=tmp_path/'hidden'
    assert storage.preflight(client,hidden)['status']=='EXISTING_ATTEMPT'
    assert storage.reserve_start(client,hidden,sleep=lambda _:None)['status']=='NOT_STARTED'
    assert client.creates==1
    fresh=FakeDrive();ready=tmp_path/'ready';ready.mkdir()
    assert storage.reserve_start(fresh,ready,sleep=lambda _:None)['status']=='START_RECEIPT_CLOUD_READ_BACK'
    assert storage.reserve_start(fresh,ready,sleep=lambda _:None)['status']=='NOT_STARTED' and fresh.creates==1
    duplicate=FakeDrive();duplicate.add(duplicate.nodes[duplicate.parent]['parents'][0],storage.REPEAT_NAME,folder=True)
    assert not storage.preflight(duplicate,hidden)['safe_to_start']

def test_complete_backup_and_cloud_core_only(tmp_path):
    client=FakeDrive();out=tmp_path/'output';populate(client,out)
    report=storage.confirm_and_backup(client,out,tmp_path/'backups')
    assert report['cloud_confirmation']['status']=='CORE_RESULTS_AND_72_MEDIA_CLOUD_READABLE'
    assert len(client.reads)==87  # No auxiliary receipt/Q/bit content GET; absent optional archives are listed.
    assert report['backup']['status']=='INCOMPLETE_BACKUP'  # Missing future-M4 archives do not invalidate core saving.
    assert report['optional_OFF_archives']['method_required'] is False
    assert len(report['optional_OFF_archives']['missing'])==8
    assert report['optional_OFF_archives']['readable_RGB24']==0
    assert report['backup']['json_count']==16 and report['backup']['media_count']==72
    with zipfile.ZipFile(report['backup_zip']) as archive:
        assert 'results/receipts/test.json' in archive.namelist()
        internal=json.loads(archive.read('backup_report.json'))
        assert internal['backup_zip']==report['backup_zip']
    assert report['computation']['cloud_evaluation']['method_status']=='M3_METHOD_CRITERIA_MET'
    assert report['method_status_modified'] is False

def test_cloud_intermediate_state_prefers_local_final_and_reports_gaps(tmp_path):
    client=FakeDrive();out=tmp_path/'output';populate(client,out)
    ev=next(i for i,v in client.nodes.items() if v['parents']==[client.parent] and v['name']=='evaluation.json')
    value=deepcopy(fixed_core()['evaluation.json']);value['status']='POST_PARTIAL';value['method_status']='TECHNICAL_UNRESOLVED'
    client.bodies[ev]=json.dumps(value).encode();client.nodes[ev]['size']=str(len(client.bodies[ev]))
    client.put('receipts/cloud_only.json',dict(status='STARTED'))
    (out/'receipts/broken.json').write_text('{broken')
    first=storage.MEDIA[0];video=next(i for i,v in client.nodes.items() if i in client.bodies and v['name']=='video.mp4')
    client.fail.add(video);(out/first).write_bytes(b'')
    report=storage.confirm_and_backup(client,out,tmp_path/'backups')
    assert report['cloud_confirmation']['status']=='CORE_OR_MEDIA_CLOUD_UNCONFIRMED'
    assert report['cloud_confirmation']['final_state_issues']
    assert report['backup']['status']=='INCOMPLETE_BACKUP' and report['backup']['media_count']==71
    assert 'receipts/cloud_only.json' in report['backup']['missing']
    assert len(client.reads)==87
    with zipfile.ZipFile(report['backup_zip']) as archive:
        assert json.loads(archive.read('results/evaluation.json'))['status']=='POST_RECORDED'
        assert report['backup']['files']['evaluation.json']['cloud_path_readable'] is True
        assert report['backup']['files']['evaluation.json']['backed_up_content_from_cloud'] is False
        assert 'unparsed_local/receipts/broken.json' in archive.namelist()
    assert (out/'evaluation.json').read_text()==json.dumps(fixed_core()['evaluation.json'])

def test_notebook_existing_cloud_attempt_skips_source_setup_worker(tmp_path,monkeypatch):
    import nbformat
    path=builder.ROOT/'notebooks'/f'{builder.STEM}.ipynb'
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    cells=[''.join(c.source) for c in nb.cells if c.cell_type=='code']
    assert len(cells)==6 and cells[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    client=FakeDrive();client.put('attempt.json',dict(status='STARTED',attempts=1))
    import types
    google=types.ModuleType('google');google.__path__=[]
    auth=types.ModuleType('google.auth');auth.__path__=[]
    transport=types.ModuleType('google.auth.transport');transport.__path__=[]
    request_module=types.ModuleType('google.auth.transport.requests');request_module.AuthorizedSession=object
    google.auth=auth;auth.transport=transport;transport.requests=request_module
    for name,module in [('google',google),('google.auth',auth),('google.auth.transport',transport),('google.auth.transport.requests',request_module)]:monkeypatch.setitem(sys.modules,name,module)
    def forbidden(*args,**kwargs):raise AssertionError('existing cloud attempt reached download/setup/worker')
    monkeypatch.setattr(urllib.request,'urlretrieve',forbidden)
    monkeypatch.setattr(subprocess,'run',forbidden);monkeypatch.setattr(subprocess,'Popen',forbidden)
    code=cells[1].replace("OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Rank3R-New-Source-Confirmation-V1/fixed-four-five-v1')",'OUTPUT = Path('+repr(str(tmp_path/'invisible-mount'))+')').replace('CLOUD_CLIENT=authenticate_drive()','CLOUD_CLIENT=INJECTED_CLIENT')
    env={'INJECTED_CLIENT':client};exec(code,env)
    assert not env['CAN_START'] and env['PREFLIGHT']['cloud_artifacts']
    exec(cells[2],env);exec(cells[3],env);exec(cells[4],env)
    assert env['SOURCE_ROOT'] is None and client.creates==0
    assert not (tmp_path/'invisible-mount').exists()
    before=path.read_bytes();builder.build();assert path.read_bytes()==before

def test_runtime_orphan_receipt_vs_fresh_handoff(tmp_path,monkeypatch):
    out=tmp_path/'out';run.dump(out/'repeat_start.json',dict(status='STARTING_FIXED_REPEAT'))
    called=[]
    def fresh(*args,**kwargs):called.append(True);return dict(status='FAKE_FIRST_ENTRY')
    monkeypatch.setattr(run,'_execute_once',fresh)
    assert run.run('unused',out)['status']=='READ_ONLY_EXISTING_ATTEMPT' and not called
    assert run._start_reserved('unused',out)['status']=='FAKE_FIRST_ENTRY' and called==[True]
    run.dump(out/'attempt.json',dict(status='STARTED',attempts=1));called.clear()
    assert run._start_reserved('unused',out)['status']=='READ_ONLY_EXISTING_ATTEMPT' and not called
    missing=tmp_path/'newtrace';run.dump(missing/'source_media/h/arm/probe.json',dict(status='STARTED'))
    assert run.run('unused',missing)['status']=='READ_ONLY_EXISTING_ATTEMPT' and not called

def test_notebook_format_source_and_handoff():
    import nbformat
    path=builder.ROOT/'notebooks'/f'{builder.STEM}.ipynb';nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    cells=[''.join(c.source) for c in nb.cells if c.cell_type=='code']
    for cell in cells:ast.parse(cell)
    assert len({c.id for c in nb.cells})==12 and len(cells)==6
    assert cells[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert 'b_rank3r_new_source_confirmation_v1_run.py' in cells[2] and 'SOURCE_OVERLAY' not in cells[2]
    assert cells[3].index('START_RECEIPT_CLOUD_READ_BACK')<cells[3].index('_notebook_reserved_main')
    assert 'FqnToConfig' in cells[3] and 'WanPipeline,AutoencoderKLWan' in cells[3]
    assert 'B-Rank3-Four-Source-Confirmation-V1/repeat-20261025-v1' in cells[1]
    assert 'CORE_RESULTS_AND_72_MEDIA_CLOUD_READABLE' in cells[1]
    before=path.read_bytes();builder.build();assert path.read_bytes()==before


@pytest.mark.parametrize('name',['physical_events.json','progress.json','physical_events.json.tmp','progress.json.tmp'])
def test_orphan_events_and_progress_are_spent_for_both_entries(tmp_path,monkeypatch,name):
    out=tmp_path/'orphan';out.mkdir();(out/name).write_text('{}')
    def forbidden(*args,**kwargs):raise AssertionError('orphan trace reached a fresh worker/backend')
    monkeypatch.setattr(run,'_execute_once',forbidden)
    before={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    assert run.run('unused',out,backend=object())['status']=='READ_ONLY_EXISTING_ATTEMPT'
    assert run._start_reserved('unused',out,backend=object())['status']=='READ_ONLY_EXISTING_ATTEMPT'
    assert before=={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}


def test_OFF_archive_actual_bytes_and_failure_are_record_only(tmp_path):
    import numpy as np
    class Tensor:
        def __init__(self):
            self.value=np.zeros((181,320,512,3),dtype=np.uint8)
            self.value[0,0,0]=[1,2,3];self.value[-1,-1,-1]=[253,254,255]
        def detach(self):return self
        def cpu(self):return self
        def contiguous(self):return self
        def numpy(self):return self.value
    class Backend:
        def validate_pixels(self,rgb,frames):
            assert frames==181
            return rgb
    host=run.SOURCES[0]['id'];record=run.archive_off_rgb24(Backend(),Tensor(),tmp_path/'valid',host)
    assert record['status']=='ARCHIVED' and record['method_required'] is False
    path=tmp_path/'valid'/record['path'];assert path.stat().st_size==88965120
    with path.open('rb') as stream:
        assert stream.read(3)==bytes([1,2,3]);stream.seek(-3,2);assert stream.read()==bytes([253,254,255])
    sidecar=json.loads((tmp_path/'valid'/record['sidecar']).read_text())
    assert sidecar['actual_bytes']==88965120 and sidecar['fps']==8
    bad=tmp_path/'blocked';bad.write_bytes(b'not a directory')
    failure=run.archive_off_rgb24(Backend(),Tensor(),bad,host)
    assert failure['status']=='ARCHIVE_MISSING' and failure['retry'] is False and failure['method_required'] is False
    assert failure['error'] and failure['sidecar_error']
