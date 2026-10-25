"""Only API/filesystem stubs. No authentication, HTTP, models or codec execution."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile
import pytest
from runtime.wan import b_rank3_repeat_storage_v1 as storage
from scripts import build_b_rank3_four_source_repeat_20261025_v1_notebook as builder

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
    counts={'native50_trajectory':8,'native_step':400,'DiT_forward':800,'payload_gradient':100,'state_construct':4,'D181':12,'codec_encode':48,'codec_probe':48,'codec_decode':48,'E41':432}
    values={n:{} for n in storage.CORE_JSON}
    values['result.json']=dict(status='COMPLETED_FIXED_ATTEMPT')
    values['evaluation.json']=dict(status='POST_RECORDED',method_status='STAGE2_METHOD_CRITERIA_MET',contribution_status='DEMONSTRATED_FIXED_SCOPE',tasks={str(i):{} for i in range(72)},quality={str(i):{} for i in range(76)})
    values['plan.json']=dict(units={'input':'input','post':'post'})
    values['ledger.json']={k:dict(status='RETURNED',attempts=1) for k in ('input','post')}
    values['planned_outputs.json']=dict(tasks={str(i):dict(status='RETURNED',readers={r:dict(slots=[{}]*32) for r in ('CDC_M','UNION_M','RANK3','RANK2','FIXED_ORIGIN')}) for i in range(72)},quality={str(i):dict(status='RETURNED') for i in range(76)})
    values['physical_counts.json']={k:dict(planned=n,started=n,returned=n,failed=0) for k,n in counts.items()}
    values['plans_complete.json']=dict(status='READ_BACK',task_plans=72,reader_plans=360)
    values['blind_complete.json']=dict(status='READ_BACK',task_decisions=72,reader_decisions=360)
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


def test_streaming_adapter_full_length_and_http_error(tmp_path):
    class Response:
        def __init__(self,fail=False):self.closed=False;self.fail=fail
        def raise_for_status(self):
            if self.fail:raise IOError('HTTP 403')
        def iter_content(self,chunk_size):
            assert chunk_size==1024*1024
            yield b'ab';yield b'cd'
        @property
        def content(self):raise AssertionError('video must not materialize in memory')
        def close(self):self.closed=True
    class Session:
        def request(self,method,url,**kw):
            assert method=='GET' and kw['params']=={'alt':'media'} and kw['stream'] is True
            return response
    response=Response();client=storage.DriveClient(Session())
    assert client.read_to(dict(id='video',size='4'),tmp_path/'video')==4 and response.closed
    response=Response()
    with pytest.raises(IOError,match='length'):client.read_to(dict(id='video',size='8'),tmp_path/'truncated')
    assert response.closed
    response=Response(True)
    with pytest.raises(IOError,match='403'):client.read_to(dict(id='video'),tmp_path/'denied')
    assert response.closed


def test_complete_backup_and_cloud_core_only(tmp_path):
    client=FakeDrive();out=tmp_path/'output';populate(client,out)
    report=storage.confirm_and_backup(client,out,tmp_path/'backups')
    assert report['cloud_confirmation']['status']=='CORE_RESULTS_AND_48_MEDIA_CLOUD_READABLE'
    assert len(client.reads)==63  # No auxiliary receipt/Q/bit content GET.
    assert report['backup']['status']=='COMPLETE_BACKUP_OF_LISTED_JSON_AND_48_MEDIA'
    assert report['backup']['json_count']==16 and report['backup']['media_count']==48
    with zipfile.ZipFile(report['backup_zip']) as archive:
        assert 'results/receipts/test.json' in archive.namelist()
        internal=json.loads(archive.read('backup_report.json'))
        assert internal['backup_zip']==report['backup_zip']
    assert report['computation']['cloud_evaluation']['method_status']=='STAGE2_METHOD_CRITERIA_MET'
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
    assert report['backup']['status']=='INCOMPLETE_BACKUP' and report['backup']['media_count']==47
    assert 'receipts/cloud_only.json' in report['backup']['missing']
    assert len(client.reads)==63
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
    code=cells[1].replace("OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Rank3-Four-Source-Confirmation-V1/repeat-20261025-v1')",'OUTPUT = Path('+repr(str(tmp_path/'invisible-mount'))+')').replace('CLOUD_CLIENT=authenticate_drive()','CLOUD_CLIENT=INJECTED_CLIENT')
    env={'INJECTED_CLIENT':client};exec(code,env)
    assert not env['CAN_START'] and env['PREFLIGHT']['cloud_artifacts']
    exec(cells[2],env);exec(cells[3],env);exec(cells[4],env)
    assert env['SOURCE_ROOT'] is None and client.creates==0
    assert not (tmp_path/'invisible-mount').exists()
    before=path.read_bytes();builder.build();assert path.read_bytes()==before


def test_saved_final_core_state_schema():
    source=Path('/home/richar/projects/Video-WM/diagnostics/b-rank3-stage2-result-20261025/saved')
    if not all((source/n).is_file() for n in storage.CORE_JSON):pytest.skip('saved completed-core fixture unavailable')
    states={}
    for name in storage.CORE_JSON:
        with (source/name).open(encoding='utf-8') as stream:states[name]=storage._core_state(name,storage._json(stream))
    assert storage._final_state_issues(states,states)==[]
    intermediate=deepcopy(states);intermediate['ledger.json']['returned']-=1
    assert any('ledger' in e for e in storage._final_state_issues(intermediate,states))


def test_failed_final_attempt_cloud_progress_must_catch_up(tmp_path):
    client=FakeDrive();out=tmp_path/'output';populate(client,out)
    final=fixed_core();final['result.json']['status']='TECHNICAL_UNRESOLVED'
    final['evaluation.json'].update(status='POST_PARTIAL',method_status='TECHNICAL_UNRESOLVED')
    final['evaluation.json']['quality']={str(i):dict(status='SCORED' if i<75 else 'FAILED') for i in range(76)}
    final['ledger.json']['post']['status']='FAILED'
    final['planned_outputs.json']['quality']['75']['status']='FAILED'
    stale=deepcopy(final);stale['ledger.json']['post']['status']='STARTED'
    stale['evaluation.json']['quality']['75']['status']='NOT_ENTERED'
    stale['planned_outputs.json']['quality']['75']['status']='NOT_ENTERED'
    for name,value in final.items():(out/name).write_text(json.dumps(value))
    def set_cloud(values):
        for ident,node in client.nodes.items():
            if node['parents']==[client.parent] and node['name'] in values:
                client.bodies[ident]=json.dumps(values[node['name']]).encode();node['size']=str(len(client.bodies[ident]))
    set_cloud(stale)
    first=storage.confirm_and_backup(client,out,tmp_path/'backups')
    assert first['cloud_confirmation']['status']=='CORE_OR_MEDIA_CLOUD_UNCONFIRMED'
    assert any('behind' in s for s in first['cloud_confirmation']['final_state_issues'])
    set_cloud(final)
    second=storage.confirm_and_backup(client,out,tmp_path/'backups')
    assert second['cloud_confirmation']['status']=='CORE_RESULTS_AND_48_MEDIA_CLOUD_READABLE'
    assert second['computation']['cloud_result']['status']=='TECHNICAL_UNRESOLVED'
    assert second['method_status_modified'] is False
