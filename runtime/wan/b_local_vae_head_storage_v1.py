"""Ordinary Drive storage and readable once-attempt receipt; no model work."""
from pathlib import Path
import json
import tempfile
import time

COMPONENTS=('Video-WM','B-Local-VAE-Head-V1','fixed-six-source-v1')
FOLDER='application/vnd.google-apps.folder'
API='https://www.googleapis.com/drive/v3/files'

def quoted(value):return "'"+str(value).replace('\\','\\\\').replace("'","\\'")+"'"

class Drive:
    def __init__(self,session):self.session=session
    def request(self,method,url,**kwargs):
        response=self.session.request(method,url,timeout=(20,180),**kwargs)
        response.raise_for_status();return response
    def children(self,parent):
        result=[];token=None
        while True:
            params=dict(q='trashed = false and '+quoted(parent)+' in parents',fields='nextPageToken,incompleteSearch,files(id,name,mimeType,size)',pageSize=1000)
            if token:params['pageToken']=token
            with self.request('GET',API,params=params) as r:value=r.json()
            if value.get('incompleteSearch'):raise IOError('incomplete Drive directory listing')
            result+=value.get('files',[]);token=value.get('nextPageToken')
            if not token:return result
    def metadata(self,ident):
        with self.request('GET',API+'/'+ident,params={'fields':'id,name,mimeType,size'}) as r:return r.json()
    def folder(self,components=COMPONENTS,create=False):
        parent='root'
        for name in components:
            found=[m for m in self.children(parent) if m['name']==name]
            if len(found)>1:raise ValueError('ambiguous Drive folder '+name)
            if found:
                if found[0]['mimeType']!=FOLDER:raise ValueError('expected Drive folder '+name)
                parent=found[0]['id']
            elif create:
                with self.request('POST',API,json=dict(name=name,mimeType=FOLDER,parents=[parent]),params={'fields':'id'}) as r:parent=r.json()['id']
            else:return None
        return parent
    def read_to(self,meta,destination):
        path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True);n=0
        with self.request('GET',API+'/'+meta['id'],params={'alt':'media'},headers={'Accept-Encoding':'identity'},stream=True) as r:
            with path.open('wb') as stream:
                for chunk in r.iter_content(1024*1024):
                    if chunk:stream.write(chunk);n+=len(chunk)
        if n==0 or ('size' in meta and n!=int(meta['size'])):raise IOError('empty/incomplete actual Drive blob read')
        return n
    def upload(self,parent,path,existing_id=None,name=None):
        """One resumable upload, no automatic retry after an uncertain write."""
        path=Path(path);metadata={'name':name or path.name}
        if existing_id is None:metadata['parents']=[parent]
        url='https://www.googleapis.com/upload/drive/v3/files'+('/'+existing_id if existing_id else '')
        with self.request('PATCH' if existing_id else 'POST',url,params={'uploadType':'resumable','fields':'id,name,size'},
            json=metadata,headers={'X-Upload-Content-Length':str(path.stat().st_size),'X-Upload-Content-Type':'application/octet-stream'}) as r:
            location=r.headers['Location']
        with path.open('rb') as stream:
            with self.request('PUT',location,data=stream,headers={'Content-Length':str(path.stat().st_size),'Content-Type':'application/octet-stream'}) as r:return r.json()

def authenticate():
    from google.colab import auth
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    auth.authenticate_user()
    credentials,_=google.auth.default(scopes=['https://www.googleapis.com/auth/drive'])
    return Drive(AuthorizedSession(credentials))

def preflight(client,mounted):
    folder=client.folder();cloud=client.children(folder) if folder else []
    local=list(Path(mounted).iterdir()) if Path(mounted).is_dir() else []
    # Any actual result artifact consumes this candidate's once-attempt scope.
    existing=bool(cloud or local)
    return dict(status='EXISTING_ATTEMPT' if existing else 'NO_EXECUTION_OBSERVED',can_start=not existing,
        folder_id=folder,cloud_names=[m['name'] for m in cloud],mounted_names=[p.name for p in local])

def reserve(client,mounted):
    state=preflight(client,mounted)
    if not state['can_start']:return dict(status='NOT_STARTED',state=state)
    folder=client.folder(create=True)
    payload=dict(status='STARTED_ONE_FIXED_ATTEMPT',unix=time.time(),retry=False,atomic_mutex=False,
        purpose='local VAE head fixed training/CAL/DEV; interrupted attempt is not restarted')
    try:
        with tempfile.TemporaryDirectory(prefix='vae-start-') as tmp:
            path=Path(tmp)/'repeat_start.json';path.write_text(json.dumps(payload))
            item=client.upload(folder,path);client.read_to(item,path)
            if json.loads(path.read_text()).get('status')!='STARTED_ONE_FIXED_ATTEMPT':raise IOError('unreadable startup receipt')
        mounted=Path(mounted);mounted.mkdir(parents=True,exist_ok=True)
        (mounted/'repeat_start.json').write_text(json.dumps(payload))
        return dict(status='START_RECEIPT_CLOUD_READ_BACK',folder_id=folder,file_id=item['id'],atomic_mutex=False)
    except BaseException as e:
        mounted=Path(mounted);mounted.mkdir(parents=True,exist_ok=True)
        (mounted/'repeat_start_unknown.json').write_text(json.dumps(dict(error=f'{type(e).__name__}: {e}',retry=False)))
        raise

def download_training(client,cfg,scratch):
    result={}
    for row in cfg['training_sources']:
        target=Path(scratch)/'inputs'/(row['id']+'.mp4')
        meta=client.metadata(row['drive_file_id']);client.read_to(meta,target)
        result[row['id']]=str(target)
    return result

def sync_results(client,local,mounted,progress=print):
    """Publish saved artifacts then read actual blobs. Does not change method status."""
    local=Path(local);root=client.folder(create=True);folders={'':root};indices={};report=dict(uploaded=0,readable=0,media=0,errors=[],missing=[],method_status_modified=False,physical_calls=0)
    def parent_for(parts):
        parent=root;prefix=''
        for name in parts:
            prefix='/'.join(filter(None,(prefix,name)))
            if prefix not in folders:
                found=[m for m in client.children(parent) if m['name']==name]
                if len(found)>1:raise ValueError('ambiguous result subfolder '+prefix)
                if found:
                    if found[0]['mimeType']!=FOLDER:raise ValueError('result path is not folder '+prefix)
                    folders[prefix]=found[0]['id']
                else:
                    with client.request('POST',API,json=dict(name=name,mimeType=FOLDER,parents=[parent]),params={'fields':'id'}) as r:folders[prefix]=r.json()['id']
            parent=folders[prefix]
        return parent
    files=[p for p in local.rglob('*') if p.is_file() and not p.name.endswith('.tmp')]
    # Final core records are uploaded last, after observations/media/checkpoint.
    files.sort(key=lambda p:(p.name in ('result.json','evaluation.json'),str(p)))
    with tempfile.TemporaryDirectory(prefix='vae-readback-') as tmp:
        for number,path in enumerate(files,1):
            rel=path.relative_to(local)
            try:
                parent=parent_for(rel.parts[:-1])
                if parent not in indices:indices[parent]=client.children(parent)
                matches=[m for m in indices[parent] if m['name']==path.name]
                if len(matches)>1:raise ValueError('ambiguous existing result blob '+str(rel))
                item=client.upload(parent,path,matches[0]['id'] if matches else None)
                report['uploaded']+=1;target=Path(tmp)/'readback';client.read_to(item,target);report['readable']+=1
                if path.suffix=='.mp4':report['media']+=1
                if path.name in ('result.json','evaluation.json'):
                    report[path.name]=json.loads(target.read_text()) if path.name=='result.json' else {k:json.loads(target.read_text()).get(k) for k in ('status','method_status','recommendation')}
            except Exception as e:report['errors'].append(dict(file=str(rel),error=f'{type(e).__name__}: {e}'))
            if number%20==0 or number==len(files):progress(f'Cloud saved/read: {number}/{len(files)}')
    for required in ('result.json','evaluation.json','plan.json','attempt.json','physical_counts.json','quality.json'):
        if not (local/required).is_file():report['missing'].append(required)
    report['status']='SAVED_ARTIFACTS_CLOUD_READABLE' if not report['errors'] and not report['missing'] else 'CLOUD_SAVE_UNCONFIRMED'
    report['expected_full_budget_media']=72;report['fixed_media_missing']=72-report['media']
    (Path(mounted)/'cloud_save_report.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False))
    return report

def read_existing(client,mounted):
    folder=client.folder();result=dict(status='READ_ONLY_EXISTING_ATTEMPT',physical_calls=0,errors=[])
    if folder is None:
        result['status']='NO_CLOUD_ATTEMPT_RECORD'
        result['errors'].append(dict(file='output_folder',error='cloud output folder missing; preparation may not have started a scientific attempt'))
        result['fixed_recovery']=recovery_snapshot({})
        result['result.json']=dict(status='NO_SAVED_COMPUTATION',stop='cloud output folder missing; no completed computation claimed',
            denominators=result['fixed_recovery']['denominators'],physical_counts=None)
        result['evaluation.json']=dict(status='UNRESOLVED',method_status='NOT_EVALUATED',recommendation='RESOLVE_RECORDED_PREPARATION_ERROR',per_source={},failures=[])
        return result
    children=client.children(folder)
    with tempfile.TemporaryDirectory(prefix='vae-existing-') as tmp:
        for name in ('result.json','evaluation.json','physical_counts.json','task_state.json','plan.json','quality.json'):
            found=[m for m in children if m['name']==name]
            try:
                if len(found)!=1:raise FileNotFoundError('expected one readable '+name)
                path=Path(tmp)/name;client.read_to(found[0],path);value=json.loads(path.read_text())
                result[name]=value if name!='evaluation.json' else {k:value.get(k) for k in ('status','method_status','recommendation','CAL','DEV','summaries','per_source','failures','denominators')}
            except Exception as e:result['errors'].append(dict(file=name,error=f'{type(e).__name__}: {e}'))
    result['fixed_recovery']=recovery_snapshot(result)
    if 'result.json' not in result:
        result['result.json']=dict(status='UNRESOLVED_INTERRUPTED_ATTEMPT',stop='final record absent; no scientific operation repeated',
            denominators=result['fixed_recovery']['denominators'],physical_counts=result.get('physical_counts.json'))
    if 'evaluation.json' not in result:
        result['evaluation.json']=dict(status='UNRESOLVED',method_status='UNRESOLVED_INTERRUPTED_ATTEMPT',recommendation='NO_RETRY_THIS_ATTEMPT',
            per_source=result['fixed_recovery']['per_source'],failures=result['fixed_recovery']['tasks'],
            null_zero_criteria=None,literal_error_count=None)
    return result

def recovery_snapshot(values):
    """Summarize already saved execution states, without scoring, BCH, or truth reads."""
    state=values.get('task_state.json',{});tasks=state.get('tasks',[]);plan=values.get('plan.json',{})
    if not tasks and plan:
        for reader,key in (('STATE_SPACE','main'),('I','I')):
            tasks.extend(dict(row,reader=reader,plan_status='UNRESOLVED',read_status='UNRESOLVED',decision_status='UNRESOLVED') for row in plan.get(key,[]))
        originals={r['task']:r for r in plan.get('main',[])}
        for row in plan.get('post_OWU',[]):
            tasks.append(dict(originals.get(row['task'],row),reader=row['mode'],plan_status='POST_ONLY',read_status='UNRESOLVED',decision_status='UNRESOLVED'))
    per_source={}
    for row in tasks:
        key=row['reader']+'/'+row.get('source','UNKNOWN');summary=per_source.setdefault(key,dict(planned_tasks=0,returned_reads=0,not_started=0,unresolved=0,information_slots=0,known_outputs=0,known_erasures=0,literal_errors=None,null_zero_criteria=None))
        summary['planned_tasks']+=1;summary['information_slots']+=row.get('information_slots',32)
        status=row.get('read_status');summary['returned_reads']+=status=='RETURNED';summary['not_started']+=status=='NOT_STARTED'
        summary['unresolved']+=status not in ('NOT_STARTED','RETURNED')
        summary['known_outputs']+=row.get('decision_status')=='OUTPUT';summary['known_erasures']+=row.get('decision_status')=='ERASURE'
    denominators=state.get('denominators') or dict(main_tasks=150,main_information_slots=4800,I_tasks=150,I_information_slots=4800,post_OWU_tasks=90,post_OWU_information_slots=2880,quality_pairs=42,quality_frames=6702,quality_adjacent=6660)
    return dict(status='SAVED_EXECUTION_STATES_ONLY',denominators=denominators,tasks=tasks,per_source=per_source,
        quality=values.get('quality.json'),physical_counts=values.get('physical_counts.json'),
        missing_state_rows=max(390-len(tasks),0),unread_is_not_zero=True,scientific_calls_this_recovery=0,
        actual_errors_and_temporal_attribution='UNKNOWN without saved final evaluation')
