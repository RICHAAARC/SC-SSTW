"""Colab repeat storage: actual Drive API reads, separate from method evaluation.

No model, codec, production runtime or network work occurs at import. The
Notebook embeds this readable module for cloud guarding before source setup.
"""
import json
from pathlib import Path
import shutil
import tempfile
import time
import uuid
import zipfile

REPEAT_NAME='repeat-20261025-v1'
DRIVE_COMPONENTS=('Video-WM','B-Rank3-Four-Source-Confirmation-V1',REPEAT_NAME)
DEFAULT_OUTPUT='/content/drive/MyDrive/'+'/'.join(DRIVE_COMPONENTS)
CORE_JSON=('result.json','evaluation.json','ledger.json','planned_outputs.json','plan.json','protocol.json',
           'background.json','thresholds.json','physical_counts.json','physical_events.json',
           'plans_complete.json','blind_complete.json','post_truth_access.json','attempt.json','progress.json')
HOSTS=tuple('b-rank3-n%02d'%n for n in range(1,5))
MEDIA=tuple(f'media/{h}/{arm}/{scope}/video.mp4' for h in HOSTS for arm in ('OFF','P0','PM')
            for scope in ('full','CROP','DELETE_LEFT_GAP','DELETE_RIGHT_GAP'))
ARTIFACT_FOLDERS=('receipts','private','media','observations','families','blind','quality')
ROOT_ARTIFACTS=set(CORE_JSON)|{'inputs.json','storage.json','candidate_stop.json','repeat_start.json','repeat_start_unknown.json'}
FOLDER_MIME='application/vnd.google-apps.folder'


def _json(stream):
    def invalid(value):raise ValueError('non-JSON numeric constant: '+value)
    return json.load(stream,parse_constant=invalid)


def _name(name):
    if not isinstance(name,str) or not name or name in ('.','..') or any(c in name for c in ('/','\\',':','\x00')):
        raise ValueError('unsafe Drive child name')
    return name


def _quoted(value):
    return "'"+str(value).replace('\\','\\\\').replace("'","\\'")+"'"


class DriveClient:
    """Small Drive v3 adapter. Writes only one ordinary startup receipt."""
    def __init__(self,session):self.session=session
    def _metadata(self,method,url,**kwargs):
        response=self.session.request(method,url,timeout=(20,120),**kwargs)
        try:
            response.raise_for_status();return response.json()
        finally:response.close()
    def quota(self):
        return self._metadata('GET','https://www.googleapis.com/drive/v3/about',params={'fields':'storageQuota'}).get('storageQuota',{})
    def children(self,parents):
        query='trashed = false and ('+' or '.join(_quoted(p)+' in parents' for p in parents)+')'
        token=None;answer=[]
        while True:
            params=dict(q=query,spaces='drive',pageSize=1000,fields='nextPageToken,incompleteSearch,files(id,name,mimeType,size,parents)')
            if token:params['pageToken']=token
            value=self._metadata('GET','https://www.googleapis.com/drive/v3/files',params=params)
            if value.get('incompleteSearch'):raise RuntimeError('Drive returned an incomplete directory listing')
            answer.extend(value.get('files',[]));token=value.get('nextPageToken')
            if not token:return answer
    def read_to(self,meta,destination):
        """Read alt=media through EOF. No media decoding and no in-memory video."""
        response=self.session.request('GET','https://www.googleapis.com/drive/v3/files/'+meta['id'],
            params={'alt':'media'},headers={'Accept-Encoding':'identity'},stream=True,timeout=(20,120))
        count=0
        try:
            response.raise_for_status()
            with Path(destination).open('wb') as stream:
                for block in response.iter_content(chunk_size=1024*1024):
                    if block:stream.write(block);count+=len(block)
            if count==0:raise IOError('empty cloud blob')
            expected=meta.get('size')
            if expected is not None and count!=int(expected):raise IOError(f'cloud read length {count} differs from listed bytes {expected}')
            return count
        finally:response.close()
    def create_start(self,parent_id,payload):
        """Exactly one create request. Unknown responses are never retried here."""
        boundary='rank3-'+uuid.uuid4().hex
        metadata=json.dumps(dict(name='repeat_start.json',parents=[parent_id],mimeType='application/json')).encode()
        body=json.dumps(payload,ensure_ascii=False).encode()
        chunks=[b'--'+boundary.encode()+b'\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n'+metadata,
                b'--'+boundary.encode()+b'\r\nContent-Type: application/json\r\n\r\n'+body,
                b'--'+boundary.encode()+b'--\r\n']
        return self._metadata('POST','https://www.googleapis.com/upload/drive/v3/files',
            params={'uploadType':'multipart','fields':'id,name,mimeType,size,parents'},
            headers={'Content-Type':'multipart/related; boundary='+boundary},data=b'\r\n'.join(chunks))


def authenticate_drive():
    from google.colab import auth
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    auth.authenticate_user()
    credentials,_=google.auth.default(scopes=['https://www.googleapis.com/auth/drive'])
    return DriveClient(AuthorizedSession(credentials))


def resolve_folder(client,components=DRIVE_COMPONENTS):
    parent='root'
    for name in components:
        matches=[v for v in client.children([parent]) if v.get('name')==name]
        if not matches:return None
        if len(matches)!=1 or matches[0].get('mimeType')!=FOLDER_MIME:
            raise ValueError('ambiguous or non-folder Drive path component: '+name)
        parent=matches[0]['id']
    return parent


def local_artifacts(output):
    output=Path(output);found=[]
    for name in ROOT_ARTIFACTS:
        for suffix in ('','.tmp'):
            p=output/(name+suffix)
            if p.exists() or p.is_symlink():found.append(name+suffix)
    for name in ARTIFACT_FOLDERS:
        folder=output/name
        if folder.is_dir():found.extend(p.relative_to(output).as_posix() for p in folder.rglob('*') if p.is_file() or p.is_symlink())
    return sorted(found)


def cloud_artifacts(client,folder):
    if folder is None:return []
    pending=[(folder,'')];found=[]
    while pending:
        parent,prefix=pending.pop()
        for item in client.children([parent]):
            name=_name(item['name']);relative=prefix+name
            if not prefix:
                if name.removesuffix('.tmp') in ROOT_ARTIFACTS:found.append(relative)
                elif name in ARTIFACT_FOLDERS and item.get('mimeType')==FOLDER_MIME:pending.append((item['id'],name+'/'))
            elif item.get('mimeType')==FOLDER_MIME:pending.append((item['id'],relative+'/'))
            else:found.append(relative)
        if found:return found  # One actual execution trace is enough to prevent a second attempt.
    return found


def preflight(client,output,components=DRIVE_COMPONENTS):
    local=local_artifacts(output);value=dict(status='UNKNOWN',safe_to_start=False,local_artifacts=local,cloud_artifacts=[],folder_id=None,errors=[],quota=None)
    if client is None:
        value['errors'].append('Drive API authentication unavailable');return value
    try:value['quota']=client.quota()
    except Exception as error:value['quota_error']=f'{type(error).__name__}: {error}'
    try:
        value['folder_id']=resolve_folder(client,components)
        value['cloud_artifacts']=cloud_artifacts(client,value['folder_id'])
        value['status']='EXISTING_ATTEMPT' if local or value['cloud_artifacts'] else 'NO_EXECUTION_OBSERVED'
        value['safe_to_start']=value['status']=='NO_EXECUTION_OBSERVED'
    except Exception as error:value['errors'].append(f'{type(error).__name__}: {error}')
    return value


def reserve_start(client,output,components=DRIVE_COMPONENTS,*,attempts=6,pause=5,sleep=time.sleep):
    """Recheck after setup; only a visible, unique mount-created folder is used.

    A receipt is an ordinary once-attempt record, not atomic cross-session exclusion.
    Folder visibility reads may repeat; create and its response are never retried.
    """
    output=Path(output);folder=None
    for index in range(attempts):
        state=preflight(client,output,components)
        if not state['safe_to_start']:return dict(status='NOT_STARTED',preflight=state)
        folder=state['folder_id']
        if folder:break
        if index+1<attempts:sleep(pause)
    if not folder:return dict(status='NOT_STARTED',error='mount-created repeat folder is not yet visible via Drive API; no folder or receipt was created')
    payload=dict(status='STARTING_FIXED_REPEAT',repeat=REPEAT_NAME,output_components=list(components),created_unix=time.time(),
                 purpose='durable one-attempt startup receipt; same fixed samples and protocol',atomic_mutex=False)
    try:
        created=client.create_start(folder,payload)
        if not isinstance(created,dict) or not isinstance(created.get('id'),str):raise ValueError('create returned no usable receipt id')
        with tempfile.TemporaryDirectory(prefix='rank3-receipt-') as temporary:
            path=Path(temporary)/'receipt.json';client.read_to(created,path)
            with path.open(encoding='utf-8') as stream:readback=_json(stream)
        if not isinstance(readback,dict) or readback.get('status')!='STARTING_FIXED_REPEAT':raise ValueError('startup receipt is not a readable start record')
        return dict(status='START_RECEIPT_CLOUD_READ_BACK',file_id=created['id'],folder_id=folder,atomic_mutex=False)
    except Exception as error:
        record=dict(status='START_UNKNOWN_NO_WORKER',error=f'{type(error).__name__}: {error}',created_unix=time.time(),retry_create=False)
        # Preserve uncertainty locally if the write may have reached Drive. Never masquerade as runtime attempt.json.
        try:
            with (output/'repeat_start_unknown.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,ensure_ascii=False)
        except Exception as local_error:record['local_record_error']=f'{type(local_error).__name__}: {local_error}'
        return record


def inventory(client,folder):
    """Directory metadata only for supplementary JSON; no per-file content GET."""
    files={};pending={folder:''};visited=set()
    while pending:
        batch=dict(list(pending.items())[:40])
        for ident in batch:pending.pop(ident);visited.add(ident)
        for item in client.children(list(batch)):
            parents=[p for p in item.get('parents',[]) if p in batch]
            if len(parents)!=1:raise ValueError('Drive file has ambiguous listed parent')
            relative=batch[parents[0]]+_name(item['name'])
            if item.get('mimeType')==FOLDER_MIME:
                if item['id'] in visited or item['id'] in pending:raise ValueError('repeated directory id')
                pending[item['id']]=relative+'/'
            else:files.setdefault(relative,[]).append(item)
    return files


def _is_json(name):return name.endswith('.json') or name.endswith('.json.tmp')


def _load_summary(path):
    try:
        with Path(path).open(encoding='utf-8') as stream:value=_json(stream)
        if not isinstance(value,dict):return dict(read_error='root record must be an object')
        return {k:value.get(k) for k in ('status','method_status','contribution_status','stop') if k in value}
    except Exception as error:return dict(read_error=f'{type(error).__name__}: {error}')


def _core_state(name,value):
    """Existing final-state semantics only; no whole-file equality or digest."""
    if not isinstance(value,dict):raise ValueError('core JSON root must be an object')
    state={k:value.get(k) for k in ('status','method_status','contribution_status') if k in value}
    if name=='plan.json':state['units']=sorted(value.get('units',{}))
    elif name=='ledger.json':
        state.update(slots=sorted(value),returned=sum(isinstance(v,dict) and v.get('status')=='RETURNED' and v.get('attempts')==1 for v in value.values()),
            finished=sum(isinstance(v,dict) and v.get('status') in ('RETURNED','FAILED','INTERRUPTED') for v in value.values()),spent=sum(isinstance(v,dict) and v.get('attempts')==1 for v in value.values()))
    elif name=='planned_outputs.json':
        for key in ('tasks','quality'):
            rows=value.get(key,{})
            state[key+'_count']=len(rows);state[key+'_returned']=sum(isinstance(v,dict) and v.get('status')=='RETURNED' for v in rows.values())
            state[key+'_finished']=sum(isinstance(v,dict) and v.get('status') in ('RETURNED','FAILED','INTERRUPTED') for v in rows.values())
        state['reader_slots_valid']=all(set(t.get('readers',{}))=={'CDC_M','UNION_M','RANK3','RANK2','FIXED_ORIGIN'} and all(len(v.get('slots',[]))==32 for v in t['readers'].values()) for t in value.get('tasks',{}).values())
    elif name=='physical_counts.json':
        state['counts']={k:{n:v.get(n) for n in ('planned','started','returned','failed')} for k,v in value.items() if isinstance(v,dict)}
    elif name in ('plans_complete.json','blind_complete.json'):
        state.update({k:value.get(k) for k in ('task_plans','reader_plans','task_decisions','reader_decisions')})
    elif name=='evaluation.json':
        state['task_count']=len(value.get('tasks',{}));state['quality_count']=len(value.get('quality',{}))
        state['quality_finished']=sum(isinstance(v,dict) and v.get('status') in ('SCORED','RETURNED','FAILED','INTERRUPTED') for v in value.get('quality',{}).values())
    elif name=='physical_events.json':state['progress']=sum((v.get('started') or 0)+(v.get('returned') or 0) for v in value.values() if isinstance(v,dict))
    return state


def _state_score(name,state):
    if name=='result.json':return (state.get('status')=='COMPLETED_FIXED_ATTEMPT',0)
    if name=='evaluation.json':return (state.get('status')=='POST_RECORDED',state.get('task_count',0)+state.get('quality_finished',0))
    if name=='ledger.json':return (False,state.get('finished',0)*2+state.get('spent',0))
    if name=='planned_outputs.json':return (False,state.get('tasks_finished',0)+state.get('quality_finished',0))
    if name=='physical_counts.json':return (False,sum((v.get('started') or 0)+(v.get('returned') or 0)+(v.get('failed') or 0) for v in state.get('counts',{}).values()))
    if name=='physical_events.json':return (False,state.get('progress',0))
    if name=='plan.json':return (False,len(state.get('units',[])))
    return (state.get('status')=='READ_BACK',0)


def _final_state_issues(states,local_states):
    expected=local_states.get('result.json',{}).get('status')=='COMPLETED_FIXED_ATTEMPT' or states.get('result.json',{}).get('status')=='COMPLETED_FIXED_ATTEMPT'
    issues=[]
    if local_states.get('result.json',{}).get('status') in ('COMPLETED_FIXED_ATTEMPT','TECHNICAL_UNRESOLVED'):
        if states.get('result.json',{}).get('status')!=local_states['result.json']['status']:issues.append('cloud result differs from the mounted final computation state')
        for name in ('evaluation.json','ledger.json','planned_outputs.json','physical_counts.json','physical_events.json','plan.json','plans_complete.json','blind_complete.json'):
            if _state_score(name,local_states.get(name,{}))>_state_score(name,states.get(name,{})):issues.append('cloud progress is behind mounted final record: '+name)
    if not expected:return issues
    def require(condition,message):
        if not condition:issues.append(message)
    require(states.get('result.json',{}).get('status')=='COMPLETED_FIXED_ATTEMPT','cloud result is not the completed computation record')
    evaluation=states.get('evaluation.json',{})
    require(evaluation.get('status')=='POST_RECORDED' and evaluation.get('task_count')==72 and evaluation.get('quality_count')==76,'cloud evaluation is not final POST_RECORDED with 72 tasks and 76 quality records')
    for field in ('method_status','contribution_status'):
        local=local_states.get('evaluation.json',{})
        if local.get('status')=='POST_RECORDED' and field in local:require(local[field]==evaluation.get(field),'cloud/local final '+field+' differ')
    outputs=states.get('planned_outputs.json',{})
    require(outputs.get('tasks_count')==outputs.get('tasks_returned')==72 and outputs.get('quality_count')==outputs.get('quality_returned')==76 and outputs.get('reader_slots_valid') is True,'cloud planned_outputs still contains unfinished or absent fixed rows')
    units=states.get('plan.json',{}).get('units',[]);ledger=states.get('ledger.json',{})
    require(bool(units) and ledger.get('slots')==units and ledger.get('returned')==len(units),'cloud ledger is not returned for the saved fixed plan')
    expected_counts={'native50_trajectory':8,'native_step':400,'DiT_forward':800,'payload_gradient':100,'state_construct':4,'D181':12,'codec_encode':48,'codec_probe':48,'codec_decode':48,'E41':432}
    counts=states.get('physical_counts.json',{}).get('counts',{})
    require(set(counts)==set(expected_counts) and all(counts[k].get('planned')==counts[k].get('started')==counts[k].get('returned')==n and counts[k].get('failed')==0 for k,n in expected_counts.items()),'cloud physical_counts are not the completed fixed budget')
    plans=states.get('plans_complete.json',{});blind=states.get('blind_complete.json',{})
    require(plans.get('status')=='READ_BACK' and plans.get('task_plans')==72 and plans.get('reader_plans')==360,'cloud plans_complete is not final')
    require(blind.get('status')=='READ_BACK' and blind.get('task_decisions')==72 and blind.get('reader_decisions')==360,'cloud blind_complete is not final')
    # attempt STARTED and progress active=post are normal final files in the successful runner.
    return issues


def confirm_and_backup(client,output,backup_directory='/content',components=DRIVE_COMPONENTS,*,progress=None):
    """Read 15 core blobs and 48 media from cloud; back up all visible JSON + 48 MP4.

    Supplementary JSON content comes from the mounted result tree. Cloud-only
    supplementary JSON is listed as missing from backup, never fetched as a new
    per-file cloud verification gate. Missing data never invokes physical work.
    """
    output=Path(output);backup_directory=Path(backup_directory);backup_directory.mkdir(parents=True,exist_ok=True)
    report=dict(status='SAVING',computation=dict(mounted_result=_load_summary(output/'result.json'),mounted_evaluation=_load_summary(output/'evaluation.json')),
        cloud_confirmation=dict(status='UNCONFIRMED',required_core_json=len(CORE_JSON),required_media=len(MEDIA),readable_core_json=0,readable_media=0,
            files={},missing=[],errors=[],supplementary_json_content_verified=False),
        backup=dict(status='INCOMPLETE',scope='all JSON visible in mounted tree or cloud directory metadata, plus fixed 48 MP4',files={},missing=[],errors=[]),
        method_status_modified=False,physical_calls=0)
    cloud=report['cloud_confirmation'];backup=report['backup'];listing={};inventory_ok=False
    local_states={};cloud_states={};valid_local=set()
    if client is not None:
        try:
            folder=resolve_folder(client,components)
            if folder is None:cloud['errors'].append('repeat output folder missing in Drive API')
            else:listing=inventory(client,folder);inventory_ok=True
        except Exception as error:cloud['errors'].append(f'{type(error).__name__}: {error}')
    else:cloud['errors'].append('Drive API unavailable; mounted files cannot establish cloud saving')
    json_local={}
    try:
        if output.is_dir():
            for path in output.rglob('*'):
                if path.is_file() and _is_json(path.name):
                    if path.is_symlink():backup['errors'].append('symlink result file not followed: '+str(path));continue
                    json_local[path.relative_to(output).as_posix()]=path
    except Exception as error:backup['errors'].append(f'local JSON inventory: {type(error).__name__}: {error}')
    for name,path in json_local.items():
        if name in CORE_JSON:
            try:
                with path.open(encoding='utf-8') as stream:local_states[name]=_core_state(name,_json(stream))
                valid_local.add(name)
            except Exception as error:backup['errors'].append(dict(path=name,error='mounted core JSON: '+str(error)))
    expected_json=set(CORE_JSON)|set(json_local)|{name for name in listing if _is_json(name)}
    for name,entries in listing.items():
        if len(entries)>1 and (_is_json(name) or name in MEDIA):backup['errors'].append('ambiguous cloud path in backup inventory: '+name)
    cloud['supplementary_json_listed']=len([p for p in listing if _is_json(p) and p not in CORE_JSON])
    required=set(CORE_JSON)|set(MEDIA)
    stamp=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'-'+uuid.uuid4().hex[:8]
    prefix=backup_directory/('b-rank3-'+REPEAT_NAME+'-'+stamp)
    temporary_zip=Path(str(prefix)+'-incomplete.zip');report_path=Path(str(prefix)+'-backup-report.json')
    try:
        with tempfile.TemporaryDirectory(prefix='rank3-cloud-read-',dir=backup_directory) as directory:
            spool=Path(directory);cloud_paths={}
            for index,name in enumerate(sorted(required),1):
                candidates=listing.get(name,[])
                if len(candidates)!=1:
                    (cloud['missing'] if not candidates else cloud['errors']).append(name if not candidates else 'ambiguous cloud path: '+name)
                    continue
                meta=candidates[0];path=spool/str(index)
                try:
                    length=client.read_to(meta,path)
                    if name.endswith('.json'):
                        with path.open(encoding='utf-8') as stream:value=_json(stream)
                        cloud_states[name]=_core_state(name,value)
                        if name in ('result.json','evaluation.json'):report['computation']['cloud_'+name.removesuffix('.json')]={k:value.get(k) for k in ('status','method_status','contribution_status','stop') if k in value}
                        cloud['readable_core_json']+=1
                    else:cloud['readable_media']+=1
                    cloud_paths[name]=path;cloud['files'][name]=dict(status='CLOUD_READ_TO_EOF',bytes=length,file_id=meta['id'])
                except Exception as error:cloud['errors'].append(dict(path=name,error=f'{type(error).__name__}: {error}'))
                if progress and (index%8==0 or index==len(required)):progress(f'Cloud blob reads: {index}/{len(required)}')
            cloud['final_state_issues']=_final_state_issues(cloud_states,local_states)
            if cloud['readable_core_json']==len(CORE_JSON) and cloud['readable_media']==len(MEDIA) and not cloud['missing'] and not cloud['errors'] and not cloud['final_state_issues']:
                cloud['status']='CORE_RESULTS_AND_48_MEDIA_CLOUD_READABLE'
            else:cloud['status']='CORE_OR_MEDIA_CLOUD_UNCONFIRMED'
            with zipfile.ZipFile(temporary_zip,'w',allowZip64=True) as archive:
                for name in sorted(expected_json|set(MEDIA)):
                    local=json_local.get(name) if _is_json(name) else output/name
                    candidate=cloud_paths.get(name);origin='Drive API full read'
                    if _is_json(name) and local is not None and local.is_file():
                        try:
                            if name not in valid_local:
                                with local.open(encoding='utf-8') as stream:_json(stream)
                                valid_local.add(name)
                            if name not in cloud_paths or _state_score(name,local_states.get(name,{}))>=_state_score(name,cloud_states.get(name,{})):
                                candidate=local;origin='mounted readable JSON; selected by saved progress semantics, no whole-file equality'
                        except Exception as error:
                            backup['errors'].append(dict(path=name,error='mounted JSON cannot be parsed: '+str(error)))
                            try:archive.write(local,'unparsed_local/'+name,compress_type=zipfile.ZIP_DEFLATED,compresslevel=3)
                            except Exception as raw_error:backup['errors'].append(dict(path=name,error='cannot preserve malformed original: '+str(raw_error)))
                    if candidate is None:
                        candidate=local;origin='mounted result file; cloud content not confirmed'
                    if candidate is None or not candidate.is_file() or candidate.is_symlink():backup['missing'].append(name);continue
                    try:
                        if name in MEDIA and name not in cloud_paths:
                            length=candidate.stat().st_size
                            if length<=0:raise IOError('empty mounted MP4')
                            entries=listing.get(name,[])
                            if len(entries)==1 and entries[0].get('size') is not None and length!=int(entries[0]['size']):raise IOError('mounted MP4 length differs from listed cloud bytes')
                        if _is_json(name) and candidate==local and name not in valid_local:raise ValueError('malformed local JSON is preserved separately, not a complete backup member')
                        archive.write(candidate,'results/'+name,compress_type=zipfile.ZIP_DEFLATED if _is_json(name) else zipfile.ZIP_STORED,compresslevel=3 if _is_json(name) else None)
                        backup['files'][name]=dict(source=origin,bytes=candidate.stat().st_size,cloud_path_readable=name in cloud_paths,backed_up_content_from_cloud=candidate==cloud_paths.get(name))
                    except Exception as error:backup['errors'].append(dict(path=name,error=f'{type(error).__name__}: {error}'))
                backup['json_count']=sum(_is_json(n) for n in backup['files']);backup['media_count']=sum(n in MEDIA for n in backup['files'])
                backup['cloud_directory_inventory_complete']=inventory_ok
                backup['status']='COMPLETE_BACKUP_OF_LISTED_JSON_AND_48_MEDIA' if inventory_ok and not backup['missing'] and not backup['errors'] and backup['media_count']==48 else 'INCOMPLETE_BACKUP'
                report['status']='STORAGE_CHECK_FINISHED';report['backup_zip']=str(prefix)+'-complete.zip' if backup['status']=='COMPLETE_BACKUP_OF_LISTED_JSON_AND_48_MEDIA' else str(temporary_zip);report['report_path']=str(report_path)
                archive.writestr('backup_report.json',json.dumps(report,ensure_ascii=False,indent=2),compress_type=zipfile.ZIP_DEFLATED)
        if backup['status']=='COMPLETE_BACKUP_OF_LISTED_JSON_AND_48_MEDIA':
            complete_zip=Path(str(prefix)+'-complete.zip');temporary_zip.rename(complete_zip);report['backup_zip']=str(complete_zip)
    except Exception as error:
        backup['status']='INCOMPLETE_BACKUP';backup['errors'].append(f'backup creation: {type(error).__name__}: {error}')
        report['status']='STORAGE_CHECK_INCOMPLETE';report['backup_zip']=str(temporary_zip) if temporary_zip.is_file() else None
    report['report_path']=str(report_path)
    try:report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    except OSError as error:report['report_write_error']=f'{type(error).__name__}: {error}'
    return report


def storage_summary(report):
    cloud=report['cloud_confirmation'];backup=report['backup']
    return dict(computation=report['computation'],cloud_save_status=cloud['status'],
        cloud_core_json=f"{cloud['readable_core_json']}/{cloud['required_core_json']}",cloud_media=f"{cloud['readable_media']}/{cloud['required_media']}",
        cloud_missing_count=len(cloud['missing']),cloud_error_count=len(cloud['errors']),cloud_final_state_issues=cloud.get('final_state_issues',[]),backup_status=backup['status'],
        backup_json_count=backup.get('json_count',0),backup_media_count=backup.get('media_count',0),backup_missing_count=len(backup['missing']),
        backup_errors=len(backup['errors']),backup_zip=report.get('backup_zip'),report_path=report.get('report_path'),
        method_status_modified=False,physical_calls=0)
