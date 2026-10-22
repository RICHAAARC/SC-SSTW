"""Fixed full-E positive-channel user run. Import is stdlib/pure-method only."""
from pathlib import Path
from collections import Counter
import json,os,time
from main.tube_state import b_time_label_local_channel_v1 as method
HOSTS=('b101-s01','b101-s03','b101-s02','b101-s04')
KEY=method.KEYS['key00']
TAU=0.4959127627330064
ROOT=Path(__file__).resolve().parents[2]
SOURCE='/content/drive/MyDrive/Video-WM/B-Time-Label-Local-Channel-V1/fixed-viewed-stage1-v1'
OUTPUT='/content/drive/MyDrive/Video-WM/B-Full-Context-Local-Read-V1/fixed-viewed-clean-v1'
REFERENCE=ROOT/'data/b_full_context_local_read_v1_reference.json.gz'
CONFIG=ROOT/'experiments/wan_state_clock/configs/b_full_context_local_read_v1.json'
MODEL='Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
def reject(x):raise ValueError('nonfinite JSON constant: '+x)
def flush(f):
    f.flush()
    try:os.fsync(f.fileno())
    except OSError as e:
        if e.errno not in (22,38,95):raise
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=1);f.write('\n');flush(f)
def execution_artifacts(output):
    out=Path(output);found=[]
    for name in ('attempt.json','ledger.json','result.json','progress.json','evaluation.json','post_reference_access.json','fixed_rows.json','fixed_pair_status.json','fixed_pair_chip_status.json','plan.json'):
        for suffix in ('','.tmp'):
            if (out/(name+suffix)).exists() or (out/(name+suffix)).is_symlink():found.append(name+suffix)
    for name in ('raw','blind','latents','markers','receipts'):
        if (out/name).exists():found.extend(str(p.relative_to(out)) for p in (out/name).rglob('*') if p.is_file() or p.is_symlink())
    return sorted(found)
def existing(output):
    out=Path(output);data={};errors={};ledger={}
    for name in ('result.json','evaluation.json'):
        try:
            if (out/name).is_file():data[name]=json.loads((out/name).read_text(),parse_constant=reject)
        except (OSError,ValueError) as e:errors[name]=str(e)
    try:
        if (out/'ledger.json').is_file():ledger=json.loads((out/'ledger.json').read_text(),parse_constant=reject)
    except (OSError,ValueError) as e:errors['ledger.json']=str(e)
    for p in (out/'receipts').rglob('*.json') if (out/'receipts').is_dir() else ():
        try:r=json.loads(p.read_text(),parse_constant=reject);ledger[r['slot']]=r
        except (OSError,ValueError,KeyError) as e:errors[str(p)]=str(e)
    return dict(status='READ_ONLY_EXISTING_ATTEMPT',physical_reentry=False,saved=data,read_errors=errors,ledger_counts=dict(Counter(r['status'] for r in ledger.values())))
def units():
    u={'vae':'VAE_load','raw_readback':'blind_raw_JSON_readback','blind_readback':'blind_final_JSON_readback','post':'private_reference_post'}
    for h in HOSTS:
        for step,kind in [('load','cached_received_RGB_load'),('prepare','RGB_input_prepare'),('E','E181'),('raw','full_projection_decode'),('final','old_tau_description')]:u[h+'/'+step]=kind
    return u
class Journal:
    def __init__(self,out,backend):
        self.out=out;self.backend=backend
        self.rows={k:dict(slot=k,kind=v,status='NOT_ENTERED',attempts=0) for k,v in units().items()};self.snapshot()
    def snapshot(self):
        save(self.out/'ledger.json',self.rows);c=Counter(r['status'] for r in self.rows.values())
        save(self.out/'progress.json',dict(planned=len(self.rows),counts=dict(c),active=next((k for k,r in self.rows.items() if r['status']=='STARTED'),None)))
    def call(self,slot,fn,deps=()):
        row=self.rows[slot]
        if row['status']!='NOT_ENTERED':raise RuntimeError('consumed slot '+slot)
        missing=[d for d in deps if self.rows[d]['status']!='RETURNED']
        if missing:
            row['unavailable_from']=missing;save(self.out/'receipts'/(slot+'.json'),row);self.snapshot();return None
        row.update(status='STARTED',attempts=1,started_unix=time.time())
        save(self.out/'receipts'/(slot+'.json'),row);self.snapshot()
        before=dict(getattr(self.backend,'blocks',{}))
        try:value=fn();row['status']='RETURNED';return value
        except BaseException as e:
            row.update(status='FAILED' if isinstance(e,Exception) else 'INTERRUPTED',error=type(e).__name__+': '+str(e))
            if not isinstance(e,Exception):raise
        finally:
            row.update(finished_unix=time.time());row['seconds']=row['finished_unix']-row['started_unix']
            if row['kind']=='E181':
                after=getattr(self.backend,'blocks',{})
                row['internal_blocks']={k:after.get(k)-before.get(k) if type(after.get(k)) is int and type(before.get(k)) is int else None for k in ('encoder','decoder')}
            save(self.out/'receipts'/(slot+'.json'),row);self.snapshot()
    def record(self,slot,fn,path,deps=()):
        def op():v=fn();save(self.out/path,v);return v
        return self.call(slot,op,deps)
def readback(out,folders=('raw','blind')):
    for folder in folders:
        for h in HOSTS:
            table=json.loads((out/folder/h/'key00.json').read_text(),parse_constant=reject)
            if not isinstance(table.get('rows'),list) or len(table['rows'])!=44:raise ValueError('actual44row blind table unavailable')
            for j,row in enumerate(table['rows'],1):
                if row.get('row')!=j or len(row.get('decoder',{}).get('code_distances',[]))!=64:raise ValueError('actual blind row/code shape')
    return dict(full_rows=176,folders=list(folders),tables=4*len(folders),private_reference_opened=False,saved_unix=time.time())
def fixed_rows(out):
    rows=[]
    for h in HOSTS:
        try:t=json.loads((out/'blind'/h/'key00.json').read_text(),parse_constant=reject)
        except (OSError,ValueError):t={}
        for j in range(1,45):
            unit=t.get('rows',[]);d=unit[j-1].get('decoder',{}) if j<=len(unit) else {}
            rows.append(dict(host=h,cell=j,status=d.get('status','NOT_ENTERED'),technical=d.get('technical_failure',True),tag_label=d.get('tag_label')))
    save(out/'fixed_rows.json',rows)
    try:pairdata=json.loads((out/'full_local_pairs.json').read_text(),parse_constant=reject)
    except (OSError,ValueError):pairdata=[]
    idx={(x.get('host'),x.get('view'),x.get('phase'),x.get('row')):x for x in pairdata if isinstance(x,dict)}
    pairs=[]
    for h in HOSTS:
        for view,phase,offset in [('clean0',0,0),('clean1',1,9),('clean2',2,18),('clean3',3,27),('clean4',0,34)]:
            for r in range(1,11):pairs.append(dict(host=h,view=view,phase=phase,row=r,source_cell=offset+r,anchor='PREFIX40' if view=='clean0' else 'OTHER160',status=idx.get((h,view,phase,r),{}).get('status','NOT_ENTERED')))
    save(out/'fixed_pair_status.json',pairs)
    save(out/'fixed_pair_chip_status.json',[dict(**p,chip=l) for p in pairs for l in range(48)])
    return rows
def run(source=SOURCE,output=OUTPUT,reference=REFERENCE,*,backend=None):
    out=Path(output)
    if execution_artifacts(out):return existing(out)
    out.mkdir(parents=True,exist_ok=True)
    try:
        with (out/'attempt.json').open('x',encoding='utf-8') as f:json.dump(dict(started_unix=time.time(),one_physical_attempt=True,full_E_plan=4),f);flush(f)
    except FileExistsError:return existing(out)
    journal=None;value=dict(status='STARTED',scientific_PASS=None,independent_confirmation=False)
    try:
        public=json.loads(CONFIG.read_text(),parse_constant=reject)
        if public.get('frozen_old_local_tau')!=TAU or public.get('hosts')!=list(HOSTS):raise ValueError('fixed public method parameters changed')
        save(out/'plan.json',dict(full_rows=176,pairs=200,anchor=40,other=160,new_full_nulls=0))
        journal=Journal(out,backend)
        # Reference gzip is neither opened nor parsed here, including when it is missing.
        if backend is None:
            from runtime.wan.b_full_context_local_read_backend_v1 import FullBackend
            backend=FullBackend(public['model_id']);journal.backend=backend
        journal.record('vae',backend.load_vae,'vae.json')
        for h in HOSTS:
            pixels=journal.call(h+'/load',lambda h=h:backend.load_received(Path(source)/'media'/h/'PT/full/received.pt'))
            rgb=journal.call(h+'/prepare',lambda:backend.prepare_full(pixels),[h+'/load'])
            def encode(h=h,rgb=rgb):
                z=backend.encode_full(rgb);backend.save_tensor(out/'latents'/h/'E181.pt',z);return z
            z=journal.call(h+'/E',encode,[h+'/prepare','vae'])
            table=journal.record(h+'/raw',lambda:backend.read_raw(z,KEY),'raw/'+h+'/key00.json',[h+'/E'])
            if table is None:table=method.unavailable_table(44,'UPSTREAM_FAILED_OR_NOT_ENTERED');save(out/'raw'/h/'key00.json',table)
            journal.snapshot()
            del z,rgb,pixels,table
        journal.record('raw_readback',lambda:readback(out,('raw',)),'markers/raw_complete.json')
        for h in HOSTS:
            def describe(h=h):
                table=json.loads((out/'raw'/h/'key00.json').read_text(),parse_constant=reject)
                return method.finalize(table,TAU)
            # All176 raw rows were persisted/read back before any old-tau description.
            final=journal.record(h+'/final',describe,'blind/'+h+'/key00.json',['raw_readback'])
            if final is None:save(out/'blind'/h/'key00.json',method.unavailable_table(44,'FINAL_DESCRIPTION_FAILED'))
        mark=journal.record('blind_readback',lambda:readback(out),'markers/blind_complete.json',['raw_readback']+[h+'/final' for h in HOSTS])
        if mark is not None:
            def post():
                save(out/'post_reference_access.json',dict(opened_unix=time.time(),new_full_blind_rows_read_back=176,reference_file=str(reference)))
                from runtime.wan.b_full_context_local_read_evaluation_v1 import evaluate
                return evaluate(out,reference,save)
            summary=journal.call('post',post,['blind_readback'])
            if summary is not None:value['post_status']=summary['status'];value['reference_issue_count']=len(summary['reference_issues']);value['post_unresolved_counts']=summary['unresolved_counts']
        value['status']='COMPLETED_POSITIVE_CHANNEL_DESCRIPTION_ONLY'
    except BaseException as e:value.update(status='FAILED' if isinstance(e,Exception) else 'INTERRUPTED',error=type(e).__name__+': '+str(e))
    finally:
        if backend is not None:
            try:backend.release_vae()
            except Exception as e:value['release_error']=str(e)
        rows=fixed_rows(out)
        if journal is not None:
            journal.snapshot();value.update(ledger_counts=dict(Counter(r['status'] for r in journal.rows.values())),
                kinds_returned=dict(Counter(r['kind'] for r in journal.rows.values() if r['status']=='RETURNED')))
            if value['status']=='COMPLETED_POSITIVE_CHANNEL_DESCRIPTION_ONLY' and (any(r['status']!='RETURNED' for r in journal.rows.values()) or any(r['technical'] for r in rows) or value.get('reference_issue_count') or any(value.get('post_unresolved_counts',{}).values())):
                value['status']='INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
        value.update(full_rows=176,pair_rows=200,pair_chip_rows=9600,prefix_anchor=40,other=160,new_full_nulls=0,
            technical_full_rows=sum(r['technical'] for r in rows),row_status_counts=dict(Counter(r['status'] for r in rows)),
            physical_reentry=False,full_FPR_validated=False,old_local_tau_descriptive_only=True)
        save(out/'result.json',value)
    return value
