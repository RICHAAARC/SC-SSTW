"""Finite compact12 repair addresses and read-only raw-record association."""
from collections import Counter
from pathlib import Path
import re

from main.baselines import public_compact12_v1 as p
from main.baselines.public_fpr001_v4 import backend, batches, read_batch
from .native_baseline_storage_v1 import read_json, write_json
from .native_baseline_storage_v4 import LazyArrays, events, Unavailable
from .native_baseline_io_v4 import empty_raw, validate_native

REBUILD_ARMS={('ucf-0642','pixelseal'),('ucf-0642','frameindex'),('ucf-0643','pixelseal')}
REPLACEMENT_ARMS={('ucf-0643','pixelseal')}


def replacement_unit(unit):
    parts=unit.split('/')
    if parts[0] not in ('writer','media','native'):return False
    arm=parts[3] if parts[0]=='native' else parts[2]
    return (parts[1],arm) in REBUILD_ARMS


def targets():
    rows=[]
    def add(shard,sid,method,arm,views):
        rows.extend(dict(shard=shard,source=sid,method=method,arm=arm,view=v) for v in views)
    add(1,'ucf-0937','wam','wam',p.VIEW_IDS[1:])
    add(2,'ucf-0641','frameindex','frameindex',('G2/T14','S/S00','S/S02'))
    for method in ('pixelseal','frameindex'):
        for arm in (method,'OFF'):add(2,'ucf-0642',method,arm,p.VIEW_IDS)
    add(2,'ucf-0643','pixelseal','OFF',('G1/T00','G2/T01','G2/T02','G2/T03','G2/T06','S/S02'))
    add(2,'ucf-0643','frameindex','OFF',('G1/T00','G2/T01','G2/T03','G2/T04','G2/T06','S/S02'))
    add(2,'ucf-0643','pixelseal','pixelseal',('G1/T00','G2/T00','G2/T08'))
    add(3,'ucf-0517','wam','OFF',(*[v for v in p.VIEW_IDS if v.startswith('G2/')],'S/S00'))
    return rows


def quality_targets():
    rows=[]
    for shard,sid,method,stages in (
        (2,'ucf-0641','frameindex',('precodec','G1','G2-identity')),
        (2,'ucf-0642','pixelseal',('precodec','G1','G2-identity')),
        (2,'ucf-0642','frameindex',('precodec','G1','G2-identity')),
        (3,'ucf-0532','frameindex',('G1','G2-identity'))):
        rows.extend(dict(shard=shard,source=sid,method=method,stage=s) for s in stages)
    return rows


def execution_targets():
    rows=targets();existing={(r['source'],r['method'],r['arm'],r['view']) for r in rows}
    for sid,method in sorted(REPLACEMENT_ARMS):
        rows.extend(dict(shard=2,source=sid,method=method,arm=method,view=v)
            for v in p.VIEW_IDS if (sid,method,method,v) not in existing)
    return rows


def execution_quality_targets():
    return quality_targets()+[dict(shard=2,source=sid,method=method,stage=stage)
        for sid,method in sorted(REPLACEMENT_ARMS) for stage in ('precodec','G1','G2-identity')]


def groups(rows=None):
    result={}
    for row in execution_targets() if rows is None else rows:
        key=(row['shard'],row['source'],backend(row['method']),row['arm'],row['view'])
        result.setdefault(key,[]).append(row)
    return result


class Locator:
    """Rebase a copied Drive tree by path, without identity or version gates."""
    def __init__(self,drive_root=None,remaps=()):
        self.drive_root=Path(drive_root) if drive_root else None
        self.remaps=sorted(((Path(a),Path(b)) for a,b in remaps),key=lambda pair:len(pair[0].parts),reverse=True)
    def __call__(self,value):
        path=Path(value)
        for old,new in self.remaps:
            if path.is_relative_to(old):return new/path.relative_to(old)
        old=Path('/content/drive/MyDrive')
        if self.drive_root and path.is_relative_to(old):return self.drive_root/path.relative_to(old)
        return path


class SavedStore:
    """Read existing unit receipts; never create directories in a source package."""
    def __init__(self,root,unit,locate=None,array_directory=None):
        self.folder=Path(root)/'units'/unit;self.unit=unit;self.rows={};self.record_errors=[]
        self.locate=locate or Locator();self.attempts=Counter();self.history=[]
        directory=Path(array_directory) if array_directory is not None else self.folder
        imported=self.folder/'old_import.json'
        if imported.is_file():
            try:
                value=read_json(imported)
                if not isinstance(value,dict) or not isinstance(value.get('events',[]),list):raise ValueError('invalid imported receipt container')
                self.history.extend(value.get('events',[]));self.record_errors.extend(value.get('read_errors',[]))
            except (OSError,ValueError,TypeError) as error:self.record_errors.append(dict(file=str(imported),error=str(error)))
        self.history.extend(events(self.folder/'calls.jsonl'))
        for row in self.history:
            if not isinstance(row,dict) or row.get('status')=='UNREADABLE_EVENT':
                self.record_errors.append(row);continue
            key=row.get('operation')
            if not isinstance(key,str):continue
            row=dict(row)
            if row.get('array_file'):
                path=self.locate(row['array_file'])
                if not path.is_absolute():path=directory/path
                row['array_file']=str(path)
            self.rows[key]=row
            if row.get('status')=='STARTED':self.attempts[row.get('physical_key',key)]+=1
        self.arrays=LazyArrays(directory,self.record_errors,self.rows)
    def row(self,key):return self.rows.get(key,dict(status='NOT_STARTED'))


def select_epoch(store,view,epoch_paths=()):
    """Use the recorded last epoch, or a single unambiguous receipt epoch."""
    own=store.folder/'views'/view['id']/'epochs.json'
    for path in (own,*map(Path,epoch_paths)):
        if not path.is_file():continue
        try:
            rows=read_json(path)
            if not isinstance(rows,list):raise ValueError('epochs must be a list')
            if not rows:continue
            value=rows[-1]
            if not isinstance(value,dict) or not isinstance(value.get('id'),str):raise ValueError('invalid last epoch')
            value=dict(value)
            if value.get('input'):value['input']=str(store.locate(value['input']))
            return value,dict(basis='SAVED_LAST_EPOCH',file=str(path))
        except (OSError,ValueError,TypeError) as error:
            return None,dict(basis='INVALID_SAVED_EPOCH',file=str(path),error=str(error))
    if store.record_errors:
        return None,dict(basis='UNREADABLE_RECEIPTS',errors=store.record_errors)
    prefix=view['id']+'/'
    malformed=[key for key in store.rows if key.startswith(prefix) and not re.fullmatch(r'epoch\d+/batch\d+',key[len(prefix):])]
    if malformed:return None,dict(basis='INVALID_RECEIPT_OPERATION_ADDRESS',operations=malformed)
    ids={key[len(prefix):].split('/')[0] for key in store.rows if key.startswith(prefix) and re.fullmatch(r'epoch\d+/batch\d+',key[len(prefix):])}
    # Missing registry plus multiple epochs does not establish which one was selected.
    # In particular never fall back to the most complete or best-scoring older epoch.
    if len(ids)!=1:
        return None,dict(basis='NO_UNIQUE_RECEIPT_EPOCH',epoch_ids=sorted(ids))
    eid=next(iter(ids))
    return dict(id=eid,input=None,whole_view_recovery=False),dict(basis='UNIQUE_RECEIPT_EPOCH',epoch_ids=[eid])


def read_raw(store,view,b,epoch):
    raw=empty_raw(view['frames'],b);details=[]
    for start,n in batches(view['frames'],read_batch(b)):
        key=f"{view['id']}/{epoch['id']}/batch{start}" if epoch else None
        receipt=store.row(key) if key else dict(status='UNKNOWN_EPOCH')
        row=dict(start=start,frames=n,operation=key,receipt_status=receipt.get('status'),observed=False)
        if key and receipt.get('status') in ('COMMITTED','RETURNED_NOT_COMMITTED'):
            try:
                record=store.arrays.get(key)
                if record is None:raise FileNotFoundError('saved batch payload unavailable')
                if record.get('header',{}).get('operation')!=key:raise ValueError('array record has another operation address')
                value=record['arrays'];validate_native(value,n,b)
                for field in ('bits','valid','logits'):
                    if field in raw:raw[field][start:start+n]=value[field]
                raw['observed'][start:start+n]=True;row['observed']=True
            except (OSError,ValueError,TypeError,KeyError,EOFError) as error:row['error']=f'{type(error).__name__}: {error}'
        details.append(row)
    return raw,details


def complete_post(value):
    return isinstance(value,dict) and isinstance(value.get('ownership'),dict) and value['ownership'].get('complete') is True


def first_file(roots,relative):
    return next((Path(root)/relative for root in roots if (Path(root)/relative).is_file()),None)


def saved_budget(ctx,shard,output):
    """Inherit original consumption for constraints, not as new repair ledger rows."""
    own=Path(output)/'physical_counts.json'
    if own.is_file():return read_json(own)
    path=ctx.parent/'shards'/f'part{shard}'/'physical_counts.json'
    if not path.is_file():raise Unavailable('original physical cost snapshot unavailable: '+str(path))
    value=read_json(path)
    if not isinstance(value.get('base'),dict) or not isinstance(value.get('recovery'),dict):raise ValueError('original physical cost counts invalid')
    write_json(Path(output)/'inherited_consumption.json',dict(source=str(path),value=value))
    return dict(base=value['base'],recovery=value['recovery'],ledger_offset=0)
