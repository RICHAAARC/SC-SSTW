"""Actual array checkpoints, recorded infrastructure recovery and runtime ownership."""
from contextlib import contextmanager
from collections import Counter, OrderedDict
from collections.abc import MutableMapping
import io
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import threading
import time
import uuid
import zipfile
import errno

from main.baselines.public_fpr001_v4 import physical_budget
from .native_baseline_storage_v1 import read_json, write_json


class Unavailable(RuntimeError): pass
class RecoveryExhausted(Unavailable): pass
class InvalidReturn(Unavailable): pass
class SpaceUnavailable(Unavailable): pass


def append_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        return stream.tell()


def events(path,offset=0):
    path=Path(path)
    if not path.is_file():return
    with path.open(encoding='utf-8') as stream:
        stream.seek(offset)
        for number,line in enumerate(stream,1):
            try:yield json.loads(line)
            except (ValueError,TypeError):yield dict(status='UNREADABLE_EVENT',line=number)


def infrastructure(error):
    if isinstance(error,(OSError,TimeoutError,subprocess.TimeoutExpired)):return True
    return type(error).__name__=='OutOfMemoryError'


def retired_recovery_gate(row):
    """Recognize only the former infrastructure stops, not real method failures."""
    if row.get('status') not in (None,'BUDGET_EXHAUSTED','FAILED_PERMANENT','INFRA_FAILED'):return False
    error=str(row.get('error',''))+' '+str(row.get('reason',''))
    loader=row.get('category') in ('model_load','lpips_load') or any(
        name+' base budget exhausted' in error for name in ('model_load','lpips_load'))
    if ' base budget exhausted' in error and not loader:return False
    return (row.get('status')=='BUDGET_EXHAUSTED' or any(text in error for text in (
        'three additional watermark loads used in this runtime',
        'model_load base budget exhausted','lpips_load base budget exhausted',
        'no same-runtime model-load retry:',
        'two extra infrastructure calls exhausted', 'fixed runtime allowance exhausted',
        'whole-view recovery needs all batches:', ' recovery budget exhausted',
        ': BUDGET_EXHAUSTED')))


def recovery_pending(row):
    return (not row.get('imported_from') or row.get('same_experiment')) and (
        row.get('status') in ('NOT_STARTED','INFRA_FAILED','STARTED','RETURNED_NOT_COMMITTED')
        or retired_recovery_gate(row))


class Budget:
    def __init__(self, output, runtime_id, limits=None, progress=None, saved=None):
        self.output=Path(output);self.runtime_id=runtime_id;self.limits=limits or physical_budget()
        self.path=self.output/'physical_calls.jsonl';self.base=Counter();self.extra=Counter();self.unknown_events=[]
        self.progress=progress;saved=saved or {}
        self.base.update(saved.get('base',{}));self.extra.update(saved.get('recovery',{}));self.offset=saved.get('ledger_offset',0)
        for processed,event in enumerate(events(self.path,self.offset),1):
            if progress and processed%10000==0:progress('RESTORE_PHYSICAL_BUDGET',processed=processed)
            if event.get('status')=='STARTED':
                (self.extra if event['recovery'] else self.base)[event['category']]+=event['count']
            elif event.get('status')=='UNREADABLE_EVENT':self.unknown_events.append(event)
        self.offset=self.path.stat().st_size if self.path.is_file() else 0
        if self.unknown_events:
            raise Unavailable('physical cost ledger has unreadable records; preserve it and repair actual records before more calls')

    def capacity(self, category, count=1, *, recovery=False):
        # Fixed scientific tasks still bound first calls. Restoring their execution
        # and loading backends are infrastructure costs, recorded without a quota.
        if recovery or category in ('model_load','lpips_load'):return True
        used=self.extra if recovery else self.base
        limit=self.limits['recovery' if recovery else 'base'][category]
        return used[category]+count<=limit

    def charge(self, category, unit, operation, *, recovery=False, count=1):
        if not self.capacity(category,count,recovery=recovery):
            raise RecoveryExhausted(f'{category} '+('recovery' if recovery else 'base')+' budget exhausted')
        event=dict(status='STARTED',runtime_id=self.runtime_id,at=time.time(),category=category,
                   unit=unit,operation=operation,recovery=recovery,count=count)
        self.offset=append_json(self.path,event)
        (self.extra if recovery else self.base)[category]+=count

    def resume_state(self):
        return dict(base=dict(self.base),recovery=dict(self.extra),ledger_offset=self.offset)

    def snapshot(self):
        result=dict(**self.resume_state(),limits=self.limits,
                    infrastructure_limits_enforced=False,
                    semantics='started physical calls include unknown return/write outcomes; not only successful calls')
        write_json(self.output/'physical_counts.json',result);return result


class ArrayLog:
    """v3 length-delimited npz records, with an operation/epoch address and complete-record recovery."""
    def __init__(self,path):self.path=Path(path)

    def scan(self):
        import numpy as np
        records={};errors=[]
        if not self.path.is_file():return records,errors
        with self.path.open('rb') as stream:
            while True:
                offset=stream.tell();prefix=stream.read(12)
                if not prefix:break
                try:
                    if len(prefix)!=12:raise ValueError('interrupted prefix')
                    h,n=struct.unpack('<IQ',prefix)
                    if not 0<h<=16384 or not 0<n<=128_000_000:raise ValueError('invalid record lengths')
                    header=json.loads(stream.read(h));data=stream.read(n)
                    if len(data)!=n:raise ValueError('interrupted payload')
                    with np.load(io.BytesIO(data),allow_pickle=False) as values:
                        arrays={k:values[k] for k in values.files}
                    key=header['operation']
                    # First complete valid response in an epoch is used, never best score.
                    records.setdefault(key,dict(header=header,arrays=arrays,offset=offset))
                except (OSError,ValueError,EOFError,KeyError,zipfile.BadZipFile) as error:
                    errors.append(dict(offset=offset,error=str(error)));break
        return records,errors

    def index(self):
        """Read only record headers, seeking over array payloads; retain torn tails."""
        records=[];errors=[]
        if not self.path.is_file():return records,errors
        with self.path.open('rb') as stream:
            size=os.fstat(stream.fileno()).st_size
            while stream.tell()<size:
                offset=stream.tell()
                try:
                    prefix=stream.read(12)
                    if len(prefix)!=12:raise ValueError('interrupted prefix')
                    h,n=struct.unpack('<IQ',prefix)
                    if not 0<h<=16384 or not 0<n<=128_000_000:raise ValueError('invalid record lengths')
                    header=json.loads(stream.read(h))
                    if not isinstance(header.get('operation'),str):raise ValueError('invalid operation address')
                    payload=stream.tell()
                    if payload+n>size:raise ValueError('interrupted payload')
                    records.append(dict(header=header,offset=offset,payload=payload,length=n,path=str(self.path)))
                    stream.seek(n,1)
                except (OSError,ValueError,KeyError,TypeError,AttributeError,struct.error) as error:
                    errors.append(dict(offset=offset,error=str(error)));break
        return records,errors

    @staticmethod
    def read_record(record):
        import numpy as np
        with Path(record['path']).open('rb') as stream:
            stream.seek(record['payload']);data=stream.read(record['length'])
        if len(data)!=record['length']:raise ValueError('interrupted payload')
        with np.load(io.BytesIO(data),allow_pickle=False) as values:
            arrays={k:values[k] for k in values.files}
        return dict(header=record['header'],arrays=arrays,offset=record['offset'])

    def append(self,operation,arrays,**metadata):
        import numpy as np
        # A torn tail is preserved; subsequent records go to a new segment, never truncate evidence.
        data=io.BytesIO();np.savez(data,**{k:np.asarray(v) for k,v in arrays.items()})
        payload=data.getvalue();header=json.dumps(dict(operation=operation,at=time.time(),**metadata),allow_nan=False).encode()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('ab') as stream:
            offset=stream.tell()
            stream.write(struct.pack('<IQ',len(header),len(payload)));stream.write(header);stream.write(payload)
            stream.flush();os.fsync(stream.fileno())
        return dict(header=json.loads(header),offset=offset,payload=offset+12+len(header),length=len(payload),path=str(self.path))


class LazyArrays(MutableMapping):
    """Trust saved operation addresses; read only the requested batch, no directory index."""
    def __init__(self,directory,errors,rows):
        self.directory=Path(directory);self.errors=errors;self.rows=rows;self.locations={};self.cache=OrderedDict()

    def location(self,key):
        if key in self.locations:return self.locations[key]
        row=self.rows.get(key,{})
        if row.get('array_record'):
            location=dict(row['array_record']);location['path']=str(self.directory/row['array_file'])
            self.locations[key]=location;return location
        if row.get('status') not in ('COMMITTED','RETURNED_NOT_COMMITTED'):raise KeyError(key)
        path=self.directory/row.get('array_file',f"arrays-{row.get('runtime_id')}.bin")
        # Legacy records lack byte offsets. Search this one necessary file up to this batch only.
        with path.open('rb') as stream:
            while True:
                offset=stream.tell();prefix=stream.read(12)
                if not prefix:break
                if len(prefix)!=12:raise OSError(f'interrupted checkpoint prefix: {path}:{offset}')
                h,n=struct.unpack('<IQ',prefix)
                if not 0<h<=16384 or not 0<n<=128_000_000:raise OSError(f'invalid checkpoint lengths: {path}:{offset}')
                header=json.loads(stream.read(h));payload=stream.tell()
                location=dict(path=str(path),header=header,offset=offset,payload=payload,length=n)
                self.locations.setdefault(header['operation'],location)
                if header['operation']==key:return location
                stream.seek(n,1)
        if row.get('status')=='COMMITTED':raise OSError(f'saved batch is unreadable: {path}:{key}')
        raise KeyError(key)

    def __getitem__(self,key):
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        location=self.location(key)
        try:value=ArrayLog.read_record(location)
        except (OSError,ValueError,EOFError,KeyError,zipfile.BadZipFile) as error:
            raise OSError(f"cannot read saved checkpoint {location['path']}:{key}: {error}") from error
        self.cache[key]=value
        while len(self.cache)>8:self.cache.popitem(last=False)
        return value

    def __setitem__(self,key,value):
        self.locations[key]=value.pop('location');self.cache[key]=value
        while len(self.cache)>8:self.cache.popitem(last=False)

    def __contains__(self,key):
        if key in self.cache or self.rows.get(key,{}).get('status')=='COMMITTED':return True
        try:self.location(key);return True
        except (KeyError,FileNotFoundError):return False

    def __delitem__(self,key):raise TypeError('checkpoint evidence cannot be deleted')
    def __iter__(self):return iter(k for k,v in self.rows.items() if v.get('status')=='COMMITTED' and v.get('array_file'))
    def __len__(self):return sum(1 for _ in self)


class UnitStore:
    def __init__(self,output,unit,budget,*,array_directory=None):
        self.output=Path(output);self.unit=unit;self.budget=budget
        self.folder=self.output/'units'/unit;self.folder.mkdir(parents=True,exist_ok=True)
        self.path=self.folder/'calls.jsonl';self.rows={};self.arrays={};self.record_errors=[];self.attempts=Counter()
        for event in events(self.path):
            if 'operation' in event:
                self.rows[event['operation']]=event
                if event['status']=='STARTED':self.attempts[event.get('physical_key',event['operation'])]+=1
        self.array_directory=Path(array_directory) if array_directory is not None else self.folder
        self.array_directory.mkdir(parents=True,exist_ok=True)
        self.arrays=LazyArrays(self.array_directory,self.record_errors,self.rows)
        self.segment=self.array_directory/f'arrays-{budget.runtime_id}.bin'

    def row(self,key):return self.rows.get(key,dict(status='NOT_STARTED',attempt=0))

    def event(self,key,status,**value):
        row=dict(self.row(key),operation=key,status=status,runtime_id=self.budget.runtime_id,at=time.time(),**value)
        append_json(self.path,row);self.rows[key]=row;return row

    def perform(self,key,category,call,*,validate=None,arrays=False,recovery=False,output_available=None,count=1,physical_key=None):
        row=self.row(key)
        if arrays and key in self.arrays:
            value=self.arrays[key]['arrays']
            if validate:validate(value)
            return value
        if row['status']=='COMMITTED' and not arrays:
            value=row.get('value')
            if output_available is None or output_available(value):return value
            recovery=True  # actual needed output missing, never a provenance/hash mismatch
        if row['status'] in ('INVALID_RETURN','FAILED_PERMANENT','BUDGET_EXHAUSTED') and not retired_recovery_gate(row):
            raise Unavailable(f'{key}: {row["status"]}')
        if row.get('runtime_id')==self.budget.runtime_id and row['status'] in ('STARTED','RETURNED_NOT_COMMITTED','INFRA_FAILED'):
            raise Unavailable('no hidden retry within this runtime: '+key)
        physical_key=physical_key or key
        attempt=self.attempts[physical_key]+1
        recovery=recovery or attempt>1
        # Start the unit receipt first, then physical cost, then actual call.
        self.event(key,'STARTED',attempt=attempt,category=category,recovery=recovery,count=count,physical_key=physical_key)
        self.attempts[physical_key]+=1
        try:self.budget.charge(category,self.unit,key,recovery=recovery,count=count)
        except RecoveryExhausted as error:
            self.event(key,'BUDGET_EXHAUSTED',error=str(error));raise
        started=time.monotonic()
        try:value=call()
        except BaseException as error:
            status='INFRA_FAILED' if infrastructure(error) or isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED_PERMANENT'
            self.event(key,status,error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-started)
            if isinstance(error,OSError) and error.errno==errno.ENOSPC:raise SpaceUnavailable('actual disk write failed: no space remains') from error
            if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
            raise Unavailable(str(error)) from error
        self.event(key,'RETURNED_NOT_COMMITTED',seconds=time.monotonic()-started)
        try:
            if validate:validate(value)
        except Exception as error:
            self.event(key,'INVALID_RETURN',error=f'{type(error).__name__}: {error}')
            raise InvalidReturn(str(error)) from error
        if arrays:
            try:location=ArrayLog(self.segment).append(key,value,attempt=attempt,runtime_id=self.budget.runtime_id)
            except BaseException as error:
                self.segment=self.array_directory/f'arrays-{self.budget.runtime_id}-{uuid.uuid4().hex}.bin'
                if isinstance(error,OSError) and error.errno==errno.ENOSPC:raise SpaceUnavailable('actual checkpoint write failed: disk full') from error
                raise
            self.arrays[key]=dict(arrays=value,header=dict(operation=key,attempt=attempt),location=location)
            self.event(key,'COMMITTED',array_file=self.segment.name,array_record={k:v for k,v in location.items() if k!='path'})
        else:self.event(key,'COMMITTED',value=value)
        return value

    def reusable(self,key):
        # Committed historical batches are trusted; payload is opened only when actually used.
        return key in self.arrays


class ViewEpochs:
    def __init__(self,store,view,cache):
        self.store=store;self.view=view;self.path=store.folder/'views'/view/'epochs.json'
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.rows=read_json(self.path) if self.path.is_file() else []

    def choose(self,starts,category):
        if self.rows:
            current=self.rows[-1];prefix=self.view+'/'+current['id']
            if all(self.store.reusable(prefix+f'/batch{start}') for start,n in starts):return current,False
            if current.get('status')=='BUDGET_EXHAUSTED':
                append_json(self.path.with_name('recovery.jsonl'),dict(
                    previous_epoch=dict(current),reason='retired infrastructure quota',runtime_id=self.store.budget.runtime_id))
            if Path(current['input']).is_file():return current,False
            if any(self.store.row(prefix+f'/batch{s}')['status'] in ('INVALID_RETURN','FAILED_PERMANENT')
                   and not retired_recovery_gate(self.store.row(prefix+f'/batch{s}')) for s,n in starts):
                raise Unavailable('method/format failure is not eligible for infrastructure retry: '+self.view)
            # If no physical reader ever started, rebuilding pixels does not rerun a reader.
            consumed=any(self.store.row(prefix+f'/batch{s}')['status']!='NOT_STARTED' for s,n in starts)
            if consumed:
                current['status']='SUPERSEDED_INPUT_LOST'
            else:return current,False
        else:consumed=False
        epoch=dict(id=f'epoch{len(self.rows)+1:02}',status='ACTIVE',whole_view_recovery=bool(consumed),created_at=time.time())
        epoch['input']=str(self.cache/(epoch['id']+'.npy'));self.rows.append(epoch);write_json(self.path,self.rows)
        return epoch,True


@contextmanager
def runtime_session(output,*,previous_runtime_stopped=False,runtime_limit=None):
    """Local flock plus explicit cross-runtime handoff. Heartbeat is not a distributed lock."""
    import fcntl
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'local-runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        owner_path=output/'runtime_owner.json';host=socket.gethostname()
        old=read_json(owner_path) if owner_path.is_file() else None
        if old and old.get('status')=='ACTIVE':
            same_host=old.get('host')==host
            alive=same_host and Path('/proc')/str(old.get('pid'))
            alive=bool(alive and alive.exists())
            if alive or (not same_host and not previous_runtime_stopped):
                raise Unavailable('prior runtime may still be active; stop it before confirming continuation; heartbeat age cannot authorize takeover')
        runtime_id=uuid.uuid4().hex
        history=list(events(output/'runtimes.jsonl'))
        # runtime_limit is accepted for old callers/checkpoints, as locating metadata only.
        record=dict(id=runtime_id,pid=os.getpid(),host=host,status='ACTIVE',started_at=time.time(),
                    ordinal=len(history),cross_vm_atomic_lock=False)
        append_json(output/'runtimes.jsonl',record);write_json(owner_path,record)
        stop=threading.Event()
        def heartbeat():
            while not stop.wait(20):
                write_json(output/'heartbeat.json',dict(runtime_id=runtime_id,at=time.time(),status='RUNNING',coordination='informational only'))
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:yield record
        finally:
            stop.set();thread.join(timeout=1)
            record.update(status='CLOSED',closed_at=time.time());write_json(owner_path,record)
