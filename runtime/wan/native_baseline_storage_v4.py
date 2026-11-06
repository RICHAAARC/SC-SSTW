"""Actual array checkpoints, bounded infrastructure recovery and single-runtime records."""
from contextlib import contextmanager
from collections import Counter
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


def events(path):
    path=Path(path)
    if not path.is_file():return
    with path.open(encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            try:yield json.loads(line)
            except (ValueError,TypeError):yield dict(status='UNREADABLE_EVENT',line=number)


def infrastructure(error):
    if isinstance(error,(OSError,TimeoutError,subprocess.TimeoutExpired)):return True
    return type(error).__name__=='OutOfMemoryError'


class Budget:
    def __init__(self, output, runtime_id, limits=None):
        self.output=Path(output);self.runtime_id=runtime_id;self.limits=limits or physical_budget()
        self.path=self.output/'physical_calls.jsonl';self.base=Counter();self.extra=Counter();self.unknown_events=[]
        for event in events(self.path):
            if event.get('status')=='STARTED':
                (self.extra if event['recovery'] else self.base)[event['category']]+=event['count']
            elif event.get('status')=='UNREADABLE_EVENT':self.unknown_events.append(event)
        if self.unknown_events:
            raise Unavailable('physical cost ledger has unreadable records; preserve it and repair actual records before more calls')

    def capacity(self, category, count=1, *, recovery=False):
        used=self.extra if recovery else self.base
        limit=self.limits['recovery' if recovery else 'base'][category]
        return used[category]+count<=limit

    def charge(self, category, unit, operation, *, recovery=False, count=1):
        if not self.capacity(category,count,recovery=recovery):
            raise RecoveryExhausted(f'{category} '+('recovery' if recovery else 'base')+' budget exhausted')
        event=dict(status='STARTED',runtime_id=self.runtime_id,at=time.time(),category=category,
                   unit=unit,operation=operation,recovery=recovery,count=count)
        append_json(self.path,event)
        (self.extra if recovery else self.base)[category]+=count

    def snapshot(self):
        result=dict(base=dict(self.base),recovery=dict(self.extra),limits=self.limits,
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

    def append(self,operation,arrays,**metadata):
        import numpy as np
        # A torn tail is preserved; subsequent records go to a new segment, never truncate evidence.
        data=io.BytesIO();np.savez(data,**{k:np.asarray(v) for k,v in arrays.items()})
        payload=data.getvalue();header=json.dumps(dict(operation=operation,at=time.time(),**metadata),allow_nan=False).encode()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('ab') as stream:
            stream.write(struct.pack('<IQ',len(header),len(payload)));stream.write(header);stream.write(payload)
            stream.flush();os.fsync(stream.fileno())


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
        for path in sorted(self.array_directory.glob('arrays-*.bin')):
            records,errors=ArrayLog(path).scan();self.record_errors.extend(dict(file=path.name,**e) for e in errors)
            for key,value in records.items():self.arrays.setdefault(key,value)
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
        if row['status'] in ('INVALID_RETURN','FAILED_PERMANENT','BUDGET_EXHAUSTED'):
            raise Unavailable(f'{key}: {row["status"]}')
        if row.get('runtime_id')==self.budget.runtime_id and row['status'] in ('STARTED','RETURNED_NOT_COMMITTED','INFRA_FAILED'):
            raise Unavailable('no hidden retry within this runtime: '+key)
        physical_key=physical_key or key
        attempt=self.attempts[physical_key]+1
        recovery=recovery or attempt>1
        if attempt>3:
            self.event(key,'BUDGET_EXHAUSTED',reason='two extra infrastructure calls exhausted')
            raise RecoveryExhausted(key)
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
            try:ArrayLog(self.segment).append(key,value,attempt=attempt,runtime_id=self.budget.runtime_id)
            except BaseException as error:
                self.segment=self.array_directory/f'arrays-{self.budget.runtime_id}-{uuid.uuid4().hex}.bin'
                if isinstance(error,OSError) and error.errno==errno.ENOSPC:raise SpaceUnavailable('actual checkpoint write failed: disk full') from error
                raise
            self.arrays[key]=dict(arrays=value,header=dict(operation=key,attempt=attempt))
            self.event(key,'COMMITTED',array_file=self.segment.name)
        else:self.event(key,'COMMITTED',value=value)
        return value

    def reusable(self,key):
        # All callers ask about actual batch checkpoints, never a progress-only receipt.
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
            if current.get('status')=='BUDGET_EXHAUSTED':raise RecoveryExhausted(self.view)
            if Path(current['input']).is_file():return current,False
            if any(self.store.row(prefix+f'/batch{s}')['status'] in ('INVALID_RETURN','FAILED_PERMANENT') for s,n in starts):
                raise Unavailable('method/format failure is not eligible for infrastructure retry: '+self.view)
            # If no physical reader ever started, rebuilding pixels does not rerun a reader.
            consumed=any(self.store.row(prefix+f'/batch{s}')['status']!='NOT_STARTED' for s,n in starts)
            if consumed:
                if (len(self.rows)>=3 or any(self.store.attempts[self.view+f'/batch{s}']>=3 for s,n in starts)
                        or not self.store.budget.capacity(category,len(starts),recovery=True)):
                    current['status']='BUDGET_EXHAUSTED';write_json(self.path,self.rows)
                    raise RecoveryExhausted('whole-view recovery needs all batches: '+self.view)
                current['status']='SUPERSEDED_INPUT_LOST'
            else:return current,False
        else:consumed=False
        epoch=dict(id=f'epoch{len(self.rows)+1:02}',status='ACTIVE',whole_view_recovery=bool(consumed),created_at=time.time())
        epoch['input']=str(self.cache/(epoch['id']+'.npy'));self.rows.append(epoch);write_json(self.path,self.rows)
        return epoch,True


@contextmanager
def runtime_session(output,*,previous_runtime_stopped=False):
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
        if len(history)>=65:raise RecoveryExhausted('64 additional runtime continuations exhausted')
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
