"""Durable completed operation values; never repeat a consumed uncertain operation."""
import json
import os
from pathlib import Path
import random
import time
import uuid
from .paper_suite_records_v1 import read,write


class BoundaryPause(Exception):pass
class UnrecoverableOperation(RuntimeError):pass


class Boundary:
    def __init__(self,max_items=None,seconds=None):
        if max_items is not None and (type(max_items) is not int or max_items<1):raise ValueError('positive task limit required')
        if seconds is not None and seconds<=0:raise ValueError('positive scheduling duration required')
        self.max_items=max_items;self.deadline=time.monotonic()+seconds if seconds else None;self.completed=0
    def before(self):
        if self.max_items is not None and self.completed>=self.max_items or self.deadline is not None and time.monotonic()>=self.deadline:
            raise BoundaryPause('saved boundary reached; next work not started')
    def returned(self):self.completed+=1


def save_value(value,folder):
    """Typed JSON tree and actual numpy/tensor files; no opaque object pickle."""
    import numpy as np
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    def array(data,suffix,writer):
        name=uuid.uuid4().hex+suffix;path=folder/name;temporary=path.with_suffix(path.suffix+'.partial')
        with temporary.open('wb') as stream:writer(stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path);return str(path)
    def encode(x):
        if x is None or type(x) in (bool,int,float,str):return dict(type='scalar',value=x)
        if isinstance(x,np.ndarray):return dict(type='numpy',path=array(x,'.npy',lambda f:np.save(f,x,allow_pickle=False)))
        if isinstance(x,np.generic):return encode(x.item())
        if isinstance(x,Path):return dict(type='path',value=str(x))
        if isinstance(x,dict):return dict(type='dict',items=[[encode(k),encode(v)] for k,v in x.items()])
        if isinstance(x,(list,tuple)):return dict(type='tuple' if isinstance(x,tuple) else 'list',items=[encode(v) for v in x])
        import torch
        if isinstance(x,torch.Tensor):
            return dict(type='tensor',device=str(x.device),path=array(x,'.pt',lambda f:torch.save(x.detach().cpu(),f)))
        raise TypeError('unsupported checkpoint return type: '+str(type(x)))
    return encode(value)


def load_value(node):
    kind=node['type']
    if kind=='scalar':return node['value']
    if kind=='path':return Path(node['value'])
    if kind=='dict':return {load_value(k):load_value(v) for k,v in node['items']}
    if kind in ('list','tuple'):
        values=[load_value(v) for v in node['items']];return tuple(values) if kind=='tuple' else values
    if kind=='numpy':
        import numpy as np
        return np.load(node['path'],allow_pickle=False)
    if kind=='tensor':
        import torch
        return torch.load(node['path'],map_location='cpu',weights_only=True).to(node['device'])
    raise ValueError('invalid checkpoint value type')


def rng_state():
    import numpy as np
    value=dict(python=random.getstate(),numpy=np.random.get_state())
    import sys
    if 'torch' in sys.modules:
        import torch
        value['torch_cpu']=torch.get_rng_state()
        if torch.cuda.is_initialized():value['torch_cuda']=torch.cuda.get_rng_state_all()
    return value


def restore_rng(value):
    import numpy as np
    random.setstate(value['python']);np.random.set_state(value['numpy'])
    if 'torch_cpu' in value:
        import torch
        torch.set_rng_state(value['torch_cpu'])
        if 'torch_cuda' in value:torch.cuda.set_rng_state_all(value['torch_cuda'])


class DurableJournal:
    def __init__(self,output,state_path,cache,boundary=None):
        self.output=Path(output);self.path=Path(state_path);self.cache=Path(cache)
        self.rows=read(self.path);self.boundary=boundary or Boundary();self.rng_restored=False
        self.current=None;self.sequence=max((row.get('commit_index',0) for row in self.rows.values()),default=0)
    def receipt(self,key):return self.output/'checkpoints/returns'/(key+'.json')
    def reusable(self,key):return self.rows[key]['status']=='RETURNED' or self.receipt(key).is_file()
    def call(self,key,function):
        row=self.rows[key];receipt=self.receipt(key)
        if receipt.is_file():
            try:
                saved=read(receipt);value=load_value(saved['value'])
            except Exception as error:raise UnrecoverableOperation('completed checkpoint unreadable; no repeat: '+key+': '+str(error)) from error
            index=saved['commit_index'];self.sequence=max(self.sequence,index)
            previous=self.output/'checkpoints/last_rng.json'
            if not previous.is_file() or read(previous).get('commit_index',0)<index:
                write(previous,dict(value=saved['rng'],operation=key,returned_at=saved['returned_at'],commit_index=index))
            if row['status']!='RETURNED':row.update(status='RETURNED',started=1,returned=1,commit_index=index,recovered_saved_return=True);write(self.path,self.rows)
            return value
        if row['status']!='NOT_STARTED':raise UnrecoverableOperation('consumed call has no completed checkpoint; no repeat: '+key)
        self.boundary.before()
        if not self.rng_restored:
            previous=self.output/'checkpoints/last_rng.json'
            if previous.is_file():restore_rng(load_value(read(previous)['value']))
            self.rng_restored=True
        row.update(status='STARTED',started=1,started_at=time.time());write(self.path,self.rows);tick=time.monotonic();self.current=key
        try:
            value=function()
            encoded=save_value(value,self.cache/'returns'/key)
            rng=save_value(rng_state(),self.cache/'rng'/uuid.uuid4().hex)
            returned_at=time.time();self.sequence+=1
            write(receipt,dict(value=encoded,rng=rng,returned_at=returned_at,commit_index=self.sequence))
            write(self.output/'checkpoints/last_rng.json',dict(value=rng,operation=key,returned_at=returned_at,commit_index=self.sequence))
            row.update(status='RETURNED',returned=1,returned_at=time.time(),commit_index=self.sequence)
        except BaseException as error:
            row.update(status='INTERRUPTED_CONSUMED' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'FAILED',
                error=f'{type(error).__name__}: {error}')
            raise
        finally:row['seconds']=time.monotonic()-tick;write(self.path,self.rows);self.current=None
        self.boundary.returned();return value

    def load_context(self,key,function,needed=True):
        """Restoring process state is explicit overhead; it cannot repeat a scientific writer/reader."""
        if not needed:return None
        if not self.reusable(key):return self.call(key,function)
        self.boundary.before()
        from .native_baseline_storage_v4 import append_json
        path=self.output/'restore_overhead.jsonl';record=dict(operation=key,kind='model_context_restore',status='STARTED',at=time.time())
        append_json(path,record);tick=time.monotonic()
        try:value=function()
        except BaseException as error:
            append_json(path,dict(record,status='FAILED',error=str(error),seconds=time.monotonic()-tick));raise
        append_json(path,dict(record,status='RETURNED',seconds=time.monotonic()-tick,
            scope='additional process/model initialization; not an additional method write/read or a reset of fixed calls'))
        self.rng_restored=False
        return value


def release_completed_cache(output,cache):
    """Only an explicitly recorded, completed new cache outside formal results."""
    import shutil
    from .b_paper_cohort_runner_v1 import scratch_allowed
    output=Path(output);cache=Path(cache).resolve()
    control=read(output/'checkpoint_control.json')
    if Path(control['cache']).resolve()!=cache:raise ValueError('cache does not match its actual saved path')
    scratch_allowed(cache,output)
    result=read(output/'result.json')
    if result.get('status')!='COMPLETED':return
    if cache.exists():shutil.rmtree(cache)
    write(output/'scratch_release.json',dict(status='COMPLETED_CACHE_RELEASED',path=str(cache),formal_results_untouched=True))


def resource_record(output,name,value):
    output=Path(output)
    write(output/'resource_history'/name/(uuid.uuid4().hex+'.json'),value)
    write(output/(name+'.json'),value)
    return value
