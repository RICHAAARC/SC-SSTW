"""Only numpy/psutil: saved raw IO and the inherited RSS watcher, no media/model path."""
from pathlib import Path
import json
import time
import numpy as np
from .b_framewise_qim_time_recovery_v1 import Resources,ResourceExceeded


class RawLimitExceeded(RuntimeError):pass


class Backend:
    def __init__(self):
        self.resources=Resources();self.started=time.monotonic();self.raw_started=None;self.codebook_started=None
    def check(self):
        self.resources.check()
        now=time.monotonic()
        if now-self.started>6300:raise ResourceExceeded('whole diagnostic 6300s limit observed')
        if self.codebook_started is not None and now-self.codebook_started>300:raise RawLimitExceeded('codebook 300s limit observed')
        if self.raw_started is not None and now-self.raw_started>600:raise RawLimitExceeded('raw 600s cumulative wall limit observed')
    def start_raw(self):self.raw_started=time.monotonic()
    def end_raw(self):self.raw_started=None
    def codebook(self):
        from main.tube_state.b_framewise_qim_time_recovery_v1 import codebook
        self.codebook_started=time.monotonic()
        try:
            result=codebook(self.check);self.check();return result
        finally:self.codebook_started=None
    def load_raw(self,path,frames):
        with np.load(path,allow_pickle=False) as saved:raw={k:saved[k] for k in ('q','d0','d1','soft','mask')}
        for key,value in raw.items():
            if value.shape!=(frames,5,2,63):raise ValueError('raw '+key+' geometry mismatch')
            if key=='mask':
                if value.dtype!=np.bool_:raise ValueError('boolean raw mask required')
            else:
                if value.dtype.kind!='f':raise ValueError('floating raw '+key+' required')
                with np.errstate(over='ignore',invalid='ignore'):raw[key]=value.astype(np.float64,copy=False)
                if not np.isfinite(raw[key]).all():raise ValueError('finite FP64 raw '+key+' required')
        return raw
    def saved_view(self,folder,frames):
        result={};errors={};folder=Path(folder)
        def rows(value,key,count):
            values=value.get(key) if isinstance(value,dict) else None
            if not isinstance(values,list) or len(values)!=count or any(not isinstance(v,dict) for v in values):
                raise ValueError('saved '+key+' row geometry mismatch')
        def graph(value):
            if not isinstance(value,dict):raise ValueError('saved H object required')
            rows(value,'nodes',frames);rows(value,'relations',frames-1)
            labels=value.get('labels')
            if not isinstance(labels,list) or len(labels)!=frames or any(not isinstance(v,list) or len(v)!=5 for v in labels):
                raise ValueError('saved H lane geometry mismatch')
        # This explicit allowlist cannot import truth-bearing evaluation files.
        for name in ('blind.json','C.json','L.json','candidates.json','semantic.json','submission.json'):
            try:
                value=json.loads((folder/name).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
                if not isinstance(value,dict):raise ValueError('public semantic object required')
                result[name]=value
                if name=='blind.json':graph(value)
                elif name=='semantic.json':
                    graph(value.get('hard'));rows(value.get('L',{}),'nodes',frames)
                elif name=='candidates.json':rows(value,'nodes',frames)
                elif name=='submission.json':rows(value,'final_nodes',frames);rows(value,'final_relations',frames-1)
            except (OSError,ValueError) as error:errors[name]=str(error)
        for name,keys in (('fused.npz',('soft','known')),('C.npz',('bounds','offsets'))):
            try:
                with np.load(folder/name,allow_pickle=False) as value:
                    arrays={k:value[k] for k in keys}
                result[name]={k:v.tolist() for k,v in arrays.items()}
                if name=='fused.npz':
                    if any(v.shape!=(frames,5,63) for v in arrays.values()) or arrays['known'].dtype!=np.bool_:
                        raise ValueError('saved fusion geometry/type mismatch')
                    if arrays['soft'].dtype.kind!='f' or not np.isfinite(arrays['soft']).all():raise ValueError('finite saved fused soft required')
                else:
                    if arrays['bounds'].shape!=(frames,2) or arrays['offsets'].shape!=(frames+1,):raise ValueError('saved C bound/offset geometry mismatch')
                    if any(v.dtype.kind not in 'iu' for v in arrays.values()):raise ValueError('integer saved C bounds/offsets required')
            except (OSError,ValueError,KeyError) as error:errors[name]=str(error)
        result['errors']=errors;return result
    def close(self):return dict(self.resources.close(),wall_seconds=time.monotonic()-self.started,
        enforcement='RSS watcher; checks between chunks and after blocking calls; no hard interruption guarantee')
