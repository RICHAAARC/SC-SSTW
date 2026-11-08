"""Direct computation cursor; completed history is trusted, never rediscovered."""
import ast
import json
from pathlib import Path
import time
from main.baselines.public_fpr001_v4 import BACKENDS, views
from .native_baseline_storage_v1 import read_json, write_json


def tasks_for(sources):
    tasks=[]
    def add(kind,pack,source=None,backend=None,arm=None,**extra):
        row=dict(kind=kind,pack=pack,source=source['id'] if source else None,backend=backend,arm=arm,**extra)
        row['id']='/'.join(str(row[k] or '-') for k in ('kind','pack','backend','source','arm'))
        tasks.append(row)
    def methods(b):return ('pixelseal','frameindex') if b=='pixelseal' else (b,)
    dev=[s for s in sources if s['split']=='DEV']
    for s in dev:add('media','DEV-bank',s,arm='OFF')
    for b in BACKENDS:
        for s in dev:
            for m in methods(b):add('media','DEV-bank',s,b,m)
        for s in dev:
            add('native','DEV-bank',s,b,'OFF')
            for m in methods(b):
                add('native','DEV-bank',s,b,m);add('quality','DEV-bank',s,b,m)
    for split in ('CAL','TEST'):
        selected=[s for s in sources if s['split']==split]
        for start in range(0,len(selected),16):
            group=selected[start:start+16];pack=f'{split}-{start//16:02}'
            for s in group:add('media',pack,s,arm='OFF')
            for b in BACKENDS:
                for s in group:
                    add('native',pack,s,b,'OFF')
                    if split!='CAL':
                        for m in methods(b):
                            add('media',pack,s,b,m);add('native',pack,s,b,m);add('quality',pack,s,b,m)
            add('cleanup',pack,source_ids=[s['id'] for s in group])
    add('post','ALL')
    return tasks


def reverse_records(path):
    """Read one existing log backwards, stopping as soon as the required record is found."""
    path=Path(path)
    if not path.is_file():return
    with path.open('rb') as stream:
        stream.seek(0,2);position=stream.tell();suffix=b''
        while position:
            size=min(position,65536);position-=size;stream.seek(position)
            lines=(stream.read(size)+suffix).split(b'\n');suffix=lines[0]
            for line in reversed(lines[1:]):
                try:
                    text=line.decode('utf-8')
                    if not text.startswith('{'):continue
                    try:row=json.loads(text)
                    except ValueError:row=ast.literal_eval(text)
                    if isinstance(row,dict):yield row
                except (ValueError,SyntaxError,UnicodeError):continue
        if suffix:
            try:
                text=suffix.decode('utf-8')
                try:row=json.loads(text)
                except ValueError:row=ast.literal_eval(text)
                if isinstance(row,dict):yield row
            except (ValueError,SyntaxError,UnicodeError):pass


def legacy_position(output,tasks):
    """Use actual compute progress and its last physical call, not RESTORE_* rows."""
    output=Path(output);anchor=None
    for row in reverse_records(output/'runner.log'):
        stage=row.get('stage','')
        if stage.startswith(('RESTORE','LOCATE')):continue
        if row.get('source') and row.get('backend') and (row.get('pack') or stage in ('READ_MISSING_VIEW','QUALITY_MISSING')):
            anchor=row;break
    def locate(kind,sid,backend,arm,operation=None):
        matches=[i for i,t in enumerate(tasks) if t['kind']==kind and t['source']==sid and t['backend']==backend and t['arm']==arm]
        if not matches:return None
        view=0
        if kind=='native' and operation:
            vid='/'.join(operation.split('/')[:2]);view=next((i for i,v in enumerate(views()) if v['id']==vid),0)
        return dict(next_task=matches[0],active_task=matches[0],view_index=view,pending={},status='ACTIVE',origin='saved physical computation')
    for row in reverse_records(output/'physical_calls.jsonl'):
        if row.get('status')!='STARTED':continue
        if anchor and row.get('at',0)<anchor.get('at',0):break
        unit=row.get('unit','').split('/')
        if len(unit)<3:continue
        if anchor and unit[1]!=anchor['source']:continue
        position=None
        if unit[0]=='native' and len(unit)==4:
            if anchor and unit[2]!=anchor['backend']:continue
            position=locate('native',unit[1],unit[2],unit[3],row.get('operation'))
        elif unit[0]=='quality':
            b='pixelseal' if unit[2]=='frameindex' else unit[2]
            position=locate('quality',unit[1],b,unit[2])
        if position is not None:return position
    if anchor:
        position=locate('quality' if anchor.get('stage')=='QUALITY_MISSING' else 'native',
                        anchor['source'],anchor['backend'],anchor.get('arm', 'OFF'),anchor.get('view'))
        if position is not None:return position
    if (output/'physical_calls.jsonl').is_file():
        raise RuntimeError('Cannot locate the interrupted computation in runner.log/physical_calls.jsonl; no historical directory scan or restart was performed')
    return dict(next_task=0,active_task=None,view_index=0,pending={},status='ACTIVE',origin='new package')


class ComputationCursor:
    def __init__(self,output,sources,budget=None):
        self.output=Path(output);self.path=self.output/'compute_checkpoint.json';self.tasks=tasks_for(sources);self.budget=budget
        if self.path.is_file():self.state=read_json(self.path)
        elif (self.output/'result.json').is_file():
            self.state=dict(next_task=len(self.tasks),active_task=None,view_index=0,pending={},status='DONE',origin='saved final result')
        else:self.state=legacy_position(self.output,self.tasks)

    def queue(self):
        pending=self.state.get('pending',{});selected={int(k):v for k,v in pending.items()}
        for index in range(self.state['next_task'],len(self.tasks)):selected.setdefault(index,0)
        if self.state.get('active_task') is not None:selected[self.state['active_task']]=self.state.get('remaining_views') if self.state.get('remaining_views') is not None else self.state.get('view_index',0)
        if selected:selected.setdefault(len(self.tasks)-1,0)
        return sorted(selected.items())

    def save(self):
        if self.budget is not None:self.state['budget']=self.budget.resume_state()
        self.state['at']=time.time()
        active=self.state.get('active_task')
        self.state['computation']=self.tasks[active] if active is not None else None
        write_json(self.path,self.state)

    def begin(self,index,view=0):
        failed=self.state.get('failed_views',[]) if self.state.get('active_task')==index else []
        self.state.update(active_task=index,view_index=view[0] if isinstance(view,list) and view else view,
                          remaining_views=view if isinstance(view,list) else None,failed_views=failed,status='ACTIVE');self.save()

    def view(self,index,remaining,failed):
        self.state.update(view_index=index,remaining_views=list(remaining),failed_views=list(failed));self.save()

    def finish(self,index,failed_views=None):
        pending=self.state.setdefault('pending',{})
        if not failed_views:pending.pop(str(index),None)
        else:pending[str(index)]=list(failed_views)
        self.state.update(next_task=max(self.state['next_task'],index+1),active_task=None,view_index=0,remaining_views=None,failed_views=[])
        if self.state['next_task']==len(self.tasks):self.state['status']='PENDING' if pending else 'DONE'
        self.save()
