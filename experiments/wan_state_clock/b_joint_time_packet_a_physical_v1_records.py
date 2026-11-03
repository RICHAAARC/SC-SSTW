"""Fixed staged operation/slot receipts. Stdlib-only recovery, no scientific reads."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import time
from .b_joint_time_packet_a_write_cpu_v1_records import dump,load

SOURCES=('n01','e02')
ARMS=('OLD','NEW','BASE')
VIEWS=('FULL','CROP','DELETE1','REPEAT1','DELETE3')
LENGTHS=(181,101,180,182,178)
LAYERS=('preclip','clip','rgb8','firstRGB')
MODEL_ID='Wan-AI/Wan2.1-T2V-1.3B-Diffusers'


def time_units():
    return [dict(id=f'{s}/{a}/{v}',source=s,arm=a,view=v,frames=n,nodes=n,edges=n-1,
                 physical_bits=n*630,coding='OLD' if a=='OLD' else 'NEW',
                 scope='DESCRIPTIVE_ONLY' if a=='BASE' else 'UNCALIBRATED_DIAGNOSTIC')
            for s in SOURCES for a in ARMS for v,n in zip(VIEWS,LENGTHS)]


def identity_units():
    return [dict(id=f'{s}/{a}/{v}/q{q}/p{p}',source=s,arm=a,view=v,frames=n,query=q,phase=p,start=w+p,
                 old_id=f'input{i}/E41/a{w+p}',raw_bit_slots=160,vote_slots=1280,
                 encoder_input_support=list(range(w+p,w+p+41)),status='NOT_STARTED')
            for s in SOURCES for a in ('OLD','NEW') for i,(v,n) in enumerate(zip(VIEWS,LENGTHS))
            for q,w in enumerate((0,(n-45)//2,n-45)) for p in range(4)]


def denominators():
    return dict(time_units=30,nodes=4932,edges=4902,physical_time_bit_slots=3107160,
                layer_streams=54,layer_frames=9276,identity_windows=240,raw_identity_bits=38400,
                coefficient_votes=307200,mature_state_evaluations=4932*92668,startup_state_evaluations=4932*4)


def plan(stage):
    rows={}
    def add(key,kind): rows[key]=dict(kind=kind,planned=1,started=0,returned=0,status='NOT_STARTED')
    if stage=='gpu1':
        add('model_load','model_load');add('e02/capture','capture');add('e02/D181','D181')
        for i in range(46):add(f'e02/baseline_head/{i:02}','baseline_head')
        for s in SOURCES:
            add(s+'/replay','native_replay')
            for i in range(46):add(f'{s}/head_replay/{i:02}','head_replay')
    elif stage=='cpu_media':
        for s in SOURCES:
            for op in ('encode','probe','decode_RGB'):add(f'{s}/NEW/first/{op}','codec_'+op.split('_')[0])
            for v in VIEWS:
                for op in ('encode','probe','decode_RGB','decode_Y'):add(f'{s}/NEW/{v}/{op}','codec_'+op.split('_')[0])
        for v in VIEWS:
            for op in ('probe','decode_RGB'):add('e02/OLD/'+v+'/'+op,'codec_'+op.split('_')[0])
        for s in SOURCES:
            for a in ARMS:
                for layer in LAYERS:add(f'{s}/{a}/layer/{layer}','layer_read' if a=='NEW' else 'layer_reuse')
        for u in time_units():add(u['id']+'/receive','time_receive')
    elif stage=='gpu2':
        add('model_load','model_load')
        for u in identity_units():
            if (u['source'],u['arm'])==('n01','OLD'):add(u['id']+'/reuse','old_cache_load')
            else:
                for op in ('E41','project','bits'):add(u['id']+'/'+op,op)
    elif stage=='post':add('evaluate','post_evaluation')
    else:raise ValueError('unknown fixed stage')
    return rows


def reserve(output,stage,config):
    output=Path(output)
    if (output/'attempt.json').exists():return False
    if output.exists() and any(output.iterdir()):raise ValueError('new stage output is not empty')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'attempt.json').open('x') as stream:
        import json
        json.dump(dict(stage=stage,status='CONSUMED_ON_START',started_utc=datetime.now(timezone.utc).isoformat(),
                       config=config,denominators=denominators(),rerun='METADATA_ONLY'),stream,indent=2)
    dump(output/'fixed_time_units.json',time_units());dump(output/'fixed_identity_units.json',identity_units())
    return True


def recover(output):
    saved={};errors={}
    for name in ('attempt','state','counts','resources','blind_terminal','result'):
        try:saved[name]=load(Path(output)/(name+'.json'))
        except (OSError,ValueError) as error:errors[name]=str(error)
    return dict(status='READ_ONLY_CONSUMED_STAGE',saved=saved,read_errors=errors,
                physical_calls_this_invocation=0,scientific_reads_this_invocation=0,denominators=denominators())


def terminal(path):
    try:
        value=load(Path(path)/'blind_terminal.json')
        return value if value.get('stage_terminal') is True else None
    except (OSError,ValueError):return None


def package_file(root,name):
    root=Path(root).resolve();relative=Path(name)
    if relative.is_absolute() or not (root/relative).resolve().is_relative_to(root):
        raise ValueError('handoff file must be relative to this package root')
    return root/relative


class Journal:
    def __init__(self,output,stage):
        self.output=Path(output);self.rows=plan(stage);self.events={};self.started=time.monotonic();self.save()

    def save(self):
        counts={k:dict(planned=n,started=0,returned=0) for k,n in Counter(x['kind'] for x in self.rows.values()).items()}
        for row in self.rows.values():
            for field in ('started','returned'):counts[row['kind']][field]+=row[field]
        dump(self.output/'state.json',self.rows);dump(self.output/'counts.json',counts)

    def call(self,key,fn):
        row=self.rows[key]
        if row['started']:raise RuntimeError('operation already consumed: '+key)
        row.update(started=1,status='STARTED');self.save();began=time.monotonic()
        try:
            value=fn();row.update(returned=1,status='RETURNED');return value
        except BaseException as error:
            if getattr(error,'physical_returned',False):row['returned']=1;row['limit_observation']='AFTER_CALL_RETURN'
            row.update(status='FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=type(error).__name__+': '+str(error));raise
        finally:row['wall_seconds']=time.monotonic()-began;self.save()

    def event(self,source,kind,done):
        key=source+'/'+kind;counter=self.events.setdefault(key,dict(started=0,returned=0))
        i=counter['returned'] if done else counter['started']
        uid=key if kind=='D181' else key+f'/{i:02}'
        if uid not in self.rows:raise RuntimeError('unplanned physical call: '+uid)
        row=self.rows[uid]
        if done:
            if row['status']!='STARTED':raise ValueError('physical return without start')
            row.update(returned=1,status='RETURNED');counter['returned']+=1
        else:
            if row['started']:raise RuntimeError('physical call already consumed: '+uid)
            row.update(started=1,status='STARTED');counter['started']+=1
        self.save()

    def finish(self,errors,**fields):
        for row in self.rows.values():
            if row['status']=='STARTED':row['status']='INTERRUPTED'
            if row['status']=='NOT_STARTED':row['reason']='dependency absent or stage ended; no later execution'
        self.save()
        result=dict(status='FIXED_STAGE_TERMINATED',stage_terminal=True,errors=errors,
                    wall_seconds=time.monotonic()-self.started,denominators=denominators(),scientific_pass=None,**fields)
        dump(self.output/'blind_terminal.json',result);dump(self.output/'result.json',result)
        return result
