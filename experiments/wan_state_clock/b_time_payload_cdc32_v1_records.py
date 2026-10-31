"""Fixed 256 logical rows, independent counts and stdlib read-only recovery."""
from pathlib import Path
from collections import Counter
import time
from .b_minimal_l_v1_records import load, dump, output_state

VIEWS = ('FULL','CROP','DELETE1','REPEAT1','DELETE3')
LENGTHS = (181,101,180,182,178)
ARMS = ('B','W','U')
LIMITS = dict(cpu_rss_bytes=8*1024**3, codec_seconds=300, model_load_seconds=900,
              E41_seconds=600, total_E41_seconds=7200, cpu_receive_save_seconds=1800)


def units(view, length):
    return [dict(id=f'input{VIEWS.index(view)}/E41/a{w+p}', query=q, phase=p, start=w+p)
            for q,w in enumerate((0,(length-45)//2,length-45)) for p in range(4)]


def plan():
    rows = {}
    def add(key,kind): rows[key]=dict(kind=kind,planned=1,started=0,returned=0,status='NOT_STARTED')
    for view in VIEWS: add(view+'/plan','plan')
    add('model_load','model_load')
    for view,length in zip(VIEWS,LENGTHS):
        for op in ('probe','decode'): add(view+'/'+op,'codec_'+op)
        for u in units(view,length):
            for op in ('E41','project','bits'): add(view+f"/q{u['query']}/p{u['phase']}/"+op,op)
        for arm in ARMS:
            for q in range(3): add(f'{view}/{arm}/q{q}/read','local_read')
            add(f'{view}/{arm}/aggregate','aggregate')
    return rows


def budgets():
    result=dict(Counter(r['kind'] for r in plan().values()))
    for k in ('codec_encode','DiT','D181','head','solver','K','KT','training','writer','time_read','BCH','C','L','submit'):
        result[k]=0
    return result


def denominators():
    return dict(logical_operations=256,E41=60,codec=10,projection_region_batches=120,
        raw_identity_bit_slots=9600,coefficient_vote_slots=76800,local_slots=1440,
        final_slots=480,per_arm_final_slots=160,views=5,arms=3,queries=3,
        existing_time_nodes=822,existing_time_relations=817,independent_sources=1)


def recovery_layout():
    """Fixed metadata skeleton, saved in attempt before any logical operation."""
    layout=[]
    def slots():
        return [dict(slot=i,value=None,status='UNRESOLVED_RECEIPT',cause_set=['MISSING_OR_UNFINISHED_RECEIPT'],
                     selected_refs=[],supporting_refs=[],missing_refs=[],conflict_refs=[]) for i in range(32)]
    for view in VIEWS:
        for arm in ARMS:
            for q in range(3):
                layout.append(dict(kind='local',path=f'local/{view}/{arm}/q{q}.json',operation=f'{view}/{arm}/q{q}/read',
                    fallback=dict(view=view,arm=arm,query=q,planned_slots=32,slots=slots())))
            layout.append(dict(kind='final',path=f'final/{view}/{arm}.json',operation=f'{view}/{arm}/aggregate',
                fallback=dict(view=view,arm=arm,planned_slots=32,slots=slots())))
    raw={u['id']:dict(status='UNRESOLVED_RECEIPT',raw_bit_slots=160,vote_slots=1280,
                     encoder_input_support=list(range(u['start'],u['start']+41)),
                     operations=[f"{v}/q{u['query']}/p{u['phase']}/{op}" for op in ('E41','project','bits')])
         for v,n in zip(VIEWS,LENGTHS) for u in units(v,n)}
    layout.append(dict(kind='raw_bank',path='raw_bank.json',operation=None,fallback=raw))
    return layout


def recover_artifacts(output,layout,state):
    """Read only receipt outputs, never bit caches, truth or scientific code."""
    from copy import deepcopy
    recovered={}
    root=Path(output).resolve()
    for item in layout:
        value=deepcopy(item['fallback']);error=None;unconfirmed=None
        try:
            path=(root/item['path']).resolve()
            if not path.is_relative_to(root):raise ValueError('receipt path outside output')
            saved=load(path)
            if item['kind']=='raw_bank':
                if not isinstance(saved,dict):raise ValueError('raw metadata object required')
                for key in value:
                    fixed=value[key];row=saved.get(key)
                    statuses=[state[op].get('status') if isinstance(state.get(op),dict) else None for op in fixed['operations']]
                    known=('RETURNED' if statuses[-1]=='RETURNED' else
                        next((s for s in reversed(statuses) if s in ('FAILED','INTERRUPTED','RESOURCE_UNRESOLVED')),None))
                    if known is None:known='DEPENDENCY_UNRESOLVED' if all(s in ('NOT_STARTED','DEPENDENCY_UNRESOLVED') for s in statuses) else 'UNRESOLVED_RECEIPT'
                    valid=(isinstance(row,dict) and known!='UNRESOLVED_RECEIPT' and row.get('status')==known and
                        all(row.get(k)==fixed[k] for k in ('raw_bit_slots','vote_slots','encoder_input_support')))
                    if valid:value[key]=dict(row,**{k:fixed[k] for k in ('raw_bit_slots','vote_slots','encoder_input_support','operations')})
                    else:
                        fixed.update(operation_status=known,unconfirmed_raw_metadata=row)
                        error='raw metadata incomplete or unconfirmed; fixed unit fields retained'
            elif not isinstance(state.get(item['operation']),dict) or state[item['operation']].get('status')!='RETURNED':
                unconfirmed=saved;error='operation receipt not returned; saved artifact retained as unconfirmed'
            else:
                rows=saved.get('slots') if isinstance(saved,dict) else None
                if not isinstance(rows,list):raise ValueError('slot receipt list required')
                for i,row in enumerate(rows[:32]):
                    if (isinstance(row,dict) and type(row.get('slot')) is int and row['slot']==i
                        and 'value' in row and (row['value'] is None or type(row['value']) is int and row['value'] in (0,1))
                        and isinstance(row.get('status'),str)):
                        value['slots'][i]=row
                    else:error='invalid slot record retained as unresolved'
                if len(rows)!=32:error='incomplete slot receipt; fixed32 retained'
        except (OSError,ValueError,TypeError) as exc:error=str(exc)
        recovered[item['path']]=dict(kind=item['kind'],value=value,error=error,unconfirmed_artifact=unconfirmed)
    return recovered


def preflight(config,output):
    try:
        source=Path(config['source']).expanduser().resolve();output=Path(output).expanduser().resolve()
        if output==source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('output overlaps read-only c01 received source')
        if not isinstance(config['model_id'],str) or not config['model_id'].strip(): raise ValueError('model location required')
        paths=[source/'receiver/received'/v/name for v in VIEWS for name in ('submission.json','received.mp4')]
        return dict(status='PREFLIGHT_READY',input_paths=[str(p) for p in paths],protected_roots=[str(source)],
                    physical_attempt_reserved=False,budgets=budgets(),denominators=denominators())
    except (OSError,ValueError,KeyError,TypeError) as error:
        return dict(status='PREFLIGHT_UNRESOLVED',error=str(error),physical_calls_this_invocation=0)


class ResourceExceeded(RuntimeError): pass


class Journal:
    def __init__(self,output):
        self.output=Path(output);self.rows=plan();self.cpu_seconds=0.;self.encode_seconds=0.;self.check=None
        self.started=time.monotonic();self.non_cpu_seconds=0.
        self.save()
    def counts(self):
        counts={k:dict(planned=v,started=0,returned=0) for k,v in budgets().items()}
        for r in self.rows.values():
            for field in ('started','returned'): counts[r['kind']][field]+=r[field]
        return counts
    def save(self):
        dump(self.output/'state.json',self.rows);dump(self.output/'counts.json',self.counts())
    def call(self,key,fn):
        row=self.rows[key]
        if row['status']!='NOT_STARTED': raise RuntimeError('fixed operation already consumed: '+key)
        if time.monotonic()-self.started-self.non_cpu_seconds>1800:
            raise ResourceExceeded('CPU receiving/saving wall limit observed before next operation')
        if self.check: self.check()
        row.update(started=1,status='STARTED');self.save();start=time.monotonic()
        kind=row['kind']
        def exceeds(elapsed):
            if kind=='E41':return elapsed>600 or self.encode_seconds+elapsed>7200
            if kind=='model_load':return elapsed>900
            if kind.startswith('codec_'):return elapsed>300
            return time.monotonic()-self.started-self.non_cpu_seconds>1800
        try:
            value=fn();row['returned']=1
            elapsed=time.monotonic()-start
            if exceeds(elapsed):
                row['limit_observation']='AFTER_CALL_RETURN'
                raise ResourceExceeded('fixed resource time exceeded; call returned, no retry')
            if self.check:
                try:self.check()
                except Exception:
                    row['limit_observation']='AFTER_CALL_RETURN';raise
            row['status']='RETURNED';return value
        except BaseException as error:
            if isinstance(error,Exception) and type(error).__name__!='ResourceExceeded' and exceeds(time.monotonic()-start):
                error=ResourceExceeded(f'elapsed limit observed after failed call: {type(error).__name__}: {error}')
                row['limit_observation']='AFTER_CALL_FAILURE'
            row.update(status='RESOURCE_UNRESOLVED' if type(error).__name__=='ResourceExceeded' else
                       'FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}')
            if type(error).__name__=='ResourceExceeded' and 'limit_observation' not in row:
                row['limit_observation']='DURING_CALL'
            raise error
        finally:
            row['wall_seconds']=time.monotonic()-start;self.save()
            if kind=='E41':self.encode_seconds+=row['wall_seconds']
            if kind in ('E41','model_load','codec_probe','codec_decode'):self.non_cpu_seconds+=row['wall_seconds']
            self.cpu_seconds=time.monotonic()-self.started-self.non_cpu_seconds
    def terminalize(self,reason):
        for row in self.rows.values():
            if row['status']=='NOT_STARTED':row.update(status='DEPENDENCY_UNRESOLVED',error=reason)
            elif row['status']=='STARTED':row.update(status='INTERRUPTED',error=reason)
        self.save()


def recover(output):
    saved={};errors={};rows=plan()
    for name in ('attempt','state','counts','resources','blind_terminal','evaluation','result'):
        try:
            value=load(Path(output)/(name+'.json'))
            if not isinstance(value,dict):raise ValueError('saved receipt object required')
            saved[name]=value
        except (OSError,ValueError) as error:errors[name]=str(error)
    for key,row in rows.items():
        actual=saved.get('state',{}).get(key)
        if isinstance(actual,dict) and actual.get('status') in ('RETURNED','FAILED','INTERRUPTED','RESOURCE_UNRESOLVED','DEPENDENCY_UNRESOLVED'):
            row.update(actual)
        else:row.update(status='UNRESOLVED_RECEIPT',error='missing/unfinished receipt; consumed without resume')
    return dict(status='READ_ONLY_EXISTING_ATTEMPT',physical_calls_this_invocation=0,saved=saved,rows=rows,
                recovered_artifacts=recover_artifacts(output,recovery_layout(),saved.get('state',{})),
                recovery_scope='record-only; unresolved receipts are not rerun or evaluated',
                read_errors=errors,denominators=denominators())
