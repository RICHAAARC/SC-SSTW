"""Fixed67 reservation and stdlib-only recovery of diagnostic records."""
from pathlib import Path
from collections import Counter
import time
from .b_minimal_l_v1_records import load,dump,output_state

VIEWS=('FULL','CROP','DELETE1','REPEAT1','DELETE3')
LENGTHS=(181,101,180,182,178)
LAYERS=('preclip','clip','rgb8','h264')
ROLES=('F','R0','R1')
LIMITS=dict(rss_bytes=8*1024**3,total_wall_seconds=6300,codebook_seconds=300,per_raw_seconds=600)


def units():
    return [dict(id='received/'+v,frames=n,kind='received',raw_path=f'time_chain/receiver/received/{v}/raw.npz',
                 saved_path=f'time_chain/receiver/received/{v}') for v,n in zip(VIEWS,LENGTHS)]+[
        dict(id='material/'+v,frames=181,kind='material',raw_path=f'time_chain/materialization/B/raw/QIM/{v}.npz') for v in LAYERS]


def plan():
    rows={}
    def add(k,kind):rows[k]=dict(kind=kind,planned=1,started=0,returned=0,status='NOT_STARTED')
    add('codebook','codebook')
    for unit in units():
        uid=unit['id'];add(uid+'/raw_load','raw_load')
        if unit['kind']=='received':add(uid+'/saved_view_load','saved_view_load')
        for op in ('arithmetic','fusion'):add(uid+'/'+op,op)
        for role in ROLES:add(uid+'/'+role+'/compatibility','compatibility')
        if unit['kind']=='received':add(uid+'/path','path')
    add('blind_save','blind_save');add('post','post');return rows


def budgets():
    return dict(Counter(r['kind'] for r in plan().values()))


def denominators():
    return dict(logical_operations=67,received=dict(raws=5,positions=822,edges=817,lanes=4110,raw_bits=517860,comparisons_upper_bound=228529152),
        material=dict(raws=4,positions=724,lanes=3620,raw_bits=456120,graphs=0,comparisons_upper_bound=201283584),
        total=dict(raws=9,positions=1546,lanes=7730,raw_bits=973980,lane_roles=23190,bit_decisions=1460970,comparisons_upper_bound=429812736),
        domain=92672,block=1024,physical_observations_added=0,
        forbidden_budget=dict(model=0,GPU=0,E41=0,codec=0,DiT=0,writer=0,solver=0,K=0,KT=0,new_C=0,new_L=0,new_G=0,payload_read=0,payload_post=0))


def skeleton():
    return {u['id']:dict(**u,roles={r:dict(status='UNRESOLVED_RECEIPT',lane_slots=u['frames']*5,
        bit_slots=u['frames']*5*63) for r in ROLES},positions=u['frames'],edges=u['frames']-1 if u['kind']=='received' else 0) for u in units()}


def preflight(config,output):
    try:
        source=Path(config['source_attempt']).expanduser().resolve();output=Path(output).expanduser().resolve()
        for target in (output,output.with_name(output.name+'-logs'),output.with_name(output.name+'-entry-cache')):
            if target==source or target.is_relative_to(source) or source.is_relative_to(target):raise ValueError('output/cache/log overlaps source attempt')
        return dict(status='PREFLIGHT_READY',protected_roots=[str(source)],input_paths=[str(source/u['raw_path']) for u in units()],
                    budgets=budgets(),denominators=denominators(),physical_attempt_reserved=False)
    except (OSError,ValueError,KeyError,TypeError) as error:return dict(status='PREFLIGHT_UNRESOLVED',error=str(error),computations_this_invocation=0)


def recover(output):
    saved={};errors={};rows=plan()
    for name in ('attempt','state','counts','resources','blind_terminal','evaluation_summary','post_progress','result'):
        try:
            value=load(Path(output)/(name+'.json'))
            if not isinstance(value,dict):raise ValueError('saved object required')
            saved[name]=value
        except (OSError,ValueError) as error:errors[name]=str(error)
    for key,row in rows.items():
        actual=saved.get('state',{}).get(key)
        if valid_receipt(actual,row):
            row.update(actual)
            if row['status']=='STARTED':row.update(status='UNFINISHED_RECEIPT',error='started receipt; return not durably recorded; consumed with no resume')
        else:row.update(status='UNRESOLVED_RECEIPT',started=None,returned=None)
    return dict(status='READ_ONLY_EXISTING_DIAGNOSTIC',computations_this_invocation=0,post_calls_this_invocation=0,
        saved=saved,rows=rows,fixed_slots=skeleton(),read_errors=errors,denominators=denominators())


def valid_receipt(actual,planned):
    if not isinstance(actual,dict) or actual.get('kind')!=planned['kind']:return False
    if actual.get('status') not in ('RETURNED','FAILED','INTERRUPTED','RESOURCE_UNRESOLVED','DEPENDENCY_UNRESOLVED','STARTED'):return False
    if any(type(actual.get(k)) is not int or actual[k] not in (0,1) for k in ('started','returned')):return False
    if actual['returned']>actual['started']:return False
    if actual['status']=='RETURNED' and (actual['started'],actual['returned'])!=(1,1):return False
    if actual['status']=='DEPENDENCY_UNRESOLVED' and (actual['started'],actual['returned'])!=(0,0):return False
    if actual['status']=='STARTED' and (actual['started'],actual['returned'])!=(1,0):return False
    return True


def summary(result,output):
    """Metadata-only compact display, also usable from the cached stdlib entry."""
    snapshot=recover(output)
    saved=snapshot['saved'];saved.update(result.get('saved',{}))
    diagnostic=saved.get('evaluation_summary') or saved.get('post_progress') or {}
    resource=saved.get('resources',{});completion_errors=resource.get('completion_errors') or {}
    recorded_status=result.get('evaluation_status',saved.get('result',{}).get('evaluation_status'))
    def aggregate(groups):
        total=Counter()
        if isinstance(groups,dict):
            for group in groups.values():
                if not isinstance(group,dict):continue
                for name,values in group.items():
                    if isinstance(values,dict):
                        for status,count in values.items():
                            if type(count) is int:total[name+'/'+status]+=count
        return dict(total)
    def bounded(value):return {str(k)[:80]:str(v)[:240] for k,v in list(value.items())[:8]} if isinstance(value,dict) else {}
    return dict(status=result.get('status'),output=str(output),
        evaluation_status='EVIDENCE_INCOMPLETE' if completion_errors or resource.get('error') else recorded_status,recorded_evaluation_status=recorded_status,
        computations_this_invocation=result.get('computations_this_invocation'),post_calls_this_invocation=result.get('post_calls_this_invocation'),
        fixed=dict(operations=67,raws=9,received_nodes=822,received_edges=817,auxiliary_positions=724,lane_roles=23190,bit_decisions=1460970),
        counts={k:{f:v.get(f) for f in ('planned','started','returned')} for k,v in saved.get('counts',{}).items() if isinstance(v,dict)},
        resources={k:resource.get(k) for k in ('peak_resident_bytes','total_wall_seconds','error','completion_errors')},
        errors=bounded(result.get('errors',saved.get('result',{}).get('errors',{}))),read_errors=bounded(snapshot['read_errors']),
        receipt_statuses=dict(Counter(row['status'] for row in snapshot['rows'].values())),
        saved_diagnostic=dict(source='evaluation_summary' if saved.get('evaluation_summary') else 'post_progress' if saved.get('post_progress') else None,
            status=diagnostic.get('status'),lane_findings=diagnostic.get('known_lane_findings',{}),
            node_counts=aggregate(diagnostic.get('node_status',{})),edge_counts=aggregate(diagnostic.get('edge_status',{})),
            event_counts=aggregate({'views':diagnostic.get('events',{})})),
        full_records='Full records remain in OUTPUT; metadata-only display does not load NPZ, truth or post.')


class Journal:
    def __init__(self,output,check):self.output=Path(output);self.rows=plan();self.check=check;self.save()
    def save(self):
        dump(self.output/'state.json',self.rows)
        counts={k:dict(planned=v,started=0,returned=0) for k,v in budgets().items()}
        for row in self.rows.values():
            for k in ('started','returned'):counts[row['kind']][k]+=row[k]
        dump(self.output/'counts.json',counts)
    def call(self,key,fn):
        row=self.rows[key]
        if row['status']!='NOT_STARTED':raise RuntimeError('fixed item already consumed')
        start=time.monotonic()
        try:
            self.check();row.update(status='STARTED',started=1);self.save()
            value=fn();row['returned']=1
            try:self.check()
            except Exception:
                row['limit_observation']='AFTER_CALL_RETURN';raise
            row['status']='RETURNED';return value
        except BaseException as error:
            row.update(status='RESOURCE_UNRESOLVED' if type(error).__name__ in ('ResourceExceeded','RawLimitExceeded') else
                       'FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}')
            raise
        finally:row['wall_seconds']=time.monotonic()-start;self.save()
    def terminalize(self,include_post=False):
        for key,row in self.rows.items():
            if not include_post and key in ('blind_save','post'):continue
            if row['status']=='NOT_STARTED':row.update(status='DEPENDENCY_UNRESOLVED',error='fixed attempt ended; no refill or resume')
            elif row['status']=='STARTED':row.update(status='INTERRUPTED',error='unfinished fixed item')
        self.save()
