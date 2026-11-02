"""Fixed37 metadata and read-only recovery. No numerical or media dependency."""
from pathlib import Path
from collections import Counter
from . import b_time_layer_diagnostic_v1_records as prior
from .b_minimal_l_v1_records import load,dump,output_state

VIEWS=prior.VIEWS
LENGTHS=prior.LENGTHS
UNITS=(('SOURCE',181),)+tuple(zip(VIEWS,LENGTHS))
LIMITS=dict(rss_bytes=8*1024**3,total_wall_seconds=6300,codec_seconds=300,reader_seconds=600)


def plan():
    kinds={view+'/old_public_load':'old_public_load' for view in VIEWS}
    kinds['shared_codebook']='shared_codebook'
    for uid,n in UNITS:
        for op in ('probe','native_Y_decode','Y_reader','F_compatibility'):kinds[uid+'/'+op]=op
        if uid!='SOURCE':kinds[uid+'/original_interval_paths']='original_interval_paths'
    kinds.update(blind_save='blind_terminal_save',post='post')
    return {k:dict(kind=v,planned=1,started=0,returned=0,status='NOT_STARTED') for k,v in kinds.items()}


def budgets():return dict(Counter(row['kind'] for row in plan().values()))
def denominators():
    return dict(operations=37,media=6,old_public_files=5,decoded_frames=1003,reader_frames=1003,
        lanes=5015,coefficient_bit_slots=631890,comparisons_upper_bound=92950016,
        source_control=dict(frames=181,lanes=905,graphs=0),
        received=dict(views=5,nodes=822,edges=817,lanes=4110),
        forbidden_budget=dict(encode=0,GPU=0,model=0,VAE=0,E41=0,writer=0,solver=0,training=0,C=0,L=0,G=0,payload=0),
        independent_new_sources=0)


def skeleton():
    return dict(scope='FIXED_DENOMINATORS_NOT_EXECUTION_STATUS',units={uid:dict(frames=n,lanes=n*5,coefficient_slots=n*630,
        comparisons_upper_bound=n*92672,nodes=n if uid!='SOURCE' else 0,edges=n-1 if uid!='SOURCE' else 0) for uid,n in UNITS})


def media_path(config,uid):
    root=Path(config['source_chain'])
    return root/('time_chain/materialization/B/QIM.mp4' if uid=='SOURCE' else 'time_chain/receiver/received/'+uid+'/received.mp4')


def input_paths(config):
    return [media_path(config,u) for u,n in UNITS]+[Path(config['source_layer'])/'received'/v/'saved_public.json' for v in VIEWS]


def preflight(config,output):
    try:
        roots=[Path(config[k]).expanduser().resolve() for k in ('source_chain','source_layer')]+[Path(config['post_source_baseline']).expanduser().resolve().parent];output=Path(output).expanduser().resolve()
        for target in (output,output.with_name(output.name+'-logs'),output.with_name(output.name+'-entry-cache')):
            if any(target==source or target.is_relative_to(source) or source.is_relative_to(target) for source in roots):raise ValueError('output/cache/log overlaps protected input')
        return dict(status='PREFLIGHT_READY',protected_roots=[str(x) for x in roots],input_paths=[str(x) for x in input_paths(config)],budgets=budgets(),denominators=denominators(),physical_attempt_reserved=False)
    except (KeyError,OSError,TypeError,ValueError) as error:return dict(status='PREFLIGHT_UNRESOLVED',error=str(error),computations_this_invocation=0)


def recover(output):
    saved={};errors={};rows=plan()
    for name in ('attempt','state','counts','resources','blind_terminal','blind_summary','post_progress','evaluation_summary','result'):
        try:
            value=load(Path(output)/(name+'.json'))
            if not isinstance(value,dict):raise ValueError('saved object required')
            saved[name]=value
        except (OSError,ValueError) as error:errors[name]=str(error)
    for key,row in rows.items():
        actual=saved.get('state',{}).get(key)
        if prior.valid_receipt(actual,row):
            row.update(actual)
            if row['status']=='STARTED':row.update(status='UNFINISHED_RECEIPT',error='started; no durable return; consumed, no resume')
        else:row.update(status='UNRESOLVED_RECEIPT',started=None,returned=None)
    return dict(status='READ_ONLY_EXISTING_DIAGNOSTIC',computations_this_invocation=0,post_calls_this_invocation=0,
        saved=saved,read_errors=errors,rows=rows,fixed_slots=skeleton(),denominators=denominators())


def summary(result,output):
    snapshot=recover(output);saved=snapshot['saved'];saved.update(result.get('saved',{}))
    resource=saved.get('resources',{});diagnostic=saved.get('evaluation_summary') or saved.get('post_progress') or {}
    recorded=result.get('evaluation_status',saved.get('result',{}).get('evaluation_status'))
    def bounded(value):return {str(k)[:90]:str(v)[:250] for k,v in list(value.items())[:8]} if isinstance(value,dict) else {}
    return dict(status=result.get('status'),output=str(output),computations_this_invocation=result.get('computations_this_invocation'),post_calls_this_invocation=result.get('post_calls_this_invocation'),
        evaluation_status='EVIDENCE_INCOMPLETE' if not resource or 'resources' in snapshot['read_errors'] or resource.get('error') or resource.get('completion_errors') else recorded,recorded_evaluation_status=recorded,
        fixed=denominators(),counts=saved.get('counts'),receipt_statuses=dict(Counter(r['status'] for r in snapshot['rows'].values())),
        resources={k:resource.get(k) for k in ('peak_resident_bytes','total_wall_seconds','error','completion_errors')},
        errors=bounded(result.get('errors',saved.get('result',{}).get('errors',{}))),read_errors=bounded(snapshot['read_errors']),
        blind=saved.get('blind_summary'),findings=diagnostic.get('counts'),post_status=diagnostic.get('status'),post_source='evaluation_summary' if saved.get('evaluation_summary') else 'post_progress',
        claim_ceiling='fixed native Y shadow H; no formal replacement, L ranking, identity/FPR or scientific PASS')


class Journal(prior.Journal):
    def __init__(self,output,check):self.output=Path(output);self.rows=plan();self.check=check;self.save()
    def save(self):
        dump(self.output/'state.json',self.rows)
        counts={k:dict(planned=v,started=0,returned=0) for k,v in budgets().items()}
        for row in self.rows.values():
            for field in ('started','returned'):counts[row['kind']][field]+=row[field]
        dump(self.output/'counts.json',counts)
