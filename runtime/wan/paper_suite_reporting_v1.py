"""Typed, fixed-slot views of already saved records. No receiver or post execution."""
from collections import Counter
from pathlib import Path
from main.tube_state import paper_suite_v1 as p
from . import paper_suite_records_v1 as rec

VERDICTS=('MATCH','MISMATCH','UNRESOLVED','INPUT_UNRESOLVED')


class SavedReader:
    def __init__(self):self.errors=[]
    def issue(self,path,reason):self.errors.append(dict(path=str(path),read_error=reason))
    def load(self,path,kind=dict):
        try:
            value=rec.read(path)
            if not isinstance(value,kind):raise ValueError(f'{kind.__name__} required, got {type(value).__name__}')
            if kind is list:
                out=[]
                for i,row in enumerate(value):
                    if isinstance(row,dict):out.append(row)
                    else:self.issue(path,f'row {i}: object required')
                return out
            return value
        except (OSError,ValueError,TypeError) as error:
            self.issue(path,f'{type(error).__name__}: {error}');return kind()
    def obj(self,value,path):
        if isinstance(value,dict):return value
        self.issue(path,'nested object required');return {}


def unique(rows,**fields):
    found=[r for r in rows if all(r.get(k)==v for k,v in fields.items())]
    return found[0] if len(found)==1 else {}


def fixed_positions(rows,n,key):
    return [unique(rows,**{key:i}) for i in range(n)]


def counts_for(rows,old=False):
    def status(r):
        if not old:return r.get('status') if r.get('status') in ('CORRECT','WRONG','UNRESOLVED') else 'TECHNICAL_UNKNOWN'
        return ('CORRECT' if r.get('original_correct') is True else 'WRONG' if r.get('original_wrong') is True
            else 'UNRESOLVED' if r.get('original_correct') is False and r.get('original_wrong') is False else 'TECHNICAL_UNKNOWN')
    return dict(Counter(map(status,rows)))


def payload_summary(rows,reader=None,path=None):
    fixed=fixed_positions(rows,32,'slot');bounds=[]
    fixed=[dict(r) for r in fixed]
    for i,r in enumerate(fixed):
        if not isinstance(r.get('status'),str) or not r['status']:
            if r and reader is not None:reader.issue(path,f'slot {i}: nonempty status string required')
            r['status']='TECHNICAL_UNKNOWN'
    for r in fixed:
        b=r.get('correct_supported_bounds')
        bounds.append(b if isinstance(b,list) and len(b)==2 and all(type(x) is int and x in (0,1) for x in b) and b[0]<=b[1] else [0,1])
    literal=[r.get('literal_correct') for r in fixed]
    observed=sum(type(r.get('value')) is int and r['value'] in (0,1) for r in fixed)
    erasures=sum(r.get('evaluability_reason') in ('EXECUTED_CONFLICT_ERASURE','EXECUTED_NO_FULL_CELL') for r in fixed)
    unknown=sum(a!=b for a,b in bounds)
    def full(values):return False if any(v is False for v in values) else True if all(v is True for v in values) else None
    return dict(planned_slots=32,observed_bits=observed,executed_erasures=erasures,
        literal_comparable=sum(type(v) is bool for v in literal),literal_errors=sum(v is False for v in literal),
        correct_supported_bounds=[sum(b[j] for b in bounds) for j in (0,1)],
        correct_supported=sum(b==[1,1] for b in bounds),support_unknown=unknown,
        complete_literal_message=full([False if r.get('evaluability_reason') in ('EXECUTED_CONFLICT_ERASURE','EXECUTED_NO_FULL_CELL') else r.get('literal_correct') for r in fixed]),
        complete_correct_attribution_message=full([bool(b[0]) if b[0]==b[1] else None for b in bounds]),
        wrong_attribution_observed=sum(r.get('wrong_attribution') is True for r in fixed),
        unsupported_output_observed=sum(r.get('unsupported_output') is True for r in fixed),
        slot_statuses=dict(Counter(r.get('status','TECHNICAL_UNKNOWN') for r in fixed)))


def query_tables(root,reader):
    root=Path(root);saved_claims=reader.load(root/'random_claims.json',list)
    saved_families=reader.load(root/'random_family.json',list);ops=reader.load(root/'operations.json')
    rows=[];families=[]
    for sid in p.SOURCES:
        for arm in p.cohort.ARMS:
            group=[]
            for view in p.VIEWS:
                for key in p.cohort.KEYS:
                    fields=dict(source=sid,arm=arm,view=view,key=key)
                    matches=[r for r in saved_claims if all(r.get(k)==v for k,v in fields.items())]
                    saved=matches[0] if len(matches)==1 else {}
                    op=reader.obj(ops.get(f'{sid}/{arm}/{view}/{key}',{}),root/'operations.json')
                    if not matches and op.get('status')=='RETURNED':
                        saved=reader.load(root/'claims'/sid/arm/view/(key+'.json'))
                    verdict=saved.get('verdict')
                    if verdict not in VERDICTS:verdict='INPUT_UNRESOLVED'
                    row=dict(fields,verdict=verdict,comparisons=saved.get('comparisons'),
                        execution_status='RETURNED' if saved.get('verdict') in VERDICTS else op.get('status','NOT_EXECUTED'),
                        read_status='SAVED_DECISION' if saved.get('verdict') in VERDICTS else 'TECHNICAL_UNKNOWN',
                        missing_reason=None if saved.get('verdict') in VERDICTS else 'missing, duplicate or invalid saved decision')
                    rows.append(row);group.append(row)
            saved=unique(saved_families,source=sid,arm=arm)
            event=p.family_event([True if r['verdict']=='MATCH' else None if r['verdict']=='INPUT_UNRESOLVED' else False for r in group],10)
            collision=saved.get('correct_query_collision') if type(saved.get('correct_query_collision')) is bool else None
            family=dict(p.nominal_family([None]*10),**{})
            for field in family:
                if field in saved:family[field]=saved[field]
            family.update(source=sid,arm=arm,family='ANY10_VIEWS_KEYS',event=event,
                correct_query_collision=False if arm=='OFF' else collision,
                false_accept_event=event if arm=='OFF' or collision is False else None,
                verdict_counts=dict(Counter(r['verdict'] for r in group)),
                nominal_status='SAVED_FAMILY' if saved else 'TECHNICAL_UNKNOWN')
            for field in ('false_query_probability_unconditional','false_query_probability_conditional'):
                family[field]=saved.get(field)
            families.append(family)
    summaries=[];panels=[]
    for panel,sids in [('ALL',p.SOURCES),('TEST-A',p.SOURCES[:4]),('TEST-B',p.SOURCES[4:])]:
        for arm in p.cohort.ARMS:
            group=[f for f in families if f['arm']==arm and f['source'] in sids]
            row=dict(panel=panel,arm=arm,family='ANY10_VIEWS_KEYS',
                **p.rate_summary([f['false_accept_event'] for f in group],[f['correct_query_collision'] is True for f in group]),
                query_collisions=sum(f['correct_query_collision'] is True for f in group),
                verdict_counts=dict(sum((Counter(f['verdict_counts']) for f in group),Counter())))
            panels.append(row)
            if panel=='ALL':summaries.append(row)
    return dict(main_random_claims=rows,main_random_family=families,
        main_random_source_statistics=summaries,main_random_panel_statistics=panels)
