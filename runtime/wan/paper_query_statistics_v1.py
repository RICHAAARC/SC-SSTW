"""One new CPU query protocol on saved evidence. No decoding, old post or truth selection."""
from collections import Counter
from pathlib import Path
import os
from main.tube_state import paper_suite_v1 as p
from main.tube_state import qmatch_v1 as qmatch
from . import paper_suite_records_v1 as rec


def unresolved(unit,reason):
    return dict(status='INPUT_UNRESOLVED',unit=unit,input_errors=[dict(reason=reason)],slots=[])


def run(cohort_root=p.COHORT_ROOT,output=p.QUERY_ROOT,random_bytes=os.urandom):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        return dict(status='READ_ONLY',saved=rec.optional(output/'result.json'),output=str(output))
    pending=[]
    for sid in p.SOURCES:
        source=Path(cohort_root)/sid/'attempt01'
        result=rec.optional(source/'result.json');attempt=rec.optional(source/'attempt.json')
        terminal={'COMPLETED','STOPPED_FIXED_ATTEMPT'}
        if not any(isinstance(value,dict) and value.get('status') in terminal for value in (result,attempt)):
            pending.append(sid)
    if pending:return dict(status='DEPENDENCY_PENDING',pending_sources=pending,output=str(output))
    if not rec.reserve(output,dict(budget=p.QUERY_BUDGET,receiver='unchanged QMatch four-state')):
        return dict(status='READ_ONLY',saved=rec.optional(output/'result.json'),output=str(output))
    plan={f'{sid}/{arm}/{view}/{key}':dict(kind='claim',status='NOT_STARTED',started=0,returned=0)
        for sid in p.SOURCES for arm in p.cohort.ARMS for view in p.VIEWS for key in p.cohort.KEYS}
    journal=rec.Journal(output,plan);rows=[];families=[];failure=None
    try:
        queries={}
        for sid in p.SOURCES:
            for arm in p.cohort.ARMS:
                data=random_bytes(4)
                if len(data)!=4:raise ValueError('uniform32 random source returned wrong length')
                queries[sid+'/'+arm]=[(b>>shift)&1 for b in data for shift in range(7,-1,-1)]
        rec.write(output/'queries.json',queries)
        for sid in p.SOURCES:
            source=Path(cohort_root)/sid/'attempt01'
            terminal=rec.optional(source/'blind_terminal.json',{})
            if not isinstance(terminal,dict):terminal={}
            for a,arm in enumerate(p.cohort.ARMS):
                evidences=[];family_rows=[]
                for i,view in enumerate(p.VIEWS):
                    for key in p.cohort.KEYS:
                        uid=f'input{a*5+i}';unit=dict(source=sid,input_id=uid,arm=arm,view=view,key_id=key)
                        ev=rec.optional(source/'evidence'/uid/(key+'.json'))
                        if terminal.get('status')!='ALL_BLIND_TERMINAL':ev=unresolved(unit,'saved blind closure unavailable')
                        if p.evidence_word(ev)[1]=='UNKNOWN':ev=unresolved(unit,'required frozen evidence unavailable or invalid')
                        evidences.append(ev);address=f'{sid}/{arm}/{view}/{key}'
                        def call():
                            value=qmatch.match_claim(ev,'random32',queries[sid+'/'+arm])
                            rec.write(output/'claims'/sid/arm/view/(key+'.json'),value);return value
                        value=journal.call(address,call)
                        row=dict(source=sid,arm=arm,view=view,key=key,verdict=value['verdict'],comparisons=value['comparisons'])
                        rows.append(row);family_rows.append(row)
                info=p.nominal_family(evidences)
                values=[True if row['verdict']=='MATCH' else None if row['verdict']=='INPUT_UNRESOLVED' else False for row in family_rows]
                families.append(dict(source=sid,arm=arm,family='ANY10_VIEWS_KEYS',event=p.family_event(values,10),
                    verdict_counts=dict(Counter(row['verdict'] for row in family_rows)),**info))
        rec.write(output/'blind_terminal.json',dict(status='ALL_RANDOM_QUERY_DECISIONS_SAVED',rows=rows,families=families))
        # Actual writer information is opened only after all query decisions have been saved.
        for family in families:
            arm,sid=family['arm'],family['source'];actual=None
            if arm!='OFF':
                writer=rec.optional(Path(cohort_root)/sid/'attempt01/private/writer.json',{})
                bits=writer.get('bits') if isinstance(writer,dict) else None
                if isinstance(bits,list) and len(bits)==32 and all(type(v) is int and v in (0,1) for v in bits):actual=''.join(map(str,bits))
            query=''.join(map(str,queries[sid+'/'+arm]))
            family['correct_query_collision']=None if arm!='OFF' and actual is None else arm!='OFF' and query==actual
            family['false_accept_event']=family['event'] if arm=='OFF' or actual is not None and query!=actual else None
            family['false_query_probability_unconditional']=None if family['unknown_inputs'] or arm!='OFF' and actual is None else len(set(family['known_words'])-({actual} if actual else set()))/2**32
            family['false_query_probability_conditional']=None if actual is None or family['unknown_inputs'] else len(set(family['known_words'])-{actual})/(2**32-1)
        summaries=[]
        for arm in p.cohort.ARMS:
            group=[f for f in families if f['arm']==arm]
            summaries.append(dict(arm=arm,family='ANY10_VIEWS_KEYS',**p.rate_summary([f['false_accept_event'] for f in group],
                [f['correct_query_collision'] is True for f in group]),
                query_collisions=sum(f['correct_query_collision'] is True for f in group),
                verdict_counts=dict(sum((Counter(f['verdict_counts']) for f in group),Counter()))))
        rec.table(output,'random_family',families);rec.table(output,'random_claims',rows);rec.table(output,'random_source_statistics',summaries)
    except BaseException as error:
        failure=f'{type(error).__name__}: {error}'
        rec.write(output/'failure.json',dict(error=failure))
        if isinstance(error,(KeyboardInterrupt,InterruptedError)):raise
    finally:
        result=dict(status='COMPLETED' if not failure else 'STOPPED_FIXED_ATTEMPT',failure=failure,
            budget=p.QUERY_BUDGET,claims_returned=len(rows),actual_bit_comparisons=sum(r['comparisons'] for r in rows),
            output=str(output),population_fpr_certified=False)
        rec.write(output/'result.json',result)
    return result
