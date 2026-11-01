"""Post-only private-message validation and layered finite-confirmation report."""
from pathlib import Path
from collections import Counter
from . import b_new_source_chain_confirmation_v1_records as records


def private_message(source,output):
    """Called only after the whole-package blind terminal is durably saved."""
    terminal=records.load(Path(output)/'blind_terminal.json')
    if terminal.get('status')!='ALL_CHAIN_BLIND_SLOTS_TERMINAL':raise ValueError('whole blind terminal required')
    path=Path(source)/'protocol.json'
    protocol=records.load(path);config=protocol['config']
    writer=config['writer'];bits=writer['bits']
    if not isinstance(bits,list) or len(bits)!=32 or any(type(b) is not int or b not in (0,1) for b in bits):
        raise ValueError('original config.writer.bits must contain 32 binary integers; no message fallback')
    if writer.get('key')!='watermark':raise ValueError('original DCT key must be watermark')
    generation=config['generation']
    for key,expected in dict(height=320,width=512,frames=181,fps=8).items():
        if type(generation.get(key)) not in (int,float) or generation[key]!=expected:
            raise ValueError('original generation geometry mismatch: '+key)
    sources=config['sources']
    if not isinstance(sources,list) or any(not isinstance(s,dict) for s in sources):raise ValueError('original sources list required')
    matched=[s for s in sources if s.get('id')=='b2-m3-c02']
    if len(matched)!=1:raise ValueError('one original c02 source record required')
    fixture=dict(message_bits=bits,message_source=str(path)+'#config.writer.bits',source_record=matched[0],
                 generation=generation,key=writer['key'],management_fields_are_reader_inputs=False)
    records.dump(Path(output)/'private/post_fixture.json',fixture)
    return fixture


def time_report(evaluation,rows):
    counts={k:Counter({s:0 for s in ('CORRECT','WRONG','UNRESOLVED','NOT_EXECUTED','UNKNOWN')}) for k in ('nodes','relations')}
    views={};bad=[]
    for view,n in zip(records.payload.VIEWS,records.payload.LENGTHS):
        value=(evaluation or {}).get('views',{}).get(view,{})
        operation=rows.get(view+'/submit',{})
        unstarted=records.operation_not_executed(operation)
        view_counts={}
        for kind,total in (('nodes',n),('relations',n-1)):
            data=value.get(kind,[]);counter=Counter({s:0 for s in counts[kind]})
            for i in range(total):
                row=data[i] if i<len(data) else {}
                statuses=[row.get('old_status'),row.get('new_status')]
                if 'WRONG' in statuses:
                    status='WRONG';bad.append(dict(view=view,kind=kind,coordinate=i,old_status=statuses[0],new_status=statuses[1]))
                elif unstarted:status='NOT_EXECUTED'
                elif 'UNKNOWN' in statuses:status='UNKNOWN'
                elif 'CORRECT' in statuses:status='CORRECT'
                elif statuses==['UNRESOLVED','NOT_ADDED']:status='UNRESOLVED'
                else:status='UNKNOWN'
                counter[status]+=1
            counts[kind].update(counter);view_counts[kind]=dict(counter)
        # A missing/corrupt file is incomplete, not proof that preservation was contradicted.
        if 'error' not in value and value.get('preservation') is False:
            bad.append(dict(view=view,kind='OLD_OUTPUT_PRESERVATION',details='saved comparison is false'))
        for error in value.get('structural_errors',[]):bad.append(dict(view=view,kind='STRUCTURE',details=error))
        views[view]=dict(counts=view_counts,error=value.get('error'),preservation=value.get('preservation'),
                         candidate_attrition=value.get('candidate_attrition'))
    return dict(counts={k:dict(v) for k,v in counts.items()},views=views,counterexamples=bad,
                evidence_complete=bool(evaluation and evaluation.get('evidence_complete')),
                startup_source_0_to_3_may_remain_unresolved=True,unresolved_is_not_recovered=True)


def effect_report(evaluation):
    views={};correction=[];new_support=[];attribution=[];other_support=[]
    for view in records.payload.VIEWS:
        data=(evaluation or {}).get('views',{}).get(view)
        if not data:views[view]=dict(status='UNKNOWN');continue
        b=data['arms']['B'];u=data['arms']['U'];comparison=dict(data['B_vs_U'])
        added=comparison['added_correct_support']
        new=[i for i in added if u['rows'][i]['value'] is None]
        attr=[i for i in added if u['rows'][i]['literal_correct'] is True and b['rows'][i]['literal_correct'] is True
              and u['rows'][i]['wrong_attribution']]
        remaining=[i for i in added if i not in new and i not in attr]
        comparison.update(B_output_count=b['output_count'],U_output_count=u['output_count'],
                          new_output_correct_support=new,attribution_only_correction=attr,other_added_correct_support=remaining)
        views[view]=comparison
        if view!='FULL':
            correction.extend((view,i) for i in comparison['wrong_to_correct'])
            new_support.extend((view,i) for i in new);attribution.extend((view,i) for i in attr)
            other_support.extend((view,i) for i in remaining)
    status=('COMMON_BIT_CORRECTION_OBSERVED' if correction else 'NEW_CORRECT_OUTPUT_SUPPORT_OBSERVED' if new_support
            else 'TIME_ATTRIBUTION_OR_SUPPORT_IMPROVEMENT' if attribution or other_support else 'NO_DISCRIMINATING_EFFECT_OBSERVED')
    return dict(status=status,views=views,edited_view_common_corrections=correction,edited_view_new_output_support=new_support,
                edited_view_attribution_only=attribution,edited_view_other_added_support=other_support,
                effect_is_separate_from_payload_transfer=True,W_zero_support_is_definitional=True)


def summarize(time_evaluation,payload_evaluation,time_rows,stage_results,errors):
    timing=time_report(time_evaluation,time_rows);payload=payload_evaluation or {}
    bad=timing['counterexamples']+[dict(kind='PAYLOAD',**r) for r in payload.get('B_counterexamples',[])]
    per_view={v:payload.get('views',{}).get(v,{}).get('arms',{}).get('B',{}) for v in records.payload.VIEWS}
    full=all(r.get('full32_supported') is True and r.get('correct_supported')==32 and
             r.get('group_support')==[True]*4 for r in per_view.values())
    supported=sum(r.get('correct_supported',0) for r in per_view.values())
    complete=bool(timing['evidence_complete'] and payload.get('evidence_complete') and not errors)
    status=('COUNTEREXAMPLE' if bad else 'EVIDENCE_INCOMPLETE' if not complete else
            'FINITE_FIVE_VIEW_PAYLOAD_TRANSFER_CONFIRMED' if full else 'PARTIAL_SUPPORTED_READBACK' if supported else 'NO_SUPPORTED_READBACK')
    return dict(status=status,evidence_complete=complete,counterexamples=bad,
        payload_transfer=dict(full_five_view_target_observed=full,correct_supported=supported,fixed_B_slots=160,
            per_view={v:dict(correct_supported=r.get('correct_supported'),full32_supported=r.get('full32_supported'),
                            output_count=r.get('output_count'),group_support=r.get('group_support')) for v,r in per_view.items()},
            missing_evaluation=payload_evaluation is None),time=timing,synchronization_effect=effect_report(payload_evaluation),
        materialization=dict(native_status=stage_results.get('time_chain',{}).get('materialization_native_status'),
            errors=stage_results.get('time_chain',{}).get('materialization_errors'),
            exact_native_target_failure_retained=True,solver_optimality_proven=False,quality_proven=False),
        rejection_capabilities=[dict(unit='time position/edge',rule='existing hard qualification, graph and inversion',new_threshold=False),
            dict(unit='ref',rule='existing exact four-frame geometry',new_threshold=False),
            dict(unit='slot',rule='existing CDC conflict erasure / technical unread',new_threshold=False)],
        not_evaluated=['existence/null/FPR','hidden wrong time','same-group position authentication','source identity','general reliability'],
        existence_test='NOT_EVALUATED',source_authenticated=False,scientific_pass=None,
        source_scope='one frozen c02 next-content confirmation; old CAL content, not globally unseen or all-component independent holdout',
        historical_failures_retained='F18/F19/F22/F24/F31/F32/F33/F34; no new remedy claimed',
        denominators=records.denominators(),errors=errors,post_physical_observations_added=0)
