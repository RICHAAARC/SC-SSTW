"""Single adjacent-origin correction candidate; no truth, source identity or payload inputs."""
from copy import deepcopy
from types import FunctionType
from main.tube_state import b_rank3r_complete_repeat_v1 as rank
from main.tube_state import b_rank3_four_source_confirmation_v1 as original
from main.tube_state import b_m3_time_path_v1 as geometry
from main.tube_state import b_cdc32_aggregation_v1 as cdc

WINDOWS=rank.WINDOWS
NAME='RANK3R_ADJACENT_ORIGIN_V1'


def unique_second(family):
    top=family['selected']
    rest=[r for r in family['members'] if (r['phase'],r['source_start'])!=(top['phase'],top['source_start'])]
    score=max(r['S'] for r in rest)
    winners=[r for r in rest if r['S']==score]
    return dict(status='UNIQUE_SECOND' if len(winners)==1 else 'SECOND_TIE',score=score,
        winners=[dict(r,origin=geometry.state_to_origin(r['phase'],r['source_start'])) for r in winners])


def select(families):
    """The original triple is checked once; at most three one-window alternatives."""
    base=rank.select(families,'RANK3R')
    result=dict(selection=deepcopy(base),original_selection=deepcopy(base),
        correction=dict(entered=False,status='ORIGINAL_BRANCH_PRESERVED',alternatives=[],legal_count=0,
            selected_window=None,minimum_delta=None,second_legal_delta=None,delta_gap=None),
        membership_checks=int(base['status'] in ('DETECT','REJECT_GEOMETRY')))
    if base['status']!='REJECT_GEOMETRY':return result
    audit=result['correction'];audit['entered']=True
    for window in WINDOWS:
        second=unique_second(families[window]);row=dict(window=window,**second,adjacent=None,legal=None,delta=None,origins=None)
        if second['status']=='UNIQUE_SECOND':
            chosen=second['winners'][0];origins=list(base['origins']);origins[list(WINDOWS).index(window)]=chosen['origin']
            adjacent=abs(chosen['origin']-base['selected'][window]['origin'])==1
            legal=False
            if adjacent:
                legal=tuple(origins) in rank.PATHS
                result['membership_checks']+=1
            row.update(adjacent=adjacent,legal=legal,origins=origins,
                delta=families[window]['selected']['S']-chosen['S'],selected=chosen)
        audit['alternatives'].append(row)
    legal=[r for r in audit['alternatives'] if r['legal'] is True];audit['legal_count']=len(legal)
    if not legal:audit['status']='NO_LEGAL_ALTERNATIVE';return result
    ordered=sorted(legal,key=lambda r:r['delta']);minimum=ordered[0]['delta']
    audit.update(minimum_delta=minimum,second_legal_delta=ordered[1]['delta'] if len(ordered)>1 else None,
        delta_gap=ordered[1]['delta']-minimum if len(ordered)>1 else None)
    if len([r for r in legal if r['delta']==minimum])!=1:
        audit['status']='EXACT_MINIMUM_DELTA_TIE'
        result['selection']=dict(base,status='METHOD_AMBIGUITY_ERASURE',detected=False)
        return result
    chosen=ordered[0];selected=deepcopy(base['selected']);selected[chosen['window']]=deepcopy(chosen['selected'])
    result['selection']=dict(base,status='DETECT',detected=True,selected=selected,origins=chosen['origins'],
        path_class=deepcopy(rank.PATHS[tuple(chosen['origins'])]))
    audit.update(status='UNIQUE_LEGAL_ALTERNATIVE' if len(legal)==1 else 'MULTIPLE_LEGAL_MINIMUM_DELTA',selected_window=chosen['window'])
    return result


def _selected_family(prepared):
    """Validate the real rank-1/rank-2 selection without changing any H137 score."""
    members=prepared['members'];checked=original.check_family(dict(status='SCORES_COMPLETE',candidates=members))
    if checked['status']!='UNIQUE_ARGMAX':raise ValueError('corrected plan requires complete unique first rank')
    rank_number=prepared['selected_rank'];selected=prepared['selected']
    if rank_number==1:expected=checked['selected']
    elif rank_number==2:
        second=unique_second(checked)
        if second['status']!='UNIQUE_SECOND':raise ValueError('corrected second rank is not unique')
        expected=second['winners'][0]
    else:raise ValueError('only unchanged first rank or unique second rank')
    if any(selected[k]!=expected[k] for k in ('phase','source_start','S','origin')):raise ValueError('selected row contradicts its actual saved score/rank')
    return members,[selected],checked['maximum'],None,True,'READY'

_selected_builder=FunctionType(cdc.plan_local.__code__,dict(cdc.plan_local.__globals__,_family=_selected_family),
    name='adjacent_origin_plan',argdefs=cdc.plan_local.__defaults__)


def plan_task(key_id,families,views,selection_record=None):
    if key_id not in ('key00','key01'):raise ValueError('key ID only; no source/arm/fixture labels')
    record=select(families) if selection_record is None else selection_record
    selection=record['selection'];changed=record['correction']['selected_window'];plans=[]
    for window,w in WINDOWS.items():
        ids=views[window][0]
        prepared=dict(query_id=ids['query_id'],input_id=ids['input_id'],key_id=key_id,
            received_query_start=w,members=families[window]['members'],receiver='RANK3R',gate=selection['status'],
            selected=selection.get('selected',{}).get(window),status='READY' if selection['detected'] else selection['status'],
            score_status='SCORES_COMPLETE' if not families[window]['issues'] else 'SCORE_UNRESOLVED')
        if changed is None:
            plan=original.plan_singleton_local(prepared,views[window])
        else:
            prepared['selected_rank']=2 if window==changed else 1
            plan=_selected_builder(prepared,views[window])
            plan.update(mapping_scope=NAME+'_SINGLETON_C',selection_gate=selection['status'],threshold_used=False,
                selected_rank=prepared['selected_rank'],saved_scores_unmodified=True)
            plan['singleton_C']=plan.pop('full_local_C')
        plan['window']=window;plans.append(plan)
    return dict(selection=selection,original_selection=record['original_selection'],locals=cdc.reconcile_plans(plans),
        correction=record['correction'],membership_checks=record['membership_checks'],message_used=False,physical_calls=0)
