"""Independent complete-replay domain. No runtime, truth or payload dependency."""
from copy import deepcopy
from main.tube_state import b_m3_time_path_v1 as geometry
from main.tube_state import b_rank3_four_source_confirmation_v1 as original
from main.tube_state import b_cdc32_aggregation_v1 as cdc

READERS=('RANK3','RANK3R')
WINDOWS={'L':0,'M':53,'R':106}

def expanded_domain():
    paths=dict(original.PATHS)
    for a in range(31):
        for gap,(lo,hi) in geometry.GAPS.items():
            for r in range(1,54):
                seams=[];origins=None
                for b in range(lo,hi+1):
                    if r>min(b,151-b):continue
                    frames=[a+u-r*(u>=b) for u in range(151)]
                    candidate=tuple(frames[w] for w in WINDOWS.values())
                    if min(frames)<0 or max(frames)>180 or any(o not in range(137) for o in candidate):continue
                    if frames[b-r:b]!=frames[b:b+r]:raise ValueError('not complete replay')
                    if any(frames[w:w+45]!=list(range(frames[w],frames[w]+45)) for w in WINDOWS.values()):raise ValueError('edited declared window')
                    if origins is not None and origins!=candidate:raise ValueError('seam aliases change origins')
                    seams.append(b);origins=candidate
                if not seams:continue
                if origins in paths:raise ValueError('nonunique origin triple')
                pair=['L','M'] if gap=='LEFT_GAP' else ['M','R']
                paths[origins]=dict(path_id=f'{gap}:{a}:{-r}',a=a,d=-r,gap=gap,
                    query_source_origins=dict(zip(WINDOWS,origins)),
                    states={k:dict(zip(('phase','source_start'),geometry.origin_to_state(v))) for k,v in zip(WINDOWS,origins)},
                    legal_seams=seams,seam_alias_interval=[seams[0],seams[-1]],seam_multiplicity=len(seams),
                    source_window_overlap_frames=max(0,r-8),source_window_gap_frames=max(0,8-r),dependent_window_pair=pair,
                    valid_received_domain=[0,150],mapping_under_h=dict(kind='COMPLETE_REPLAY_CONDITIONAL',a=a,d=-r,formula='a+u+d*I[u>=b]',unobserved_frames_are_conditional=True),
                    source_relation='SINGLE_INPUT_CONDITIONAL',exact_seam_certified=False,
                    alias_policy='distinct observations retained; conflicts erased; source aliases labelled post-only; each slot counted once')
    if len(paths)!=4247 or sum(p['seam_multiplicity'] for p in paths.values())!=35743:
        raise ValueError('complete replay geometry is inconsistent')
    return paths

PATHS=expanded_domain()

def select(families,reader):
    if reader=='RANK3':return original.select(families,'RANK3')
    if reader!='RANK3R':raise ValueError('only the adopted two readers')
    if any(families[w]['status']=='TECHNICAL_UNRESOLVED' for w in WINDOWS):
        return dict(status='TECHNICAL_UNRESOLVED',detected=None,selected={},windows=list(WINDOWS))
    if any(families[w]['status']=='METHOD_AMBIGUITY_ERASURE' for w in WINDOWS):
        return dict(status='METHOD_AMBIGUITY_ERASURE',detected=False,selected={},windows=list(WINDOWS))
    selected={w:deepcopy(families[w]['selected']) for w in WINDOWS}
    origins=tuple(selected[w]['origin'] for w in WINDOWS);accepted=origins in PATHS
    return dict(status='DETECT' if accepted else 'REJECT_GEOMETRY',detected=accepted,selected=selected,
        windows=list(WINDOWS),origins=list(origins),path_class=deepcopy(PATHS[origins]) if accepted else None,
        endpoint_projection_only=False,exact_seam_certified=False,domain_classes=4247)

def plan_task(key_id,families,views):
    if key_id not in ('key00','key01'):raise ValueError('public key ID required, no source/arm/fixture labels')
    result={}
    for reader in READERS:
        selection=select(families,reader);plans=[]
        for window,w in WINDOWS.items():
            ids=views[window][0]
            prepared=dict(query_id=ids['query_id'],input_id=ids['input_id'],key_id=key_id,
                received_query_start=w,members=families[window]['members'],receiver=reader,gate=selection['status'],
                selected=selection.get('selected',{}).get(window),status='READY' if selection['detected'] else selection['status'],
                score_status='SCORES_COMPLETE' if not families[window]['issues'] else 'SCORE_UNRESOLVED')
            plan=original.plan_singleton_local(prepared,views[window]);plan['window']=window;plans.append(plan)
        result[reader]=dict(selection=selection,locals=cdc.reconcile_plans(plans),message_used=False,physical_calls=0)
    return result

def regress_saved(planned,evaluation,expected_tasks):
    """Membership only. Saved exact-support flags are old post-evaluation evidence.

    Never select new observations with old outcomes; no Q/S/bits are recomputed.
    """
    if evaluation.get('status')!='POST_RECORDED':raise ValueError('old finalized evaluation required')
    if set(planned.get('tasks',{}))!=set(expected_tasks) or set(evaluation.get('tasks',{}))!=set(expected_tasks):
        raise ValueError('all old72 fixed tasks required')
    rows={};loss=0;null_detect=0;positive_detect=0;supported=0;membership_checks=0
    for task in expected_tasks:
        if planned['tasks'][task].get('status')!='RETURNED':raise ValueError('old task not returned: '+task)
        saved=planned['tasks'][task]['readers']['RANK3']['path_context']
        post=evaluation['tasks'][task]['readers']['RANK3']
        positive=task.split('/')[1]=='PM' and task.endswith('/key00')
        if len(post.get('slots',[]))!=32:raise ValueError('old fixed32 missing: '+task)
        status=saved.get('status')
        if status=='TECHNICAL_UNRESOLVED':raise ValueError('old technical family: '+task)
        if status=='METHOD_AMBIGUITY_ERASURE':old=new=False;unchanged=True
        else:
            selected=saved.get('selected',{})
            if set(selected)!=set(WINDOWS):raise ValueError('old complete selected triple missing')
            triple=tuple(selected[k]['origin'] for k in WINDOWS)
            old=saved.get('detected');new=triple in PATHS;membership_checks+=1
            if type(old) is not bool or old != (post['detection_status']=='DETECT'):raise ValueError('old selection disagrees with saved decision')
            unchanged=not old or original.PATHS.get(triple) is not None and PATHS.get(triple)==original.PATHS[triple]
        flags=[v.get('actual_supported_correct') for v in post['slots']]
        if any(type(v) is not bool for v in flags):raise ValueError('old actual-support flags missing')
        count=sum(flags) if positive else 0;supported+=count
        lost=count if not new or not unchanged else 0;loss+=lost
        if positive:positive_detect+=new
        else:null_detect+=new
        rows[task]=dict(positive=positive,old_detect=old,expanded_detect=new,singleton_ref_inputs_unchanged=unchanged,
            old_supported_correct=count,lost_supported=lost)
    passed=positive_detect==12 and supported==384 and loss==0 and null_detect==0
    return dict(status='OLD72_REGRESSION_PASSED' if passed else 'OLD72_REGRESSION_FAILED',tasks=rows,
        positive_tasks=12,null_tasks=60,old_supported_correct=supported,preserved_supported_correct=supported-loss,
        lost_supported=loss,positive_detect=positive_detect,null_detect=null_detect,domain_classes=4247,seam_aliases=35743,
        planned_membership_slots=72,membership_checks=membership_checks,new_Q_S_or_bit_reads=0,physical_calls=0,domain_shrink_retry=False,
        evidence_ceiling='expanded membership and unchanged old singleton reference inputs, not repeat-video performance')
