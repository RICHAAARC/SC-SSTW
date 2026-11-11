"""Saved-only post: full time slots, physical words and attributed payload pairs."""
from collections import Counter
from pathlib import Path
import json
import time
from . import b_t12_saved_candidate_media_v1_records as rec


def transition(old,new):
    return dict(transition=old+'->'+new,repair=old=='WRONG' and new=='CORRECT',
        error_removed_unresolved=old=='WRONG' and new=='UNRESOLVED',
        old_correct_lost=old=='CORRECT' and new in ('WRONG','UNRESOLVED'),
        old_correct_retention_unknown=old=='CORRECT' and new=='UNKNOWN',
        candidate_wrong=new=='WRONG',comparison_known=old!='UNKNOWN' and new!='UNKNOWN')


def optional(path,errors):
    try: return rec.read(path)
    except (OSError,ValueError,TypeError) as error: errors.append(dict(file=str(path),error=str(error)));return None


def indexed(rows,field,count,*,uid=None):
    if not isinstance(rows,list): return None
    if any(not isinstance(r,dict) for r in rows): return None
    selected=[r for r in rows if uid is None or r.get('input_id')==uid]
    if len(selected)!=count or any(not isinstance(r,dict) or type(r.get(field)) is not int for r in selected): return None
    result={r[field]:r for r in selected}
    return result if set(result)==set(range(count)) else None


def evaluate(output,config,operations):
    from runtime.wan.b_relative_time_payload_evaluation_v1 import slot_rows
    from runtime.wan.b_paper_cohort_evaluation_v1 import bounds_for_slot
    from runtime.wan.b_joint_time_relative_evaluation_v1 import singleton,edge_pair
    from main.tube_state.b_joint_time_relative_validation_v1 import construction_maps
    from main.tube_state import b_joint_time_packet_a_v1 as packet
    from runtime.wan.b_relative_time_payload_records_v1 import _decoded_index
    output=Path(output);old=Path(config['original']);errors=[]
    if rec.read(output/'blind_terminal.json')['status']!='ALL_BLIND_TERMINAL': raise ValueError('blind outputs must close before truth access')
    rec.write(output/'truth_access.json',dict(after_all_blind=True,time=time.time()))
    maps=construction_maps();old_maps=optional(old/'private/attack_truth.json',errors)
    writer=optional(old/'private/writer.json',errors);message=writer.get('bits') if isinstance(writer,dict) else None
    if not isinstance(message,list) or len(message)!=32 or any(type(b) is not int or b not in (0,1) for b in message):
        message=None;errors.append(dict(error='original writer binary32 unavailable'))
    old_ops=optional(old/'state.json',errors) or {}
    if not isinstance(old_ops,dict) or any(not isinstance(r,dict) for r in old_ops.values()):
        errors.append(dict(error='original state format invalid'));old_ops={}
    old_nodes=optional(old/'node_results.json',errors);old_edges=optional(old/'edge_results.json',errors)
    old_quality=optional(old/'quality.json',errors) or {};raw=optional(output/'raw_bank.json',errors) or {}
    if not isinstance(old_quality,dict): errors.append(dict(error='original quality format invalid'));old_quality={}
    if not isinstance(raw,dict): errors.append(dict(error='saved raw bank format invalid'));raw={}
    nodes=[];edges=[];payload=[];local_slots=[];quality=[];physical=[];f_roles=[];claims=[];expected_codes={}
    def returned(key): return operations.get(key,{}).get('status')=='RETURNED' and not operations.get(key,{}).get('artifact_error')
    def observed(path,key): return optional(output/path,errors) if returned(key) else None
    for item in rec.items():
        uid,n,view=item['id'],item['length'],item['view'];mapping=maps[view]
        compatible=isinstance(old_maps,dict) and old_maps.get(view)==mapping
        old_time_returned=old_ops.get(f'time/{uid}/relative_compare',{}).get('status')=='RETURNED'
        prior_nodes=indexed(old_nodes,'received_frame',n,uid=uid) if compatible and old_time_returned else None
        prior_edges=indexed(old_edges,'received_edge',n-1,uid=uid) if compatible and old_time_returned else None
        relative=observed(f'blind/{uid}/relative.json',uid+'/relative')
        valid=bool(isinstance(relative,dict) and isinstance(relative.get('conditional_nodes'),list)
            and isinstance(relative.get('conditional_edges'),list) and len(relative['conditional_nodes'])==n
            and len(relative['conditional_edges'])==n-1
            and all(isinstance(row,dict) for row in relative['conditional_nodes']+relative['conditional_edges']))
        for index,t in enumerate(mapping):
            row=relative['conditional_nodes'][index] if valid else None
            candidate=row.get('conditional_candidates') if row else None;value=singleton(candidate)
            usable=bool(valid and row and 'conditional_candidates' in row and
                (candidate is None or isinstance(candidate,list) and all(type(v) is int for v in candidate)))
            if valid and not usable: errors.append(dict(file=f'blind/{uid}/relative.json',received_frame=index,error='invalid node candidates'))
            status=('CORRECT' if value==t else 'WRONG' if value is not None else 'UNRESOLVED') if usable else 'UNKNOWN'
            prior=prior_nodes[index] if prior_nodes else None
            old_status=prior.get('status','UNKNOWN') if prior and prior.get('truth_n')==t else 'UNKNOWN'
            if old_status not in ('CORRECT','WRONG','UNRESOLVED'): old_status='UNKNOWN'
            nodes.append(dict(input_id=uid,view=view,received_frame=index,truth=t,output=value,candidates=candidate,
                old_status=old_status,status=status,startup=t<=4,second_or_third_frame=t in (1,2),**transition(old_status,status)))
        for index in range(n-1):
            target=(mapping[index],mapping[index+1]);edge=relative['conditional_edges'][index] if valid else None
            usable=bool(edge and type(edge.get('exact_singleton_pair')) is bool and isinstance(edge.get('pairs'),list)
                and all(isinstance(pair,dict) and type(pair.get('left')) is int and type(pair.get('right')) is int for pair in edge['pairs'])
                and (not edge['exact_singleton_pair'] or len(edge['pairs'])==1))
            if valid and not usable: errors.append(dict(file=f'blind/{uid}/relative.json',received_edge=index,error='invalid edge pairs'))
            value=edge_pair(edge) if usable else None
            status=('CORRECT' if value==target else 'WRONG' if value is not None else 'UNRESOLVED') if usable else 'UNKNOWN'
            prior=prior_edges[index] if prior_edges else None
            old_status=prior.get('status','UNKNOWN') if prior and prior.get('truth_pair')==list(target) else 'UNKNOWN'
            if old_status not in ('CORRECT','WRONG','UNRESOLVED'): old_status='UNKNOWN'
            edges.append(dict(input_id=uid,view=view,received_edge=index,truth=list(target),output=value,
                old_status=old_status,status=status,startup=any(t<=4 for t in target),**transition(old_status,status)))
        # Physical R0/R1 retained separately from derived F; no new reader calls.
        hard=observed(f'blind/{uid}/packets.json',uid+'/hard')
        old_hard=optional(old/f'blind/{uid}/packets.json',errors) if compatible and old_ops.get(f'time/{uid}/hard_receive',{}).get('status')=='RETURNED' else None
        def observations(value,index):
            if not isinstance(value,dict) or not isinstance(value.get('nodes'),list) or len(value['nodes'])!=n or not isinstance(value['nodes'][index],dict): return {}
            rows=value['nodes'][index].get('observations',[])
            if not isinstance(rows,list) or len(rows)!=15 or any(not isinstance(r,dict) for r in rows): return {}
            result={(r.get('lane'),r.get('role')):r for r in rows}
            return result if len(result)==15 else {}
        def word(row,expected):
            if not isinstance(row,dict): return dict(status='UNKNOWN',errors=None,erasures=None,correct=None)
            bits,known=row.get('raw_bits'),row.get('known')
            if not isinstance(bits,list) or len(bits)!=63 or not isinstance(known,list) or len(known)!=63 or any(type(b) is not int or b not in (0,1) for b in bits) or any(type(k) is not bool for k in known):
                return dict(status='UNKNOWN',errors=None,erasures=None,correct=None)
            return dict(status=row.get('status'),msg=row.get('msg'),candidate_n=row.get('candidate_n'),
                errors=sum(k and b!=t for b,t,k in zip(bits,expected_codes[expected],known)) if expected is not None else None,
                erasures=sum(not k for k in known),correct=(row.get('msg')==expected) if expected is not None else None)
        for index,t in enumerate(mapping):
            current,prior=observations(hard,index),observations(old_hard,index)
            for lane in range(5):
                j=t-(t-lane)%5;active=j>=0;message_id=8*j+t-j if active else None
                if active and message_id not in expected_codes: expected_codes[message_id]=packet.encode_raw24(message_id)
                for role in ('R0','R1','F'):
                    before,after=word(prior.get((lane,role)),message_id),word(current.get((lane,role)),message_id)
                    entry=dict(input_id=uid,view=view,received_frame=index,source_frame=t,lane=lane,role=role,
                        active=active,expected_msg=message_id,old=before,new=after,
                        old_correct_lost=before['correct'] is True and after['correct'] is False,
                        old_correct_retention_unknown=before['correct'] is True and after['correct'] is None,
                        comparison_known=before['correct'] is not None and after['correct'] is not None)
                    (f_roles if role=='F' else physical).append(entry)
        frozen=observed(f'plans/{uid}.json',uid+'/plan')
        frozen_time=frozen.get('time') if isinstance(frozen,dict) else None
        if frozen is not None and (not isinstance(frozen_time,dict) or 'status' not in frozen_time):
            errors.append(dict(file=f'plans/{uid}.json',error='time status unavailable'));frozen_time=None
        for strategy,key in (('B_NEW','key00'),('K_NEW','key01'),('U_NEW','key00')):
            for query in range(3):
                address=f'{uid}/{strategy}/q{query}'
                saved=observed(f'local/{uid}/{strategy}/q{query}.json',address)
                ordered=indexed(saved.get('slots'),'slot',32) if isinstance(saved,dict) else None
                if saved is not None and ordered is None: errors.append(dict(file=f'local/{uid}/{strategy}/q{query}.json',error='local slot addresses invalid'))
                for index in range(32):
                    row=ordered[index] if ordered is not None else dict(slot=index,value=None,status='UNKNOWN',cause_set=['LOCAL_ARTIFACT_UNAVAILABLE'],selected_refs=[],supporting_refs=[],missing_refs=[],conflict_refs=[])
                    local_slots.append(dict(input_id=uid,view=view,strategy=strategy,key=key,query=query,slot=index,
                        observation=row,observation_available=ordered is not None,operation=operations.get(address,{})))
            final=observed(f'final/{uid}/{strategy}.json',f'{uid}/{strategy}/aggregate')
            final_index=indexed(final.get('slots'),'slot',32) if isinstance(final,dict) else None
            try: current=indexed(slot_rows(dict(final,slots=[final_index[i] for i in range(32)]),mapping,raw,message,key),'slot',32) if final_index is not None and message is not None else None
            except (ValueError,KeyError,TypeError,IndexError,AttributeError) as error: errors.append(dict(file=f'final/{uid}/{strategy}',error=str(error)));current=None
            prior=optional(old/f'post/{uid}/{strategy}.json',errors) if compatible and old_ops.get(f'{uid}/{strategy}/aggregate',{}).get('status')=='RETURNED' else None
            prior=indexed(prior,'slot',32)
            for index in range(32):
                saved_slot=final_index[index] if final_index is not None else None
                readable=bool(saved_slot is not None and 'value' in saved_slot and isinstance(saved_slot.get('status'),str)
                    and (saved_slot['value'] is None or type(saved_slot['value']) is int and saved_slot['value'] in (0,1)))
                row=current[index] if current else dict(saved_slot if readable else dict(slot=index,value=None,status='UNKNOWN'),
                    literal_correct=None,correct_supported=None,wrong_attribution=None,
                    supporting_refs=saved_slot.get('supporting_refs',[]) if readable else [])
                time_ready=bool(strategy=='U_NEW' or frozen_time and frozen_time.get('status')=='READY')
                bounds,reason=bounds_for_slot(row,returned=current is not None,time_ready=time_ready,truth_available=message is not None)
                correct=row.get('correct_supported') if bounds[0]==bounds[1] else None
                before=prior[index] if prior else None
                old_correct=before.get('correct_supported') if before else None
                payload.append(dict(input_id=uid,view=view,strategy=strategy,key=key,slot=index,new=dict(row,correct_supported=correct,
                    correct_supported_bounds=bounds,evaluability_reason=reason),old=before,
                    old_correct_supported_lost=old_correct is True and correct is False,
                    old_correct_retention_unknown=old_correct is True and correct is None,
                    comparison_known=old_correct is not None and correct is not None,
                    literal_changed=(before.get('value')!=row.get('value')) if before and before.get('value') is not None and row.get('value') is not None else None))
        measured=observed(f'quality/{uid}.json',uid+'/quality')
        previous_quality=old_quality.get('NEW_P0/'+view) if compatible else None
        quality.append(dict(input_id=uid,view=view,reference=item['reference'],new=measured,old=previous_quality,
            comparison_status='MEASURED_COMMON_P0_REFERENCE' if measured is not None and isinstance(previous_quality,dict) and previous_quality.get('status')=='RETURNED' else 'UNKNOWN',
            SSIM='NOT_MEASURED',LPIPS='NOT_MEASURED',quality_pass=None))
        for key in ('key00','key01'):
            for claim in ('q0','q1'):
                value=observed(f'claims/{uid}/{key}/{claim}.json',f'{uid}/{key}/{claim}')
                verdict=value.get('verdict') if isinstance(value,dict) else None
                if verdict not in ('MATCH','MISMATCH','UNRESOLVED','INPUT_UNRESOLVED'):
                    if value is not None: errors.append(dict(file=f'claims/{uid}/{key}/{claim}.json',error='claim verdict unavailable'))
                    verdict='UNKNOWN'
                claims.append(dict(input_id=uid,key=key,claim=claim,result=value,status=verdict))
    # Fixed raw terminal payload rows even for skipped or failed E41/key calls.
    raw_rows=[]
    for item in rec.items():
        for bank in rec.banks(item):
            bid=bank['view_id']
            for key in ('key00','key01'):
                decoded=observed(f'bank/{bid}/{key}.json',bid+'/'+key);index={};bad={}
                if decoded is not None:
                    try: index,bad=_decoded_index(decoded,bid,key)
                    except (ValueError,KeyError,TypeError) as error: errors.append(dict(file=f'bank/{bid}/{key}',error=str(error)))
                for r in range(1,11):
                    for region in ('R0','R1'):
                        for bit in range(8):
                            row=index.get((r,region,bit));error=bad.get((r,region,bit))
                            raw_rows.append(dict(view_id=bid,key=key,local_row=r,region=region,local_bit=bit,
                                observation=row,status='UNKNOWN' if row is None or error else row.get('status'),error=error))
    for name,rows in [('node_results',nodes),('edge_results',edges),('payload_pairs',payload),('local_terminal_slots',local_slots),('physical_words',physical),('derived_F',f_roles),('raw_terminal_slots',raw_rows)]:
        path=output/'post'/(name+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as stream:
            for row in rows: stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    rec.write(output/'post/quality.json',quality);rec.write(output/'post/claims.json',claims)
    time_rows=nodes+edges;known=all(r['comparison_known'] for r in time_rows)
    b=[r for r in payload if r['strategy']=='B_NEW']
    wrong=[r for r in time_rows if r['old_status']=='WRONG']
    repairs=all(r['status']=='CORRECT' for r in wrong)
    regressions=any(r['old_correct_lost'] or r['candidate_wrong'] for r in time_rows)
    payload_known=all(r['comparison_known'] for r in b)
    payload_bad=any(r['old_correct_supported_lost'] or r['new'].get('wrong_attribution') is True or r['new'].get('literal_correct') is False for r in b)
    supported=bool(wrong) and known and repairs and not regressions and payload_known and not payload_bad
    cpu_result=optional(Path(config['candidate'])/'result.json',errors)
    if not isinstance(cpu_result,dict):
        errors.append(dict(error='saved CPU history summary unavailable'));cpu_result=None
    return dict(label=rec.LABEL,scientific_pass=None,denominators=rec.DENOMINATORS,
        system_observation='T12_KNOWN_ERROR_LOCAL_REPAIR_SUPPORTED' if supported else 'PARTIAL_DEGRADED_OR_UNRESOLVED',
        old_wrong_slots=len(wrong),repaired=sum(r['repair'] for r in time_rows),
        wrong_to_unresolved=sum(r['error_removed_unresolved'] for r in time_rows),
        old_correct_time_lost=sum(r['old_correct_lost'] for r in time_rows),candidate_wrong=sum(r['candidate_wrong'] for r in time_rows),
        old_correct_time_retention_unknown=sum(r['old_correct_retention_unknown'] for r in time_rows),
        node_statuses=dict(Counter(r['status'] for r in nodes)),edge_statuses=dict(Counter(r['status'] for r in edges)),
        old_wrong_all=[{k:r[k] for k in ('input_id','view','status','old_status') if k in r}|{k:r[k] for k in ('received_frame','received_edge') if k in r} for r in wrong],
        startup_nodes=[r for r in nodes if r['startup']],startup_edges=[r for r in edges if r['startup']],
        B_old_correct_supported_lost=sum(r['old_correct_supported_lost'] for r in b),
        B_old_correct_retention_unknown=sum(r['old_correct_retention_unknown'] for r in b),
        B_correct_supported=sum(r['new'].get('correct_supported') is True for r in b),
        B_unknown=sum(r['new'].get('correct_supported') is None for r in b),
        quality=quality,claims=dict(Counter(r['status'] for r in claims)),read_errors=errors,
        native=optional(output/'native/result.json',errors),cpu_saved_result=cpu_result,
        declared_cpu_history='RUN02_BUDGET_FAILURE_AND_NUMERICAL_UNRESOLVED_RETAINED',
        certificate='NO_NEW_KKT_ENVELOPE_OR_ORIGINAL_BUDGET_CLAIM',scope='one known t12 candidate; no generalization, authentication or population FPR')
