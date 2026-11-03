"""One post pass, opened only after both time and identity blind terminal saves."""
from pathlib import Path
from collections import Counter
from . import b_joint_time_packet_a_physical_v1_records as rec
from .b_joint_time_packet_a_gpu_v1 import wait


def _at(rows,index):
    value=rows[index] if isinstance(rows,list) and index<len(rows) else None
    return value if isinstance(value,dict) else {}


def _object(value):return value if isinstance(value,dict) else {}


def _state(value):return type(value) is int and 0<=value<92672


def _coordinate(row,key,index):return type(row.get(key)) is int and row[key]==index


def _old_node(row,index):
    if not _coordinate(row,'received_frame',index) or not all(k in row for k in ('output_index','interval','origin')):return False
    value,bounds,origin=(row[k] for k in ('output_index','interval','origin'))
    if value is not None and not _state(value):return False
    if bounds is not None and (not isinstance(bounds,list) or len(bounds)!=2 or not all(_state(x) for x in bounds) or bounds[0]>bounds[1]):return False
    if value is None:return origin=='UNRESOLVED'
    if origin=='OLD':return bounds==[value,value]
    return origin=='NEW' and bounds is not None and bounds[0]<=value<=bounds[1]


def _candidate(row,index,field):
    value=row.get(field)
    return value if _coordinate(row,'received_frame',index) and field in row and _state(value) else None


def _set(value):
    return value is None or (isinstance(value,list) and len(value)<=3 and all(_state(x) for x in value) and value==sorted(set(value)))


def _packet_node(row,index,coding):
    if not _coordinate(row,'received_frame',index) or not all(k in row for k in ('compatible_n','constraint_count','formal_output','observations','lanes')):return False
    candidate=row['compatible_n'];count=row['constraint_count']
    if not _set(candidate) or type(count) is not int or not 0<=count<=5 or row['formal_output'] is not None:return False
    observations,lanes=row['observations'],row['lanes']
    if not isinstance(observations,list) or len(observations)!=15 or not isinstance(lanes,list) or len(lanes)!=5:return False
    slots=set()
    for observation in observations:
        if not isinstance(observation,dict) or type(observation.get('lane')) is not int or observation['lane'] not in range(5):return False
        if observation.get('role') not in ('R0','R1','F') or observation.get('status') not in (
                'ERASURE','PARTIAL_ERASURE_UNRESOLVED','UNRESOLVED_CODE','INVALID_PACKET','LEGAL_OLD_J_CANDIDATE','LEGAL_LOCAL_CANDIDATE'):return False
        if 'candidate_n' not in observation or any(observation.get(k) is not None and not _state(observation[k]) for k in ('candidate_n','candidate_j')):return False
        slot=(observation['lane'],observation['role'])
        if slot in slots:return False
        slots.add(slot)
        for key,kind in (('known',bool),('raw_bits',int)):
            bits=observation.get(key)
            if not isinstance(bits,list) or len(bits)!=63 or any(type(b) is not kind or kind is int and b not in (0,1) for b in bits):return False
    if any(not isinstance(lane,dict) or not _coordinate(lane,'lane',i) or not isinstance(lane.get('candidate_union'),list) or not _set(lane['candidate_union']) for i,lane in enumerate(lanes)):return False
    if coding=='OLD':return row.get('status')=='OLD_LOCAL_J_DIAGNOSTIC_ONLY' and candidate is None and count==0
    expected=('NO_VALID_LOCAL_CANDIDATES' if candidate is None else 'CONSTRAINT_EMPTY_SET' if not candidate else 'STRUCTURAL_SINGLETON' if len(candidate)==1 else 'STRUCTURAL_AMBIGUITY')
    return row.get('status')==expected and ((candidate is None and count==0) or (candidate is not None and count>0))


def _old_edge(row,index):
    if not _coordinate(row,'received_left',index) or not all(k in row for k in ('output_endpoints','status','kind')):return False
    if row['status'] not in ('EXACT','CONDITIONAL_EXACT','AMBIGUOUS','NO_COMPATIBLE_PATH','LOCAL_CONFLICT_UNRESOLVED'):return False
    if row['kind'] not in (None,'NORMAL','GAP','REPEAT','AMBIGUOUS'):return False
    pair=row['output_endpoints']
    if pair is None:return True
    if not isinstance(pair,list) or len(pair)!=2 or not all(_state(x) for x in pair) or pair[0]>pair[1]:return False
    kind='REPEAT' if pair[0]==pair[1] else 'NORMAL' if pair[1]==pair[0]+1 else 'GAP'
    return row['status'] in ('EXACT','CONDITIONAL_EXACT') and row['kind']==kind


def _packet_edge(row,index):
    if not _coordinate(row,'received_edge',index) or not all(k in row for k in ('left_candidates','right_candidates','pairs','status','formal_output')):return False
    left,right=row['left_candidates'],row['right_candidates']
    if not _set(left) or not _set(right) or row['formal_output'] is not None:return False
    expected=[dict(left=a,right=b,delta=b-a,relation='INVERSION' if b<a else 'REPEAT' if b==a else 'NORMAL' if b==a+1 else 'GAP') for a in (left or []) for b in (right or [])]
    pairs=row['pairs']
    if not isinstance(pairs,list) or len(pairs)!=len(expected):return False
    if any(not isinstance(pair,dict) or any(pair.get(k)!=v or type(pair.get(k)) is not type(v) for k,v in item.items()) for pair,item in zip(pairs,expected)):return False
    singleton=left is not None and right is not None and len(left)==len(right)==1
    status=('UNRESOLVED' if not expected else 'INVERSION' if singleton and expected[0]['delta']<0 else 'UNCALIBRATED_EXACT_'+expected[0]['relation'] if singleton else 'AMBIGUOUS')
    return row['status']==status


def score_report(folder,mapping,competitors):
    """Read saved scores only; no receiver/score recalculation or candidate changes."""
    import numpy as np
    info=rec.load(folder/'scores.json');n=info['frames'];epsilon=1e-12
    if n!=len(mapping):raise ValueError('saved score/truth length mismatch')
    queries=[];values=[];minimum=np.full(n,np.inf);best_count=np.zeros(n,np.int64)
    for r,t in enumerate(mapping):
        row={'truth':t,'minus1':None if t is None else t-1,'plus1':None if t is None else t+1,**competitors[r]}
        queries.append(row);values.append({k:None for k in row})
    expected=4
    for chunk in info['chunks']:
        lo,hi=chunk['start'],chunk['stop']
        if lo!=expected or not lo<hi<=92672:raise ValueError('saved full-domain chunk gap/overlap')
        expected=hi;score=np.load(folder/chunk['file'],mmap_mode='r',allow_pickle=False)
        if score.shape!=(n,hi-lo) or score.dtype!=np.float64:raise ValueError('saved FP64 scores shape mismatch')
        finite=np.isfinite(score);row_min=np.where(finite,score,np.inf).min(axis=1);minimum=np.minimum(minimum,row_min)
        for r,row in enumerate(queries):
            for name,state in row.items():
                if type(state) is int and lo<=state<hi:
                    value=float(score[r,state-lo]);values[r][name]=value if np.isfinite(value) else None
    if expected!=92672:raise ValueError('incomplete mature domain')
    less=np.zeros(n,np.int64);ties=np.zeros(n,np.int64)
    for chunk in info['chunks']:
        score=np.load(folder/chunk['file'],mmap_mode='r',allow_pickle=False)
        for r in range(n):
            if np.isfinite(minimum[r]):best_count[r]+=int(np.count_nonzero(np.abs(score[r]-minimum[r])<=epsilon))
            truth=values[r]['truth']
            if truth is not None:
                less[r]+=int(np.count_nonzero(score[r]<truth-epsilon));ties[r]+=int(np.count_nonzero(np.abs(score[r]-truth)<=epsilon))
    with np.load(folder/'startup.npz',allow_pickle=False) as start:
        startup={k:start[k] for k in ('numerator','denominator','partial','active')}
    out=[]
    for r,t in enumerate(mapping):
        item=dict(received_frame=r,truth_state=t,truth_is_startup=type(t) is int and t<4,
            mature_minimum=float(minimum[r]) if np.isfinite(minimum[r]) else None,mature_best_count=int(best_count[r]),
            truth_rank=None if values[r]['truth'] is None else int(less[r])+1,truth_tie_count=int(ties[r]),
            queried_states=queries[r],queried_mature_scores=values[r],startup=[])
        for state in range(4):
            value=float(startup['partial'][r,state])
            item['startup'].append(dict(n=state,active=startup['active'][state].tolist(),
                numerator=float(startup['numerator'][r,state]),denominator=int(startup['denominator'][r,state]),
                partial=value if np.isfinite(value) else None,mature_rank=None))
        out.append(item)
    return out


def time_report(config,cpu,terminal,output,mappings,errors):
    rows=[];edges=[];summaries={};history={}
    for source in rec.SOURCES:
        for view,length in zip(rec.VIEWS,rec.LENGTHS):
            key=source+'/'+view;history[key]={}
            for label in ('H','L','final'):
                try:history[key][label]=rec.load(config['old_time'][key][label])
                except (KeyError,OSError,ValueError) as error:errors['history/'+key+'/'+label]=str(error)
    rec.dump(output/'historical_H_L_unchanged.json',history)
    for unit in rec.time_units():
        uid=unit['id'];source,arm,view,n=(unit[k] for k in ('source','arm','view','frames'))
        mapping=mappings.get(view,[None]*n);old=history[source+'/'+view]
        old_h,old_l,old_final_graph=(_object(old.get(k)) for k in ('H','L','final'))
        competitor=[]
        old_nodes=[]
        for r in range(n):
            baseline=_at(old_final_graph.get('final_nodes'),r);available=_old_node(baseline,r)
            old_nodes.append((baseline,available))
            competitor.append(dict(old_H=_candidate(_at(old_h.get('nodes'),r),r,'output_index'),
                old_L=_candidate(_at(old_l.get('nodes'),r),r,'model_index'),
                old_final=baseline['output_index'] if available else None))
        packet={};ranks=[{} for _ in range(n)]
        folder=cpu/'received'/uid
        evidence_status=terminal.get('time_units',{}).get(uid,{}).get('status','UNRESOLVED')
        if evidence_status!='BLIND_SAVED':errors['time/'+uid]='blind receiver did not complete within its fixed attempt; partial artifacts retained'
        try:
            saved_packet=rec.load(folder/'packets.json')
            if not isinstance(saved_packet,dict):raise ValueError('saved packet object required')
            packet=saved_packet
        except (OSError,ValueError,KeyError,TypeError) as error:errors['packets/'+uid]=str(error)
        try:ranks=score_report(folder,mapping,competitor)
        except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError) as error:errors['scores/'+uid]=str(error)
        counts=Counter()
        for r,t in enumerate(mapping):
            node=_at(packet.get('nodes'),r)
            local_available=packet.get('coding')==unit['coding'] and _packet_node(node,r,unit['coding'])
            candidate=node.get('compatible_n') if local_available else None
            singleton=candidate[0] if candidate and len(candidate)==1 else None
            baseline,baseline_available=old_nodes[r]
            old_final=competitor[r]['old_final']
            old_status=('BASELINE_UNAVAILABLE' if not baseline_available else 'UNRESOLVED' if old_final is None
                        else 'UNKNOWN' if t is None else 'CORRECT' if old_final==t else 'WRONG')
            local=('UNKNOWN' if not local_available else 'OLD_J_LOCAL_ONLY' if arm=='OLD' else 'UNRESOLVED' if singleton is None
                   else 'UNKNOWN' if t is None else 'CORRECT' if singleton==t else 'WRONG')
            if arm=='BASE':local='DESCRIPTIVE_GEOMETRY_'+local
            counts['local/'+local]+=1;counts['startup' if type(t) is int and t<4 else 'nonstartup_or_unknown']+=1
            counts['baseline/'+old_status]+=1
            packets=[]
            for obs in node.get('observations',[]) if local_available else []:
                expected=None if t is None else t if arm!='OLD' else t-(t-obs['lane'])%5
                value=obs.get('candidate_j') if arm=='OLD' else obs.get('candidate_n')
                valid=value is not None
                status=('LEGAL_GEOMETRIC_MATCH' if value==expected else 'LEGAL_GEOMETRIC_MISMATCH') if valid and expected is not None else obs['status']
                counts['packet/'+status]+=1;packets.append(dict(obs,post_geometry_status=status,time_payload_present=arm!='BASE'))
            unions=[x['candidate_union'] for x in node.get('lanes',[]) if x.get('candidate_union')] if local_available else []
            row=dict(unit=uid,source=source,arm=arm,view=view,received_frame=r,truth=t,local_status=local,
                evidence_status=evidence_status,
                baseline_available=baseline_available,local_record_available=local_available,
                unavailable_baseline_row=baseline if not baseline_available else None,
                unavailable_local_row=node if not local_available else None,
                original_final_status=old_status,original_final=old_final,local_candidate=singleton,
                intersection=candidate,intersection_excludes_truth=None if t is None or candidate is None or arm=='OLD' else t not in candidate,
                nonempty_lane_union_contains_truth=None if t is None or arm=='OLD' or not local_available else [t in x for x in unions],
                packet_observations=packets,rank=ranks[r],formal_output=None,
                diagnostic_corrected_old_wrong=arm=='NEW' and old_status=='WRONG' and local=='CORRECT',
                diagnostic_damaged_old_correct=arm=='NEW' and old_status=='CORRECT' and local=='WRONG',
                diagnostic_lost_old_correct=arm=='NEW' and old_status=='CORRECT' and local=='UNRESOLVED')
            rows.append(row)
        for r in range(n-1):
            a,b=mapping[r:r+2];observed=_at(packet.get('edges'),r)
            old_edge=_at(old_final_graph.get('final_relations'),r)
            if arm=='OLD':
                available=_old_edge(old_edge,r)
                item=dict(received_edge=r,status='OLD_FORMAL_READ_ONLY',saved=old_edge,formal_semantics_changed=False)
                endpoints=old_edge.get('output_endpoints') if available else None;kind=old_edge.get('kind') if available else None
            else:
                available=packet.get('coding')==unit['coding'] and _packet_edge(observed,r)
                item=observed or dict(received_edge=r,status='UNKNOWN',pairs=[],formal_output=None)
                exact=available and item.get('status','').startswith('UNCALIBRATED_EXACT_')
                pair=item.get('pairs',[{}])[0] if exact else {}
                endpoints=[pair['left'],pair['right']] if exact else None;kind=pair.get('relation')
            correct=None if a is None or b is None or endpoints is None else endpoints==[a,b]
            status=('UNKNOWN' if not available else 'UNRESOLVED' if endpoints is None else 'UNKNOWN'
                    if a is None or b is None else 'CORRECT' if correct else 'WRONG')
            counts['edge/'+status]+=1
            false_gap=kind=='GAP' and correct is False;false_repeat=kind=='REPEAT' and correct is False
            counts['false_GAP']+=false_gap;counts['false_REPEAT']+=false_repeat
            edges.append(dict(unit=uid,source=source,arm=arm,view=view,truth_endpoints=[a,b],geometry_status=status,
                              record_available=available,evidence_status=evidence_status,
                              false_gap=false_gap,false_repeat=false_repeat,diagnostic=item,
                              BASE_is_descriptive_only=arm=='BASE'))
        summaries[uid]=dict(unit,counts=dict(counts))
    rec.dump(output/'time_nodes.json',rows);rec.dump(output/'time_edges.json',edges)
    preservation={}
    for source in rec.SOURCES:
        selected=[x for x in rows if x['source']==source and x['arm']=='NEW']
        preservation[source]=dict(fixed_nodes=822,old_correct=sum(x['original_final_status']=='CORRECT' for x in selected),
            old_wrong=sum(x['original_final_status']=='WRONG' for x in selected),
            baseline_unavailable=sum(x['original_final_status']=='BASELINE_UNAVAILABLE' for x in selected),
            new_local_unknown=sum(x['local_status']=='UNKNOWN' for x in selected),
            corrected_old_wrong=sum(x['diagnostic_corrected_old_wrong'] for x in selected),
            damaged_old_correct=sum(x['diagnostic_damaged_old_correct'] for x in selected),
            lost_old_correct=sum(x['diagnostic_lost_old_correct'] for x in selected),
            comparison='OLD formal baseline versus NEW uncalibrated local diagnostic; no promotion to formal acceptance')
    return dict(units=summaries,preservation=preservation,nodes=len(rows),edges=len(edges),expected_n01_old_error_count=23,
                startup_retained=True,scientific_pass=None)


def identity_report(config,gpu,output,mappings,errors):
    try:
        raw=rec.load(gpu/'addressed_raw_bits.json')
        if not isinstance(raw,list):raise ValueError('saved blind raw bit list required')
    except (OSError,ValueError) as error:errors['identity_raw']=str(error);raw=[]
    messages={}
    for source in rec.SOURCES:
        try:
            writer=rec.load(config['writer_protocol'][source])['config']['writer'];bits=writer['bits']
            if writer.get('key')!='watermark' or len(bits)!=32 or any(type(x) is not int or x not in (0,1) for x in bits):
                raise ValueError('actual original watermark key and 32 writer bits required')
            messages[source]=bits
        except (OSError,ValueError,KeyError,TypeError) as error:errors['message/'+source]=str(error)
    indexed={};rows=[]
    for unit in rec.identity_units():
        # Fixed addresses are reconstructed independently of successful rows.
        from main.tube_state.b_joint_time_packet_a_identity_v1 import empty
        for item in empty(unit,'missing blind raw slot'):
            indexed[tuple(item[k] for k in ('source','arm','view','query','phase','region','row','bit'))]=item
    seen=set();invalid=[]
    for item in raw:
        try:
            key=tuple(item[k] for k in ('source','arm','view','query','phase','region','row','bit'))
            if key not in indexed:raise ValueError('unexpected blind identity address')
            if key in seen:
                indexed[key].update(value=None,status='UNRESOLVED',cause='DUPLICATE_BLIND_ADDRESS')
                raise ValueError('duplicate blind identity address')
            if item['value'] is not None and (type(item['value']) is not int or item['value'] not in (0,1)):
                raise ValueError('invalid saved raw identity value')
            seen.add(key);indexed[key]=dict(item)
        except (KeyError,ValueError,TypeError) as error:invalid.append(dict(saved=item,error=str(error)))
    if invalid:errors['identity_invalid_rows']=str(len(invalid));rec.dump(output/'invalid_identity_rows.json',invalid)
    units={u['id']:u for u in rec.identity_units()}
    for item in indexed.values():
        unit=units[f"{item['source']}/{item['arm']}/{item['view']}/q{item['query']}/p{item['phase']}"]
        start=unit['start'];received=list(range(start+4*item['row']-3,start+4*item['row']+1))
        mapping=mappings.get(item['view']);frames=None if mapping is None else [mapping[i] for i in received]
        t=(frames[0]+3)//4 if frames else None
        exact=t in range(1,45) and frames==list(range(4*t-3,4*t+1)) if t is not None else False
        group=(t-1)//11 if exact else None
        expected=messages[item['source']][group*8+item['bit']] if group is not None and item['source'] in messages else None
        context=None if mapping is None else [mapping[i] for i in unit['encoder_input_support']]
        rows.append(dict(item,received_cell=received,actual_source_frames=frames,true_cell=t if exact else None,true_group=group,
            geometry='EXACT_NATIVE_FOUR_FRAME_CELL' if exact else 'UNKNOWN_OR_MIXED_CELL',
            actual_encoder_source_mapping=context,context_kind='UNKNOWN' if context is None else 'CONTIGUOUS' if all(b==a+1 for a,b in zip(context,context[1:])) else 'MIXED',
            expected_bit=expected,literal_error=None if expected is None or item['value'] is None else item['value']!=expected,
            declared_group=None,false_attribution='NO_GROUP_ASSIGNMENT_BY_RAW_READER',native_181_context_equivalent=False))
    table={tuple(x[k] for k in ('source','view','query','phase','region','row','bit'))+(x['arm'],):x for x in rows}
    comparison={}
    for source in rec.SOURCES:
        for view in rec.VIEWS:
            pairs=[(x,table[tuple(x[k] for k in ('source','view','query','phase','region','row','bit'))+('NEW',)])
                   for x in rows if x['source']==source and x['view']==view and x['arm']=='OLD']
            common=[(a,b) for a,b in pairs if a['value'] is not None and b['value'] is not None]
            comparable=[(a,b) for a,b in common if a['expected_bit'] is not None and b['expected_bit']==a['expected_bit']]
            comparison[source+'/'+view]=dict(fixed_raw_pairs=1920,common_readable=len(common),common_comparable=len(comparable),
                OLD_common_errors=sum(a['literal_error'] is True for a,b in comparable),NEW_common_errors=sum(b['literal_error'] is True for a,b in comparable),
                OLD_common_BER=sum(a['literal_error'] is True for a,b in comparable)/len(comparable) if comparable else None,
                NEW_common_BER=sum(b['literal_error'] is True for a,b in comparable)/len(comparable) if comparable else None,
                added_readable=sum(a['value'] is None and b['value'] is not None for a,b in pairs),
                lost_readable=sum(a['value'] is not None and b['value'] is None for a,b in pairs),
                common_geometry_unresolved=len(common)-len(comparable),false_attribution='NO_BLIND_GROUP_ASSIGNMENT',
                added_support_is_not_BER=True)
    rec.dump(output/'identity_rows.json',rows)
    return dict(raw_slots=len(rows),comparisons=comparison,full32_gate=False,formal_aggregation=False)


def run(config,output):
    output=Path(output).expanduser().resolve()
    if (output/'attempt.json').exists():return rec.recover(output)
    cpu,gpu=Path(config['cpu_media']),Path(config['gpu2'])
    time_terminal,identity_terminal=rec.terminal(cpu),rec.terminal(gpu)
    if time_terminal is None or identity_terminal is None:return wait('both time and identity blind terminal receipts must precede truth')
    if not rec.reserve(output,'post',config):return rec.recover(output)
    journal=rec.Journal(output,'post');errors={};results={}
    def evaluate():
        mappings={}
        try:
            mappings=rec.load(cpu/'private/attack_truth.json')['views']
            for v,n in zip(rec.VIEWS,rec.LENGTHS):
                if len(mappings[v])!=n or any(type(x) is not int or not 0<=x<181 for x in mappings[v]):raise ValueError('invalid actual attack mapping')
        except (OSError,ValueError,KeyError,TypeError) as error:errors['truth']=str(error);mappings={}
        try:results['time']=time_report(config,cpu,time_terminal,output,mappings,errors)
        except Exception as error:errors['time_post']=type(error).__name__+': '+str(error)
        try:results['identity']=identity_report(config,gpu,output,mappings,errors)
        except Exception as error:errors['identity_post']=type(error).__name__+': '+str(error)
        rec.dump(output/'evaluation.json',dict(results=results,errors=errors,denominators=rec.denominators(),scientific_pass=None,
            claim_ceiling='two seen sources, fixed media and uncalibrated diagnostics only; no general FPR, source authentication or long-video recovery'))
    try:journal.call('evaluate',evaluate)
    except BaseException as error:errors['stage']=type(error).__name__+': '+str(error)
    return journal.finish(errors,results=results,physical_calls=0,post_attempts=1)
