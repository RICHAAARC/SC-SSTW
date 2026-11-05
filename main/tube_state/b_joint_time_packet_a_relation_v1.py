"""Pure received-message S-or-T witnesses. No decoding, scores, truth or paths."""
from collections import defaultdict
from itertools import combinations

N=92672
COPY_STATUSES=('LEGAL_LOCAL_CANDIDATE','INVALID_PACKET','UNRESOLVED_CODE','ERASURE','PARTIAL_ERASURE_UNRESOLVED')


def counters():
    return dict(physical_slots_parsed=0,raw_bit_fields_validated=0,known_flags_validated=0,known_bits=0,
        valid_physical_seeds=0,vertices=0,S_pair_checks=0,S_pairs=0,T_age_existence_checks=0,
        T_witnessed_age_groups=0,sort_groups=0,sort_items=0,adjacent_endpoint_pairs=0,
        bch_decode_calls=0,bit_distance_calls=0,template_encodings=0,hinge_calls=0,
        model_calls=0,codec_calls=0,identity_calls=0,training_calls=0)


def unresolved(r,reason='not evaluated'):
    return dict(received_frame=r,status='INPUT_OR_EXECUTION_UNRESOLVED',reason=reason,
        input_complete=False,relation_complete=False,evaluable=False,U=None,W=None,observed_W=None,candidates=None,vertex_ids=[],
        physical_records=[],known_count=None,W_witness_types={},formal_output=None,presence_claim=False)


def physical(node,r,counts):
    if not isinstance(node,dict) or type(node.get('received_frame')) is not int or node['received_frame']!=r:
        raise ValueError('received frame address mismatch')
    observations=node.get('observations')
    if not isinstance(observations,list):raise ValueError('physical observation list required')
    slots=set();records=[];dedup={};known_count=0
    for offset,obs in enumerate(observations):
        if not isinstance(obs,dict):raise ValueError('observation object required')
        role=obs.get('role')
        if role=='F':continue
        lane=obs.get('lane');status=obs.get('status')
        if type(lane) is not int or not 0<=lane<5 or role not in ('R0','R1'):
            raise ValueError('physical lane/R0/R1 address required')
        if (lane,role) in slots:raise ValueError('duplicate physical copy slot')
        if status not in COPY_STATUSES:raise ValueError('saved physical decoding status unavailable or unknown')
        bits,known=obs.get('raw_bits'),obs.get('known')
        if (not isinstance(bits,list) or len(bits)!=63 or any(type(v) is not int or v not in (0,1) for v in bits)
                or not isinstance(known,list) or len(known)!=63 or any(type(v) is not bool for v in known)):
            raise ValueError('63 integer hard bits and boolean known flags required')
        if 'physical_copy' in obs and obs['physical_copy'] is not True:raise ValueError('R0/R1 marked nonphysical')
        k=sum(known);candidate=obs.get('candidate_n');msg=obs.get('msg')
        slots.add((lane,role));counts['physical_slots_parsed']+=1
        counts['raw_bit_fields_validated']+=63;counts['known_flags_validated']+=63;counts['known_bits']+=k;known_count+=k
        reference=dict(received_frame=r,lane=lane,role=role,observation_index=offset)
        record=dict(reference,status=status,known_count=k,candidate_n=candidate,msg=msg)
        records.append(record)
        if status in ('ERASURE','PARTIAL_ERASURE_UNRESOLVED'):
            if candidate is not None or msg is not None or (k!=0 if status=='ERASURE' else not 0<k<63):
                raise ValueError('saved erasure status contradicts physical known/message fields')
            continue
        if k!=63:raise ValueError('saved decoded status requires all known bits in that copy')
        if status=='UNRESOLVED_CODE':
            if candidate is not None or msg is not None:raise ValueError('unresolved code has a saved message')
            continue
        if type(msg) is not int or not 0<=msg<1<<24:raise ValueError('saved raw24 message required')
        j,age=msg>>3,msg&7;n=j+age
        legal=0<=j<N and age<=4 and n<N and j%5==lane
        if status=='INVALID_PACKET':
            if legal or candidate is not None:raise ValueError('invalid-packet status contradicts public fields')
            continue
        if not legal or type(candidate) is not int or candidate!=n or age!=(n-lane)%5 or j!=n-age:
            raise ValueError('legal exact n disagrees with public joint fields')
        for field,value in (('j',j),('age',age),('n',n)):
            if field in obs and (type(obs[field]) is not int or obs[field]!=value):
                raise ValueError('saved '+field+' disagrees with raw24 message')
        counts['valid_physical_seeds']+=1
        uid=f'r{r}/j{j}/a{age}'
        vertex=dedup.setdefault(uid,dict(id=uid,r=r,lane=lane,j=j,age=age,n=n,copies=[],S_partners=[],T_groups=[],S=False,T=False))
        vertex['copies'].append(reference)
    if len(slots)!=10:raise ValueError('all ten physical copy records required')
    return records,known_count,dedup


def relation(left,right):
    delta=right-left
    return 'INVERSION' if delta<0 else 'REPEAT' if delta==0 else 'NORMAL' if delta==1 else 'GAP'


def edge(left,right,r,counts,check):
    check();a,b=left['W'],right['W'];pairs=[]
    observed_a,observed_b=left.get('observed_W',a),right.get('observed_W',b)
    complete=left['evaluable'] and right['evaluable']
    for x in observed_a or []:
        for y in observed_b or []:
            counts['adjacent_endpoint_pairs']+=1
            pairs.append(dict(left=x,right=y,delta=y-x,relation=relation(x,y)))
    exact=complete and a is not None and b is not None and len(a)==len(b)==1
    status=('INPUT_OR_EXECUTION_UNRESOLVED' if not complete
        else 'UNCALIBRATED_EXACT_'+pairs[0]['relation'] if exact else 'RELATION_AMBIGUITY' if pairs else 'NO_ENDPOINT_PAIR')
    return dict(received_edge=r,status=status,left_candidates=a,right_candidates=b,pairs=pairs,
                observed_left_candidates=observed_a,observed_right_candidates=observed_b,relation_complete=complete,
                relation_types=sorted({p['relation'] for p in pairs}),exact_singleton_pair=exact,formal_output=None)


def diagnose(packet,frames,counts,check=lambda:None):
    if not isinstance(packet,dict) or packet.get('coding')!='NEW':raise ValueError('saved NEW packet semantics required')
    source=packet.get('nodes')
    if not isinstance(source,list) or len(source)!=frames:raise ValueError('fixed received frame count required')
    nodes=[unresolved(r) for r in range(frames)];vertices={};s_groups={};t_groups={}
    edges=[dict(received_edge=r,status='INPUT_OR_EXECUTION_UNRESOLVED',reason='not evaluated',
                left_candidates=None,right_candidates=None,pairs=[],relation_types=[],exact_singleton_pair=False,formal_output=None)
           for r in range(frames-1)]
    try:
        for r,node in enumerate(source):
            check()
            try:records,known,values=physical(node,r,counts)
            except (ValueError,TypeError,KeyError) as error:
                nodes[r]=dict(unresolved(r,str(error)),unavailable_input_record=node);continue
            vertices.update(values);counts['vertices']+=len(values)
            nodes[r].update(input_complete=True,physical_records=records,known_count=known,
                            U=sorted({v['n'] for v in values.values()}),vertex_ids=list(values))
        unavailable_frames=[node['received_frame'] for node in nodes if not node['input_complete']]
        complete=not unavailable_frames
        for node in nodes:
            check();ids=node['vertex_ids']
            for a,b in combinations(ids,2):
                counts['S_pair_checks']+=1;v,w=vertices[a],vertices[b]
                if v['n']==w['n'] and v['j']!=w['j'] and v['lane']!=w['lane']:
                    counts['S_pairs']+=1;v['S_partners'].append(b);w['S_partners'].append(a);v['S']=w['S']=True
            for uid in ids:
                v=vertices[uid];key=f"r{v['r']}/n{v['n']}"
                s_groups.setdefault(key,dict(r=v['r'],n=v['n'],vertices=[]))['vertices'].append(uid)
                v['S_group']=key
        for uid,v in vertices.items():
            group=t_groups.setdefault(str(v['j']),dict(j=v['j'],lane=v['lane'],ages={str(a):[] for a in range(5)}))
            group['ages'][str(v['age'])].append(dict(r=v['r'],vertex=uid))
        for group in t_groups.values():
            check()
            for members in group['ages'].values():
                if members:
                    members.sort(key=lambda x:x['r']);counts['sort_groups']+=1;counts['sort_items']+=len(members)
        for v in vertices.values():
            check();ages=t_groups[str(v['j'])]['ages']
            for age in range(5):
                if age==v['age']:continue
                counts['T_age_existence_checks']+=1;members=ages[str(age)]
                before=age<v['age'];example=members[0 if before else -1] if members else None
                if example is not None and (example['r']<v['r'] if before else example['r']>v['r']):
                    counts['T_witnessed_age_groups']+=1;v['T']=True
                    v['T_groups'].append(dict(j=v['j'],age=age,received_relation='BEFORE' if before else 'AFTER',
                        group=str(v['j']),existence_example=example,all_partners='every group member satisfying strict received_relation'))
        for node in nodes:
            check()
            if not node['input_complete']:continue
            witnessed=defaultdict(set)
            for uid in node['vertex_ids']:
                v=vertices[uid];v['witness_type']='BOTH' if v['S'] and v['T'] else 'S_ONLY' if v['S'] else 'T_ONLY' if v['T'] else 'UNWITNESSED'
                if v['S']:witnessed[v['n']].add('S')
                if v['T']:witnessed[v['n']].add('T')
            W=sorted(witnessed)
            node.update(W=W if complete else None,candidates=W if complete else None,observed_W=W,
                relation_complete=complete,W_witness_types={str(n):sorted(types) for n,types in witnessed.items()},
                evaluable=complete,status='INPUT_OR_EXECUTION_UNRESOLVED' if not complete
                else 'NO_LEGAL_SEED' if not node['U'] else 'NO_RELATION_WITNESS' if not W
                else 'RELATION_SINGLETON' if len(W)==1 else 'RELATION_AMBIGUITY')
            if complete:node.pop('reason',None)
            else:node['reason']='view has unavailable physical frames; T partner set is incomplete'
        for r,(left,right) in enumerate(zip(nodes,nodes[1:])):edges[r]=edge(left,right,r,counts,check)
    except BaseException as error:
        error.partial=dict(nodes=nodes,edges=edges,vertices=vertices,S_groups=s_groups,T_groups=t_groups,
                           relation_complete=False,formal_output=None);raise
    return dict(nodes=nodes,edges=edges,vertices=vertices,S_groups=s_groups,T_groups=t_groups,
                relation_complete=complete,unavailable_received_frames=unavailable_frames,formal_output=None,presence_claim=False,
                T_representation='full sorted age groups plus strict directional references; no expanded T edge list or selected path')
