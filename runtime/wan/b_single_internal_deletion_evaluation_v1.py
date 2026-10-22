"""Card103 private post only. Stdlib arithmetic; never imported by pure receiver."""
from collections import Counter

def mapping(private):
    a,b,d=(private[k] for k in ('a','b','d'))
    return [a+r+(d if r>=b else 0) for r in range(121)]

def quality_coordinates(source_frames):
    frames=[dict(frame=t,source_frame=f) for t,f in enumerate(source_frames)]
    adjacent=[dict(frame_before=t-1,frame_after=t,source_frame_before=source_frames[t-1],source_frame_after=source_frames[t],
        spans_private_deletion_seam=source_frames[t]!=source_frames[t-1]+1) for t in range(1,len(source_frames))]
    return frames,adjacent

def geometry(private):
    frames=mapping(private);kept=set(frames)
    a,b,d=(private[k] for k in ('a','b','d'))
    gap=set(range(a+b,a+b+d));segment=set(range(a,a+121+d))
    cells=[]
    for j in range(1,45):
        ss=set(range(4*j-3,4*j+1));retained=len(ss&kept);missing_gap=len(ss&gap);boundary=len(ss-segment)
        cells.append(dict(cell=j,group=(j-1)//11,retained_frames=retained,
            INTERNAL_GAP=missing_gap,SOURCE_SEGMENT_BOUNDARY=boundary,
            cause='BOTH' if missing_gap and boundary else 'INTERNAL_GAP' if missing_gap else 'SOURCE_SEGMENT_BOUNDARY' if boundary else 'NONE',
            presence='FULL' if retained==4 else 'PARTIAL' if retained else 'ABSENT'))
    groups=[]
    for g in range(4):
        cc=cells[11*g:11*g+11];full=sum(x['presence']=='FULL' for x in cc);partial=sum(x['presence']=='PARTIAL' for x in cc)
        ng=sum(x['INTERNAL_GAP'] for x in cc);nb=sum(x['SOURCE_SEGMENT_BOUNDARY'] for x in cc)
        groups.append(dict(group=g,full_cells=full,partial_cells=partial,absent_cells=11-full-partial,
            retained_frames=sum(x['retained_frames'] for x in cc),INTERNAL_GAP=ng,SOURCE_SEGMENT_BOUNDARY=nb,
            cause='BOTH' if ng and nb else 'INTERNAL_GAP' if ng else 'SOURCE_SEGMENT_BOUNDARY' if nb else 'NONE',
            presence='PRESENT' if full else 'PARTIAL_ONLY' if partial else 'STRUCTURAL_MISSING'))
    observed=set()
    for origin in (a,a+76+d):
        p=(-origin)%4;s=(origin+p)//4+1;observed.update(range(s,s+10))
    for g in groups:g['nominal_observed']=any((j-1)//11==g['group'] for j in observed)
    return dict(source_frames=frames,cells=cells,groups=groups,nominal_observed_cells=sorted(observed))

def references(decision,anchor=None):
    result=[]
    for group in decision.get('groups',[]):
        for r in group.get('references',group.get('conditional_references',[])):
            row=dict(r,group=group['group'])
            if anchor is not None:
                row['anchor']=anchor
                w=0 if anchor=='left' else 76
                row['received_interval']=[w+x for x in row['received_interval_inclusive']]
            result.append(row)
    return result

def reference_truth(ref,frames):
    lo,hi=ref['received_interval'];actual=[frames[r] for r in range(lo,hi+1)]
    cells=sorted({(f-1)//4+1 for f in actual});groups=sorted({(c-1)//11 for c in cells if 1<=c<=44})
    valid=all(1<=x<=176 for x in actual)
    whole=valid and len(cells)==1 and actual==list(range(4*cells[0]-3,4*cells[0]+1))
    kind='SINGLE_SOURCE_FULL' if whole else 'MIXED_SAME_GROUP' if valid and len(groups)==1 else 'MIXED_GROUP_UNDEFINED'
    return dict(actual_frames=actual,actual_cells=cells,actual_groups=groups,kind=kind,
        exact_canonical=whole and cells[0]==ref['canonical_cell'],
        actual_group=groups[0] if valid and len(groups)==1 else None)

def assess(decision,meta,private,bits,anchor=None,parallel=None):
    geo=geometry(private);target=meta['arm'] in ('P0','PM') and meta['key']=='key00'
    rr=references(decision,anchor)
    if parallel is not None:rr=references(parallel['left'],'left')+references(parallel['right'],'right')
    refs=[dict(r,actual_post=reference_truth(r,geo['source_frames'])) for r in rr]
    messages=[];counts=Counter(message_slots=32)
    for slot,row in enumerate(decision['message']):
        g=slot//8;x=row['decoded'];read=x is not None;pg=geo['groups'][g];present=pg['presence']=='PRESENT'
        group_refs=[r for r in refs if r['group']==g and (parallel is None or (read and parallel[r['anchor']]['message'][slot]['decoded']==x))]
        # Union support comes only from anchors that emitted this exact slot/value.
        group_refs=[r for r in group_refs if r.get('values',[None]*8)[slot%8]==x] if read else []
        correct=x==bits[slot] if read and target and present else None
        supported=any(r['actual_post']['exact_canonical'] for r in group_refs)
        group_supported=any(r['actual_post']['actual_group']==g for r in group_refs)
        mm=dict(row,slot=slot,group=g,bit=slot%8,physical=pg['presence'],
            PRESENT_UNOBSERVED=present and not pg['nominal_observed'],
            literal_correct=correct,keyed_target_available=target and present,
            OFF_no_target_output=read and meta['arm']=='OFF',
            wrong_key_output=read and meta['arm'] in ('P0','PM') and meta['key']!='key00',
            false_missing_group_output=read and pg['presence']=='STRUCTURAL_MISSING',
            partial_only_output=read and pg['presence']=='PARTIAL_ONLY',
            exact_canonical_support_exists=supported if read else None,
            canonical_unsupported_output=read and not supported,
            supported_canonical_correct=(correct is True and supported) if read and target and present else None,
            support_reference_indices=[i for i,r in enumerate(refs) if r in group_refs],
            wrong_group_output=read and bool(group_refs) and not group_supported,
            actual_group_support_exists=group_supported if read else None)
        messages.append(mm)
        counts['read']+=read;counts['erased']+=not read;counts['literal_errors']+=correct is False
        counts['literal_correct']+=correct is True;counts['correct']+=mm['supported_canonical_correct'] is True;counts['physical_present']+=present;counts['structural_missing']+=pg['presence']=='STRUCTURAL_MISSING'
        counts['partial_only']+=pg['presence']=='PARTIAL_ONLY';counts['present_unobserved']+=mm['PRESENT_UNOBSERVED']
        for k in ('OFF_no_target_output','wrong_key_output','false_missing_group_output','partial_only_output','canonical_unsupported_output','wrong_group_output'):counts[k]+=mm[k]
        counts['technical_message_slots']+=str(row.get('status','')).startswith(('TECHNICAL','CALIBRATION','UNRESOLVED'))
    return dict(message=messages,counts=dict(counts),references_post=refs,
        geometry=geo,metadata_post_only=meta,physical_presence_not_blindly_claimed=True)

def compare(new,old):
    a,b=new['message'],old['message']
    common=[i for i in range(32) if a[i]['decoded'] is not None and b[i]['decoded'] is not None and a[i]['keyed_target_available']]
    ne=sum(a[i]['literal_correct'] is False for i in common);oe=sum(b[i]['literal_correct'] is False for i in common)
    result=dict(common_indices=common,common_count=len(common),new_errors=ne,old_errors=oe,
        new_common_BER=ne/len(common) if common else None,old_common_BER=oe/len(common) if common else None)
    for label,field in [('literal','literal_correct'),('supported_canonical','supported_canonical_correct')]:
        retained=[i for i in range(32) if a[i][field] is True and b[i][field] is True]
        lost=[i for i in range(32) if b[i][field] is True and a[i][field] is not True]
        added=[i for i in range(32) if a[i][field] is True and b[i][field] is not True]
        result[label]=dict(correct_retained=retained,correct_lost=lost,correct_added=added)
    result.update(result['supported_canonical'])
    result['primary_coverage']='supported_canonical; literal reported separately'
    return result

def physical_pair(host,private,bits):
    rows=[];counts=Counter(raw_bit_slots=160)
    for anchor,o in [('left',private['a']),('right',private['a']+76+private['d'])]:
        p=(-o)%4;s=(o+p)//4+1
        aa=host['q02']['anchor_bits'][anchor].get(str(p),[])
        bb=host['q04']['anchor_bits'][anchor].get(str(p),[])
        for r in range(10):
            g=(s+r-1)//11
            for bit in range(8):
                i=8*r+bit;x=aa[i] if i<len(aa) else None;y=bb[i] if i<len(bb) else None
                common=x is not None and y is not None;target=bits[8*g+bit]
                counts['common']+=common;counts['P0_errors']+=common and x!=target;counts['PM_errors']+=common and y!=target
                counts['correct_lost']+=common and x==target and y!=target;counts['correct_added']+=common and x!=target and y==target
                rows.append(dict(anchor=anchor,phase=p,cell=s+r,bit=bit,P0=x,PM=y,target_post_only=target))
    return dict(counts=dict(counts),rows=rows,oracle_phase_post_only=True,new_blind_BER=False)

def source_selection_cost(pack,assessed,private,bits):
    count=Counter(original_raw_bit_slots=160)
    for anchor,o in [('left',private['a']),('right',private['a']+76+private['d'])]:
        p=(-o)%4;s=(o+p)//4+1;cache=pack['anchor_bits'].get(anchor,{}).get(str(p),[])
        for r in range(10):
            cell=s+r;g=(cell-1)//11
            for bit in range(8):
                old=cache[8*r+bit] if 8*r+bit<len(cache) else None
                mm=assessed['message'][8*g+bit]
                exact=mm.get('selected_cell')==cell and mm['exact_canonical_support_exists'] is True and mm['decoded'] is not None
                new=mm['decoded'] if exact else None;target=bits[8*g+bit]
                count['original_read']+=old is not None;count['new_exact_read']+=new is not None
                count['unselected_or_erased_raw_bits']+=new is None
                count['correct_retained']+=old==target and new==target
                count['correct_lost']+=old==target and new!=target
                count['correct_added']+=old!=target and new==target
                common=old is not None and new is not None
                count['common']+=common;count['old_common_errors']+=common and old!=target;count['new_common_errors']+=common and new!=target
    result=dict(count)
    for prefix in ('old','new'):result[prefix+'_common_BER']=result.get(prefix+'_common_errors',0)/count['common'] if count['common'] else None
    return result

def comparison_summaries(rows):
    names=('continuous','unaligned','independent_left','independent_right','independent_union')
    def empty():return {k:Counter() for k in names}
    all_queries,positive=empty(),empty()
    classes={arm+'_'+key:dict(query_slots=0,message_slots=0,comparisons=empty()) for arm in ('OFF','P0','PM') for key in ('correct_key','wrong_key')}
    for row in rows:
        meta=row['main']['metadata_post_only'];label=meta['arm']+('_correct_key' if meta['key']=='key00' else '_wrong_key')
        scope=classes[label];scope['query_slots']+=1;scope['message_slots']+=32
        for target in [all_queries,scope['comparisons']]+([positive] if row['positive'] else []):
            for name,c in row['comparisons'].items():
                target[name]['query_slots']+=1;target[name]['message_slots']+=32
                for k in ('correct_retained','correct_lost','correct_added'):
                    target[name][k]+=len(c[k]);target[name]['literal_'+k]+=len(c['literal'][k])
                for k in ('common_count','new_errors','old_errors'):target[name][k]+=c[k]
    def plain(v):return {k:dict(x) for k,x in v.items()}
    return dict(comparisons=plain(positive),comparisons_primary_scope='positive_only_supported_canonical',
        comparisons_positive_only=plain(positive),comparisons_all_queries=plain(all_queries),
        comparisons_by_query_class={k:dict(query_slots=v['query_slots'],message_slots=v['message_slots'],comparisons=plain(v['comparisons'])) for k,v in classes.items()})

def evaluate(saved,private_roster,writer_bits):
    hosts={};total=Counter()
    for h,queries in saved.items():
        private=private_roster[h];qq={}
        for q,pack in queries.items():
            meta=pack['metadata_post_only'];positive=meta==dict(arm='PM',key='key00')
            main=assess(pack['main'],meta,private,writer_bits)
            controls={name:assess(pack[name],meta,private,writer_bits) for name in ('continuous','unaligned')}
            for anchor in ('left','right'):controls['independent_'+anchor]=assess(pack['independent'][anchor],meta,private,writer_bits,anchor=anchor)
            controls['independent_union']=assess(pack['independent_union'],meta,private,writer_bits,parallel=pack['independent'])
            C=pack['main']['C'];true_in=any(r['a']==private['a'] and r['d']==private['d'] and r['b_min']<=private['b']<=r['b_max'] for r in C)
            detected=pack['main']['detection']
            row=dict(main=main,controls=controls,positive=positive,true_path_in_C=true_in,
                detected=detected,C_pairs=len(C),C_paths=pack['main']['C_paths'],
                coarse_jump_only=pack['main']['coarse_jump_only'],seam_exactly_observed=False,
                ranges={k:[min(r[k] for r in C),max(r[k] for r in C)] if C else None for k in ('a','d')},
                retained_seam_band=[45,76] if any(r['d']>0 for r in C) else None,continuous_present=any(r['d']==0 for r in C),
                endpoint_correspondence_post={name:dict(true_origin=true,retained_origins=sorted({origin(r) for r in C}),
                    true_origin_in_set=any(origin(r)==true for r in C),true_phase=(-true)%4,
                    retained_phases=sorted({(-origin(r))%4 for r in C})) for name,true,origin in [
                    ('left',private['a'],lambda r:r['a']),('right',private['a']+76+private['d'],lambda r:r['a']+76+r['d'])]},
                common_canonical_cells=sum(len(g['common_cells']) for g in pack['main']['groups']),
                chosen_cells=sum(g['selected_cell'] is not None for g in pack['main']['groups']),
                source_bit_selection_cost=source_selection_cost(pack,main,private,writer_bits) if positive else None,
                comparisons={k:compare(main,v) for k,v in controls.items()})
            qq[q]=row;total['query_slots']+=1;total['positive_queries' if positive else 'null_queries']+=1
            total['positive_detected' if positive else 'null_detected']+=detected
            total['positive_true_path_in_C']+=positive and true_in
            for k,v in main['counts'].items():total['all_'+k]+=v
            if positive:
                for k,v in main['counts'].items():total['positive_'+k]+=v
        hosts[h]=dict(queries=qq,physical_P0_PM_true_phase_pair=physical_pair(queries,private,writer_bits))
    return dict(status='DEVELOPMENT_POST_ONLY',hosts=hosts,totals=dict(total),
        **comparison_summaries([row for host in hosts.values() for row in host['queries'].values()]),
        fixed_evaluation=dict(hosts=2,queries=12,null_queries=10,positive_queries=2,joint_message_slots=384,
            positive_slots=64,physical_present=56,structural_missing=8,present_unobserved=16,nominal_observed=40,
            logical_paths=703452,all_control_message_plans=1920,physical_true_phase_raw_bit_slots=320,
            true_phase_cell_views=40,full_source_cell_slots=88,full_source_bit_slots=704),
        development_only=True,CFAR=False,scientific_PASS=None,authenticated=False,exact_seam_claim=False)
