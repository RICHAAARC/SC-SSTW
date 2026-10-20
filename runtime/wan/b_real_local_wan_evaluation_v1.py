"""Post-only three-layer local geometry, mixed truth, and fixed32 message evaluation."""
from collections import Counter
from main.tube_state import b_real_local_wan_v1 as method


def geometry(start,stop):
    """Preparation truth only. All45 frames enter E; local latent0 is unread."""
    if stop-start!=45:raise ValueError('fixed45-frame truth span required')
    present=set(range(start,stop));source=[]
    for j in range(1,45):
        overlap=sorted(present&set(range(4*j-3,4*j+1)))
        source.append(dict(source_cell=j,group=(j-1)//11,visible_frames=overlap,
            physical_status='FULL' if len(overlap)==4 else 'PARTIAL' if overlap else 'ABSENT'))
    rows=[]
    for r in range(1,12):
        frames=list(range(start+4*r-3,start+4*r+1));counts=Counter((f-1)//4+1 for f in frames)
        members=sorted(counts);groups=sorted({(j-1)//11 for j in members})
        rows.append(dict(local_row=r,nominal_frames=frames,nominal_source_cells=members,
            nominal_frame_counts={str(j):counts[j] for j in members},source_identity_kind='SINGLE_CELL' if len(members)==1 else 'MIXED_UNDEFINED',
            true_source_cell=members[0] if len(members)==1 else None,true_group=groups[0] if len(groups)==1 else None,
            nominal_groups=groups,bit_reference_kind='SINGLE_CELL' if len(members)==1 else 'GROUP_ONLY' if len(groups)==1 else 'MIXED_GROUP_UNDEFINED'))
    groups=[g for g in range(4) if any(s['group']==g and s['physical_status']=='FULL' for s in source)]
    return dict(start=start,stop=stop,all45_frames_enter_E=True,first_frame_context_only=start,first_latent_excluded=0,
        source_cells=source,observed_cells=rows,physically_present_message_groups=groups,
        present_message_bits=8*len(groups),structurally_missing_message_bits=32-8*len(groups),
        causal_history='reset, not equal to full-history',frame_membership_is_not_linear_q_mixture=True)


def evaluate(case_metadata,observations,matches,mappings,votes,payload,reference_q,reference_matches,reference_raw,truth):
    bits=truth.get('bits');tables={};pairs={}
    for case,meta in case_metadata.items():
        geo=geometry(meta['start'],meta['stop']);physical={s['source_cell']:s for s in geo['source_cells']}
        for key in method.KEY_IDS:
            name=case+'/'+key;mapping=mappings[name];raw=votes[name];conditioned=payload[name];match=matches[name]
            old_bits=reference_raw.get(meta['arm']+'/'+key)
            local=[];cells=[]
            for i,g in enumerate(geo['observed_cells']):
                hypothesis=mapping['cells'][i];j=hypothesis['eligible_source_cell'];claimed_g=hypothesis['eligible_group']
                single=g['true_source_cell'];reference_g=g['true_group']
                cell=dict(**g,claimed_source_cell=j,claimed_group=claimed_g,
                    conditional_cell_correct=None if single is None or j is None else j==single,
                    nominal_overlap_compatible=None if j is None else j in g['nominal_source_cells'],
                    unsupported_single_cell_claim=j is not None and single is None,
                    claimed_physical_status=None if j is None else physical[j]['physical_status'],
                    conditional_group_compatible=None if reference_g is None or claimed_g is None else reference_g==claimed_g)
                cells.append(cell)
                for b in range(8):
                    value=raw['local_cells'][i]['bits'][b]['decoded']
                    local.append(dict(local_row=i+1,slot=b,raw_decoded=value,truth_reference_kind=g['bit_reference_kind'],
                        true_source_cell=single,true_group=reference_g,claimed_source_cell=j,claimed_group=claimed_g,
                        raw_reference_correct=None if value is None or reference_g is None or bits is None else value==bits[8*reference_g+b],
                        claimed_message_bit_correct=None if value is None or claimed_g is None or bits is None else value==bits[8*claimed_g+b],
                        reference_undefined=reference_g is None,raw_unread=value is None,
                        full_history_same_cell_bit=None if single is None or old_bits is None else old_bits[i][b],
                        raw_changed_from_full_history=None if single is None or old_bits is None or value is None or old_bits[i][b] is None else value!=old_bits[i][b]))
            source=[]
            for j in range(1,45):
                claims=[c['local_row'] for c in cells if c['claimed_source_cell']==j]
                correct=[c['local_row'] for c in cells if c['claimed_source_cell']==j and c['true_source_cell']==j]
                mixed=any(c['true_source_cell'] is None for c in cells)
                source.append(dict(**physical[j],claimed_local_rows=claims,correct_single_cell_support=correct,
                    conditional_false_source_coverage=None if mixed else bool(claims) and not bool(correct),
                    claim_to_absent_source=bool(claims) and physical[j]['physical_status']=='ABSENT',
                    claim_to_partial_source=bool(claims) and physical[j]['physical_status']=='PARTIAL',
                    mixed_identity_unverifiable=mixed and bool(claims)))
            message=[]
            for b,value in enumerate(conditioned['decoded_bits']):
                exists=b//8 in geo['physically_present_message_groups']
                message.append(dict(slot=b,decoded=value,structurally_present=exists,
                    correct_present=None if value is None or not exists or bits is None else value==bits[b],
                    conditional_false_missing_group_coverage=value is not None and not exists,
                    matches_truth_without_source_support=None if value is None or exists or bits is None else value==bits[b]))
            readable=[row for row in message if row['structurally_present'] and row['decoded'] is not None]
            cell_evaluable=[c for c in cells if c['true_source_cell'] is not None]
            refkey=meta['arm']+'/'+key;old=reference_matches[refkey]
            ref_compare=None
            if len(cell_evaluable)==11:
                old_q=reference_q.get(refkey);new_q=observations[name]
                if old_q is not None and new_q['status']=='SCORE_OBSERVED':
                    ref_compare=dict(scope='known-origin cross-run full-history versus reset; not a receiver input',
                        per_cell=[dict(local_row=i+1,source_cell=16+i,full_q=old_q[i][15+i],local_q=new_q['pooled_q'][i][15+i],
                            local_minus_full_q=new_q['pooled_q'][i][15+i]-old_q[i][15+i]) for i in range(11)],
                        full_sequence=old,full_raw_bits=reference_raw.get(refkey),no_reference_FFT=True)
            target_start=16 if len(cell_evaluable)==11 else None
            target_score=None;target_rank=None
            if target_start is not None and match['comparison_status']=='COMPLETE':
                target_score=match['scores'][target_start-1]['score'];target_rank=1+sum(x['score']>target_score for x in match['scores'])
            tables[name]=dict(case=case,arm=meta['arm'],key=key,preparation_geometry_post_only=geo,
                observation_status=observations[name]['status'],sequence=match,post_singleton_true_start=target_start,
                true_start_score=target_score,true_start_rank=target_rank,source_cell_rows=source,observed_cells=cells,
                cell_denominator=11,singleton_geometry_rows=len(cell_evaluable),mixed_identity_rows=11-len(cell_evaluable),
                cell_identity_evaluable=bool(cell_evaluable),
                conditional_singleton_correct_fraction=None if not cell_evaluable else sum(c['conditional_cell_correct'] is True for c in cell_evaluable)/11,
                conditional_correct_singleton_cells=sum(c['conditional_cell_correct'] is True for c in cells),
                conditional_wrong_singleton_cells=sum(c['conditional_cell_correct'] is False for c in cells),
                unread_singleton_cells=sum(c['claimed_source_cell'] is None for c in cell_evaluable),
                unsupported_mixed_singleton_claims=sum(c['unsupported_single_cell_claim'] for c in cells),
                local_bit_denominator=88,local_bits=local,raw_defined_bit_references=sum(not x['reference_undefined'] for x in local),
                undefined_bit_references=sum(x['reference_undefined'] for x in local),
                raw_defined_errors=None if bits is None else sum(x['raw_reference_correct'] is False for x in local),
                raw_single_cell_readable_bits=sum(x['truth_reference_kind']=='SINGLE_CELL' and x['raw_decoded'] is not None for x in local),
                raw_group_only_readable_bits=sum(x['truth_reference_kind']=='GROUP_ONLY' and x['raw_decoded'] is not None for x in local),
                raw_single_cell_errors=None if bits is None else sum(x['truth_reference_kind']=='SINGLE_CELL' and x['raw_reference_correct'] is False for x in local),
                raw_group_only_errors=None if bits is None else sum(x['truth_reference_kind']=='GROUP_ONLY' and x['raw_reference_correct'] is False for x in local),
                raw_unread_bits=sum(x['raw_unread'] for x in local),
                message_denominator=32,message=message,structurally_present_bits=16,structurally_missing_bits=16,
                readable_present_bits=len(readable),present_bit_errors=None if bits is None else sum(x['correct_present'] is False for x in readable),
                present_BER=None if bits is None or not readable else sum(x['correct_present'] is False for x in readable)/len(readable),
                unread_present_bits=sum(x['structurally_present'] and x['decoded'] is None for x in message),
                conditional_false_missing_group_bits=sum(x['conditional_false_missing_group_coverage'] for x in message),
                reference_comparison=ref_compare,identity_status='UNCALIBRATED',phase_status='UNCALIBRATED',trusted_full_message=None)
    for number in (0,1):
        for key in method.KEY_IDS:
            ids=[next(c for c,m in case_metadata.items() if m['arm']==arm and m['variant']==number)+'/'+key for arm in ('P0','PM')]
            a,b=[tables[x] for x in ids]
            common=[i for i in range(32) if a['message'][i]['structurally_present'] and a['message'][i]['decoded'] is not None and b['message'][i]['decoded'] is not None]
            pairs[f'variant{number}/{key}']=dict(common_present_indices=common,
                P0_common_present_BER=None if bits is None or not common else sum(a['message'][i]['correct_present'] is False for i in common)/len(common),
                PM_common_present_BER=None if bits is None or not common else sum(b['message'][i]['correct_present'] is False for i in common)/len(common),
                added_present_readability=[i for i in range(32) if a['message'][i]['structurally_present'] and a['message'][i]['decoded'] is None and b['message'][i]['decoded'] is not None],
                lost_present_readability=[i for i in range(32) if a['message'][i]['structurally_present'] and a['message'][i]['decoded'] is not None and b['message'][i]['decoded'] is None],
                new_physical_BER_gain_claim=False)
    return dict(status='POST_ONLY',tables=tables,paired_arms=pairs,truth_status=truth['status'],
        fixed_denominators=dict(clip_key_cases=8,observed_cells=88,source_cells=352,local_bits=704,message_bits=256,
            singleton_geometry_rows=44,mixed_identity_rows=44,local_defined_single_cell_bits=352,local_group_only_bits=320,
            local_mixed_group_undefined_bits=32,present_message_bits=128,missing_message_bits=128),
        identity_status='UNCALIBRATED',phase_status='UNCALIBRATED',accepted_identity_count=None,FPR=None,trusted_full_messages=None,scientific_pass=None)
