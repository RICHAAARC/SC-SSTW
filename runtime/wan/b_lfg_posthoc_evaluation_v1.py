"""Post-only saved message/payload arithmetic. No receiver/PN/BCH/model calls."""
from pathlib import Path
from collections import Counter
from main.tube_state import b_lfg_posthoc_plan_v1 as p

def evaluate(output,original_protocol):
    from runtime.wan.b_lfg_posthoc_reference_v1 import read,save,message_path,payload_path
    out=Path(output)
    if not (out/'markers/blind_complete.json').is_file():raise RuntimeError('all blind products not saved/read back')
    issues=[];true_code=[None]*512;true_info=[None]*432;original=[None]*32
    try:
        ref=read(out/'private/writer_reference.json');c,m=ref['codeword_bits'],ref['information_bits']
        if len(c)!=512 or len(m)!=432 or any(type(x) is not int or x not in (0,1) for x in c+m):raise ValueError('fixed fixture512/432 binary geometry')
        true_code,true_info=c,m
    except (OSError,ValueError,KeyError,TypeError) as e:issues.append('new diagnostic reference: '+str(e))
    try:
        bits=read(original_protocol)['config']['writer']['bits']
        if len(bits)!=32 or any(type(x) is not int or x not in (0,1) for x in bits):raise ValueError('original32 writer bits required')
        original=bits
    except (OSError,ValueError,KeyError,TypeError) as e:issues.append('original net32 reference: '+str(e))
    rows=[];summary={}
    for t in p.tables():
        view,h,arm,key=t;raw=read(message_path(out,'raw',t));bch=read(message_path(out,'blind',t))
        got=raw['raw_codeword_bits'];info=bch['information_bits'];corrected=bch.get('corrected_codeword_bits')
        common_code=[(a,b) for a,b in zip(got,true_code) if a is not None and b is not None]
        common_info=[(a,b) for a,b in zip(info,true_info) if a is not None and b is not None]
        full_truth=all(x is not None for x in true_info)
        row=dict(view=view,host=h,arm=arm,key=key,positive=arm=='MSG' and key=='key00',score_status=raw['status'],BCH_status=bch['status'],BCH_success=bch.get('BCH_success'),
            raw_coded_denominator=512,raw_coded_read=sum(x is not None for x in got),raw_common_bits=len(common_code),raw_bit_errors=sum(a!=b for a,b in common_code) if common_code else None,
            raw_BER=sum(a!=b for a,b in common_code)/len(common_code) if common_code else None,raw_codeword_exact=all(a==b for a,b in zip(got,true_code)) if len(common_code)==512 else None,
            information_denominator=432,information_read=sum(x is not None for x in info),information_common_bits=len(common_info),information_errors=sum(a!=b for a,b in common_info) if common_info else None,
            BCH_correction_count=bch.get('correction_count'),corrected_codeword_exact=corrected==true_code if corrected is not None and all(x is not None for x in true_code) else None,
            full_message_exact=(bch.get('BCH_success') is True and len(common_info)==432 and all(a==b for a,b in common_info)) if full_truth else None,
            message_is_new_user_payload=False,authentication=False)
        rows.append(row);group='/'.join((view,arm,key));s=summary.setdefault(group,dict(planned_messages=0,raw_coded_denominator=0,raw_common_bits=0,raw_bit_errors=0,information_denominator=0,information_common_bits=0,information_errors=0,BCH_success=0,full_message_exact=0,unknown_truth=0,technical_score=0,statuses={}))
        s['planned_messages']+=1;s['raw_coded_denominator']+=512;s['information_denominator']+=432
        for name in ('raw_common_bits','raw_bit_errors','information_common_bits','information_errors'):s[name]+=row[name] or 0
        s['BCH_success']+=row['BCH_success'] is True;s['full_message_exact']+=row['full_message_exact'] is True;s['unknown_truth']+=row['full_message_exact'] is None;s['technical_score']+=row['score_status']=='TECHNICAL_UNRESOLVED';s['statuses'][row['BCH_status']]=s['statuses'].get(row['BCH_status'],0)+1
    for s in summary.values():s['raw_common_BER']=s['raw_bit_errors']/s['raw_common_bits'] if s['raw_common_bits'] else None;s['information_common_BER']=s['information_errors']/s['information_common_bits'] if s['information_common_bits'] else None
    payloads={};group_pairs=[];cell_pairs=[]
    for view in p.MEDIA_VIEWS:
        for h in p.HOSTS:
            for arm in p.ARMS:
                for key in p.KEYS:payloads[(view,h,arm,key)]=read(payload_path(out,view,h,arm,key))
            base=payloads[(view,h,'P0','key00')];marked=payloads[(view,h,'MSG','key00')]
            def compare(a,b,target):
                common=target is not None and a is not None and b is not None
                return dict(baseline=a,marked=b,target_post_only=target,common=common,baseline_error=common and a!=target,marked_error=common and b!=target,correct_retained=target is not None and a==target and b==target,correct_lost=target is not None and a==target and b!=target,correct_added=target is not None and a!=target and b==target,added_coverage=a is None and b is not None,lost_coverage=a is not None and b is None,undefined=not common)
            for i in range(32):group_pairs.append(dict(view=view,host=h,bit=i,**compare(base['decoded_bits'][i],marked['decoded_bits'][i],original[i])))
            for cell in range(44):
                for bit in range(8):
                    index=8*(cell//11)+bit
                    cell_pairs.append(dict(view=view,host=h,cell=cell+1,bit=index,**compare(base['local_cells'][cell]['bits'][bit]['decoded'],marked['local_cells'][cell]['bits'][bit]['decoded'],original[index])))
    def payload_summary(items):
        names=('common','baseline_error','marked_error','correct_retained','correct_lost','correct_added','added_coverage','lost_coverage','undefined');s={k:sum(bool(x[k]) for x in items) for k in names};s['planned']=len(items);n=s['common'];s['baseline_common_BER']=s['baseline_error']/n if n else None;s['marked_common_BER']=s['marked_error']/n if n else None;return s
    per_view={view:dict(groups=payload_summary([r for r in group_pairs if r['view']==view]),cells=payload_summary([r for r in cell_pairs if r['view']==view])) for view in p.MEDIA_VIEWS}
    save(out/'message_post_rows.json',rows);save(out/'payload_group_pairs.json',group_pairs);save(out/'payload_cell_pairs.json',cell_pairs)
    return dict(status='POST_TRUTH_UNRESOLVED' if issues else 'POST_DESCRIPTION_ONLY',reference_issues=issues,message_summary=summary,payload_by_view=per_view,payload_total=dict(groups=payload_summary(group_pairs),cells=payload_summary(cell_pairs)),fixed_denominators=p.fixed(),
        BCH_success_is_not_message_exact=True,authentication=False,scientific_PASS=None,independent_confirmation=False,author_equivalent_reproduction=False,user_net_payload_bits=32,new_user_capacity_claim=False,temporal_ID_or_localization_or_sync_claim=False)
