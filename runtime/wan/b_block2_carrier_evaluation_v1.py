"""Post-only saved JSON evaluation. No latent/media/model/FFT or demodulation."""
from pathlib import Path
import json
from main.tube_state import b_block2_carrier_v1 as m

def read(p):
    def bad(v):raise ValueError('nonfinite JSON '+v)
    return json.loads(Path(p).read_text(),parse_constant=bad)

def compare(base,candidate,target):
    defined=target in (0,1) and type(target) is int
    a=base in (0,1) and type(base) is int;b=candidate in (0,1) and type(candidate) is int
    return dict(baseline=base,candidate=candidate,target_post_only=target,common=defined and a and b,
        baseline_error=defined and a and base!=target,candidate_error=defined and b and candidate!=target,
        correct_retained=defined and a and b and base==target and candidate==target,
        correct_lost=defined and a and base==target and (not b or candidate!=target),
        correct_added=defined and b and candidate==target and (not a or base!=target),
        both_wrong=defined and a and b and base!=target and candidate!=target,
        added_coverage=not a and b,lost_coverage=a and not b,undefined=not defined or not a or not b)

def summarize(rows):
    names=('common','baseline_error','candidate_error','correct_retained','correct_lost','correct_added','both_wrong','added_coverage','lost_coverage','undefined')
    result={k:sum(r[k] for r in rows) for k in names};result['planned_bits']=len(rows)
    n=result['common']
    result['baseline_common_BER']=sum(r['baseline_error'] for r in rows if r['common'])/n if n else None
    result['candidate_common_BER']=sum(r['candidate_error'] for r in rows if r['common'])/n if n else None
    return result

def evaluate(output,protocol):
    out=Path(output)
    if not (out/'markers/blind_complete.json').is_file():raise RuntimeError('blind products not saved/read back')
    issues=[];bits=[None]*32
    try:
        b=read(protocol)['config']['writer']['bits']
        if not isinstance(b,list) or len(b)!=32 or any(type(x) is not int or x not in (0,1) for x in b):raise ValueError('actual writer bits must be 32 binary integers')
        bits=b
    except (OSError,ValueError,KeyError,TypeError) as e:issues.append(f'{type(e).__name__}: {e}')
    tag_rows=[];chip_rows=[];summary={};lookup={}
    for t in m.tables():
        layer,h,arm,family,key=t;name='/'.join(t)
        raw=read(out/'raw'/(name+'.json'));final=read(out/'blind'/(name+'.json'))
        positive=arm in ('RANDOM','BLOCK2') and key=='key00'
        for i,(r,f) in enumerate(zip(raw['rows'],final['rows']),1):
            d=f['decoder'];technical=d.get('technical_failure',False)
            target=m.core.packet(i-1,m.book(family,m.KEYS['key00'])) if positive and not technical else (None,)*48
            errors=[]
            if positive:
                for l,(actual,expected) in enumerate(zip(d['physical_bits'],target)):
                    err=None if actual is None or expected is None else actual!=expected;errors.append(err)
                    chip_rows.append(dict(layer=layer,host=h,arm=arm,family=family,key=key,cell=i,chip=l,kind='pilot' if l in m.core.PILOT_CHIPS else 'data',physical_bit=actual,writer_bit_post_only=expected,error=err,technical=technical,q=r['projections'][l] if r['projections'] is not None else None))
            row=dict(layer=layer,host=h,arm=arm,family=family,key=key,cell=i,positive=positive,null=not positive,technical=technical,
                status=d['status'],pilot_P=d['pilot_P'],pilot_H=d['pilot_H'],pilot_detected=d['pilot_detected'],code_status=d['code_status'],
                nearest=d['nearest'],nearest_ids=d['nearest_ids'],d_min=d['nearest_distance'],d_true=d['code_distances'][i-1] if positive else None,
                tag_label=d['tag_label'],correct_ID=positive and d['tag_label']==i,wrong_legal_ID=positive and d['tag_label'] is not None and d['tag_label']!=i,
                false_legal_null=not positive and d['tag_label'] is not None,false_presence_null=not positive and d['pilot_detected'] is True,
                pilot_bit_errors=sum(errors[l] is True for l in m.core.PILOT_CHIPS) if positive else None,
                data_bit_errors=sum(errors[l] is True for l in m.core.DATA_CHIPS) if positive else None,
                pilot_erased=sum(errors[l] is None for l in m.core.PILOT_CHIPS) if positive else None,
                data_erased=sum(errors[l] is None for l in m.core.DATA_CHIPS) if positive else None)
            tag_rows.append(row);lookup[(layer,h,arm,key,i)]=row
            group='/'.join((layer,family,arm,key));s=summary.setdefault(group,dict(planned_rows=0,technical=0,correct_ID=0,wrong_legal_ID=0,false_legal_null=0,false_presence_null=0,pilot_bit_errors=0,data_bit_errors=0,pilot_erased=0,data_erased=0))
            s['planned_rows']+=1
            for k in list(s)[1:]:s[k]+=row[k] or 0
    carrier_pairs=[]
    for h in m.HOSTS:
        for i in range(1,45):carrier_pairs.append(dict(host=h,cell=i,RANDOM=lookup[('full',h,'RANDOM','key00',i)],BLOCK2=lookup[('full',h,'BLOCK2','key00',i)]))
    payload=[];payload_summary={};payload_raw={}
    for h in m.HOSTS:
        for arm in ('P0','RANDOM','BLOCK2'):
            for key in m.KEYS:
                p=read(out/'payload'/h/arm/(key+'.json'));payload_raw[(h,arm,key)]=p
                # Wrong-key output retained separately; not an authentication estimate.
                for j,actual in enumerate(p['decoded_bits']):payload.append(dict(host=h,arm=arm,key=key,kind='group',bit=j,decoded=actual,target_post_only=bits[j] if key=='key00' else None,status=p['status']))
    pair_groups=[];pair_cells=[]
    for h in m.HOSTS:
        base=payload_raw[(h,'P0','key00')]
        for a in ('RANDOM','BLOCK2'):
            p=payload_raw[(h,a,'key00')];groups=[];cells=[]
            for j in range(32):groups.append(dict(host=h,arm=a,bit=j,**compare(base['decoded_bits'][j],p['decoded_bits'][j],bits[j])))
            for t in range(44):
                for j in range(8):
                    bit=8*(t//11)+j
                    cells.append(dict(host=h,arm=a,cell=t+1,slot=j,bit=bit,**compare(base['local_cells'][t]['bits'][j]['decoded'],p['local_cells'][t]['bits'][j]['decoded'],bits[bit])))
            pair_groups.extend(groups);pair_cells.extend(cells)
            payload_summary[h+'/'+a]=dict(groups=summarize(groups),cells=summarize(cells))
    from runtime.wan.b_full_context_local_read_v1 import save
    save(out/'tag_post_rows.json',tag_rows);save(out/'tag_post_chips.json',chip_rows);save(out/'carrier_full_pairs.json',carrier_pairs)
    save(out/'payload_post_rows.json',payload);save(out/'payload_group_pairs.json',pair_groups);save(out/'payload_cell_pairs.json',pair_cells)
    return dict(status='POST_TRUTH_UNRESOLVED' if issues else 'POST_DESCRIPTION_ONLY',reference_issues=issues,
        tag_summary=summary,payload_summary=payload_summary,payload_total=dict(groups=summarize(pair_groups),cells=summarize(pair_cells)),
        fixed_denominators=dict(tag_rows=4224,positive_rows=704,null_rows=3520,primary_full_pairs=176,payload_group_pairs=256,payload_cell_pairs=2816,quality_pairs=24),
        independent_confirmation=False,old_tau_descriptive_only=True,full_FPR_validated=False,authentication_claim=False,scientific_PASS=None,
        payload_scope='known public complete clean span; full-context cell votes are not cropped-local or tag-guided recovery')
