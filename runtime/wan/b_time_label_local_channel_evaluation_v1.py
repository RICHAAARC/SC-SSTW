"""Card106 private post evaluation. Stdlib saved data only; imported after marker."""
from collections import Counter
from pathlib import Path
import json
from main.tube_state import b_time_label_local_channel_v1 as layout

def cross_map(private):
    return [private['a']+i+(private['d'] if i>=private['b'] else 0) for i in range(121)]

def geometry(private,view,phase,row):
    lo,hi=layout.public_interval(phase,row)
    if view=='cross_center':
        mapping=cross_map(private);received=list(range(38+lo,38+hi+1));source=[mapping[i] for i in received]
        crossed=any(b!=a+1 for a,b in zip(source,source[1:]))
    else:
        origin=layout.CLEAN_ORIGINS[int(view[-1])]
        received=list(range(lo,hi+1));source=[origin+i for i in received];crossed=False
    cells=sorted({(x-1)//4+1 for x in source})
    status='CROSS_SEAM' if crossed else 'GUARD' if any(not 1<=x<=176 for x in source) else 'FULL' if len(cells)==1 else 'MIXED'
    return dict(status=status,source_frames=source,received_interval_relative45=[lo,hi],
        source_cells=cells,true_label=cells[0] if status=='FULL' else None,nominal_not_full_VAE_causal_support=True)

def assess(decoder,geo,arm,key):
    accepted=decoder.get('tag_label') is not None
    positive=arm=='PT' and key=='key00';null=not positive
    full=geo['status']=='FULL';expected=geo['true_label']
    correct=accepted and positive and full and decoder['tag_label']==expected
    technical=decoder.get('technical_failure',False) or decoder.get('status')=='TECHNICAL_UNRESOLVED'
    return dict(presence_null=null,positive_material=positive,nominal_geometry=geo['status'],
        expected_label_post_only=expected,accepted=accepted,correct_FULL_tag=correct,
        wrong_legal_FULL=accepted and positive and full and not correct,
        unsupported_geometry_legal=accepted and positive and not full,
        false_legal_null=accepted and null,false_presence_null=decoder.get('pilot_detected') is True and null,
        detected_label_erasure=decoder.get('status')=='DETECTED_LABEL_ERASURE',
        FULL_positive_no_correct=positive and full and not correct,
        technical=technical,code_only_reject=decoder.get('status')=='CODE_ONLY_REJECT')

def evaluate(out,cfg,writer_bits):
    out=Path(out);rows=[];summary={};strata={};geometry_counts={};by_host={}
    def add(host,role,scope,arm,key,unit,geo,view=None,phase=None):
        decoder=unit['decoder'];flags=assess(decoder,geo,arm,key)
        row=dict(host=host,role=role,scope=scope,arm=arm,key=key,view=view,phase=phase,row=unit['row'],
            status=decoder['status'],pilot_P=decoder.get('pilot_P'),pilot_H=decoder.get('pilot_H'),
            pilot_detected=decoder.get('pilot_detected'),code_status=decoder.get('code_status'),
            nearest_distance=decoder.get('nearest_distance'),tag_label=decoder.get('tag_label'),
            code_id_candidate=decoder.get('code_id_candidate'),geometry_post_only=geo,**flags)
        rows.append(row)
        for bucket,name in [(summary,role+'/'+scope),(strata,role+'/'+scope+'/'+arm+'/'+key),
                            (geometry_counts,role+'/'+scope+'/'+geo['status']),(by_host,host+'/'+scope)]:
            c=bucket.setdefault(name,Counter());c['planned_rows']+=1;c['accepted']+=flags['accepted']
            for k,v in flags.items():
                if type(v) is bool:c[k]+=v
    for host in cfg['private_roster']:
        h,role=host['id'],host['role']
        for arm in layout.ARMS:
            for key in layout.KEYS:
                table=json.loads((out/f'blind/native/{h}/{arm}/{key}.json').read_text())
                for unit in table['rows']:
                    j=unit['row'];geo=dict(status='FULL',source_frames=list(range(4*j-3,4*j+1)),true_label=j)
                    add(h,role,'native',arm,key,unit,geo)
            for view in layout.VIEWS:
                for p in range(4):
                    for key in layout.KEYS:
                        table=json.loads((out/f'blind/local/{h}/{arm}/{view}/p{p}/{key}.json').read_text())
                        for unit in table['rows']:
                            add(h,role,'local',arm,key,unit,geometry(host,view,p,unit['row']),view,p)
    payload_rows=[];payload_count=Counter()
    for host in cfg['private_roster']:
        if host['role']!='evaluation':continue
        h=host['id']
        for view in layout.VIEWS:
            for p in range(4):
                vals={arm:json.loads((out/f'payload/{h}/{arm}/{view}/p{p}/key00.json').read_text())['bits'] for arm in ('P0','PT')}
                for r in range(1,11):
                    geo=geometry(host,view,p,r);full=geo['status']=='FULL'
                    for bit in range(8):
                        x,y=vals['P0'][8*(r-1)+bit],vals['PT'][8*(r-1)+bit]
                        target=writer_bits[8*((geo['true_label']-1)//11)+bit] if full else None
                        common=full and x is not None and y is not None
                        row=dict(host=h,view=view,phase=p,row=r,bit=bit,geometry=geo['status'],
                            source_cell=geo['true_label'],P0=x,PT=y,target_post_only=target,
                            status='FULL_TARGET' if full else 'UNDEFINED_CELL_TARGET',common=common)
                        payload_rows.append(row);payload_count['planned_pair_bits']+=1
                        payload_count['FULL_target_bits']+=full;payload_count['undefined_cell_target_bits']+=not full
                        if view=='cross_center':payload_count['cross_planned']+=1;payload_count['cross_undefined']+=not full
                        payload_count['common_FULL']+=common
                        payload_count['P0_common_errors']+=common and x!=target;payload_count['PT_common_errors']+=common and y!=target
                        if full:
                            payload_count['P0_read']+=x is not None;payload_count['PT_read']+=y is not None
                            payload_count['correct_retained']+=x==target and y==target
                            payload_count['correct_lost']+=x==target and y!=target
                            payload_count['correct_added']+=x!=target and y==target
    common=payload_count['common_FULL']
    payload_summary=dict(payload_count,P0_common_BER=payload_count['P0_common_errors']/common if common else None,
                         PT_common_BER=payload_count['PT_common_errors']/common if common else None)
    return dict(status='SEEN_LOCAL_CHANNEL_POST_ONLY',summary={k:dict(v) for k,v in summary.items()},
        strata={k:dict(v) for k,v in strata.items()},geometry_counts={k:dict(v) for k,v in geometry_counts.items()},
        by_host={k:dict(v) for k,v in by_host.items()},rows=rows,payload_summary=payload_summary,payload_rows=payload_rows,
        fixed_denominators=dict(native_rows=1056,local_rows=5760,eval_local_rows=2880,eval_null_rows=2400,
            eval_PT_correct_rows=480,eval_PT_correct_FULL=118,eval_PT_correct_MIXED=350,
            eval_PT_correct_GUARD=6,eval_PT_correct_CROSS_SEAM=6,unique_IDs_per_host=44,
            payload_pair_bits=3840,payload_FULL_bits=944,payload_cross_undefined=496),
        scientific_PASS=None,independent_confirmation=False,authentication_claim=False,state_space_executed=False)
