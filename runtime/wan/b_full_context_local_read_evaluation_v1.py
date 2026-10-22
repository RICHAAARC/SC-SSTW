"""Post-only saved-reference comparison. No receiver, cosine, model or tensor imports."""
from pathlib import Path
from collections import Counter,defaultdict
import gzip,json,math
HOSTS=('b101-s01','b101-s03','b101-s02','b101-s04')
WINDOWS=(('clean0',0,0),('clean1',1,9),('clean2',2,18),('clean3',3,27),('clean4',0,34))
PILOT=tuple(i for i in range(48) if i%3==2)
DATA=tuple(i for i in range(48) if i%3!=2)
DI={x:i for i,x in enumerate(DATA)}
def obj(x):return x if isinstance(x,dict) else {}
def at(x,i):return x[i] if isinstance(x,list) and 0<=i<len(x) else None
def finite(x):return type(x) in (int,float) and math.isfinite(x)
def bit(x):return type(x) is int and x in (0,1)
def reject(x):raise ValueError('nonfinite JSON constant: '+x)
def relation(full_correct,local_correct):
    if type(full_correct) is not bool or type(local_correct) is not bool:return 'UNDEFINED'
    if full_correct:return 'BOTH_CORRECT_RETAINED' if local_correct else 'FULL_ONLY_CORRECT_ADDED'
    return 'LOCAL_ONLY_CORRECT_LOST' if local_correct else 'BOTH_WRONG'
def geometry(q,qN,rec,delta):
    if not all(finite(v) for v in (q,qN,rec.get('target_projection'),delta)) or delta<=0:return dict(status='UNRESOLVED')
    try:
        D=q-qN;e=(q-rec['target_projection'])/delta
        if not finite(D) or not finite(e):return dict(status='UNRESOLVED')
        w=e-2*math.floor((e+1)/2);M=.5-abs(w)
        return dict(status='SCORED',D=D,epsilon_L=e,w=w,signed_same_parity_margin=M,nearest_half_integer_boundary_distance=abs(M))
    except (ArithmeticError,ValueError):return dict(status='UNRESOLVED')
def evaluate(output,reference_file,save):
    out=Path(output);issues=[]
    try:
        with gzip.open(reference_file,'rt',encoding='utf-8') as f:data=json.load(f,parse_constant=reject)
    except (OSError,ValueError,EOFError) as e:
        data={};issues.append(dict(kind='REFERENCE_UNAVAILABLE',error=type(e).__name__+': '+str(e)))
    def index(name,key):
        rr=data.get(name,[])
        if not isinstance(rr,list):rr=[];issues.append(dict(kind='REFERENCE_LIST_UNAVAILABLE',field=name))
        result={}
        for r in rr:
            if isinstance(r,dict):
                k=key(r)
                if k in result:issues.append(dict(kind='DUPLICATE_REFERENCE',field=name,key=list(k)))
                else:result[k]=r
        return result
    native=index('native_packets',lambda x:(x.get('host'),x.get('row')))
    refs=index('native_reference',lambda x:(x.get('host'),x.get('cell'),x.get('chip')))
    local=index('local_packets',lambda x:(x.get('host'),x.get('view'),x.get('phase'),x.get('row')))
    localrows=index('local_rows',lambda x:(x.get('host'),x.get('view'),x.get('phase'),x.get('row')))
    oldchips=index('local_chips',lambda x:(x.get('host'),x.get('view'),x.get('phase'),x.get('row'),x.get('chip')))
    protocol=obj(data.get('protocol'));delta=protocol.get('delta')
    full=[];chips=[];full_index={};chip_index={}
    for h in HOSTS:
        try:
            raw=json.loads((out/'raw'/h/'key00.json').read_text(),parse_constant=reject)
            final=json.loads((out/'blind'/h/'key00.json').read_text(),parse_constant=reject)
        except (OSError,ValueError) as e:
            raw=final={};issues.append(dict(kind='BLIND_FILE_UNAVAILABLE',host=h,error=str(e)))
        for j in range(1,45):
            unit=obj(at(raw.get('rows'),j-1));dec=obj(unit.get('decoder'));fin=obj(obj(at(final.get('rows'),j-1)).get('decoder'))
            nr=obj(native.get((h,j)));nd=obj(obj(nr.get('saved_unit')).get('decoder'));word=nd.get('unmasked_bits')
            technical=dec.get('technical_failure') is True or dec.get('status')=='TECHNICAL_UNRESOLVED'
            native_ok=isinstance(word,list) and len(word)==32 and all(bit(b) for b in word) and at(nd.get('code_distances'),j-1)==0
            rowchips=[]
            for l in range(48):
                ref=obj(refs.get((h,j,l)));rec=obj(ref.get('reference_record'));qN=ref.get('q_N');qF=at(unit.get('projections'),l)
                expected=rec.get('bit') if l in PILOT else at(word,DI[l])
                actual=at(dec.get('physical_bits'),l) if l in PILOT else at(dec.get('unmasked_bits'),DI[l])
                valid_ref=(bit(expected) and (native_ok if l in DI else True))
                err=(actual!=expected) if valid_ref and bit(actual) and not technical else None
                arr=dec.get('physical_bits') if l in PILOT else dec.get('unmasked_bits');position=l if l in PILOT else DI[l]
                present=isinstance(arr,list) and position<len(arr)
                geo=geometry(qF,qN,rec,delta)
                if ref.get('projection_reference_status')!='AVAILABLE':geo=dict(status='REFERENCE_UNRESOLVED')
                c=dict(host=h,cell=j,chip=l,ROI=l//12,channel=4+l%12,kind='pilot' if l in PILOT else 'data',
                    q_native=qN,q_full=qF,writer_physical_bit=rec.get('bit'),saved_physical_bit=at(dec.get('physical_bits'),l),
                    reference_bit=expected,saved_comparison_bit=actual,bit_error=err,
                    saved_symbol_erasure=present and not technical and arr[position] is None,source_technical=technical,QIM=geo)
                chips.append(c);rowchips.append(c);chip_index[h,j,l]=c
            errors=[c['bit_error'] for c in rowchips if c['kind']=='data']
            realdistance=sum(errors) if all(type(v) is bool for v in errors) else None
            stored_true=at(dec.get('code_distances'),j-1)
            status='SCORED' if not technical and native_ok and all(c['QIM']['status']=='SCORED' and type(c['bit_error']) is bool for c in rowchips) else 'UNRESOLVED'
            if realdistance is not None and stored_true!=realdistance:
                status='UNRESOLVED';issues.append(dict(kind='TRUE_DISTANCE_DISAGREEMENT',host=h,cell=j))
            f=dict(host=h,role='calibration' if h in HOSTS[:2] else 'evaluation',cell=j,status=status,
                saved_raw_decoder=dec,under_old_local_tau=fin,stored_true_distance=stored_true,real_RM_distance=realdistance,
                stored_nearest_distance=dec.get('nearest_distance'),saved_nearest_ids=dec.get('nearest_ids'),
                nearest_unique_true=(dec.get('nearest_ids')==[j-1] if not technical else None),
                true_code_readable=(stored_true<=3 if type(stored_true) is int and not technical else None),
                old_rule_passed=(fin.get('tag_label') is not None if not technical else None),
                correct_tag=(fin.get('tag_label')==j if not technical and native_ok else None),
                pilot_P=dec.get('pilot_P'),pilot_H=dec.get('pilot_H'),pilot_known_errors=sum(c['bit_error'] is True for c in rowchips if c['kind']=='pilot'),
                real_RM_known_errors=sum(v is True for v in errors))
            full.append(f);full_index[h,j]=f
    pairs=[];pairchips=[]
    for h in HOSTS:
        for view,phase,offset in WINDOWS:
            for r in range(1,11):
                j=offset+r;k=(h,view,phase,r);lp=obj(local.get(k));lr=obj(localrows.get(k));ld=obj(obj(lp.get('saved_unit')).get('decoder'))
                f=full_index[h,j];oldfinal=obj(lr.get('saved_final_state'))
                local_ok=bool(oldfinal) and oldfinal.get('technical') is not True
                pair=dict(host=h,role=f['role'],view=view,phase=phase,row=r,source_cell=j,anchor='PREFIX40' if view=='clean0' else 'OTHER160',
                    status='SCORED' if f['status']=='SCORED' and local_ok else 'UNRESOLVED',
                    full=f,local_saved_state=oldfinal,local_stored_true_distance=lr.get('stored_true_distance'),
                    local_stored_nearest_distance=lr.get('stored_nearest_distance'),
                    local_true_code_readable=(lr['stored_true_distance']<=3 if type(lr.get('stored_true_distance')) is int else None),
                    local_correct_tag=(oldfinal.get('tag_label')==j if local_ok else None),
                    local_old_rule_passed=(oldfinal.get('tag_label') is not None if local_ok else None))
                pair['row_correct_tag_transition']=relation(f['correct_tag'],pair['local_correct_tag'])
                pair['true_code_readable_transition']=relation(f['true_code_readable'],pair['local_true_code_readable'])
                counts=Counter();pair_undefined=False
                for l in range(48):
                    fc=chip_index[h,j,l];oc=obj(oldchips.get(k+(l,)))
                    actual=at(ld.get('physical_bits'),l) if l in PILOT else at(ld.get('unmasked_bits'),DI[l])
                    full_actual=fc['saved_comparison_bit'];local_error=oc.get('bit_error')
                    fcorr=not fc['bit_error'] if type(fc['bit_error']) is bool else None
                    lcorr=not local_error if type(local_error) is bool else None
                    state=relation(fcorr,lcorr);counts[state]+=1
                    qL=at(obj(lp.get('saved_unit')).get('projections'),l);qF=fc['q_full']
                    if state=='UNDEFINED' or not finite(qL) or not finite(qF):pair_undefined=True
                    pairchips.append(dict(host=h,view=view,phase=phase,row=r,source_cell=j,anchor=pair['anchor'],chip=l,ROI=l//12,
                        channel=4+l%12,kind=fc['kind'],full_saved_bit=full_actual,local_saved_bit=actual,
                        saved_bit_agreement=(full_actual==actual if bit(full_actual) and bit(actual) else None),
                        full_correct=fcorr,local_correct=lcorr,truth_transition=state,
                        full_minus_local_projection=qF-qL if finite(qF) and finite(qL) else None,
                        full_QIM=fc['QIM'],old_native_to_local_QIM=oc))
                if pair_undefined:pair['status']='UNRESOLVED'
                pair['bit_transitions']=dict(counts);pairs.append(pair)
    nullindex=index('null_saved_states',lambda x:tuple(obj(x.get('coordinate')).get(k) for k in ('host','arm','key','view','phase','row')))
    nulls=[]
    for h in HOSTS:
        for arm,key in (('OFF','key00'),('OFF','key01'),('P0','key00'),('P0','key01'),('PT','key01')):
            for view in ('clean0','clean1','clean2','clean3','clean4','cross_center'):
                for phase in range(4):
                    for r in range(1,11):
                        value=nullindex.get((h,arm,key,view,phase,r))
                        nulls.append(dict(scope='HISTORICAL_LOCAL_NOT_FULL_CONTROL',saved=value,status='SAVED_STATE' if value and value.get('status')=='SAVED_STATE' and obj(value.get('saved_state')).get('status') is not None else 'UNRESOLVED'))
    def fullsum(rows):
        c=Counter(planned=len(rows));dist=Counter()
        for x in rows:
            c['unresolved']+=x['status']!='SCORED'
            for k in ('true_code_readable','old_rule_passed','correct_tag','nearest_unique_true'):c[k]+=x[k] is True
            dist[str(x['stored_true_distance'])]+=1
        return dict(counts=dict(c),true_distance_histogram=dict(dist))
    groups={'all_unique176':fullsum(full)}
    for h in HOSTS:groups[h]=fullsum([x for x in full if x['host']==h])
    bitgroups=defaultdict(Counter);rowgroups=defaultdict(Counter)
    for x in pairchips:
        for k in ('all',x['kind'],x['anchor'],x['host']):bitgroups[k][x['truth_transition']]+=1
    for x in pairs:
        for k in ('all',x['anchor'],x['host']):
            rowgroups[k]['planned']+=1;rowgroups[k]['status/'+x['status']]+=1
            rowgroups[k]['correct_tag/'+x['row_correct_tag_transition']]+=1
            rowgroups[k]['true_code/'+x['true_code_readable_transition']]+=1
    summary=dict(status='POSITIVE_CHANNEL_FULL_READ_WITH_HISTORICAL_LOCAL_REFERENCE',full=groups,
        bit_truth_transitions={k:dict(v) for k,v in bitgroups.items()},row_transitions={k:dict(v) for k,v in rowgroups.items()},
        fixed_denominators=dict(full_rows=176,full_chip_rows=8448,pairs=200,pair_chip_rows=9600,prefix_anchor=40,other=160,historical_nulls=4800,new_full_nulls=0),
        reference_issues=issues,unresolved_counts=dict(full=sum(x['status']!='SCORED' for x in full),pairs=sum(x['status']!='SCORED' for x in pairs),historical_nulls=sum(x['status']!='SAVED_STATE' for x in nulls)),scientific_PASS=None,independent_confirmation=False,full_FPR_validated=False,
        limitation='Full versus historical local changes context/start/history/length and runtime; no isolated reset/VAE/codec causality. Old local tau is descriptive only.')
    for name,value in [('full_post_rows.json',full),('native_to_full_chips.json',chips),('full_local_pairs.json',pairs),
        ('full_local_chip_pairs.json',pairchips),('historical_null_states.json',nulls),('evaluation.json',summary)]:save(out/name,value)
    return summary
