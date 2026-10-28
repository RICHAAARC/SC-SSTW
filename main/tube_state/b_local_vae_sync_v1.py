"""Fixed soft temporal state-space planning, then one cached BCH read."""
from functools import cache
import numpy as np
from main.tube_state import b_local_vae_code_v1 as code

@cache
def domain():
    paths=[];metadata=[];r=np.arange(151)
    for a in range(181):
        values=a+r
        if values[-1]<=180:paths.append(values);metadata.append(dict(a=a,d=0,b=None))
        for d in (8,-8):
            for b in range(1,151):
                values=a+r+d*(r>=b)
                if values.min()>=0 and values.max()<=180:
                    paths.append(values);metadata.append(dict(a=a,d=d,b=b))
    counts={d:sum(x['d']==d for x in metadata) for d in (0,8,-8)}
    if counts!={0:31,8:3450,-8:5794}:raise ValueError('fixed complete151 path domain mismatch')
    return np.asarray(paths,dtype=np.int16),metadata

def evidence(logits,key):
    x=np.asarray(logits,dtype=float)
    if x.shape!=(151,32) or not np.isfinite(x).all():raise ValueError('finite151x32 pilot logits required')
    signed=2*np.asarray(code.pilots(key),dtype=float)-1
    return np.tanh(x/2)@signed.T

def earliest(rows):
    result=[];seen=set()
    for r,t,e in rows:
        if t not in seen:result.append(dict(received_frame=int(r),source_frame=int(t),e=float(e)));seen.add(t)
    return result

def presence(mean,support):
    source=[x['source_frame'] for x in support]
    span=max(source)-min(source) if source else 0
    return bool(mean>=8 and len(source)>=24 and span>=64),span

def plan(pilot_logits,key,reader='STATE_SPACE'):
    if reader not in ('STATE_SPACE','I'):raise ValueError('fixed main or independent baseline only')
    e=evidence(pilot_logits,key);path_scores=None;retained=[];edit_status=None;sides=True
    if reader=='I':
        maximum=e.max(axis=1);labels=e.argmax(axis=1)
        unique=(e==maximum[:,None]).sum(axis=1)==1
        support=earliest((r,int(labels[r]),maximum[r]) for r in range(151) if unique[r] and maximum[r]>=16)
        mean=float(maximum.mean());edit_status='NO_PATH_MODEL'
    else:
        paths,metadata=domain()
        scores=e[np.arange(151)[None,:],paths].sum(axis=1)
        best=float(scores.max());retained=np.flatnonzero(scores>=best-8).tolist()
        chosen=paths[retained];consistent=(chosen==chosen[0]).all(axis=0);labels=chosen[0]
        support=earliest((r,int(labels[r]),e[r,labels[r]]) for r in range(151) if consistent[r] and e[r,labels[r]]>=16)
        edited=[metadata[i]['d']!=0 for i in retained]
        kinds={metadata[i]['d'] for i in retained}
        if all(edited):
            sides=all(sum(x['received_frame']<metadata[i]['b'] for x in support)>=8 and
                      sum(x['received_frame']>=metadata[i]['b'] for x in support)>=8 for i in retained)
            edit_status='DELETE8' if kinds=={8} else 'REPEAT8' if kinds=={-8} else 'EDIT_TYPE_UNRESOLVED'
        else:edit_status='NO_EDIT' if kinds=={0} else 'EDIT_TYPE_UNRESOLVED_MIXED'
        mean=best/151;path_scores=scores.tolist()
    detected,span=presence(mean,support);detected=detected and sides
    return dict(status='DETECT' if detected else 'REJECT',detected=detected,reader=reader,
        best_mean_e=mean,support=support,unique_source_frames=len(support),source_span=span,
        all_edited_paths_have_eight_support_each_side=sides,edit_status=edit_status,
        retained_paths=retained,path_scores=path_scores,path_domain=9275 if reader=='STATE_SPACE' else None,
        payload_used=False,exact_edit_location_claim=False,information_slots=32)

def read_payload(payload_logits,key,support,*,detected=True):
    x=np.asarray(payload_logits,dtype=float)
    base=dict(planned_information_slots=32,support=support,padding_used=False,authentication=False,
              information_bits=[None]*32,raw_information_bits=[None]*32,BCH_calls=0)
    if x.shape!=(151,64) or not np.isfinite(x).all():return dict(base,status='TECHNICAL_UNREAD',error='nonfinite or wrong-shaped cached payload logits')
    if not detected or not support:return dict(base,status='ERASURE',reason='EXISTENCE_OR_SUPPORT')
    values=[]
    for row in support:
        r=row['received_frame'];block=row.get('block',row['source_frame']//8)
        if type(r) is not int or r not in range(151):raise ValueError('invalid frozen support reference')
        values.append(code.unscramble_logits(x[r],key,block))
    mean=np.asarray(values).mean(axis=0)
    if not np.isfinite(mean).all():return dict(base,status='TECHNICAL_UNREAD',error='nonfinite aggregate')
    if (mean==0).any():
        base['raw_information_bits']=[None if v==0 else int(v>0) for v in mean[:32]]
        return dict(base,status='ERASURE',reason='EXACT_ZERO_CODE_BIT',mean_logits=mean.tolist())
    hard=(mean>0).astype(int).tolist();decoded=code.decode(hard)
    return dict(base,status=decoded['status'],information_bits=decoded['message'] or [None]*32,
        raw_information_bits=hard[:32],mean_logits=mean.tolist(),BCH_calls=1,BCH=decoded,
        aggregation='equal weight per earliest distinct supported source frame')

def oracle_support(actual_frames,mode):
    if len(actual_frames)!=151 or mode not in ('O','W_GROUP','U'):raise ValueError('fixed post-only diagnostic')
    rows=earliest((r,t,0.) for r,t in enumerate(actual_frames))
    for row in rows:
        t=row['source_frame'];r=row['received_frame']
        row['block']=t//8 if mode=='O' else (t//8+1)%23 if mode=='W_GROUP' else r//8
    return rows
