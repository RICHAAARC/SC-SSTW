"""Fixed M_Y observation and original H graph; no IO, truth or old anchors."""
import numpy as np
from . import b_framewise_qim_time_recovery_v1 as hard

MODEL = dict(name='M_Y',sample_bits=8,range='limited',matrix='601',gain=255/219,
             offset=16,clip=False,round_to_uint8=False,coefficient=[1,2])


def profile(stream):
    unknown=(None,'','unknown','unspecified','reserved')
    range_value=stream.get('color_range');matrix=stream.get('color_space')
    conflicts=[];unconfirmed=[]
    if stream.get('pix_fmt')=='yuvj420p':conflicts.append(dict(field='pixel_format_range',actual='full'))
    for name,value,accepted in (('range',range_value,('tv','mpeg')),('matrix',matrix,('smpte170m','bt470bg'))):
        if value in unknown:unconfirmed.append(name)
        elif value not in accepted:conflicts.append(dict(field=name,actual=value))
    return dict(status='MODEL_APPLICABILITY_CONFLICT' if conflicts else 'MODEL_UNCONFIRMED' if unconfirmed else 'METADATA_CONSISTENT',
                conflicts=conflicts,unconfirmed=unconfirmed,model=dict(MODEL),historical_conversion_proven=False)


def read_y(y,check=lambda:None):
    if not isinstance(y,np.ndarray) or y.dtype!=np.uint8 or y.ndim!=3 or y.shape[1:]!=(320,512) or not len(y):
        raise ValueError('native uint8 Y [R,320,512] required')
    x=np.arange(8,dtype=np.float64)+.5
    basis=.25*np.cos(np.pi*x[:,None]/8)*np.cos(2*np.pi*x[None,:]/8)
    q=np.empty((len(y),5,2,63),dtype=np.float64)
    for r,frame in enumerate(y):
        check()
        for replica in range(2):
            for lane in range(5):
                top,left=200+64*replica,16+96*lane
                expanded=(frame[top:top+56,left:left+72].astype(np.float64)-16)*255/219
                q[r,lane,replica]=np.einsum('iyjx,yx->ij',expanded.reshape(7,8,9,8),basis).reshape(63)
    distances=[]
    for bit in (0,1):
        low=np.floor((q-16*bit)/32)*32+16*bit
        center=np.where(q-low<=16,low,low+32)
        distances.append(np.abs(q-center))
    d0,d1=distances;soft=(d0*d0-d1*d1)/256
    if not all(np.isfinite(v).all() for v in (q,d0,d1,soft)):raise ValueError('nonfinite native Y observation')
    mask=np.isfinite(q);merged,known=hard.fuse(soft,mask)
    return dict(q=q,d0=d0,d1=d1,soft=soft,mask=mask,known=mask&(soft!=0),bit=soft>0,
                F_soft=merged,F_known=known,F_bit=merged>0)


def validate_labels(compatible,frames):
    labels=compatible.get('labels')
    if not isinstance(labels,list) or len(labels)!=frames:raise ValueError('complete frame labels required')
    for frame in labels:
        if not isinstance(frame,list) or len(frame)!=5:raise ValueError('complete five lanes required')
        for lane,row in enumerate(frame):
            if not isinstance(row,dict) or type(row.get('lane')) is not int or row['lane']!=lane:raise ValueError('lane coordinate mismatch')
            hits=row.get('labels');u=row.get('erasures')
            if not isinstance(hits,list) or any(type(j) is not int or not 0<=j<hard.DOMAIN or j%5!=lane for j in hits):raise ValueError('invalid public labels')
            if len(set(hits))!=len(hits) or type(u) is not int or not 0<=u<=63:raise ValueError('invalid hits/erasures')
            expected='COMPATIBLE' if len(hits)==1 else 'IMPLEMENTATION_UNRESOLVED' if len(hits)>1 else 'NO_COMPATIBLE_CODE'
            if row.get('status')!=expected:raise ValueError('failed/missing label is not an empty compatible set')
    return labels


def path(compatible,frames):
    labels=validate_labels(compatible,frames)
    result=hard.interval_paths(labels)
    return dict(result,labels=labels,formal_output=False,scope='Y_H_SHADOW_ORIGINAL_RULE')
