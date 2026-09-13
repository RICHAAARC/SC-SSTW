"""Common spatial latent relocation. No runtime, observer, or oracle inputs."""
import math
ARMS=('OFF1','OFF2','X_PLUS','X_MINUS','Y_PLUS','Y_MINUS')

def translate_latent(z, dx, dy, torch):
    """Same bilinear border-replicated map on B,C,T slices; +x right,+y down."""
    if z.ndim!=5 or min(z.shape[-2:])<2: raise ValueError('expected B,C,T,H,W')
    if not all(math.isfinite(v) and abs(v)<=.25 for v in (dx,dy)): raise ValueError('displacement')
    if not bool(torch.isfinite(z).all()): raise ValueError('nonfinite latent')
    if dx==dy==0: return z.clone()
    b,c,t,h,w=z.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=z.device,dtype=torch.float32),torch.arange(w,device=z.device,dtype=torch.float32),indexing='ij')
    grid=torch.stack((2*(xx-dx)/(w-1)-1,2*(yy-dy)/(h-1)-1),dim=-1)
    x=z.permute(0,2,1,3,4).reshape(b*t,c,h,w).float()
    v=torch.nn.functional.grid_sample(x,grid[None].expand(b*t,-1,-1,-1),mode='bilinear',padding_mode='border',align_corners=True)
    return v.reshape(b,t,c,h,w).permute(0,2,1,3,4).to(z.dtype).contiguous()

def mean(rows): return [sum(p[d] for p in rows)/len(rows) for d in range(2)]
def sub(a,b): return [a[d]-b[d] for d in range(2)]
def norm(p): return math.hypot(*p)
def rms(rows): return math.sqrt(sum(norm(p)**2 for p in rows)/len(rows))
def singular2(columns):
    a,b=columns[0];c,d=columns[1]
    tr=a*a+b*b+c*c+d*d;det=abs(a*d-b*c)
    hi=math.sqrt(max(0.,(tr+math.sqrt(max(0.,tr*tr-4*det*det)))/2))
    return [hi,det/hi if hi else 0.]

def evaluate_response(data,delta,budget):
    if set(data)!=set(ARMS): raise ValueError('six-arm denominator required')
    lengths={len(r) for r in data.values()}
    if len(lengths)!=1: return dict(status='OPERATIONAL_BLOCKED',reason='AXIS_LENGTH_MISMATCH')
    n=lengths.pop()
    for rows in data.values():
        for i,r in enumerate(rows):
            if r['sample_index']!=i or r['time_seconds']!=i/5: return dict(status='OPERATIONAL_BLOCKED',reason='AXIS_MAPPING_MISMATCH')
            if r['valid'] and (r['q'] is None or not all(math.isfinite(v) for v in r['q'])): raise ValueError('nonfinite q')
    idx=[i for i in range(1,n) if all(data[a][i]['valid'] for a in ARMS)]
    result=dict(total_rows=n,noninitial_denominator=max(0,n-1),common_indices=idx,
        common_fraction=len(idx)/max(1,n-1),valid_counts={a:sum(r['valid'] for r in data[a]) for a in ARMS},
        excluded=[dict(sample_index=i,reasons={a:data[a][i]['reason'] for a in ARMS if not data[a][i]['valid']}) for i in range(n) if i not in idx])
    if not idx: return dict(result,status='OPERATIONAL_BLOCKED',reason='NO_COMMON_VALID')
    q={a:[data[a][i]['q'] for i in idx] for a in ARMS}
    floor=rms([sub(a,b) for a,b in zip(q['OFF1'],q['OFF2'])]);f=max(floor,budget['numeric_floor'])
    axes={};columns=[]
    for axis in ('X','Y'):
        p=q[axis+'_PLUS'];m=q[axis+'_MINUS']
        odd=[[v/2 for v in sub(a,b)] for a,b in zip(p,m)];c=mean(odd);columns.append(c);length=norm(c)
        by_off={}
        for k in ('OFF1','OFF2'):
            plus=mean([sub(a,b) for a,b in zip(p,q[k])]);minus=mean([sub(a,b) for a,b in zip(m,q[k])])
            projection=[sum(v[d]*c[d] for d in range(2))/length if length else 0. for v in (plus,minus)]
            by_off[k]=dict(even_rows=[[(a[d]+b[d])/2-o[d] for d in range(2)] for a,b,o in zip(p,m,q[k])],mean_plus=plus,mean_minus=minus,plus_projection=projection[0],minus_projection=projection[1],opposite=projection[0]>f and projection[1]<-f)
        axes[axis]=dict(mean_odd=c,odd_rows=odd,by_off=by_off,
            even_rows=[[(a[d]+b[d]-o1[d]-o2[d])/2 for d in range(2)] for a,b,o1,o2 in zip(p,m,q['OFF1'],q['OFF2'])],
            temporal_odd_drift_rms=rms([sub(v,c) for v in odd]))
    sv=singular2(columns);condition=sv[0]/sv[1] if sv[1]>0 else None
    checks=dict(coverage=len(idx)>=budget['minimum_common_pairs'] and result['common_fraction']>=budget['common_valid_fraction'],
        two_direction_effect=sv[1]>budget['minimum_singular_to_floor']*f,
        condition=condition is not None and condition<=budget['maximum_condition'],
        signed_response=all(v['opposite'] for a in axes.values() for v in a['by_off'].values()))
    result.update(axes=axes,C_columns=columns,J_columns=[[x/delta for x in c] for c in columns],singular_values_C=sv,condition=condition,
        off_repeat_rms=floor,effective_floor=f,checks=checks,status='ENGINEERING_RESPONSE_READY' if all(checks.values()) else ('OPERATIONAL_BLOCKED' if not checks['coverage'] else 'ENGINEERING_RESPONSE_NOT_MET'))
    return result
