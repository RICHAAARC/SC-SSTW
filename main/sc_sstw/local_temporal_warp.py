"""Fixed content-derived local video-time warp; no observer or runtime inputs."""
import math
ARMS=('OFF1','OFF2','X_PLUS','X_MINUS','Y_PLUS','Y_MINUS')
MASK_X=(5,8,51,54)
MASK_Y=(6,9,36,39)

def axis_taper(x,bounds,torch):
    a,b,c,d=bounds
    up=.5-.5*torch.cos(math.pi*((x-a)/(b-a)).clamp(0,1))
    down=.5+.5*torch.cos(math.pi*((x-c)/(d-c)).clamp(0,1))
    return up*down

def warp_latent(z,arm,torch):
    if arm not in ARMS: raise ValueError('fixed arm required')
    if z.ndim!=5 or tuple(z.shape[2:])!=(13,40,64): raise ValueError('expected B,C,13,40,64')
    if not bool(torch.isfinite(z).all()): raise ValueError('nonfinite latent')
    yy,xx=torch.meshgrid(torch.arange(40,device=z.device,dtype=torch.float32),torch.arange(64,device=z.device,dtype=torch.float32),indexing='ij')
    mask=axis_taper(xx,MASK_X,torch)*axis_taper(yy,MASK_Y,torch)
    r=(torch.arange(13,device=z.device,dtype=torch.float32)-6)/6
    sign=-1 if arm.endswith('MINUS') else 1
    dx=sign*r if arm.startswith('X') else torch.zeros_like(r)
    dy=sign*r if arm.startswith('Y') else torch.zeros_like(r)
    sx=xx[None]-dx[:,None,None]*mask;sy=yy[None]-dy[:,None,None]*mask
    grid=torch.stack((2*sx/63-1,2*sy/39-1),-1)
    if arm.startswith('OFF'):return z.clone(),grid,mask
    b,c,t,h,w=z.shape
    v=z.permute(0,2,1,3,4).reshape(b*t,c,h,w).float()
    sampled=torch.nn.functional.grid_sample(v,grid.repeat(b,1,1,1),mode='bilinear',padding_mode='border',align_corners=True)
    sampled=sampled.reshape(b,t,c,h,w).permute(0,2,1,3,4).to(z.dtype)
    # Exact identity outside the declared intervention and at the zero-ramp slice.
    active=(mask[None,None,None]>0)&((dx.abs()+dy.abs())[None,None,:,None,None]>0)
    return torch.where(active,sampled,z).contiguous(),grid,mask

def reference_rows(qrows,annotations,width=512,height=320):
    """External evaluation only; annotations are locked from actual images."""
    if len(qrows)!=len(annotations):raise ValueError('full original axis required')
    result=[]
    for i,(q,a) in enumerate(zip(qrows,annotations)):
        if a['sample_index']!=i or q['sample_index']!=i:raise ValueError('index mismatch')
        if a.get('time_seconds')!=i/5 or q.get('time_seconds',i/5)!=i/5:raise ValueError('time mismatch')
        if type(a.get('alignment_confirmed')) is not bool:raise ValueError('alignment bool required')
        point=a.get('p_pixel')
        if point is not None:
            if len(point)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in point) or not (0<=point[0]<width and 0<=point[1]<height):raise ValueError('invalid p')
            radius=a.get('subjective_radius_px')
            if isinstance(radius,bool) or not isinstance(radius,(int,float)) or not math.isfinite(radius) or radius<0:raise ValueError('subjective radius required')
        prev=annotations[i-1] if i else None
        p=a.get('p_pixel');pp=prev.get('p_pixel') if prev else None
        reference_pair_valid=bool(i>0 and a.get('status')=='REVIEWED' and prev.get('status')=='REVIEWED' and p is not None and pp is not None and a.get('alignment_confirmed',False) and prev.get('alignment_confirmed',False))
        valid=reference_pair_valid and q['valid']
        midpoint=[(p[d]+pp[d])/2/[width-1,height-1][d] for d in range(2)] if reference_pair_valid else None
        d=[q['q'][j]-midpoint[j] for j in range(2)] if valid else None
        result.append(dict(sample_index=i,time_seconds=i/5,p_pixel=p,p_prev_pixel=pp,q=q['q'],q_valid=q['valid'],q_reason=q['reason'],midpoint=midpoint,d=d,d_norm=math.hypot(*d) if d else None,eligible=bool(valid),reference_pair_valid=reference_pair_valid,previous_reference_status=prev.get('status') if prev else None,previous_reference_reason=prev.get('reason') if prev else None,reference_status=a.get('status'),subjective_radius_px=a.get('subjective_radius_px'),missing_reason=a.get('reason'),actual_displacement_pixel=[p[j]-pp[j] for j in range(2)] if reference_pair_valid else None))
    return result
