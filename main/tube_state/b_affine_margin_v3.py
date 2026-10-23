"""Isolated V3 native construct and received-only inference. No runtime/model imports."""
from collections import Counter
import math,time,warnings
import numpy as np
from .b_lowrate_local_time_label_v1 import book
from .b_lfg_posthoc_reference_v1 import haar
ALPHA=.15;M=5120;U64=2.**-53;U32=2.**-24
WORDS=np.array([[(g&1)^(sum(((g>>b)&1)*((i>>(b-1))&1) for b in range(1,6))&1) for i in range(32)] for g in range(64)],dtype=np.int8)
def gam(n):return n*U64/(1-n*U64)
def up(x):return np.nextafter(x,np.inf)
def down(x):return np.nextafter(x,-np.inf)
def blocklow(x):return x.reshape(12,2,20,2,32,2).sum(axis=(1,3,5))/math.sqrt(8)
def construct(source,old_source,record=None):
 import cvxpy as cp
 for z in (source,old_source):
  if not isinstance(z,np.ndarray) or z.dtype!=np.float32 or z.shape!=(16,46,40,64) or not np.isfinite(z).all():raise ValueError('finite native FP32[16,46,40,64] required')
 host='b101-s01';bk=book('watermark');prows=[];crows=[];urows=[];brows=[];hrows=[];inputs=[];solvecalls=0;casts=0;actdots=0
 def dump(name,value):
  if record:record(name,value)
 dump('RUN_STATE.json',dict(status='HOST_ATTEMPT_STARTED',host=host,completed_hosts=len(hrows),solver_calls=solvecalls,physical_calls=0))
 roles=[];state='UNRESOLVED_INPUT';solver_info=None;certificate=None;after=np.full((21,48),np.nan);before=np.full((21,48),np.nan);actual=[]
 try:
  z=source.astype(np.float64);old=old_source.astype(np.float64)
  if not (np.array_equal(z[:4],old[:4]) and np.array_equal(z[:,[0,1,44,45]],old[:,[0,1,44,45]])):raise ValueError('OLD support differs; cap contract unavailable')
  low=haar(z)['LLL'];sigma=low.std((1,2,3),ddof=0);witness=[]
  for rn,cs,N,base in [('pilot',slice(4,8),16,0),('label',slice(8,16),32,16)]:
   pn=bk['pilot' if rn=='pilot' else 'data'].astype(float);sc=sigma[cs];C=len(sc);sb=float(sc.mean())
   if not np.all(sc>0):raise ValueError('ZERO_SCALE')
   h=np.array([np.tensordot(pn,low[cs,g].reshape(C,640),axes=((1,2),(0,1)))/(C*640) for g in range(1,22)]);H=h/sb;before[:,base:base+N]=h
   B=np.array([2*(np.array(bk['pilot_bits'])^(g%2))-1 if rn=='pilot' else 2*(WORDS[g]^np.array(bk['mask']))-1 for g in range(1,22)],dtype=float)
   elo=[];ehi=[];plo=[];phi=[];err=[];caps=[];pks=[];rad=[];herror=[]
   for gi in range(21):
    zz=z[cs,2*(gi+1):2*(gi+1)+2];oo=old[cs,2*(gi+1):2*(gi+1)+2];dd=oo-zz;eo=np.square(dd).sum((1,2,3));po=np.abs(dd).max((1,2,3));public=640*ALPHA**2*sc**2
    lo=down(np.minimum(public/(1+gam(8)),eo/(1+gam(2*M+4))));hi=up(np.minimum(public/(1-gam(8)),eo/(1-gam(2*M+4))))
    zmax=np.abs(zz).max((1,2,3));omax=np.abs(oo).max((1,2,3));pl=down(po-gam(2)*(zmax+omax));ph=up(po+gam(2)*(zmax+omax))
    e32=up(U32/(1-U32)*(zmax+ph)+float(np.nextafter(np.float32(0),np.float32(1))));e64=up(gam(N+8)*sc*math.sqrt(N)*ALPHA/math.sqrt(8)+gam(2)*(zmax+ph));ee=up(e32+e64)
    radi=(np.sqrt(lo)-math.sqrt(M)*ee)/(math.sqrt(640)*sc);rad.append(float(radi.min()));elo.append(lo);ehi.append(hi);plo.append(pl);phi.append(ph);err.append(ee);caps.append(np.minimum(public,eo));pks.append(po);herror.append(float(up(gam(C*640+32)*math.sqrt(8)*zmax.max()/sb)))
    for k,c in enumerate(range(cs.start,cs.stop)):crows.append(dict(host=host,g=gi+1,role=rn,channel=c,original_energy_cap=float(min(public[k],eo[k])),cap_lower=float(lo[k]),cap_upper=float(hi[k]),OLD_peak=float(po[k]),peak_lower=float(pl[k]),preplanned_cast_and_formation_reserve=float(ee[k]),intended_L2_limit=float(math.sqrt(lo[k])-math.sqrt(M)*ee[k]),intended_peak_limit=float(pl[k]-ee[k]),actual_energy=None,actual_peak=None))
   role=dict(name=rn,cs=cs,N=N,base=base,pn=pn,sc=sc,C=C,sb=sb,h=h,H=H,B=B,elo=np.array(elo),ehi=np.array(ehi),plo=np.array(plo),phi=np.array(phi),err=np.array(err),rad=np.array(rad),caps=np.array(caps),pks=np.array(pks),beta=ALPHA*sb/math.sqrt(N))
   roles.append(role);Rorig=np.sqrt(role['ehi']/(640*sc[None]**2)).min(1)
   for i in range(21):
    for j in range(i+1,21):
     v=H[i]-H[j];bd=B[i]-B[j];nb=float(bd@bd);nv=float(v@v);dot=float(v@bd);ev=math.sqrt(N)*(herror[i]+herror[j])+gam(2)*float(np.linalg.norm(np.abs(H[i])+np.abs(H[j])))
     nlo=max(0,nv*(1-gam(N+2))-2*math.sqrt(nv)*ev-ev*ev)
     crosshi=max(0,dot+gam(N+2)*float(np.abs(v*bd).sum())+ev*math.sqrt(nb))
     dlo=down(math.sqrt(max(0,nlo-crosshi*crosshi/nb))) if nb>0 else down(math.sqrt(nlo));rup=up(Rorig[i]+Rorig[j]);gap=float(dlo-rup);cert=bool(gap>0)
     row=dict(host=host,role=rn,g1=i+1,g2=j+1,distance_lower=float(dlo),radius_sum_upper=float(rup),separation_gap=gap,certifies_original_affine_domain_empty=cert,source_h_error_bound_max=max(herror[i],herror[j]));prows.append(row)
     if cert:witness.append(row)
  if witness:
   certificate=max(witness,key=lambda x:x['separation_gap']);state='NO_AFFINE_DOMAIN_CERTIFIED'
  elif any(np.any(r['rad']<=0) or np.any(r['plo']-r['err']<=0) for r in roles):state='RESERVE_DOMAIN_EMPTY_NOT_ORIGINAL_INFEASIBILITY'
  else:
   t=cp.Variable();cons=[t>=0];variables=[]
   for rr in roles:
    aa=cp.Variable();cc=cp.Variable(rr['N']);uu=aa*rr['B']+cp.vstack([cc]*21)-rr['H'];cons+=[aa>=t*ALPHA/math.sqrt(rr['N']),cp.norm(uu,2,axis=1)<=rr['rad']]
    for ci in range(rr['C']):
     val=uu@rr['pn'][:,ci,:];bound=(rr['plo'][:,ci]-rr['err'][:,ci])*math.sqrt(8)/rr['sc'][ci];cons+=[val<=bound[:,None],val>=-bound[:,None]]
    variables.append((aa,cc,uu))
   problem=cp.Problem(cp.Maximize(t),cons);solvecalls+=1;dump('RUN_STATE.json',dict(status='ONE_SOLVER_ATTEMPT_STARTED',host=host,solver_calls=solvecalls))
   try:
    with warnings.catch_warnings(record=True) as ww:
     problem.solve(solver='CLARABEL',max_iter=200,tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-9,verbose=False)
    solver_info=dict(status=problem.status,t_estimate=float(t.value) if t.value is not None else None,iterations=problem.solver_stats.num_iters,solve_seconds=problem.solver_stats.solve_time,warnings=[str(x.message) for x in ww],optimality='UNRESOLVED_NO_RIGOROUS_DUAL_CERTIFICATE')
    if problem.status not in ('optimal','optimal_inaccurate') or t.value is None:state='SOLVER_UNRESOLVED'
    elif float(t.value)<=0:state='NO_POSITIVE_NUMERIC_MARGIN_NOT_GLOBAL_INFEASIBILITY'
    else:
     precast=True;deltas=[];solver_info['primal']=[dict(A=float(a.value),center=np.asarray(c.value).tolist(),u=np.asarray(u.value).tolist()) for a,c,u in variables];solver_info['precast_slacks']=[]
     for rr,(aa,cc,uu) in zip(roles,variables):
      rr['A']=float(aa.value);rr['Ccenter']=np.array(cc.value);rr['u']=np.array(uu.value);rr['a']=rr['A']*rr['sb'];rr['center']=rr['Ccenter']*rr['sb']
      for gi in range(21):
       un=rr['u'][gi];dl=np.tensordot(un,rr['pn'],axes=(0,0))*rr['sc'][:,None]/math.sqrt(8);eideal=640*rr['sc']**2*float(un@un);pideal=np.abs(dl).max(1)
       solver_info['precast_slacks'].append(dict(role=rr['name'],g=gi+1,energy_L2_slack=(np.sqrt(rr['elo'][gi])-math.sqrt(M)*rr['err'][gi]-np.sqrt(eideal)*(1+gam(rr['N']+4))).tolist(),peak_slack=(rr['plo'][gi]-rr['err'][gi]-pideal*(1+gam(rr['N']+8))).tolist()))
       precast &= bool(np.all(np.sqrt(eideal)*(1+gam(rr['N']+4))<=np.sqrt(rr['elo'][gi])-math.sqrt(M)*rr['err'][gi]) and np.all(pideal*(1+gam(rr['N']+8))<=rr['plo'][gi]-rr['err'][gi]))
     if not precast:state='PRECAST_BOUND_UNRESOLVED_NO_SCALING'
     else:
      casts+=1;marked=source.copy();after=before.copy();actualpass=True;allmargin=[]
      for gi in range(21):
       dd=[]
       for rr in roles:
        v=np.tensordot(rr['u'][gi],rr['pn'],axes=(0,0))*rr['sc'][:,None]/math.sqrt(8);dd.append(np.repeat(np.repeat(v.reshape(rr['C'],20,32),2,1),2,2)[:,None].repeat(2,1))
       cast=(z[4:16,2*(gi+1):2*(gi+1)+2]+np.concatenate(dd)).astype(np.float32).astype(float);marked[4:16,2*(gi+1):2*(gi+1)+2]=cast.astype(np.float32);delta=cast-z[4:16,2*(gi+1):2*(gi+1)+2];ld=blocklow(delta);offset=0
       for rr in roles:
        c=rr['C'];ee=np.square(delta[offset:offset+c]).sum((1,2,3));pp=np.abs(delta[offset:offset+c]).max((1,2,3));r=rr['h'][gi]+np.tensordot(rr['pn'],ld[offset:offset+c].reshape(c,640),axes=((1,2),(0,1)))/(c*640);actdots+=rr['N'];after[gi,rr['base']:rr['base']+rr['N']]=r;centered=r-rr['center'];signed=rr['B'][gi]*centered;res=r-(rr['a']*rr['B'][gi]+rr['center']);e0=int(np.count_nonzero(signed<0));zero=int(np.count_nonzero(centered==0));margin=float(signed.min());actualpass&=bool(np.all(ee*(1+gam(2*M+4))<=rr['elo'][gi]) and np.all(pp<=rr['plo'][gi]) and margin>0 and e0==0 and zero==0);allmargin.append(margin/rr['beta'])
        row=dict(host=host,g=gi+1,role=rr['name'],min_actual_centered_signed_margin=margin,actual_e0=e0,remaining_hard_error_budget=7-e0 if rr['name']=='label' else None,exactzero=zero,actual_model_residual_Linf=float(np.abs(res).max()),intended_gain=rr['a'],beta=rr['beta'])
        if rr['name']=='label':
         bits=(centered>0).astype(np.int8)^np.array(bk['mask']);dist=np.sum(WORDS!=bits,axis=1);row.update(distances64=dist.tolist(),target_distance=int(dist[gi+1]),other63RM_min=int(min(dist[k] for k in range(64) if k!=gi+1)),other20valid_min=int(min(dist[k] for k in range(1,22) if k!=gi+1)))
        actual.append(row)
        for k in range(rr['N']):brows.append(dict(host=host,g=gi+1,role=rr['name'],chip=k,actual_raw_score=float(r[k]),writer_center_post_only=float(rr['center'][k]),actual_centered_signed_margin=float(signed[k]),actual_model_residual=float(res[k])))
        for cr in crows:
         if cr['host']==host and cr['g']==gi+1 and cr['role']==rr['name']:
          ci=cr['channel']-rr['cs'].start;cr.update(actual_energy=float(ee[ci]),actual_peak=float(pp[ci]))
        offset+=c
      state='ADMITTED_ALL21' if actualpass else 'ACTUAL_FP32_REJECT_NO_RETRY';solver_info['actual_verified_t_lower_bound']=float(min(allmargin)) if actualpass else None
   except Exception as ex:state='SOLVER_EXCEPTION_UNRESOLVED';solver_info=dict(error=repr(ex))
 except Exception as ex:
  inputs.append(dict(host=host,status='UNRESOLVED_INPUT',error=repr(ex)));state='UNRESOLVED_INPUT'
 admitted=state=='ADMITTED_ALL21'
 receipt=dict(status=state,solver=solver_info,certificate=certificate,pairs=prows,channels=crows,chips=brows,roles=actual,solver_calls=solvecalls,host_materialization_rounds=casts,unit_FP32_block_casts=21*casts,actual_correlations=actdots,fullhost_admitted=admitted,RM_error_reserve_scope='label only; pilot has coherent margin, no RM-r7',no_retry=True)
 if admitted:
  if not np.array_equal(marked[:4],source[:4]) or not np.array_equal(marked[:,[0,1,44,45]],source[:,[0,1,44,45]]):raise ValueError('protected payload/boundary changed')
 return (marked if admitted else None),receipt

# Fixed descriptive references. Never calibrated on the real s01 observation.
TAU_D=.06249999999999999;TAU_L=.06844171267044827;RHO=7.771561172376096e-16
EPS=4096*np.finfo(float).eps/(1-4096*np.finfo(float).eps)
WINDOWS=(0,26,52)
def state_grid():
 grid=[(o,0,104) for o in range(78)]
 for d in range(1,25):grid.extend((o,d,s) for o in range(78-d) for s in range(1,104))
 return np.array(grid,dtype=np.int16).T

def source_cell(sf):
 if len(sf)!=4 or any(x is None for x in sf):return None
 if list(sf)!=list(range(sf[0],sf[0]+4)) or not (1<=sf[0]<=sf[-1]<=176):return None
 cells={(x+3)//4 for x in sf}
 return next(iter(cells)) if len(cells)==1 else None

def payload_records(tables):
 records={}
 for wi in range(3):
  for p in range(8):
   table=tables[wi*8+p];signed=table.get('signed_votes');zeros=table.get('zero_coefficient_mask')
   if signed is not None:
    signed=np.asarray(signed);zeros=np.asarray(zeros)
    if signed.shape!=(4,11,240) or zeros.shape!=signed.shape or not np.isin(signed,[-1,1]).all() or not np.isin(zeros,[0,1]).all():raise ValueError('local votes/zero mask geometry')
   for l in range(1,12):
    bits=[]
    for b in range(8):
     seq=[] if signed is None else (signed[:,l-1,b::8]>0).astype(int).reshape(-1).tolist();zz=120 if signed is None else int(zeros[:,l-1,b::8].sum());tie=bool(seq and sum(seq)*2==len(seq))
     bits.append(dict(votes=seq,readable=bool(seq) and not tie and zz<len(seq),tie=tie,zero_coefficients=zz,technical=signed is None))
    lo=WINDOWS[wi]+p+4*l-3;records[(wi,p,l)]=dict(lo=lo,hi=lo+3,bits=bits)
 return records

def recover(records,common,tag_g,require_tags=True,phase_zero=False):
 refs=[]
 for ref,r in records.items():
  if phase_zero and ref[1]!=0:continue
  t=source_cell(common[r['lo']:r['hi']+1])
  if t is None or (require_tags and t//2 not in tag_g):continue
  refs.append((t,ref))
 chosen={}
 for t,ref in sorted(refs):chosen.setdefault(t,ref)
 out=[None]*32;legacy=[None]*32;states=[]
 for seg in range(4):
  for b in range(8):
   cells=[(t,ref) for t,ref in sorted(chosen.items()) if (t-1)//11==seg];seq=[];ls=[]
   for c in range(4):
    for t,ref in cells:
     bit=records[ref]['bits'][b];ls.extend(bit['votes'][c*30:(c+1)*30])
     if require_tags and not bit['readable']:continue
     seq.extend(bit['votes'][c*30:(c+1)*30])
   i=8*seg+b
   if ls:legacy[i]=Counter(ls).most_common(1)[0][0]
   if seq and (not require_tags or sum(seq)*2!=len(seq)):out[i]=Counter(seq).most_common(1)[0][0]
   states.append(dict(bit=i,status='READ' if out[i] is not None else 'ERASURE',vote_count=len(seq)))
 return dict(decoded=out,legacy_decoded=legacy,planned_bits=32,read=sum(x is not None for x in out),bits=states,selected_refs=[dict(source_cell=t,window=ref[0],phase=ref[1],latent_row=ref[2]) for t,ref in sorted(chosen.items())],truth_inputs=False,message_inputs=False)

def support(paths,O,D,S,xpos,yp,yd,key,gp,gl,mp,mz):
 if not len(paths):return [None]*104,set(),dict(paths=0,rows=[]),dict(candidate_c_pilot=np.empty((0,16)),candidate_c_label=np.empty((0,32)),corrected_pilot=np.empty((0,96,16)),corrected_label=np.empty((0,96,32)),candidate_pure_ids=np.empty((0,96),dtype=np.int16))
 bk=book(key);pb=np.array(bk['pilot_bits']);mask=np.array(bk['mask']);baseD=(2*(WORDS[0:22:2]^mask)-1).astype(float)
 cp=yp.mean(0)[None]-gp[paths,None]*mp[paths,None]*(2*pb-1)[None];cd=yd.mean(0)[None]-gl[paths,None]*(mz[paths]@baseD)
 maps=np.arange(104)[None]+O[paths,None]+D[paths,None]*(np.arange(104)[None]>=S[paths,None]);lo=maps.min(0);hi=maps.max(0);common=[int(lo[i]) if lo[i]==hi[i] else None for i in range(104)]
 corrected_p=yp[None]-cp[:,None,:];corrected_d=yd[None]-cd[:,None,:];pure_ids=np.zeros((len(paths),96),dtype=np.int16);rows=[];supported=[]
 for k,xx in enumerate(xpos):
  sf=xx[None]+O[paths,None]+D[paths,None]*(xx[None]>=S[paths,None]);gg=(sf+3)//8;pure=(np.diff(sf,axis=1)==1).all(1)&(gg==gg[:,0,None]).all(1)&(gg[:,0]>=1)&(gg[:,0]<=21);pure_ids[:,k]=np.where(pure,gg[:,0],0)
  g=int(gg[0,0]);shared=bool(pure.all() and (gg==g).all())
  if not shared:rows.append(dict(row=k,status='NOT_COMMON_PURE'));continue
  dd=corrected_d[:,k];bits=(dd>0).astype(np.int8)^mask;dist=np.sum(bits!=WORDS[g],axis=1);rm=bool((dd!=0).all() and (dist<=7).all());pm=np.mean((2*(pb^(g%2))-1)[None]*corrected_p[:,k],axis=1);po=bool((pm>0).all())
  rows.append(dict(row=k,g=g,status='SUPPORTED' if rm and po else 'CORRECTED_REFUSED',max_target_distance=int(dist.max()),min_pilot_mean=float(pm.min()),all_candidate_RM=rm,all_candidate_pilot=po))
  if rm and po:supported.append((k,g))
 cg={g for k,g in supported};piece=np.zeros((len(paths),2),dtype=bool)
 for n,ii in enumerate(paths):
  for side in (0,1):
   gs={g for k,g in supported if (D[ii]==0 and side==0) or (D[ii]>0 and (xpos[k,-1]<S[ii] if side==0 else xpos[k,0]>=S[ii]))};piece[n,side]=len(gs)>=2
 for x in range(104):
  side=((D[paths]>0)&(x>=S[paths])).astype(int)
  if not piece[np.arange(len(paths)),side].all():common[x]=None
 return common,cg,dict(paths=len(paths),rows=rows,common_g=sorted(cg)),dict(candidate_c_pilot=cp,candidate_c_label=cd,corrected_pilot=corrected_p,corrected_label=corrected_d,candidate_pure_ids=pure_ids)

def infer_many(queries,save_arrays):
 """Only received raw tables, keys and public geometry. No writer/P0/truth argument."""
 O,D,S=state_grid();Q=len(O);xpos=np.array([np.arange(a+p+8*j-3,a+p+8*j+5) for a in WINDOWS for p in range(8) for j in range(1,5)],dtype=np.int16)
 F=len(queries);valid=[];YP=[];YD=[];CP=[];CD=[];EP=[];ED=[];bad=np.zeros(F,bool)
 for i,q in enumerate(queries):
  yp=np.asarray(q['pilot'],float);yd=np.asarray(q['label'],float);missing=q.get('missing',False)
  if yp.shape!=(96,16) or yd.shape!=(96,32):raise ValueError('fixed96x48 received geometry')
  good=not missing and np.isfinite(yp).all() and np.isfinite(yd).all();valid.append(bool(good))
  if not good:yp=np.zeros((96,16));yd=np.zeros((96,32))
  pc=yp-yp.mean(0);dc=yd-yd.mean(0);bk=book(q['key']);base=(2*(WORDS[0:22:2]^np.array(bk['mask']))-1).astype(float);ep=float(np.square(pc).sum());ed=float(np.square(dc).sum());rawp=float(np.square(yp).sum());rawd=float(np.square(yd).sum());bad[i]=(ep>0 and ep<=EPS*rawp) or (ed>0 and ed<=EPS*rawd)
  YP.append(yp);YD.append(yd);CP.append(pc@(2*np.array(bk['pilot_bits'])-1));CD.append((dc@base.T).reshape(-1));EP.append(ep);ED.append(ed)
 CP=np.array(CP);CD=np.array(CD);EP=np.array(EP);ED=np.array(ED);PV=np.zeros((F,Q));LV=np.zeros((F,Q));GP=np.zeros((F,Q));GL=np.zeros((F,Q));MP=np.zeros(Q);MZ=np.zeros((Q,11))
 for lo in range(0,Q,2048):
  hi=min(Q,lo+2048);n=hi-lo;sf=xpos[None]+O[lo:hi,None,None]+D[lo:hi,None,None]*(xpos[None]>=S[lo:hi,None,None]);gg=(sf+3)//8
  if not ((gg>=1)&(gg<=21)).all():raise ValueError('public tag support outside native domain')
  z=np.zeros((n,96,11));ri=np.arange(n)[:,None];oi=np.arange(96)[None,:]
  for k in range(8):z[ri,oi,gg[:,:,k]//2]+=(1-2*(gg[:,:,k]%2))/8
  q=z.sum(2);MP[lo:hi]=q.mean(1);MZ[lo:hi]=z.mean(1);exP=16*(np.square(q).sum(1)-np.square(q.sum(1))/96);exL=32*(np.square(z).sum((1,2))-np.square(z.sum(1)).sum(1)/96)
  for ex,cross,e0,V,G in [(exP,q@CP.T,EP,PV,GP),(exL,z.reshape(n,-1)@CD.T,ED,LV,GL)]:
   den=ex[:,None]*e0[None];v=np.divide(np.maximum(cross,0)**2,den,out=np.zeros_like(cross),where=den>0);g=np.divide(np.maximum(cross,0),ex[:,None],out=np.zeros_like(cross),where=ex[:,None]>0);bad|=np.any(~np.isfinite(v)|(v>1+EPS)|(v < -EPS),axis=0);V[:,lo:hi]=np.clip(v,0,1).T;G[:,lo:hi]=g.T
 results=[]
 for i,q in enumerate(queries):
  H=np.array([],int);R=np.maximum(1-PV[i],1-LV[i]);status='UNRESOLVED_INPUT' if not valid[i] else 'NUMERICALLY_UNRESOLVED' if bad[i] else 'DESCRIPTIVE_REFERENCE_ONLY'
  if valid[i] and not bad[i]:
   H=np.flatnonzero((PV[i]>TAU_D+EPS)&(LV[i]>TAU_L+EPS)&(R<=RHO+2*EPS))
   status='COMPATIBLE_REFERENCE_ONLY' if len(H) else 'NO_COMPATIBLE_REFERENCE'
  bk=book(q['key']);baseD=(2*(WORDS[0:22:2]^np.array(bk['mask']))-1).astype(float);allcp=YP[i].mean(0)[None]-GP[i,:,None]*MP[:,None]*(2*np.array(bk['pilot_bits'])-1)[None];allcd=YD[i].mean(0)[None]-GL[i,:,None]*(MZ@baseD)
  common,cg,audit,arrays=support(H,O,D,S,xpos,YP[i],YD[i],q['key'],GP[i],GL[i],MP,MZ);records=payload_records(q['payload']);E=recover(records,common,cg);U=recover(records,list(range(104)),set(),False,True)
  save_arrays(q['id'],dict(O=O,d=D,s=S,D_score=PV[i],L_score=LV[i],R_score=R,gain_pilot=GP[i],gain_label=GL[i],all_candidate_c_pilot=allcp,all_candidate_c_label=allcd,H=H,**arrays))
  results.append(dict(id=q['id'],status=status,model_domain='MODEL_DOMAIN_UNVALIDATED',reference_only=True,scientific_PASS=None,truth_inputs=False,message_inputs=False,rho=RHO,tauD=TAU_D,tauL=TAU_L,gamma=EPS,states=Q,common_frames=common,common_g=sorted(cg),reference_set_paths=len(H),corrected_row_evidence=audit,payload_E=E,payload_U=U,diagnostic_acceptance_is_not_authentication=True))
 return results
