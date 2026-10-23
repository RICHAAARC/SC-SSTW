"""Truth-only post evaluation; never called before blind_complete is durable."""
from pathlib import Path
import numpy as np
from main.tube_state import b_affine_margin_plan_v3 as p
from main.tube_state import b_affine_margin_v3 as core
from runtime.wan.b_lfg_posthoc_reference_v1 import save,read
from runtime.wan.b_lowrate_local_time_label_evaluation_v1 import compare,pair_summary

def geometry(attack):return [33+r+(13 if attack=='delete13_s52' and r>=52 else 0) for r in range(104)]
def query(out,a,attack,k):
 yp=[];yd=[];tables=[]
 for wi in range(3):
  for phase in range(8):
   v=f'{a}/{attack}/w{wi}/p{phase}';r=read(out/'raw'/v/(k+'.json'));tables.append(read(out/'payload'/v/(k+'.json')))
   for row in r['rows']:yp.append(row['pilot_scores']);yd.append(row['data_scores'])
 return np.array(yp,float),np.array(yd,float),core.payload_records(tables)
def reference_frames(ref):return list(range((0,26,52)[ref['window']]+ref['phase']+4*ref['latent_row']-3,(0,26,52)[ref['window']]+ref['phase']+4*ref['latent_row']+1))
def post_bits(value,records,true_map,message,require_tags=True):
 rows=[]
 for i,bit in enumerate(value['decoded']):
  seg,b=divmod(i,8);used=[]
  for ref in value['selected_refs']:
   if (ref['source_cell']-1)//11!=seg:continue
   rec=records[(ref['window'],ref['phase'],ref['latent_row'])]
   if require_tags and not rec['bits'][b]['readable']:continue
   sf=[true_map[x] for x in reference_frames(ref)];used.append(core.source_cell(sf)!=ref['source_cell'])
  rows.append(dict(bit=i,decoded=bit,target=message[i],read=bit is not None,error=None if bit is None or message[i] is None else bit!=message[i],false_attribution=bit is not None and any(used)))
 return rows

def classify_bit(row,variant,blind,records,true_map):
 seg,b=divmod(row['bit'],8);left=1+44*seg;right=44*(seg+1);present=sum(left<=x<=right for x in true_map);geometric=[];acquired=[]
 for ref,rec in records.items():
  t=core.source_cell(true_map[rec['lo']:rec['hi']+1])
  if t is None or (t-1)//11!=seg:continue
  geometric.append(ref)
  if not rec['bits'][b]['technical']:acquired.append(ref)
 if row['read']:kind='READ'
 elif not present:kind='PHYSICALLY_ABSENT_GROUP'
 elif not geometric:kind='NO_COMPLETE_OBSERVATION'
 elif not acquired:kind='TECHNICAL_UNREAD'
 elif variant=='E' and not blind['reference_set_paths']:kind='NO_COMPATIBLE_REFERENCE'
 elif variant=='O_admitted' and not blind.get('oracle_true_in_reference',False):kind='ORACLE_PATH_NOT_ADMITTED'
 elif variant=='E' and not any((r['source_cell']-1)//11==seg for r in blind['payload_E']['selected_refs']):kind='NO_COMMON_SOURCE_SUPPORT'
 else:kind='TIE_ZERO_OR_SUPPORTED_READ_ERASURE'
 row.update(coverage_class=kind,source_group_frames_present=present,geometric_full_ref_count=len(geometric),acquired_full_ref_count=len(acquired))
 return row

def corrected_description(yp,yd,cp,cd,key,frame_sets):
 bk=core.book(key);mask=np.array(bk['mask']);pb=np.array(bk['pilot_bits']);out=[]
 for j,frames in enumerate(frame_sets):
  rp=yp[j]-cp;rd=yd[j]-cd
  if not np.isfinite(rp).all() or not np.isfinite(rd).all():out.append(dict(row=j,status='TECHNICAL_UNRESOLVED'));continue
  gs={(x+3)//8 for x in frames};pure=len(gs)==1 and list(frames)==list(range(frames[0],frames[0]+8)) and 1<=next(iter(gs))<=21;g=next(iter(gs)) if pure else None
  zero=int(np.count_nonzero(rd==0));bits=(rd>0).astype(np.int8)^mask;dist=np.sum(core.WORDS!=bits,axis=1);nearest=np.flatnonzero(dist==dist.min()).tolist();value=nearest[0] if not zero and len(nearest)==1 and dist[nearest[0]]<=7 and 1<=nearest[0]<=21 else None
  out.append(dict(row=j,status='POST_CORRECTED_DIAGNOSTIC',nominal_kind='FULL' if pure else 'MIXED_OR_SEAM',true_g_post_only=g,corrected_pilot=rp.tolist(),corrected_label=rd.tolist(),label_exactzeros=zero,distances64=dist.tolist(),nearest_RM_words=nearest,decoded_valid_ID=value,target_distance=int(dist[g]) if g is not None else None,min_target_label_margin=float(((2*(core.WORDS[g]^mask)-1)*rd).min()) if g is not None else None,pilot_coherent_mean=float(np.mean((2*(pb^(g%2))-1)*rp)) if g is not None else None,pilot_min_margin=float(((2*(pb^(g%2))-1)*rp).min()) if g is not None else None,receiver_output_changed=False))
 return out

def full_diagnostic(out):
 rows=[]
 for a in p.ARMS:
  for k,key in p.KEYS.items():
   raw=read(out/'raw'/a/'full'/(k+'.json'));yp=np.array([x['pilot_scores'] for x in raw['rows']],float);yd=np.array([x['data_scores'] for x in raw['rows']],float)
   if raw['status']!='OBSERVED' or not np.isfinite(yp).all() or not np.isfinite(yd).all():rows.append(dict(arm=a,key=k,status='TECHNICAL_UNRESOLVED',layer='full_MP4_E181'));continue
   bk=core.book(key);B_p=np.array([2*(np.array(bk['pilot_bits'])^(g%2))-1 for g in range(1,22)]);B_d=np.array([2*(core.WORDS[g]^np.array(bk['mask']))-1 for g in range(1,22)]);centers=[];gains=[]
   for Y,B in [(yp,B_p),(yd,B_d)]:
    x=B-B.mean(0);y=Y-Y.mean(0);gain=max(0,float((x*y).sum()))/float(np.square(x).sum());centers.append(Y.mean(0)-gain*B.mean(0));gains.append(gain)
   obs=corrected_description(yp,yd,*centers,key,[list(range(8*g-3,8*g+5)) for g in range(1,22)])
   rows.append(dict(arm=a,key=k,status='POST_PUBLIC_GRID_DIAGNOSTIC',layer='full_MP4_E181_only',gain_pilot=gains[0],gain_label=gains[1],fitted_c_pilot=centers[0].tolist(),fitted_c_label=centers[1].tolist(),rows=obs,authentication=False,receiver_output_changed=False))
 return rows

def summarize_alignment(align,message):
 query_summary={}
 for name in p.families():
  arm,attack,key=name.split('/');one={}
  for variant in ('U','E','O_geom','O_admitted'):
   rr=[x for x in align if x['id']==name and x['variant']==variant];known=[x for x in rr if x['read'] and x['target'] is not None];cs=[x for x in known if not x['false_attribution']]
   one[variant]=dict(planned=32,read=sum(x['read'] for x in rr),read_target_known=len(known),errors=sum(x['error'] is True for x in known),correct_source_read=len(cs),correct_source_errors=sum(x['error'] is True for x in cs),correct_source_BER=sum(x['error'] is True for x in cs)/len(cs) if cs else None,false_attributed_bits=sum(x['false_attribution'] for x in rr),coverage_classes={v:sum(x['coverage_class']==v for x in rr) for v in sorted({x['coverage_class'] for x in rr})})
  query_summary[name]=dict(arm=arm,attack=attack,key=key,descriptive_control=arm=='P0' or key=='key01',variants=one)
 controls=[x for x in query_summary.values() if x['descriptive_control']]
 control_outputs=dict(queries=len(controls),reference_E_output_bits=sum(x['variants']['E']['read'] for x in controls),authentication_FPR=None,meaning='Descriptive P0/wrong-key reference outputs, not certification or FPR')
 alignment_pairs=[]
 index={(x['id'],x['variant'],x['bit']):x for x in align}
 for a in ('OLD','NEW'):
  for attack in p.ATTACKS:
   for k in p.KEYS:
    for bit in range(32):
     x=index[(f'P0/{attack}/{k}','E',bit)];y=index[(f'{a}/{attack}/{k}','E',bit)];bx=x['decoded'] if not x['false_attribution'] else None;by=y['decoded'] if not y['false_attribution'] else None
     alignment_pairs.append(dict(arm=a,attack=attack,key=k,bit=bit,baseline_false_attribution=x['false_attribution'],marked_false_attribution=y['false_attribution'],**compare(bx,by,message[bit])))
 return query_summary,control_outputs,alignment_pairs

def evaluate(output,protocol):
 out=Path(output)
 if not (out/'markers/blind_complete.json').is_file():raise ValueError('blind save/readback required')
 issues=[];message=[None]*32
 try:
  message=read(protocol)['config']['writer']['bits']
  if len(message)!=32 or any(type(x) is not int or x not in (0,1) for x in message):raise ValueError('original32 bits')
 except (OSError,ValueError,KeyError,TypeError) as e:issues.append(str(e));message=[None]*32
 private=read(out/'private/preparation.json')
 if (private['origin'],private['delete_frames'],private['seam_received'])!=(33,13,52):raise ValueError('fixed preparation geometry')
 group=[];cells=[];local=[];align=[];metadata=[];corrected_post=[]
 for k in p.KEYS:
  base=read(out/'payload/P0/full'/(k+'.json'))
  for a in ('OLD','NEW'):
   marked=read(out/'payload'/a/'full'/(k+'.json'))
   for b in range(32):group.append(dict(arm=a,key=k,bit=b,**compare(base['decoded_bits'][b],marked['decoded_bits'][b],message[b])))
   for t in range(44):
    for b in range(8):cells.append(dict(arm=a,key=k,cell=t+1,bit=b,**compare(base['local_cells'][t]['bits'][b]['decoded'],marked['local_cells'][t]['bits'][b]['decoded'],message[8*(t//11)+b])))
 for a in p.ARMS:
  for attack in p.ATTACKS:
   true=geometry(attack)
   for k,key in p.KEYS.items():
    name=f'{a}/{attack}/{k}';blind=read(out/'blind'/name/'decision.json');yp,yd,records=query(out,a,attack,k)
    with np.load(out/'blind'/name/'scores.npz') as scores:
     target=(scores['O']==33)&(scores['d']==(13 if attack=='delete13_s52' else 0))&((scores['d']==0)|(scores['s']==52));ix=int(np.flatnonzero(target)[0]);contains=bool(ix in scores['H']);wrong=sum(v is not None and v!=true[i] for i,v in enumerate(blind['common_frames']));blind['oracle_true_in_reference']=contains
     framesets=[true[w+ph+8*j-3:w+ph+8*j+5] for w in (0,26,52) for ph in range(8) for j in range(1,5)]
     corrected_post.append(dict(id=name,status='POST_TRUE_PATH_DIAGNOSTIC_ONLY',true_path_in_reference=contains,rows=corrected_description(yp,yd,scores['all_candidate_c_pilot'][ix],scores['all_candidate_c_label'][ix],key,framesets),receiver_output_changed=False))
     metadata.append(dict(id=name,status=blind['status'],model_domain='MODEL_DOMAIN_UNVALIDATED',H=blind['reference_set_paths'],true_path_in_reference_set=contains,common_frames=sum(v is not None for v in blind['common_frames']),wrong_common_frames=wrong,D_true=float(scores['D_score'][ix]),L_true=float(scores['L_score'][ix]),R_true=float(scores['R_score'][ix]),actual_authentication=False))
     og=core.recover(records,true,set(range(1,22)),False);oa=dict(decoded=[None]*32,selected_refs=[])
     if contains and np.isfinite(yp).all() and np.isfinite(yd).all():
      xx=np.array([np.arange(w+ph+8*j-3,w+ph+8*j+5) for w in (0,26,52) for ph in range(8) for j in range(1,5)])
      sf=np.array(true)[xx];gg=(sf+3)//8;z=np.zeros((96,11))
      for t in range(8):z[np.arange(96),gg[:,t]//2]+=(1-2*(gg[:,t]%2))/8
      cp,cg,_,_=core.support(np.array([0]),np.array([33]),np.array([13 if attack=='delete13_s52' else 0]),np.array([52 if attack=='delete13_s52' else 104]),xx,yp,yd,key,np.array([scores['gain_pilot'][ix]]),np.array([scores['gain_label'][ix]]),np.array([z.sum(1).mean()]),z.mean(0)[None]);oa=core.recover(records,cp,cg)
     for variant,value in [('U',blind['payload_U']),('E',blind['payload_E']),('O_geom',og),('O_admitted',oa)]:
      for row in post_bits(value,records,true,message,variant in ('E','O_admitted')):align.append(dict(id=name,variant=variant,**classify_bit(row,variant,blind,records,true)))
 for attack in p.ATTACKS:
  true=geometry(attack)
  for k in p.KEYS:
   _,_,baseline=query(out,'P0',attack,k)
   for a in ('OLD','NEW'):
    _,_,marked=query(out,a,attack,k)
    for ref,rec in baseline.items():
     sf=[true[x] for x in range(rec['lo'],rec['hi']+1)];cell=core.source_cell(sf)
     for b in range(8):
      x=rec['bits'][b]['votes'];y=marked[ref]['bits'][b]['votes'];bx=None if not x else __import__('collections').Counter(x).most_common(1)[0][0];by=None if not y else __import__('collections').Counter(y).most_common(1)[0][0];target=None if cell is None else message[8*((cell-1)//11)+b]
      local.append(dict(arm=a,key=k,attack=attack,ref=list(ref),bit=b,geometry='FULL' if cell is not None else 'MIXED_OR_SEAM',**compare(bx,by,target)))
 save(out/'post_true_path_corrected_rows.json',corrected_post);save(out/'post_full_readback_corrected_rows.json',full_diagnostic(out))
 for name,value in [('full_group_bit_pairs',group),('full_cell_bit_pairs',cells),('local_raw_bit_pairs',local),('alignment_bit_rows',align),('metadata_post_rows',metadata)]:save(out/(name+'.json'),value)
 summary={}
 for variant in ('U','E','O_geom','O_admitted'):
  rr=[r for r in align if r['variant']==variant];summary[variant]=dict(planned=len(rr),read=sum(r['read'] for r in rr),errors=sum(r['error'] is True for r in rr),false_attributed_bits=sum(r['false_attribution'] for r in rr),coverage_classes={name:sum(r['coverage_class']==name for r in rr) for name in sorted({r['coverage_class'] for r in rr})})
 query_summary,control_outputs,alignment_pairs=summarize_alignment(align,message)
 save(out/'alignment_correct_source_pairs.json',alignment_pairs)
 return dict(status='POST_TRUTH_UNRESOLVED' if issues else 'POST_DESCRIPTION_ONLY',issues=issues,host=p.HOST,model_domain='MODEL_DOMAIN_UNVALIDATED',scientific_PASS=None,rho_recalibrated=False,alignment=summary,alignment_by_query=query_summary,descriptive_control_outputs=control_outputs,alignment_correct_source_pairs={f'{a}/{attack}/{k}':pair_summary([x for x in alignment_pairs if x['arm']==a and x['attack']==attack and x['key']==k]) for a in ('OLD','NEW') for attack in p.ATTACKS for k in p.KEYS},full_group_by_arm_key={f'{a}/{k}':pair_summary([x for x in group if x['arm']==a and x['key']==k]) for a in ('OLD','NEW') for k in p.KEYS},full_group={k:pair_summary([r for r in group if r['key']==k]) for k in p.KEYS},full_cell={k:pair_summary([r for r in cells if r['key']==k]) for k in p.KEYS},local={k:pair_summary([r for r in local if r['key']==k]) for k in p.KEYS},metadata=metadata,full_group_absence_tested=False,E181_layer='MP4 readback only; RGB8 not separately encoded',causal_limit='Cannot separate all VAE/codec/reset effects with this physical budget',historical_background='V3 four hosts/84 units: s01/s04 admitted, s02/s03 numeric unresolved; only s01 physically planned here')
