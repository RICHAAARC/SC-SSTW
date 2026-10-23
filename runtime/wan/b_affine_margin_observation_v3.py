"""User-only fixed one-attempt V3 runner. Reentry is saved JSON read-only."""
from pathlib import Path
from collections import Counter
import json,time,os
from main.tube_state import b_affine_margin_plan_v3 as p
from runtime.wan.b_lfg_posthoc_reference_v1 import save,read,flush,existing,execution_artifacts
from runtime.wan.b_lowrate_local_time_label_v1 import Journal as PreviousJournal,payload_empty

def units():
 u={'PN':'PN_initialize','load/P0':'native_input_load','load/OLD':'native_input_load','construct':'native_construct_once','vae':'VAE_load','blind':'received_only_inference','blind_readback':'blind_saved_readback','post':'post_truth_evaluation'}
 for a in p.ARMS:
  if a=='NEW':u['native/NEW/save']='native_save'
  for name,kind in [('D','D181'),('quantize','RGB8_quantize'),('encode','MP4_encode'),('probe','MP4_probe'),('read','MP4_decode')]:u[a+'/full/'+name]=kind
  u['quality/'+a+'/clip']='post_clipping'
  for attack in p.ATTACKS:
   for name,kind in [('prepare','pixel_edit104'),('encode','MP4_encode'),('probe','MP4_probe'),('read','MP4_decode')]:u[a+'/'+attack+'/'+name]=kind
 for v in p.views():
  u[v+'/E']='E181' if v.endswith('/full') else 'E45';u[v+'/LLL']='received_Haar_LLL';u[v+'/FFT']='payload_FFT'
  if not v.endswith('/full'):u[v+'/prepare']='pixel_window45'
  for k in p.KEYS:u[v+'/tag/'+k]='tag_correlations';u[v+'/payload/'+k]='payload_keyed_read'
 for a in ('OLD','NEW'):
  for view in ('full',)+p.ATTACKS:u['quality/'+a+'/'+view]='post_paired_quality'
 return u
class Journal(PreviousJournal):
 def __init__(self,out,backend):
  self.out=out;self.backend=backend;self.rows={k:dict(slot=k,kind=v,status='NOT_ENTERED',attempts=0) for k,v in units().items()};self.snapshot()

def array_save(path,values):
 import numpy as np
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**values);flush(f)
 tmp.replace(path)
def raw_path(v,k):return 'raw/'+v+'/'+k+'.json'
def pay_path(v,k):return 'payload/'+v+'/'+k+'.json'
def empty_raw(local):
 return dict(status='TECHNICAL_UNRESOLVED',rows=[dict(row=j,status='TECHNICAL_UNRESOLVED',pilot_scores=[None]*16,data_scores=[None]*32) for j in (range(1,5) if local else range(1,22))],truth_inputs=False,message_inputs=False)
def placeholders(out):
 for v in p.views():
  local=not v.endswith('/full')
  for k in p.KEYS:save(out/raw_path(v,k),empty_raw(local));save(out/pay_path(v,k),payload_empty(local))
 for name in p.families():save(out/'blind'/name/'decision.json',dict(status='NOT_ENTERED',model_domain='MODEL_DOMAIN_UNVALIDATED',planned_bits=32,decoded=[None]*32))

def seal_products(out,j):
 # Turn every failed/unentered physical product into an explicit saved technical row.
 # This is bookkeeping, never a retry of model/FFT/keyed reads.
 for v in p.views():
  local=not v.endswith('/full')
  for k in p.KEYS:
   for part,path in [('tag',raw_path(v,k)),('payload',pay_path(v,k))]:
    slot=v+'/'+part+'/'+k;r=j.rows[slot]
    value=read(out/path) if r['status']=='RETURNED' else empty_raw(local) if part=='tag' else payload_empty(local)
    value.update(journal_slot=slot,journal_status=r['status'],journal_attempts=r['attempts'])
    save(out/path,value)

def verify_products(out,j):
 for v in p.views():
  local=not v.endswith('/full')
  for k in p.KEYS:
   for part,path in [('tag',raw_path(v,k)),('payload',pay_path(v,k))]:
    slot=v+'/'+part+'/'+k;r=j.rows[slot];value=read(out/path);receipt=read(out/'receipts'/(slot+'.json'))
    if (value.get('journal_slot'),value.get('journal_status'),value.get('journal_attempts'))!=(slot,r['status'],r['attempts']) or receipt['status']!=r['status'] or receipt['attempts']!=r['attempts']:raise ValueError('saved technical status disagrees with operation receipt: '+slot)
    if r['status']!='RETURNED':
     if value['status']!='TECHNICAL_UNRESOLVED':raise ValueError('failed product masquerades as observed: '+slot)
     if part=='payload' and value['signed_votes'] is not None:raise ValueError('failed payload still has votes')
    elif part=='tag':
     import math
     if [row['row'] for row in value['rows']]!=list(range(1,5) if local else range(1,22)) or value['status']!='OBSERVED' or any(len(row['pilot_scores'])!=16 or len(row['data_scores'])!=32 or row['status']!='OBSERVED' or not all(type(x) in (float,int) and math.isfinite(x) for x in row['pilot_scores']+row['data_scores']) for row in value['rows']):raise ValueError('returned tag product malformed')
    else:
     import numpy as np
     shape=(4,11 if local else 44,240)
     if np.shape(value['signed_votes'])!=shape or np.shape(value['zero_coefficient_mask'])!=shape or not np.isin(value['signed_votes'],[-1,1]).all() or not np.isin(value['zero_coefficient_mask'],[0,1]).all():raise ValueError('returned payload product malformed')
     if len(value['local_cells'])!=(11 if local else 44) or any(len(x['bits'])!=8 for x in value['local_cells']):raise ValueError('fixed local payload cells')
     if not local and (len(value['decoded_bits'])!=32 or any(x is not None and x not in (0,1) for x in value['decoded_bits'])):raise ValueError('original full32 slots')
 return True

def queries_from_saved(out):
 queries=[]
 for a in p.ARMS:
  for attack in p.ATTACKS:
   for k,key in p.KEYS.items():
    yp=[];yd=[];payload=[];missing=False
    for wi in range(3):
     for phase in range(8):
      v=f'{a}/{attack}/w{wi}/p{phase}';raw=read(out/raw_path(v,k));payload.append(read(out/pay_path(v,k)));missing|=raw['status']!='OBSERVED'
      if [r['row'] for r in raw['rows']]!=[1,2,3,4]:raise ValueError('fixed four tag rows')
      for r in raw['rows']:yp.append(r['pilot_scores']);yd.append(r['data_scores']);missing|=r['status']!='OBSERVED'
    queries.append(dict(id=f'{a}/{attack}/{k}',pilot=yp,label=yd,payload=payload,missing=missing,key=key))
 return queries

def verify_blind(out,j):
 verify_products(out,j)
 queries_from_saved(out)
 import numpy as np
 for name in p.families():
  result=read(out/'blind'/name/'decision.json')
  if result['model_domain']!='MODEL_DOMAIN_UNVALIDATED' or len(result['payload_E']['decoded'])!=32 or len(result['common_frames'])!=104:raise ValueError('fixed blind product geometry')
  for variant in ('payload_E','payload_U'):
   value=result[variant]
   if len(value['decoded'])!=32 or any(x is not None and x not in (0,1) for x in value['decoded']):raise ValueError('fixed 32 blind bit slots')
   for ref in value['selected_refs']:
    if ref['source_cell'] not in range(1,45) or ref['window'] not in range(3) or ref['phase'] not in range(8) or ref['latent_row'] not in range(1,12):raise ValueError('blind reference geometry')
  with np.load(out/'blind'/name/'scores.npz') as z:
   H=z['H'];nh=len(H)
   if H.ndim!=1 or len(np.unique(H))!=nh or np.any((H<0)|(H>=161994)) or nh!=result['reference_set_paths']:raise ValueError('all H not persisted')
   shapes={k:(161994,) for k in ('O','d','s','D_score','L_score','R_score','gain_pilot','gain_label')}
   shapes.update(all_candidate_c_pilot=(161994,16),all_candidate_c_label=(161994,32),candidate_c_pilot=(nh,16),candidate_c_label=(nh,32),corrected_pilot=(nh,96,16),corrected_label=(nh,96,32),candidate_pure_ids=(nh,96))
   for key,shape in shapes.items():
    value=z[key] # Materialize/decompress every promised array, not just its directory metadata.
    if value.shape!=shape:raise ValueError('blind array geometry: '+key)
    if result['status'] not in ('NUMERICALLY_UNRESOLVED',) and not np.isfinite(value).all():raise ValueError('nonfinite returned blind array: '+key)
   if result['corrected_row_evidence']['paths']!=nh or len(result['corrected_row_evidence']['rows'])!=(96 if nh else 0):raise ValueError('corrected row evidence not saved')
 return dict(status='BLIND_SAVED_READBACK',queries=12,truth_opened=False,model_domain='MODEL_DOMAIN_UNVALIDATED')

def run(p0=p.P0,old=p.OLD,output=p.OUTPUT,truth=p.TRUTH,*,backend=None):
 out=Path(output)
 if execution_artifacts(out):return existing(out)
 out.mkdir(parents=True,exist_ok=True)
 try:
  with (out/'attempt.json').open('x') as f:json.dump(dict(one_attempt=True,started_unix=time.time(),native_construct_budget=1,physical_reentry=False),f);flush(f)
 except FileExistsError:return existing(out)
 j=None;result=dict(status='STARTED',model_domain='MODEL_DOMAIN_UNVALIDATED',scientific_PASS=None,physical_reentry=False)
 try:
  j=Journal(out,backend);placeholders(out);save(out/'plan.json',dict(fixed=p.fixed(),host=p.HOST,arms=p.ARMS,keys=list(p.KEYS),E181_layer='full_MP4_readback_only',local_layer='edited104_MP4_readback_then_reset_E45',rho_policy='frozen_toy_reference_only_no_real_certification'))
  # Private preparation is not passed into the inference interface.
  save(out/'private/preparation.json',dict(origin=33,delete_frames=13,seam_received=52,source_frames=181,receiver_access=False))
  if backend is None:
   from runtime.wan.b_affine_margin_backend_v3 import Backend
   backend=Backend(p.MODEL);j.backend=backend
  j.record('PN',backend.initialize_PN,'PN/parameters.json')
  source=j.call('load/P0',lambda:backend.load_native(Path(p0)));previous=j.call('load/OLD',lambda:backend.load_native(Path(old)))
  new=j.call('construct',lambda:backend.construct_new(source,previous,lambda name,value:save(out/'construction'/name,value)),['load/P0','load/OLD','PN'])
  j.call('native/NEW/save',lambda:backend.save_tensor(out/'native'/'NEW.pt',new),['construct'])
  j.record('vae',backend.load_vae,'vae.json')
  def encoded(v,pixels,dep,local):
   ep=v+'/E'
   def encode():
    z=backend.encode_E45(pixels) if local else backend.encode_full(backend.prepare_full(pixels));backend.save_tensor(out/'latents'/(v+'.pt'),z);return z
   z=j.call(ep,encode,[dep,'vae']);low=j.call(v+'/LLL',lambda:backend.tag_lll(z,local),[ep])
   for k,key in p.KEYS.items():j.record(v+'/tag/'+k,lambda key=key:backend.tag_scores(low,key,list(range(1,5) if local else range(1,22))),raw_path(v,k),[v+'/LLL','PN'])
   spectrum=j.call(v+'/FFT',lambda:backend.local_payload_spectrum(z) if local else backend.payload_spectrum(z),[ep])
   for k,key in p.KEYS.items():j.record(v+'/payload/'+k,lambda key=key:backend.local_payload_votes(spectrum,key) if local else backend.payload_votes(spectrum,key),pay_path(v,k),[v+'/FFT'])
   del z,low,spectrum
  def codec(v,pixels,dep,length):
   path=out/'media'/v/'video.mp4';base=path.parent
   j.record(v+'/encode',lambda:backend.save_mp4(pixels,path),str((base/'encode.json').relative_to(out)),[dep])
   probe=j.record(v+'/probe',lambda:backend.probe(path),str((base/'probe.json').relative_to(out)),[v+'/encode'])
   def decode():
    rgb=backend.read_media(path,probe,length);backend.save_tensor(base/'readback.pt',rgb);return rgb
   return j.call(v+'/read',decode,[v+'/probe'])
  for a,z,dep in [('P0',source,'load/P0'),('OLD',previous,'load/OLD'),('NEW',new,'native/NEW/save')]:
   v=a+'/full'
   def decode():
    values=backend.decode_PT(z)
    for layer,x in values.items():backend.save_tensor(out/'media'/v/(layer+'.pt'),x)
    return values
   decoded=j.call(v+'/D',decode,[dep,'vae'])
   def quantize():
    rgb=backend.quantize(decoded['float_rgb']);backend.save_tensor(out/'media'/v/'rgb8.pt',rgb);return rgb
   rgb=j.call(v+'/quantize',quantize,[v+'/D']);del decoded
   full=codec(v,rgb,v+'/quantize',181);encoded(v,full,v+'/read',False)
   for attack in p.ATTACKS:
    av=a+'/'+attack;received=j.call(av+'/prepare',lambda:backend.received104(full,attack),[v+'/read']);received=codec(av,received,av+'/prepare',104)
    for wi in range(3):
     for phase in range(8):
      lv=f'{av}/w{wi}/p{phase}';pixels=j.call(lv+'/prepare',lambda:backend.window45(received,wi,phase),[av+'/read']);encoded(lv,pixels,lv+'/prepare',True);del pixels
    del received
   del rgb,full
  del source,previous,new
  def inference():
   # Only saved received matrices, keys, and received payload votes enter this API.
   seal_products(out,j);verify_products(out,j)
   qs=queries_from_saved(out)
   rows=backend.infer(qs,lambda name,values:array_save(out/'blind'/name/'scores.npz',values))
   if len(rows)!=12 or {x['id'] for x in rows}!=set(p.families()):raise ValueError('twelve blind families retained')
   for r in rows:save(out/'blind'/r['id']/'decision.json',r)
   return dict(status='BLIND_PRODUCTS_SAVED',queries=12,model_domain='MODEL_DOMAIN_UNVALIDATED')
  j.record('blind',inference,'markers/blind_products.json')
  j.record('blind_readback',lambda:verify_blind(out,j),'markers/blind_complete.json',['blind'])
  if j.rows['blind_readback']['status']=='RETURNED':
   save(out/'post_truth_access.json',dict(opened_unix=time.time(),all_blind_saved_and_read_back=True))
   from runtime.wan.b_affine_margin_evaluation_v3 import evaluate
   j.record('post',lambda:evaluate(out,Path(truth)),'evaluation.json',['blind_readback'])
   for a in p.ARMS:j.record('quality/'+a+'/clip',lambda a=a:backend.preclip(backend.load_tensor(out/'media'/a/'full/preclip.pt')),f'quality/{a}/clipping.json',[a+'/full/D','blind_readback'])
   for a in ('OLD','NEW'):
    for view in ('full',)+p.ATTACKS:
     j.record('quality/'+a+'/'+view,lambda a=a,view=view:backend.paired_quality(backend.load_tensor(out/'media'/'P0'/view/'readback.pt'),backend.load_tensor(out/'media'/a/view/'readback.pt'),'same-runtime P0 to '+a+' '+view),f'quality/{a}/{view}.json',['P0/'+view+'/read',a+'/'+view+'/read','blind_readback'])
  result['status']='COMPLETED_DESCRIPTION_ONLY' if all(x['status']=='RETURNED' for x in j.rows.values()) else 'INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
 except BaseException as e:result.update(status='FAILED' if isinstance(e,Exception) else 'INTERRUPTED',error=type(e).__name__+': '+str(e))
 finally:
  if backend is not None:
   try:backend.release_vae()
   except Exception as e:result['release_error']=str(e)
  if j is not None:
   j.snapshot();result.update(ledger_status=dict(Counter(x['status'] for x in j.rows.values())),kinds_started=dict(Counter(x['kind'] for x in j.rows.values() if x['attempts'])),kinds_returned=dict(Counter(x['kind'] for x in j.rows.values() if x['status']=='RETURNED')))
  result.update(fixed_denominators=p.fixed(),host=p.HOST,user_net_bits=32,sourceprep=0,independent_confirmation=False,rho_recalibrated=False,scientific_PASS=None);save(out/'result.json',result)
 return result
