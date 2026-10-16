"""Fixed four-source temporal-payload user run, with finite physical attempts."""
from __future__ import annotations
from collections import Counter
import io
import json
from pathlib import Path
import time
import numpy as np
from experiments.paper_results_v1.resume_state import StateJournal, atomic_bytes, read_state, state_exists, unit_attempt
from main.tube_state import segmented_payload_v1 as method
from main.tube_state import video_trajectory_temporal_edit_receiver_v1 as temporal
from main.tube_state.receiver_controls_v1 import select_score_path

KEY='watermark'
BITS=[int(b) for b in f'{0xA6D39C5E:032b}']
VIEWS=('CROP_MID','DELETE_S1')
BLIND=('CENTERED_U','RAW_U','UNALIGNED')
RGB_SHAPE=(181,320,512,3)
LIMITS=dict(generation_load=4,trajectory=4,native_decode=4,framewise_encode=8,clock_write=4,
            framewise_decode=8,codec=4,selector=16,wan_encode=36,segment_read=36,
            wan_decode_load=2,framewise_load=2,wan_reader_load=2)
CALL_LIMITS=dict(transformer_conditional=200,transformer_unconditional=200,native_step=200,
                 payload_gradient=100,framewise_encode_batch=184,framewise_decode_batch=184,
                 mp4_save=4,mp4_probe=4,mp4_readback=4)
OUTPUT_DEFAULT='/content/drive/MyDrive/Video-WM/Paper-Results-V1-Segmented-Payload-V1/resumable-segmented-payload-v1'


def cases():
    source=Path(__file__).with_name('independent_four_v1.adopted.json')
    rows=[{k:r[k] for k in ('case_id','seed','prompt','negative_prompt')} for r in json.loads(source.read_text())['cases']]
    if [r['case_id'] for r in rows]!=[f'independent_20261015_{i:02d}' for i in range(1,5)]:
        raise ValueError('fixed four-source roster requires exactly the four adopted distinct case IDs in order')
    if any(type(r['seed'])!=int or not isinstance(r['prompt'],str) or not r['prompt'] or not isinstance(r['negative_prompt'],str) for r in rows):
        raise ValueError('source prompt/seed has invalid actual format')
    return rows


def view_indices(view):
    if view=='CLEAN':return list(range(181))
    if view=='CROP_MID':return list(range(37,126))
    if view=='DELETE_S1':return list(range(45))+list(range(89,181))
    raise ValueError('unknown fixed received view')


def planned_rows():
    rows=[]
    for case in cases():
        cid=case['case_id']
        for view,arms in [('CLEAN',('ORACLE',)),*[(v,(*BLIND,'ORACLE')) for v in VIEWS]]:
            for arm in arms:
                rows.append(dict(slot_id=f'{cid}/{view}/{arm}',case_id=cid,view=view,receiver=arm,
                    oracle=arm=='ORACLE',planned_bits=32,planned_segments=4,status='PENDING',reason=None))
    return rows


def fixed_semantics():
    return dict(cases=cases(),key=KEY,bits=BITS,views={v:view_indices(v) for v in ('CLEAN',*VIEWS)},
        segment_times=[[1,11],[12,22],[23,33],[34,44]],guards=[0,45],channels=[0,1,2,3],
        alpha=.5,mask_count=44160,reference_eta=200.,eta=5520.,active=list(range(25,50)),steps=50,cfg=5.,
        clock_margin=.5,framewise_batch=8,codec=dict(crf=18,pixel_format='yuv420p',fps=8),
        receiver_arms=list(BLIND),transition='U',full_cell_support=True,clean_exact_before_usage=True,
        target_assignment='all original channels: bits[8*segment(t)+coordinate_index%8]',
        receiver_source_time='anchor//4 + receiver_wan_time',
        cell_support='all source positions 4*t-3 through 4*t must occur in estimated received path',
        read_transform='fft2 ortho real; strict >0; channel,time,coordinate order; Counter first-encounter ties',
        physical_limits=LIMITS,call_limits=CALL_LIMITS,attempts_per_unit=1)


def compare_known(saved,current,prefix='method'):
    if not isinstance(saved,dict):return
    for key,value in saved.items():
        if key not in current:continue
        if isinstance(value,dict) and isinstance(current[key],dict):compare_known(value,current[key],prefix+'.'+key)
        elif value!=current[key]:raise ValueError('known fixed method differs at '+prefix+'.'+key)


def unknown_method_fields(saved,current,prefix='method'):
    if not isinstance(saved,dict):return [prefix]
    missing=[]
    for key,value in current.items():
        if key not in saved:missing.append(prefix+'.'+key)
        elif isinstance(value,dict):missing.extend(unknown_method_fields(saved[key],value,prefix+'.'+key))
    return missing


def write_json(path,value):
    atomic_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())


def save_array(path,array):
    buffer=io.BytesIO();np.save(buffer,array,allow_pickle=False);atomic_bytes(path,buffer.getvalue())


def finite_array(path,shape,dtype=np.float32):
    value=np.load(path,allow_pickle=False)
    if value.dtype!=dtype or tuple(value.shape)!=tuple(shape) or not np.isfinite(value).all():
        raise ValueError(f'expected finite {np.dtype(dtype)} {tuple(shape)} at {path}')
    return value


class Store:
    def __init__(self,output):
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.path=self.output/'segmented_payload_state.json';self.journal=StateJournal(self.path)
        had_existing_state=state_exists(self.path)
        self.data=read_state(self.path) if had_existing_state else dict(phases={},rows=planned_rows(),
            artifacts=[],physical_inputs=[],unit_attempts={},counts={},events=[],clean_readability={})
        expected=planned_rows()
        if len(expected)!=36:raise ValueError('fixed receiver plan must contain exactly 36 rows')
        if [r['slot_id'] for r in self.data['rows']]!=[r['slot_id'] for r in expected]:
            raise ValueError('actual fixed 36-row plan differs')
        for saved,current in zip(self.data['rows'],expected):
            for key in ('case_id','view','receiver','oracle','planned_bits','planned_segments'):
                if saved.get(key)!=current[key]:raise ValueError('fixed row method differs: '+saved['slot_id']+'/'+key)
        current=fixed_semantics();saved=self.data.get('fixed_method_semantics')
        unknown=unknown_method_fields(saved,current)
        if unknown and had_existing_state:
            self.data.setdefault('notes',[]).append(dict(status='PRIOR_METHOD_RECORD_UNAVAILABLE',fields=unknown,
                meaning='available old evidence retained; no retrospective verification of missing method fields claimed'))
        compare_known(saved,current)
        self.data['fixed_method_semantics']=current;self.journal.seed_cache(self.data)
        for attempts in self.data['unit_attempts'].values():
            for attempt in attempts:
                if attempt['status'] in ('STARTED','RUNNING'):
                    attempt.update(status='INTERRUPTED',reason='previous process ended before durable completion; attempt consumed')
        for artifact in self.data['artifacts']:
            if artifact['status']=='STARTED':artifact.update(status='INTERRUPTED',reason='physical attempt consumed')
        self.save()

    def save(self):self.journal.save(self.data)

    def record(self,identifier,kind=None):
        item=next((a for a in self.data['artifacts'] if a['artifact_id']==identifier),None)
        if item is None:
            item=dict(artifact_id=identifier,kind=kind,status='PENDING');self.data['artifacts'].append(item)
        elif kind is not None:item['kind']=kind
        return item

    def started(self,identifier):return bool(self.data['unit_attempts'].get(identifier))

    def used(self,kind):
        return sum(len(a) for a in self.data['unit_attempts'].values() if a and a[0].get('kind')==kind)

    def count(self,kind,returned):
        values=self.data['counts'].setdefault(kind,dict(started=0,returned=0))
        key='returned' if returned else 'started'
        if not returned and kind in CALL_LIMITS and values[key]>=CALL_LIMITS[kind]:
            raise RuntimeError('physical call budget exhausted: '+kind)
        values[key]+=1;self.save()

    def once(self,identifier,kind,function):
        """Persist STARTED before any call. Neither exceptions nor missing files reset it."""
        item=self.record(identifier,kind)
        if self.started(identifier):return None
        if kind in LIMITS and self.used(kind)>=LIMITS[kind]:
            item.update(status='BUDGET_EXHAUSTED',reason='physical unit cap consumed');self.save();return None
        try:
            with unit_attempt(self,identifier,kind=kind) as attempt:
                attempt['status']='STARTED';item.update(status='STARTED',reason=None);self.save()
                value=function(item)
                item['status']='AVAILABLE' if item.get('path') else 'COMPLETE';self.save()
                return value
        except BaseException as exc:
            item.update(status='FAILED' if isinstance(exc,Exception) else 'INTERRUPTED',reason=f'{type(exc).__name__}: {exc}')
            self.save()
            if not isinstance(exc,Exception):raise
            return None

    def saved(self,identifier,loader):
        item=self.record(identifier)
        if item['status'] not in ('AVAILABLE','MISSING'):return None
        try:
            value=loader(Path(item['path']))
            item.update(status='AVAILABLE',read_error=None)
            return value
        except Exception as exc:
            item.update(status='MISSING',read_error=f'{type(exc).__name__}: {exc}');self.save();return None

    def array(self,identifier,shape,dtype=np.float32):
        return self.saved(identifier,lambda p:finite_array(p,shape,dtype))

    def json(self,identifier):return self.saved(identifier,lambda p:json.loads(p.read_text()))

    def make_array(self,identifier,kind,shape,function,dtype=np.float32):
        value=self.array(identifier,shape,dtype)
        if value is not None:return value
        def call(item):
            array=np.asarray(function())
            if array.dtype!=dtype or tuple(array.shape)!=tuple(shape) or not np.isfinite(array).all():
                raise ValueError('computed array has wrong geometry, dtype, or nonfinite values')
            path=self.output/'data'/f'{identifier}.npy';save_array(path,array);item['path']=str(path)
            return array
        return self.once(identifier,kind,call)

    def make_json(self,identifier,kind,function):
        value=self.json(identifier)
        if value is not None:return value
        def call(item):
            value=function();path=self.output/'records'/f'{identifier}.json';write_json(path,value);item['path']=str(path)
            return value
        return self.once(identifier,kind,call)

    def block(self,identifier,reason,status='BLOCKED'):
        item=self.record(identifier)
        if not self.started(identifier):item.update(status=status,reason=reason);self.save()

    def load(self,kind,function):
        for index in range(LIMITS[kind]):
            identifier=f'load/{kind}/{index}'
            if not self.started(identifier):return self.once(identifier,kind,lambda item:function())
        self.data['events'].append(dict(kind=kind,status='BUDGET_EXHAUSTED',at=time.time()));self.save()
        return None


class Backend:
    """Real calls are lazy and only invoked by a user's worker process."""
    def __init__(self,wan,framewise,device):self.wan,self.framewise,self.device=wan,framewise,device

    def load_generation(self,case):
        import torch
        from runtime.wan.conditional_joint import generation
        dtype=torch.float32 if self.device=='cpu' else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        config=dict(model=dict(id=str(self.wan)),generation=dict(case,frames=181,height=320,width=512,steps=50,max_sequence_length=512))
        return generation.prepare_generation(config,load_vae=False,device=self.device,model_dtype=dtype)

    def generate(self,prepared,store,cid):
        from runtime.wan.segmented_payload_v1 import run_trajectory
        pipe,z,prompt,negative,dtype=prepared;records=[]
        terminal,receipt=run_trajectory(pipe,z,pipe.scheduler,prompt,negative,dtype,KEY,BITS,store.count,records.append)
        write_json(store.output/'records'/cid/'trajectory_details.json',dict(receipt=receipt,steps=records))
        return terminal.float().numpy()

    def load_wan(self):
        from runtime.wan.conditional_joint.generation import load_frozen_vae
        return load_frozen_vae(dict(model=dict(id=str(self.wan))),device=self.device)

    def decode_native(self,model,z):
        import torch
        from runtime.wan.conditional_joint.vae import decode_normalized_latent,quantize_rgb8_no_codec
        rgb=decode_normalized_latent(model,torch.from_numpy(z).to(self.device))
        if not bool(torch.isfinite(rgb).all()):raise ValueError('native Wan decode returned nonfinite RGB')
        return quantize_rgb8_no_codec(rgb).numpy()

    def load_framewise(self):
        from experiments.paper_results_v1.real_backends import LocalFramewiseBackend
        return LocalFramewiseBackend(dict(local_snapshot_path=str(self.framewise),batch_frames=8,device=self.device))

    @staticmethod
    def batch_observer(store,kind):
        def observe(status,start,stop,error):
            if status in ('STARTED','RETURNED'):store.count(kind,status=='RETURNED')
            if status=='FAILED':
                store.data['events'].append(dict(kind=kind,start=start,stop=stop,status=status,error=str(error)));store.save()
        return observe

    def encode_framewise(self,model,rgb,store):
        import torch
        return model.encode(torch.from_numpy(rgb),batch_observer=self.batch_observer(store,'framewise_encode_batch'))

    def decode_framewise(self,model,y,store):
        return model.decode(y,batch_observer=self.batch_observer(store,'framewise_decode_batch')).numpy()

    def codec(self,rgb,store,cid):
        import torch
        from runtime.wan.fixed_rgb_media import save_raster,mp4_roundtrip
        folder=store.output/'media'/cid;folder.mkdir(parents=True,exist_ok=True)
        raster=folder/'candidate_PRE.rgb8';save_raster(torch.from_numpy(rgb),raster)
        def event(kind,value):write_json(folder/(kind+'_receipt.json'),value)
        return mp4_roundtrip(raster,None,folder/'candidate.mp4',folder/'candidate_POST.rgb8',count=store.count,event=event).numpy()

    def encode_wan(self,model,rgb):
        import torch
        from runtime.wan.conditional_joint.vae import reencode_rgb24_readback
        return reencode_rgb24_readback(model,torch.from_numpy(rgb).float().div(255)).detach().cpu().float().numpy()

    @staticmethod
    def read(z,path):
        import torch
        return method.read_segments(torch.from_numpy(z),KEY,path)

    @staticmethod
    def close(model):
        import gc
        import torch
        if hasattr(model,'close'):model.close()
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def phase_generate(store,backend):
    for case in cases():
        cid=case['case_id'];identifier=cid+'/terminal'
        if store.array(identifier,(1,16,46,40,64)) is not None or store.started(identifier):continue
        prepared=store.once('load/generation/'+cid,'generation_load',lambda item:backend.load_generation(case))
        if prepared is None:
            store.block(identifier,'generation prepare/load unavailable; its one attempt is consumed');continue
        try:store.make_array(identifier,'trajectory',(1,16,46,40,64),lambda:backend.generate(prepared,store,cid))
        finally:
            backend.close(prepared);del prepared


def phase_native(store,backend):
    model=None;load_tried=False
    for case in cases():
        cid=case['case_id'];identifier=cid+'/native'
        if store.array(identifier,RGB_SHAPE,np.uint8) is not None or store.started(identifier):continue
        z=store.array(cid+'/terminal',(1,16,46,40,64))
        if z is None:store.block(identifier,'new segmented terminal unavailable');continue
        if not load_tried:model=store.load('wan_decode_load',backend.load_wan);load_tried=True
        if model is None:
            store.block(identifier,'native Wan model load unavailable','BUDGET_EXHAUSTED' if store.used('wan_decode_load')>=2 else 'BLOCKED');continue
        store.make_array(identifier,'native_decode',RGB_SHAPE,lambda:backend.decode_native(model,z),np.uint8)
    if model is not None:backend.close(model)


def quality(rgb,reference,label):
    squared=[];absolute=[]
    for left,right in zip(rgb,reference):
        delta=left.astype(np.float64)-right
        squared.append(float(np.mean(delta*delta)));absolute.append(float(np.mean(np.abs(delta))))
    mse=float(np.mean(squared))
    return dict(reference='this run segmented NATIVE_PRE',comparison=label,frames=len(rgb),
        rmse_rgb8=float(np.sqrt(mse)),mae_rgb8=float(np.mean(absolute)),psnr_db=10*np.log10(255**2/mse) if mse else None,
        psnr_infinite=mse==0,frame_mse=squared,meaning='clock residual' if label=='candidate_PRE' else 'clock plus codec relative to N_PRE')


def phase_framewise(store,backend):
    from main.tube_state.video_trajectory_payload_framewise_sync_v1 import apply_projection_margin
    model=None;load_tried=False
    def need_model():
        nonlocal model,load_tried
        if not load_tried:model=store.load('framewise_load',backend.load_framewise);load_tried=True
        return model
    for case in cases():
        cid=case['case_id'];native=store.array(cid+'/native',RGB_SHAPE,np.uint8)
        def fw_array(name,kind,shape,producer,dtype=np.float32,dependency=True):
            identifier=cid+'/'+name;value=store.array(identifier,shape,dtype)
            if value is not None or store.started(identifier):return value
            if not dependency:return None
            if need_model() is None:
                store.block(identifier,'framewise model load unavailable','BUDGET_EXHAUSTED' if store.used('framewise_load')>=2 else 'BLOCKED');return None
            return store.make_array(identifier,kind,shape,producer,dtype)
        y=fw_array('writer_y','framewise_encode',(181,4,40,64),lambda:backend.encode_framewise(model,native,store),dependency=native is not None)
        def clock_write():
            written,receipt=apply_projection_margin(y,KEY,target_margin=.5)
            write_json(store.output/'records'/cid/'clock_write.json',receipt)
            return written
        written=store.array(cid+'/written_y',(181,4,40,64))
        if written is None and y is not None:written=store.make_array(cid+'/written_y','clock_write',(181,4,40,64),clock_write)
        p=fw_array('recon','framewise_decode',RGB_SHAPE,lambda:backend.decode_framewise(model,y,store),np.uint8,dependency=y is not None)
        m=fw_array('margin','framewise_decode',RGB_SHAPE,lambda:backend.decode_framewise(model,written,store),np.uint8,dependency=written is not None)
        candidate=store.array(cid+'/candidate',RGB_SHAPE,np.uint8)
        if candidate is None and all(x is not None for x in (native,p,m)):
            candidate=store.make_array(cid+'/candidate','residual',RGB_SHAPE,
                lambda:np.clip(native.astype(np.int16)+m.astype(np.int16)-p.astype(np.int16),0,255).astype(np.uint8),np.uint8)
        received=store.array(cid+'/received',RGB_SHAPE,np.uint8)
        if received is None and candidate is not None:
            received=store.make_array(cid+'/received','codec',RGB_SHAPE,lambda:backend.codec(candidate,store,cid),np.uint8)
        if received is None:continue
        fw_array('received_y','framewise_encode',(181,4,40,64),lambda:backend.encode_framewise(model,received,store))
        if native is not None and candidate is not None:
            write_json(store.output/'records'/cid/'quality.json',[quality(candidate,native,'candidate_PRE'),quality(received,native,'candidate_POST')])
    if model is not None:backend.close(model)


def plan_from_scores(q):
    estimate=select_score_path(q,'U')
    if estimate['status']!='ESTIMATED':return dict(status='UNRESOLVED',estimate=estimate,plan=None,truth_inputs=False,message_inputs=False)
    plan=method.segment_plan(estimate['path'])
    return dict(status=plan['operation']['status'],estimate=estimate,plan=plan,truth_inputs=False,message_inputs=False)


def phase_plan(store,backend=None):
    # Every blind plan is committed before any correct-time payload answer exists.
    graph=artifact_graph(store)
    payload_boundary_crossed=any(store.started('read/'+r['slot_id']) for r in store.data['rows'])
    for case in cases():
        cid=case['case_id'];y=store.array(cid+'/received_y',(181,4,40,64))
        for view in VIEWS:
            n=len(view_indices(view))
            for arm in BLIND:
                identifier=f'plan/{cid}/{view}/{arm}'
                if store.json(identifier) is not None or store.started(identifier):continue
                if payload_boundary_crossed:
                    store.make_json(identifier,'unavailable_plan',lambda:dict(status='MISSING',plan=None,
                        reason='prior payload attempt preceded this unsaved blind slot; slot frozen without reading answers',
                        truth_inputs=False,message_inputs=False));continue
                if y is None:
                    if graph[cid]['received_y']['possible']:
                        store.block(identifier,'received clock Y may still be produced by an unstarted unit; all clean reads wait')
                    else:
                        store.make_json(identifier,'unavailable_plan',lambda:dict(status='MISSING',plan=None,
                            reason='clock Y unavailable and no reachable unstarted producer within remaining budgets; blind slot frozen',
                            truth_inputs=False,message_inputs=False))
                    continue
                if arm=='UNALIGNED':
                    store.make_json(identifier,'unaligned_plan',lambda:dict(status='SUPPORTED',plan=method.segment_plan(list(range(n))),truth_inputs=False,message_inputs=False))
                    continue
                def select():
                    evidence=temporal.score_framewise(y[view_indices(view)],KEY)
                    q=np.asarray(evidence['signed_projection'],np.float64)/np.asarray(evidence['rho'],np.float64)
                    if arm=='CENTERED_U':q=q-np.median(q,axis=0)
                    path=store.output/'observations'/cid/view/(arm+'.npy');save_array(path,q)
                    return dict(plan_from_scores(q),observation=str(path))
                store.make_json(identifier,'selector',select)
    if blind_plans_complete(store):
        store.data['phases']['blind_plans_committed']=dict(at=time.time(),payload_answer_used=False)
    else:store.data['phases'].pop('blind_plans_committed',None)
    store.save()
    if not blind_plans_complete(store):return
    for row in store.data['rows']:
        if row['oracle']:
            identifier='plan/'+row['slot_id'];path=view_indices(row['view'])
            store.make_json(identifier,'oracle_plan',lambda path=path:dict(status='SUPPORTED',plan=method.segment_plan(path),truth_inputs=True,message_inputs=False,role='separate correct-time diagnostic'))


def physical_input(store,row,plan):
    mapping=plan['operation']['received_index_map']
    # Equality of the actual map within this exact source and received view; R alone is insufficient.
    record=next((p for p in store.data['physical_inputs'] if p['artifact_id']==row['case_id']+'/received'
        and p['view']==row['view'] and p['received_index_map']==mapping),None)
    if record is None:
        record=dict(physical_id=f'wan_input_{len(store.data["physical_inputs"]):03d}',artifact_id=row['case_id']+'/received',
            view=row['view'],received_index_map=mapping,output_frames=len(mapping),logical_slots=[],key_reads=[],status='PLANNED')
        store.data['physical_inputs'].append(record)
    if row['slot_id'] not in record['logical_slots']:record['logical_slots'].append(row['slot_id'])
    row['physical_id']=record['physical_id'];store.save();return record


def saved_plan(store,identifier):
    def load(path):
        value=json.loads(path.read_text())
        row=next(r for r in store.data['rows'] if identifier=='plan/'+r['slot_id'])
        if value['truth_inputs'] is not row['oracle'] or value['message_inputs'] is not False:
            raise ValueError('saved plan blind/oracle input declarations differ')
        if value['status'] in ('UNRESOLVED','MISSING'):
            if value['plan'] is not None:raise ValueError('unresolved plan must not contain a path')
        elif value['status'] in ('SUPPORTED','UNSUPPORTED'):
            plan=value['plan'];expected=method.segment_plan(plan['estimated_path'])
            if len(plan['estimated_path'])!=len(view_indices(row['view'])):raise ValueError('saved plan received-view length differs')
            if plan!=expected or value['status']!=plan['operation']['status']:
                raise ValueError('saved plan has inconsistent actual mapping or segment support')
        else:raise ValueError('invalid selector plan status')
        return value
    return store.saved(identifier,load)


def blind_plans_complete(store):
    for row in store.data['rows']:
        if row['oracle']:continue
        identifier='plan/'+row['slot_id']
        if saved_plan(store,identifier) is None and not store.started(identifier):return False
    return True


def saved_read(store,identifier):
    def load(path):
        value=json.loads(path.read_text());bits=value['decoded_bits'];plan=value['plan']
        row=next(r for r in store.data['rows'] if identifier=='read/'+r['slot_id'])
        if not isinstance(bits,list) or len(bits)!=32 or any(b is not None and (type(b)!=int or b not in (0,1)) for b in bits):
            raise ValueError('saved segmented read requires 32 binary or missing entries')
        if value['decoded_bit_count']!=sum(b is not None for b in bits):raise ValueError('inconsistent decoded bit count')
        if plan!=method.segment_plan(plan['estimated_path']):raise ValueError('saved read mapping/support mismatch')
        if len(plan['estimated_path'])!=len(view_indices(row['view'])):raise ValueError('saved read received-view length differs')
        if value['oracle'] is not row['oracle'] or value['truth_inputs'] is not row['oracle']:
            raise ValueError('saved read blind/oracle declarations differ')
        with np.load(value['votes_path'],allow_pickle=False) as arrays:
            votes=arrays['signed_votes'];zeros=arrays['zero_coefficient_mask']
            reconstructed=method.decode_signed_votes(plan,votes,zeros)
        for key in ('status','decoded_bits','decoded_bit_count','segments','net_payload_bits','vote_order'):
            if value[key]!=reconstructed[key]:raise ValueError('saved read differs from actual votes at '+key)
        return value
    return store.saved(identifier,load)


def phase_reader(store,backend):
    if not blind_plans_complete(store):
        store.data['phases']['reader_waiting_for_blind_plans']=True;store.save();return
    store.data['phases'].pop('reader_waiting_for_blind_plans',None)
    model=None;load_tried=False
    def read_row(row):
        nonlocal model,load_tried
        identifier='read/'+row['slot_id'];saved=saved_read(store,identifier)
        if saved is not None:row.update(status='READ',reason=None);return saved
        if store.started(identifier):row.update(status='MISSING',reason='reader attempt consumed; saved read unavailable');return None
        selection=saved_plan(store,'plan/'+row['slot_id'])
        if selection is None:row.update(status='MISSING',reason='saved plan unavailable');return None
        if selection['status']!='SUPPORTED':row.update(status=selection['status'],reason='fixed selector/visible-span result');return None
        plan=selection['plan'];physical=physical_input(store,row,plan);pid=physical['physical_id']
        shape=(1,16,(physical['output_frames']-1)//4+1,40,64)
        z=store.array(pid,shape)
        if z is None and not store.started(pid):
            received=store.array(row['case_id']+'/received',RGB_SHAPE,np.uint8)
            if received is None:
                exhausted=any(a['artifact_id'].startswith(row['case_id']+'/') and a['status']=='BUDGET_EXHAUSTED' for a in store.data['artifacts'])
                row.update(status='BUDGET_EXHAUSTED' if exhausted else 'MISSING',reason='candidate RGB readback unavailable; source model budget exhausted' if exhausted else 'candidate RGB readback unavailable')
                return None
            if not load_tried:model=store.load('wan_reader_load',backend.load_wan);load_tried=True
            if model is None:
                row.update(status='BUDGET_EXHAUSTED' if store.used('wan_reader_load')>=2 else 'BLOCKED',reason='reader Wan model load unavailable');return None
            rgb=received[view_indices(row['view'])][physical['received_index_map']]
            z=store.make_array(pid,'wan_encode',shape,lambda:backend.encode_wan(model,rgb))
        if z is None:
            row.update(status='MISSING',reason='physical encode failed, interrupted, or unreadable; no repeated attempt');return None
        physical['status']='AVAILABLE';store.save()
        def read():
            value=backend.read(z,plan['estimated_path'])
            arrays={key:value.pop(key) for key in ('signed_votes','zero_coefficient_mask')}
            buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
            path=store.output/'votes'/(row['slot_id']+'.npz');atomic_bytes(path,buffer.getvalue());value['votes_path']=str(path)
            value.update(oracle=row['oracle'],truth_inputs=row['oracle'])
            return value
        result=store.make_json(identifier,'segment_read',read)
        row.update(status='READ' if result else 'FAILED',reason=None if result else 'segment reader attempt failed')
        store.save();return result
    try:
        for case in cases():
            cid=case['case_id'];rows=[r for r in store.data['rows'] if r['case_id']==cid]
            clean=next(r for r in rows if r['view']=='CLEAN');result=read_row(clean)
            exact=result is not None and result['decoded_bits']==BITS
            store.data['clean_readability'][cid]=dict(status='EXACT' if exact else 'UNREADABLE' if result else 'UNAVAILABLE',
                fully_decoded=result is not None and result['decoded_bit_count']==32,truth_used_only_for_execution_prerequisite=True)
            store.save()
            for row in rows:
                if row is clean:continue
                if saved_read(store,'read/'+row['slot_id']) is not None:
                    row.update(status='READ',reason=None)
                elif row['status']=='NOT_RUN_SOURCE_UNREADABLE':pass
                elif result is not None and not exact:
                    row.update(status='NOT_RUN_SOURCE_UNREADABLE',reason='clean correct-time four bytes are not fully exact; fixed row retained')
                elif result is None:
                    row.update(status='BUDGET_EXHAUSTED' if clean['status']=='BUDGET_EXHAUSTED' else 'MISSING',reason='clean correct-time read unavailable')
                else:read_row(row)
                store.save()
    finally:
        if model is not None:backend.close(model)


def report(store):
    evaluated=[]
    for row in store.data['rows']:
        result=dict(row);truth=view_indices(row['view']);truth_plan=method.segment_plan(truth)
        true_missing=[s['segment'] for s in truth_plan['segments'] if not s['cells']]
        result.update(true_missing_segments=true_missing,structural_missing_bits=8*len(true_missing),
            eligible_planned_bits=32-8*len(true_missing),
            nominal_true_cells=[s['observed_cells'] for s in truth_plan['segments']])
        selection=saved_plan(store,'plan/'+row['slot_id'])
        if selection and selection.get('plan'):
            plan=selection['plan'];estimated=plan['estimated_path']
            result.update(path_exact=estimated==truth,absolute_path_error=sum(abs(a-b) for a,b in zip(estimated,truth)),
                blind_claimed_segments=[s['segment'] for s in plan['segments'] if s['cells']],
                falsely_claimed_missing_segments=[s['segment'] for s in plan['segments'] if s['cells'] and s['segment'] in true_missing])
        read=saved_read(store,'read/'+row['slot_id'])
        if read is None and store.started('read/'+row['slot_id']):
            result.update(status='MISSING',reason='saved read evidence unavailable; attempt remains consumed')
        if read:
            decoded=read['decoded_bits'];present=[i for i in range(32) if i//8 not in true_missing]
            scored=[i for i in present if decoded[i] is not None]
            errors=sum(decoded[i]!=BITS[i] for i in scored)
            result.update(decoded_bits=decoded,eligible_planned_bits=len(present),scored_present_bits=len(scored),
                missing_present_bits=len(present)-len(scored),wrong_present_bits=errors,
                bit_error_rate=errors/len(scored) if scored else None,
                false_attribution_bits=sum(decoded[i] is not None for i in range(32) if i//8 in true_missing),
                decoded_segment_hex=[None if any(x is None for x in decoded[g*8:g*8+8]) else f'{int("".join(map(str,decoded[g*8:g*8+8])),2):02X}' for g in range(4)],
                exact_present_segments=sum(all(decoded[8*g+j]==BITS[8*g+j] for j in range(8)) for g in range(4) if g not in true_missing))
            result['matches_other_expected_byte']=[dict(claimed_segment=g,matching_segment=other)
                for g in range(4) for other in range(4) if g!=other and decoded[8*g:8*g+8]==BITS[8*other:8*other+8]]
        evaluated.append(result)
    groups={}
    for label in ('CLEAN','ATTACK_ORACLE',*BLIND):
        rows=[r for r in evaluated if ('CLEAN' if r['view']=='CLEAN' else 'ATTACK_ORACLE' if r['oracle'] else r['receiver'])==label]
        eligible=sum(r['eligible_planned_bits'] for r in rows)
        scored=sum(r.get('scored_present_bits',0) for r in rows)
        errors=sum(r.get('wrong_present_bits',0) for r in rows)
        groups[label]=dict(planned_rows=len(rows),planned_segments=4*len(rows),planned_bits=32*len(rows),
            read_rows=sum(r['status']=='READ' for r in rows),unread_rows=sum(r['status']!='READ' for r in rows),
            status_counts=dict(Counter(r['status'] for r in rows)),structural_missing_bits=sum(r['structural_missing_bits'] for r in rows),
            eligible_planned_bits=eligible,scored_present_bits=scored,unread_present_bits=eligible-scored,
            wrong_present_bits=errors,observed_bit_error_rate=errors/scored if scored else None,
            complete_present_denominator_bit_error_rate=errors/eligible if scored==eligible else None,
            false_attribution_bits=sum(r.get('false_attribution_bits',0) for r in rows))
    summary=dict(status='SEGMENTED_PAYLOAD_USER_RUN_RECORD',planned_rows=36,planned_segments=144,planned_bits=1152,
        blind_rows=24,attack_oracle_rows=8,clean_oracle_rows=4,independent_sources=4,
        blind_structural_missing_bits=192,attack_oracle_structural_missing_bits=64,
        status_counts=dict(Counter(r['status'] for r in evaluated)),clean_readability=store.data['clean_readability'],
        by_receiver=groups,
        physical_attempts={kind:store.used(kind) for kind in LIMITS},physical_limits=LIMITS,
        physical_calls=store.data['counts'],physical_call_limits=CALL_LIMITS,
        exact_input_aliases=sum(max(0,len(p['logical_slots'])-1) for p in store.data['physical_inputs']),
        setup_requirements=setup_requirements(store),
        claim_ceiling='Fixed four-source use demonstration. Missing true segments cannot be recovered; false blind claims are attribution errors. Nominal Wan-time grouping is not a proof of VAE temporal locality. No tuning or scientific PASS.')
    write_json(store.output/'row_results.json',evaluated);write_json(store.output/'summary.json',summary);store.save();return summary


def artifact_graph(store):
    """Actual readable arrays and producer reachability, with no payload answers."""
    graph={}
    for case in cases():
        cid=case['case_id'];nodes={};fw=store.used('framewise_load')<2
        def node(name,shape,dtype,can_produce):
            def readable(path):
                array=np.load(path,allow_pickle=False,mmap_mode='r')
                if array.dtype!=dtype or tuple(array.shape)!=tuple(shape):raise ValueError('saved array geometry or dtype differs')
                if np.issubdtype(dtype,np.floating) and not np.isfinite(array).all():raise ValueError('saved array is nonfinite')
                return True
            available=store.saved(cid+'/'+name,readable) is True
            future=not available and not store.started(cid+'/'+name) and bool(can_produce)
            nodes[name]=dict(available=available,future=future,possible=available or future)
            return available or future
        terminal=node('terminal',(1,16,46,40,64),np.float32,not store.started('load/generation/'+cid) and store.used('generation_load')<4)
        native=node('native',RGB_SHAPE,np.uint8,terminal and store.used('wan_decode_load')<2)
        y=node('writer_y',(181,4,40,64),np.float32,native and fw)
        written=node('written_y',(181,4,40,64),np.float32,y)
        p=node('recon',RGB_SHAPE,np.uint8,y and fw)
        m=node('margin',RGB_SHAPE,np.uint8,written and fw)
        candidate=node('candidate',RGB_SHAPE,np.uint8,native and p and m)
        received=node('received',RGB_SHAPE,np.uint8,candidate)
        node('received_y',(181,4,40,64),np.float32,received and fw)
        graph[cid]=nodes
    return graph


def setup_requirements(store):
    """Only prepare models for reachable unstarted units with readable dependencies."""
    need=dict(wan_full=False,wan_vae=False,framewise=False,ffmpeg=False,torch=False)
    graph=artifact_graph(store)
    for case in cases():
        cid=case['case_id'];nodes=graph[cid]
        need['wan_full']|=nodes['terminal']['future']
        need['wan_vae']|=nodes['native']['future']
        need['framewise']|=any(nodes[name]['future'] for name in ('writer_y','recon','margin','received_y'))
        need['ffmpeg']|=nodes['received']['future']
        received=nodes['received']['possible']
        clean_state=store.data['clean_readability'].get(cid,{}).get('status')
        clean_row=next(r for r in store.data['rows'] if r['case_id']==cid and r['view']=='CLEAN')
        clean_saved=saved_read(store,'read/'+clean_row['slot_id'])
        if clean_saved is not None and clean_saved['decoded_bits']!=BITS:continue
        if clean_saved is None and store.started('read/'+clean_row['slot_id']):continue
        clean_plan=saved_plan(store,'plan/'+clean_row['slot_id'])
        if clean_plan is None and store.started('plan/'+clean_row['slot_id']):continue
        clean_physical=next((p for p in store.data['physical_inputs'] if clean_row.get('physical_id')==p['physical_id']),None)
        if clean_saved is None and clean_physical is not None and store.started(clean_physical['physical_id']):
            shape=(1,16,(clean_physical['output_frames']-1)//4+1,40,64)
            if store.array(clean_physical['physical_id'],shape) is None:continue
        pending_reads=[r for r in store.data['rows'] if r['case_id']==cid and r['status']!='NOT_RUN_SOURCE_UNREADABLE' and not store.started('read/'+r['slot_id'])]
        if clean_state=='UNREADABLE':pending_reads=[]
        for row in pending_reads:
            selection=saved_plan(store,'plan/'+row['slot_id'])
            if selection is None and store.started('plan/'+row['slot_id']):continue
            if selection is not None and selection['status']!='SUPPORTED':continue
            physical=next((p for p in store.data['physical_inputs'] if row.get('physical_id')==p['physical_id']),None)
            if physical is None and selection is not None:
                physical=next((p for p in store.data['physical_inputs'] if p['artifact_id']==cid+'/received' and p['view']==row['view'] and p['received_index_map']==selection['plan']['operation']['received_index_map']),None)
            if physical is not None and store.started(physical['physical_id']):
                shape=(1,16,(physical['output_frames']-1)//4+1,40,64)
                need['torch']|=store.array(physical['physical_id'],shape) is not None
                continue
            need['wan_vae']|=received and store.used('wan_reader_load')<2
    need['torch']|=need['wan_full'] or need['wan_vae'] or need['framewise']
    return need


PHASES=dict(generate=phase_generate,native=phase_native,framewise=phase_framewise,plan=phase_plan,reader=phase_reader)


def run_phase(output,phase,backend=None):
    store=Store(output)
    with store.journal.periodic():
        try:
            if phase in PHASES:PHASES[phase](store,backend)
            elif phase not in ('init','report'):raise ValueError('unknown internal phase')
            store.data['phases'][phase]=dict(status='RETURNED',at=time.time())
        except BaseException as exc:
            store.data['phases'][phase]=dict(status='FAILED',reason=f'{type(exc).__name__}: {exc}',at=time.time());raise
        finally:store.save();report(store)
    return store
