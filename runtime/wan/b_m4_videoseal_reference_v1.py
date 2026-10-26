"""Fixed native VideoSeal external reference. Imports never load models or media."""
from collections import Counter
from pathlib import Path
import errno,json,math,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/b_m4_videoseal_reference_v1.json'
DEFAULT_INPUT='/content/drive/MyDrive/Video-WM/B-Rank3R-New-Source-Confirmation-V1/fixed-four-five-v1'
DEFAULT_OUTPUT='/content/drive/MyDrive/Video-WM/B-M4-VideoSeal-Reference-V1/fixed-four-five-v1'
VENDOR=Path('/content/video-wm-m4/vendor/videoseal')
HOSTS=tuple('b-rank3r-m0'+str(i) for i in range(1,5))
STARTS=(2,11,17,6)
FIXTURES={'CROP':(0,None),'DELETE_LEFT_GAP':(8,49),'DELETE_RIGHT_GAP':(8,102),'REPEAT_LEFT_GAP':(-8,49),'REPEAT_RIGHT_GAP':(-8,102)}
SCOPES=('full',*FIXTURES)
ARMS=('OFF','VideoSeal')
BUDGET=dict(model_load=1,embed=4,quantize=4,codec_encode=24,codec_probe=48,codec_decode=48,edit=20,extract=48,quality=28)
INTERNAL_BUDGET=dict(embedder=8,extractor=248,JND=8)
DENOMINATORS=dict(extract_tasks=48,all_bits=12288,per_arm_bits=6144,per_arm_full_bits=1024,per_arm_fixture_bits=5120,
    quality_records=28,quality_frames=4468,quality_adjacent=4440,new_WM_MP4=24,embed_frames=724,extract_frames=7488)
EMBED_INTERPOLATION=dict(mode='bilinear',align_corners=False,antialias=True)
EXTRACT_INTERPOLATION=dict(mode='bilinear',align_corners=False,antialias=False)


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False);stream.flush()
        try:os.fsync(stream.fileno())
        except OSError as error:
            if error.errno not in (errno.EINVAL,errno.ENOSYS,errno.ENOTSUP):raise
    temp.replace(path)


def read(path):
    def invalid(value):raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=invalid)


def frames_for(host,scope):
    if scope=='full':return list(range(181))
    d,b=FIXTURES[scope];a=STARTS[HOSTS.index(host)]
    return [a+u+(d if b is not None and u>=b else 0) for u in range(151)]


def tasks():
    return [dict(id=f'{h}/{arm}/{scope}',source=h,arm=arm,fixture=scope,frames=181 if scope=='full' else 151,planned_bits=256)
        for h in HOSTS for scope in SCOPES for arm in ARMS]


def qualities():
    return [dict(id=f'{h}/{stage}/{scope}',source=h,stage=stage,fixture=scope,frames=181 if scope=='full' else 151,
        adjacent=180 if scope=='full' else 150) for h in HOSTS for stage,scope in [('precodec','full'),*[('readback',s) for s in SCOPES]]]


def plan_units():
    units={'inputs':'input','model_load':'model_load'}
    for h in HOSTS:
        units[h+'/raw']='raw_input';units[h+'/embed']='embed';units[h+'/quantize']='quantize'
        units['quality/'+h+'/precodec/full']='quality'
        for scope in SCOPES:
            if scope!='full':units[h+'/'+scope+'/edit']='edit'
            units[h+'/VideoSeal/'+scope+'/encode']='codec_encode'
            for arm in ARMS:
                for operation,kind in [('probe','codec_probe'),('decode','codec_decode'),('extract','extract')]:units[f'{h}/{arm}/{scope}/{operation}']=kind
            units['quality/'+h+'/readback/'+scope]='quality'
    units.update(model_release='cleanup',blind_complete='readback',post='evaluation')
    return units


def empty_extract(status='NOT_ENTERED'):
    return dict(status=status,planned_bits=256,native_logits=None,mean_bit_logits=None,hard_bits=None,
        slots=[dict(bit=i,value=None,status=status) for i in range(256)],owner_message_used=False)


def planned_outputs():
    return dict(tasks={t['id']:dict(t,**empty_extract()) for t in tasks()},
        quality={q['id']:dict(q,status='NOT_ENTERED',unread_frames=q['frames'],unread_adjacent=q['adjacent'],frame_rows=[],adjacent_rows=[]) for q in qualities()})


def execution_artifacts(output):
    output=Path(output);names=('attempt.json','plan.json','protocol.json','ledger.json','result.json','evaluation.json','planned_outputs.json',
        'physical_counts.json','physical_events.json','progress.json','blind_complete.json','post_truth_access.json','model.json','release.json','B_reference.json','inputs.json','repeat_start.json','repeat_start_unknown.json')
    found=[n+s for n in names for s in ('','.tmp') if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for name in ('receipts','media','extractions','quality'):
        p=output/name
        if p.is_dir():found.extend(str(v.relative_to(output)) for v in p.rglob('*') if v.is_file() or v.is_symlink())
    return sorted(found)


def recover_ledger(output):
    output=Path(output);units=plan_units();ordered=list(units)
    ledger={slot:dict(slot=slot,kind=kind,status='NOT_ENTERED',attempts=0) for slot,kind in units.items()};errors={}
    def valid(row,slot,receipt=False):
        if not isinstance(row,dict) or row.get('slot')!=slot or row.get('kind')!=units[slot]:return False
        status=row.get('status');attempt=row.get('attempts')
        return type(attempt) is int and ((not receipt and status=='NOT_ENTERED' and attempt==0) or
            (status in ('STARTED','RETURNED','FAILED','INTERRUPTED') and attempt==1))
    try:
        saved=read(output/'ledger.json')
        if not isinstance(saved,dict):raise ValueError('ledger root must be an object')
        for slot,row in saved.items():
            if slot not in units:errors[slot]='unknown saved ledger slot';continue
            if valid(row,slot):ledger[slot]=row
            else:ledger[slot].update(status='UNRESOLVED_LEDGER',attempts=1);errors[slot]='invalid saved ledger status/attempt/address'
    except Exception as error:errors['ledger.json']=str(error)
    folder=output/'receipts'
    for path in sorted(folder.iterdir()) if folder.is_dir() else []:
        if not (path.name.endswith('.json') or path.name.endswith('.json.tmp')):continue
        name=path.name.removesuffix('.tmp').removesuffix('.json');key=str(path.relative_to(output))
        if not name.isdigit() or not 0<=int(name)<len(ordered):errors[key]='unknown numeric receipt address';continue
        slot=ordered[int(name)]
        try:
            row=read(path)
            if not valid(row,slot,receipt=True):raise ValueError('invalid receipt status/attempt/address')
            if path.name.endswith('.tmp'):raise ValueError('pending atomic receipt; attempt consumption unresolved')
            ledger[slot]=row
        except Exception as error:
            errors[key]=str(error);ledger[slot].update(status='UNRESOLVED_RECEIPT',attempts=1,error=str(error))
    return dict(ledger=ledger,counts=dict(Counter(r['status'] for r in ledger.values())),read_errors=errors,physical_reentry=False)


def existing(output):
    recovery=recover_ledger(output);saved={};errors={}
    for name in ('result.json','evaluation.json','planned_outputs.json'):
        try:saved[name]=read(Path(output)/name)
        except Exception as error:errors[name]=str(error)
    return dict(status='READ_ONLY_EXISTING_ATTEMPT',saved=saved,recovery=recovery,read_errors=errors,physical_calls=0)


def load_config(path=CONFIG):
    cfg=read(path);messages=cfg['messages'];rows=cfg['tasks'];maps=cfg['maps']
    if [h['id'] for h in cfg['sources']]!=list(HOSTS) or [h['a'] for h in cfg['sources']]!=list(STARTS):raise ValueError('fixed M3 source roster/start geometry required')
    if len(messages)!=4 or {m['source'] for m in messages}!=set(HOSTS):raise ValueError('four source message records required')
    bits=messages[0]['canonical_bits']
    if len(bits)!=256 or any(type(v) is not int or v not in (0,1) for v in bits) or any(m['canonical_bits']!=bits or m['message_id']!='owner256' for m in messages):raise ValueError('one shared native256 owner message required')
    if [t['id'] for t in rows]!=[t['id'] for t in tasks()]:raise ValueError('fixed48 extraction roster required')
    if len(maps)!=24 or any(m['source_frames']!=frames_for(m['source'],m['fixture']) for m in maps):raise ValueError('fixed original source-frame maps required')
    model=cfg['model']
    fixed=dict(card='videoseal_1.0',nbits=256,scaling_w=.2,scaling_i=1.,attenuation='jnd_1_1',chunk_size=32,step_size=4,
        lowres_attenuation=False,model_dtype='float32',inference_mode=True,model_eval=True,autocast=False)
    if any(model.get(k)!=v or (isinstance(v,bool) and type(model.get(k)) is not bool) for k,v in fixed.items()):raise ValueError('fixed native VideoSeal method fields required')
    defaults=model.get('effective_official_defaults',{})
    if defaults.get('video_mode')!='repeat' or defaults.get('clamp') is not True or model.get('embed_interpolation')!=EMBED_INTERPOLATION or model.get('extract_interpolation')!=EXTRACT_INTERPOLATION:raise ValueError('fixed repeat/clamp/interpolation semantics required')
    return cfg


def native_prediction(logits,frames):
    """Official avg, on the native float32 device, before CPU/list conversion."""
    import torch
    if not isinstance(logits,torch.Tensor) or logits.dtype!=torch.float32 or logits.ndim<2 or tuple(logits.shape[:2])!=(frames,257) or any(n!=1 for n in logits.shape[2:]):raise ValueError('native float32 F x257 with only optional singleton trailing axes required')
    if not bool(torch.isfinite(logits).all()):raise ValueError('nonfinite native logits')
    means=logits[:,1:].mean(dim=0)
    if not bool(torch.isfinite(means).all()):raise ValueError('nonfinite native float32 frame mean')
    means=means.reshape(256);bits=(means>0).to(torch.int64).cpu().tolist()
    return dict(status='RETURNED',planned_bits=256,native_shape=list(logits.shape),native_dtype='float32',
        native_logits=logits.detach().cpu().tolist(),mask_column_used_for_message=False,
        mean_bit_logits=means.detach().cpu().tolist(),hard_bits=bits,slots=[dict(bit=i,value=b,status='READ') for i,b in enumerate(bits)],
        aggregation='native float32 frame mean then >0',owner_message_used=False,extra_detector_calls=0)


def quantize(rgb):
    import numpy as np,torch
    if not isinstance(rgb,torch.Tensor) or rgb.dtype!=torch.float32 or tuple(rgb.shape)!=(181,3,320,512) or not bool(torch.isfinite(rgb).all()):raise ValueError('finite unchanged FCHW float32 embed output required')
    if float(rgb.min())<0 or float(rgb.max())>1:raise ValueError('official clamped output outside[0,1]')
    x=rgb.detach().cpu().permute(0,2,3,1).contiguous().numpy()
    return np.rint(np.clip(x,0,1)*255).astype(np.uint8)


def quality_pair(left,right):
    import numpy as np
    if left.dtype!=np.uint8 or right.dtype!=np.uint8 or left.shape!=right.shape or left.ndim!=4 or left.shape[-1]!=3:raise ValueError('paired same-shape uint8 RGB required')
    frames=[];adj=[];previous=None;total=0.;adj_total=0.;size=int(np.prod(left.shape[1:]))
    def metric(square,count):
        rms=math.sqrt(square/count)
        return dict(rms=rms,psnr=20*math.log10(1/rms) if rms else None,psnr_status='FINITE' if rms else 'POSITIVE_INFINITY')
    for i,(a,b) in enumerate(zip(left,right)):
        e=(b.astype(np.float64)-a.astype(np.float64))/255.;s=float(np.square(e).sum());total+=s
        frames.append(dict(frame=i,**metric(s,size)))
        if previous is not None:
            d=float(np.square(e-previous).sum());adj_total+=d;adj.append(dict(before=i-1,after=i,**metric(d,size)))
        previous=e
    return dict(status='RETURNED',frames=len(left),adjacent=max(0,len(left)-1),unread_frames=0,unread_adjacent=0,frame_rows=frames,adjacent_rows=adj,
        whole=metric(total,len(left)*size),temporal_residual_rms=math.sqrt(adj_total/(max(1,len(adj))*size)) if adj else None,
        quality_pass=None,ROI=None)


class VideoSealBackend:
    def __init__(self):self.model=None;self.handles=[];self.event=lambda *a:None;self.previous_cwd=None
    def load(self,config,event):
        import torch
        self.event=event
        if not (VENDOR/'videoseal/utils/cfg.py').is_file():raise FileNotFoundError('available official VideoSeal source required in '+str(VENDOR))
        self.previous_cwd=Path.cwd();os.chdir(VENDOR);sys.path.insert(0,str(VENDOR))
        import videoseal
        self.model=videoseal.load('videoseal_1.0').float().eval().to('cuda' if torch.cuda.is_available() else 'cpu')
        actual=dict(dtype=str(next(self.model.parameters()).dtype),floating_dtypes=sorted({str(p.dtype) for p in (*self.model.parameters(),*self.model.buffers()) if p.is_floating_point()}),
            device=str(self.model.device),cwd=str(Path.cwd()),scaling_w=getattr(getattr(self.model,'blender',None),'scaling_w',None),
            scaling_i=getattr(getattr(self.model,'blender',None),'scaling_i',None),chunk_size=getattr(self.model,'chunk_size',None),step_size=getattr(self.model,'step_size',None),
            video_mode=getattr(self.model,'video_mode',None),clamp=getattr(self.model,'clamp',None),model_eval=not self.model.training,
            attenuation_present=getattr(self.model,'attenuation',None) is not None,attenuation_config='jnd_1_1')
        self.load_observation=dict(status='OBSERVED_FIXED_METHOD',card='videoseal_1.0',**actual)
        expected=dict(dtype='torch.float32',floating_dtypes=['torch.float32'],scaling_w=.2,scaling_i=1.,chunk_size=32,step_size=4,video_mode='repeat',clamp=True,model_eval=True,attenuation_present=True)
        mismatches={key:dict(expected=value,actual=actual[key]) for key,value in expected.items() if actual[key]!=value}
        if mismatches:
            self.load_observation.update(status='FIXED_METHOD_UNRESOLVED',mismatches=mismatches)
            raise ValueError('loaded VideoSeal method parameters differ: '+repr(mismatches))
        def hooks(module,name):
            self.handles.append(module.register_forward_pre_hook(lambda m,args:event(name,False,int(args[0].shape[0]))))
            self.handles.append(module.register_forward_hook(lambda m,args,value:event(name,True,int(args[0].shape[0]))))
        hooks(self.model.embedder,'embedder');hooks(self.model.detector,'extractor')
        attenuation=self.model.attenuation
        self.original_heatmaps=attenuation.heatmaps
        def measured_heatmaps(*args,**kwargs):
            count=int(args[0].shape[0]);event('JND',False,count);value=self.original_heatmaps(*args,**kwargs);event('JND',True,count);return value
        attenuation.heatmaps=measured_heatmaps
        return dict(self.load_observation,status='LOADED',lowres_attenuation=False,model_execution_not_part_of_implementation_checks=True)
    @staticmethod
    def pixels(rgb,frames):
        import numpy as np
        if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or tuple(rgb.shape)!=(frames,320,512,3):raise ValueError('actual fixed uint8 THWC RGB required')
        return rgb
    def input_info(self,source,cfg):
        from runtime.wan.b_gamma2_fixed_validation_v1 import read_public_protocol
        public=read_public_protocol(source/'protocol.json')['config'];hosts={h['id']:h['a'] for h in public['sources']}
        if any(hosts.get(h)!=a for h,a in zip(HOSTS,STARTS)):raise ValueError('OFF source starts disagree with fixed fixture maps')
        return dict(status='INPUT_GEOMETRY_OBSERVED',sources=hosts,original_protocol_payload_read=False,OFF_reencode=False)
    def raw(self,source,host):
        import numpy as np
        folder=source/'m4_inputs'/host/'OFF';sidecar=read(folder/'precodec.json');path=folder/'precodec.rgb24'
        if sidecar.get('status')!='ARCHIVED' or sidecar.get('dtype')!='uint8' or sidecar.get('shape')!=[181,320,512,3] or sidecar.get('fps')!=8:raise ValueError('actual OFF RGB24 sidecar invalid')
        if path.stat().st_size!=88965120:raise ValueError('OFF RGB24 incomplete; encoded media cannot replace it')
        return self.pixels(np.fromfile(path,dtype=np.uint8).reshape(181,320,512,3),181)
    def tensor(self,rgb):
        import torch
        self.pixels(rgb,len(rgb));return torch.from_numpy(rgb.copy()).permute(0,3,1,2).to(device=self.model.device,dtype=torch.float32)/255.
    def embed(self,rgb,bits):
        import torch
        x=self.tensor(rgb);message=torch.tensor([bits],device=self.model.device,dtype=torch.float32)
        with torch.inference_mode():return self.model.embed(x,msgs=message,is_video=True,lowres_attenuation=False,interpolation=dict(EMBED_INTERPOLATION))['imgs_w']
    def quantize(self,rgb):return quantize(rgb)
    def encode(self,rgb,path):
        import torch
        from runtime.wan.io import encode_rgb8
        self.pixels(rgb,len(rgb));encode_rgb8(torch.from_numpy(rgb),path,8,18)
        return dict(path=str(path),bytes=path.stat().st_size,codec='libx264',crf=18,fps=8,pixel_format='yuv420p',input='prequantized uint8; no second quantization')
    def probe(self,path,frames):
        result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames','-of','json',str(path)],check=True,capture_output=True,text=True)
        value=json.loads(result.stdout)['streams'][0]
        if value.get('codec_name')!='h264' or value.get('pix_fmt')!='yuv420p' or (value.get('width'),value.get('height'))!=(512,320) or value.get('r_frame_rate')!='8/1' or int(value.get('nb_frames',-1))!=frames:raise ValueError('actual fixed H264 yuv420p dimensions/fps/frame count required')
        return value
    def decode(self,path,probe,frames):
        import numpy as np
        result=subprocess.run(['ffmpeg','-v','error','-threads','1','-noautorotate','-i',str(path),'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-'],check=True,capture_output=True)
        if len(result.stdout)!=frames*320*512*3:raise ValueError('decoded RGB24 length mismatch; no truncation or padding')
        return self.pixels(np.frombuffer(result.stdout,dtype=np.uint8).reshape(frames,320,512,3).copy(),frames)
    def edit(self,rgb,indices):return self.pixels(rgb,181)[indices].copy()
    def extract(self,rgb):
        import torch
        x=self.tensor(rgb)
        with torch.inference_mode():logits=self.model.detect(x,is_video=True,interpolation=dict(EXTRACT_INTERPOLATION))['preds']
        return native_prediction(logits,len(rgb))
    def quality(self,left,right):return quality_pair(left,right)
    def release(self):
        for handle in self.handles:handle.remove()
        if self.model is not None and hasattr(self,'original_heatmaps'):self.model.attenuation.heatmaps=self.original_heatmaps
        self.handles=[];self.model=None
        if self.previous_cwd is not None:os.chdir(self.previous_cwd)
        import gc;gc.collect()
        return dict(status='RELEASED')


class Journal:
    def __init__(self,out):
        self.out=Path(out);self.units=plan_units();self.data={s:dict(slot=s,kind=k,status='NOT_ENTERED',attempts=0) for s,k in self.units.items()}
        self.counts={k:dict(planned=n,started=0,returned=0,failed=0) for k,n in BUDGET.items()}
        self.internal={k:dict(planned=n,started=0,returned=0,frames_started=0,frames_returned=0) for k,n in INTERNAL_BUDGET.items()};self.save()
    def save(self):
        dump(self.out/'ledger.json',self.data);dump(self.out/'physical_counts.json',self.counts);dump(self.out/'physical_events.json',self.internal)
    def event(self,name,returned,frames):
        row=self.internal[name];field='returned' if returned else 'started'
        if not returned and row['started']>=row['planned']:raise RuntimeError('fixed internal forward budget spent: '+name)
        row[field]+=1;row['frames_'+field]+=frames;dump(self.out/'physical_events.json',self.internal)
    def call(self,slot,function):
        row=self.data[slot];kind=row['kind']
        if row['attempts']:raise RuntimeError('fixed unit already spent: '+slot)
        row.update(status='STARTED',attempts=1);receipt=self.out/'receipts'/('%03d.json'%list(self.units).index(slot));dump(receipt,row)
        if kind in self.counts:self.counts[kind]['started']+=1
        self.save();dump(self.out/'progress.json',dict(active=slot,counts=dict(Counter(v['status'] for v in self.data.values()))))
        try:value=function()
        except BaseException as error:
            row.update(status='FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}')
            if kind in self.counts:self.counts[kind]['failed']+=1
            dump(receipt,row);self.save();raise
        row['status']='RETURNED'
        if kind in self.counts:self.counts[kind]['returned']+=1
        dump(receipt,row);self.save();return value
    def record(self,slot,function,path):
        def produce():
            value=function();dump(path,value);return value
        return self.call(slot,produce)


def _start_reserved(input_directory,output,**kwargs):
    artifacts=execution_artifacts(output)
    if any(a not in ('repeat_start.json',) for a in artifacts):return existing(output)
    return _execute_once(input_directory,output,**kwargs)


def run(input_directory=DEFAULT_INPUT,output=DEFAULT_OUTPUT,**kwargs):
    if execution_artifacts(output):return existing(output)
    return _execute_once(input_directory,output,**kwargs)


def _execute_once(input_directory,output,*,config_path=CONFIG,backend=None):
    out=Path(output);source=Path(input_directory);out.mkdir(parents=True,exist_ok=True)
    with (out/'attempt.json').open('x') as stream:json.dump(dict(status='STARTED',attempts=1,started_unix=time.time(),budget=BUDGET),stream);stream.flush()
    outputs=planned_outputs();dump(out/'plan.json',dict(units=plan_units(),tasks=tasks(),quality=qualities(),denominators=DENOMINATORS,budget=BUDGET,internal_budget=INTERNAL_BUDGET))
    dump(out/'planned_outputs.json',outputs);journal=Journal(out);stop=None;cfg=None
    for task in tasks():dump(out/'extractions'/(task['id']+'.json'),empty_extract())
    for item in qualities():dump(out/'quality'/(item['id']+'.json'),outputs['quality'][item['id']])
    backend=VideoSealBackend() if backend is None else backend
    try:
        cfg=load_config(config_path);dump(out/'protocol.json',dict(stage='M4_FIXED_VIDEOSEAL_REFERENCE',config=cfg,owner_message_use='writer and post only'))
        journal.record('inputs',lambda:backend.input_info(source,cfg),out/'inputs.json')
        def load_model():
            try:value=backend.load(cfg,journal.event)
            except BaseException as error:
                observation=getattr(backend,'load_observation',{})
                # Actual invalid parameters are evidence, not invalid JSON that hides the original error.
                def safe(value):
                    if isinstance(value,float) and not math.isfinite(value):return repr(value)
                    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
                    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
                    if value is None or isinstance(value,(str,int,float,bool)):return value
                    return repr(value)
                dump(out/'model.json',dict(status='MODEL_LOAD_UNRESOLVED',error=f'{type(error).__name__}: {error}',observed=safe(observation)))
                raise
            dump(out/'model.json',value);return value
        journal.call('model_load',load_model)
        for host in HOSTS:
            off_raw=journal.call(host+'/raw',lambda:backend.raw(source,host))
            offpath=source/'media'/host/'OFF'/'full'/'video.mp4'
            offprobe=journal.record(host+'/OFF/full/probe',lambda:backend.probe(offpath,181),out/'media'/host/'OFF'/'full'/'probe.json')
            off_full=journal.call(host+'/OFF/full/decode',lambda:backend.decode(offpath,offprobe,181))
            writer_bits=next(m['canonical_bits'] for m in cfg['messages'] if m['source']==host)
            watermarked=journal.call(host+'/embed',lambda:backend.embed(off_raw,writer_bits))
            wm_raw=journal.call(host+'/quantize',lambda:backend.quantize(watermarked));del watermarked
            qid=host+'/precodec/full';outputs['quality'][qid]=journal.record('quality/'+qid,lambda:backend.quality(off_raw,wm_raw),out/'quality'/(qid+'.json'));del off_raw
            wm_full=None
            for scope in SCOPES:
                length=181 if scope=='full' else 151
                wm_input=wm_raw if scope=='full' else journal.call(host+'/'+scope+'/edit',lambda:backend.edit(wm_full,frames_for(host,scope)))
                wmpath=out/'media'/host/'VideoSeal'/scope/'video.mp4'
                journal.record(host+'/VideoSeal/'+scope+'/encode',lambda:backend.encode(wm_input,wmpath),wmpath.parent/'codec.json')
                if scope!='full':
                    offpath=source/'media'/host/'OFF'/scope/'video.mp4'
                    offprobe=journal.record(host+'/OFF/'+scope+'/probe',lambda:backend.probe(offpath,length),out/'media'/host/'OFF'/scope/'probe.json')
                    off=journal.call(host+'/OFF/'+scope+'/decode',lambda:backend.decode(offpath,offprobe,length))
                else:off=off_full
                wmprobe=journal.record(host+'/VideoSeal/'+scope+'/probe',lambda:backend.probe(wmpath,length),wmpath.parent/'probe.json')
                wm=journal.call(host+'/VideoSeal/'+scope+'/decode',lambda:backend.decode(wmpath,wmprobe,length))
                if scope=='full':wm_full=wm
                for arm,rgb in [('OFF',off),('VideoSeal',wm)]:
                    task=host+'/'+arm+'/'+scope
                    record=journal.record(task+'/extract',lambda rgb=rgb:backend.extract(rgb),out/'extractions'/(task+'.json'))
                    outputs['tasks'][task].update(record)
                qid=host+'/readback/'+scope;outputs['quality'][qid]=journal.record('quality/'+qid,lambda:backend.quality(off,wm),out/'quality'/(qid+'.json'))
                del wm_input,wm,off
            del wm_full,wm_raw,off_full
    except BaseException as error:stop=dict(type=type(error).__name__,message=str(error),interrupted=not isinstance(error,Exception),retry=False)
    finally:
        try:journal.record('model_release',backend.release,out/'release.json')
        except BaseException as error:
            if stop is None:stop=dict(type=type(error).__name__,message=str(error),retry=False)
        for task in tasks():
            slot=task['id']+'/extract';state=journal.data[slot]['status']
            if state!='RETURNED':outputs['tasks'][task['id']].update(empty_extract(state));dump(out/'extractions'/(task['id']+'.json'),empty_extract(state))
        for item in qualities():
            slot='quality/'+item['id'];state=journal.data[slot]['status']
            if state!='RETURNED':outputs['quality'][item['id']]['status']=state;dump(out/'quality'/(item['id']+'.json'),outputs['quality'][item['id']])
        dump(out/'planned_outputs.json',outputs)
    try:
        def extraction_readback():
            rows=[read(out/'extractions'/(t['id']+'.json')) for t in tasks()]
            if any(len(r.get('slots',[]))!=256 for r in rows):raise ValueError('fixed256 extraction slots absent')
            return dict(status='READ_BACK',tasks=48,all_bits=12288,statuses=dict(Counter(r['status'] for r in rows)),message_accuracy_not_evaluated=True)
        journal.record('blind_complete',extraction_readback,out/'blind_complete.json')
        journal.record('post',lambda:evaluate(out,source),out/'evaluation.json')
    except BaseException as error:
        if stop is None:stop=dict(type=type(error).__name__,message=str(error),retry=False)
    complete=stop is None and all(v['status']=='RETURNED' for v in journal.data.values()) and all(v['started']==v['returned']==v['planned'] for v in journal.internal.values())
    result=dict(status='COMPLETED_FIXED_ATTEMPT' if complete else 'TECHNICAL_UNRESOLVED',stop=stop,denominators=DENOMINATORS,
        physical_counts=journal.counts,observed_internal=journal.internal,ledger_counts=dict(Counter(v['status'] for v in journal.data.values())),output=str(out),performance_is_not_stop_trigger=True)
    dump(out/'result.json',result);return result


def B_reference(source):
    """Validate saved result structure only; no B receiver, bits or metric recomputation."""
    observed={}
    try:
        old=read(source/'evaluation.json')
        if not isinstance(old,dict):raise ValueError('B evaluation must be an object')
        observed={k:old.get(k) for k in ('status','method_status','contribution_status')}
        if old.get('status')!='POST_RECORDED' or any(not isinstance(old.get(k),str) or not old[k] for k in ('method_status','contribution_status')):raise ValueError('B post/status fields absent or invalid')
        readers=('CDC_M','RANK3','RANK3R');rows=old.get('tasks',{})
        roster={f'{h}/{arm}/{fixture}/{key}' for h in HOSTS for arm in ('OFF','P0','PM') for fixture in FIXTURES for key in ('key00','key01')}
        if not isinstance(rows,dict) or set(rows)!=roster:raise ValueError('B fixed four-source120 task roster missing or invalid')
        required_counts=('planned','emitted','literal_errors','actual_supported_correct','false_attribution','null_outputs','technical_unread','conflict_slots')
        sums={f'{reader}/{kind}':Counter() for reader in readers for kind in ('positive','null','wrong_key_PM')}
        detected={key:Counter() for key in sums}
        for name,row in rows.items():
            host,arm,fixture,key=name.split('/');positive=arm=='PM' and key=='key00'
            if not isinstance(row,dict) or row.get('positive') is not positive or set(row.get('readers',{}))!=set(readers):raise ValueError('B task positive/reader schema invalid: '+name)
            for reader in readers:
                judged=row['readers'][reader];slots=judged.get('slots',[]);counts=judged.get('counts',{});status=judged.get('detection_status')
                if not isinstance(slots,list) or len(slots)!=32 or [v.get('slot') for v in slots]!=list(range(32)):raise ValueError('B fixed32 slot addresses missing: '+name+'/'+reader)
                if not isinstance(counts,dict) or counts.get('planned')!=32 or any(type(counts.get(k)) is not int or not 0<=counts[k]<=32 for k in required_counts):raise ValueError('B saved counts absent or invalid: '+name+'/'+reader)
                if counts['literal_errors']>counts['emitted'] or counts['actual_supported_correct']>counts['emitted'] or counts['emitted']+counts['technical_unread']>32:raise ValueError('B saved counts contradict their denominators')
                if not isinstance(status,str) or not status:raise ValueError('B detection status absent')
                kinds=['positive' if positive else 'null']+(['wrong_key_PM'] if arm=='PM' and key=='key01' else [])
                for kind in kinds:
                    address=reader+'/'+kind;sums[address].update({k:counts[k] for k in required_counts});detected[address][status]+=1
        totals=old.get('totals',{});detection=old.get('detection',{});denom=old.get('denominators',{})
        expected=dict(sources=4,readers=3,tasks_per_reader=120,slots_per_reader=3840,positive_tasks=20,positive_slots=640,null_tasks=100,null_slots=3200)
        if any(denom.get(k)!=v for k,v in expected.items()):raise ValueError('B saved fixed denominators missing or invalid')
        for name,counts in sums.items():
            if any(totals.get(name,{}).get(k)!=v for k,v in counts.items()):raise ValueError('B totals disagree with saved task counts: '+name)
            expected_tasks=100 if name.endswith('/null') else 20
            if detection.get(name,{}).get('planned')!=expected_tasks or detection[name].get('statuses')!=dict(detected[name]):raise ValueError('B detection totals disagree with saved task statuses: '+name)
        quality=old.get('quality',{});full={}
        for host in HOSTS:
            name=host+'/quality/full/OFF_to_PM/readback';value=quality.get(name,{})
            if value.get('status')!='SCORED' or value.get('frame_denominator')!=181 or value.get('adjacent_denominator')!=180 or len(value.get('frames',[]))!=181 or len(value.get('adjacent',[]))!=180:raise ValueError('B saved full/readback quality incomplete: '+host)
            for field in ('whole','adjacent_residual'):
                rms=value.get(field,{}).get('rms')
                if type(rms) not in (int,float) or not math.isfinite(rms) or rms<0:raise ValueError('B saved quality RMS invalid: '+host+'/'+field)
            full[name]=value
        summary={k:old.get(k) for k in ('status','method_status','contribution_status','checks','totals','detection','per_source','subsets','paired_summary','net_actual_supported_gain_vs_CDC_M','denominators')}
        return dict(status='SAVED_ORIGINAL_M3',summary=summary,quality_full_OFF_to_PM_readback=full,
            structure_observed=dict(tasks_per_reader=120,readers=3,slots_per_reader=3840,positive_slots=640,null_slots=3200,full_quality=4),
            fixture_OFF_to_PM='N/A: not saved under same definition',precodec_OFF_to_PM='N/A: not saved under same definition',
            full_blind_payload='N/A',payload_bits=32,readers=list(readers),actual_method_and_contribution_preserved=True,rerun=False)
    except Exception as error:return dict(status='B_REFERENCE_UNRESOLVED',error=str(error),observed_status_fields=observed,rerun=False)


def evaluate(out,source):
    marker=read(out/'blind_complete.json')
    if marker.get('status')!='READ_BACK' or marker.get('tasks')!=48:raise ValueError('all48 observed/explicitly unresolved extractions must be persisted first')
    dump(out/'post_truth_access.json',dict(opened_unix=time.time(),blind_tasks=48,owner_message_access='post only; embed writer was separate'))
    cfg=read(out/'protocol.json')['config'];truth=cfg['messages'][0]['canonical_bits'];rows={}
    for task in tasks():
        observed=read(out/'extractions'/(task['id']+'.json'));bits=observed.get('hard_bits')
        available=observed['status']=='RETURNED' and isinstance(bits,list) and len(bits)==256 and all(type(v) is int and v in (0,1) for v in bits)
        slots=[dict(bit=i,value=bits[i] if available else None,correct=bits[i]==truth[i] if available else None,status='READ' if available else observed['status']) for i in range(256)]
        correct=sum(r['correct'] is True for r in slots);count=256 if available else 0;errors=count-correct
        rows[task['id']]=dict(task,status=observed['status'],available=count,correct=correct,errors=errors,unread=256-count,slots=slots,
            bit_accuracy=correct/count if count else None,BER=errors/count if count else None,BER_defined=bool(count),correct_over_fixed=correct/256,
            exact_match=correct==256 if available else None,path='N/A',exact_support='N/A',wrong_key='N/A',DETECT='N/A')
    summary={}
    for arm in ARMS:
        for scope in ('all','full','fixture'):
            values=[v for v in rows.values() if v['arm']==arm and (scope=='all' or (v['fixture']=='full')==(scope=='full'))]
            counts={k:sum(v[k] for v in values) for k in ('planned_bits','available','correct','errors','unread')};n=counts['available']
            summary[arm+'/'+scope]=dict(counts,tasks=len(values),bit_accuracy=counts['correct']/n if n else None,BER=counts['errors']/n if n else None,
                correct_over_fixed=counts['correct']/counts['planned_bits'],exact_matches=sum(v['exact_match'] is True for v in values),technical_tasks=sum(v['available']<256 for v in values))
    quality={q['id']:read(out/'quality'/(q['id']+'.json')) for q in qualities()};reference=B_reference(source);dump(out/'B_reference.json',reference)
    complete=all(v['available']==256 for v in rows.values()) and all(v['status']=='RETURNED' for v in quality.values())
    return dict(status='POST_RECORDED',reference_status='COMPLETE_EXTERNAL_REFERENCE' if complete and reference['status']=='SAVED_ORIGINAL_M3' else 'TECHNICAL_UNRESOLVED',
        tasks=rows,summary=summary,quality=quality,B_reference=reference,denominators=DENOMINATORS,
        OFF_interpretation='unembedded owner-message matching observation; not false detection',payload_bits=256,
        owner_message_scope='one shared owner256; cross-source bit accuracies are correlated in message dimension',quality_pass=None,population_FPR=None,
        method_ranking=None,original_M3_status_fields_preserved=(reference['status']=='SAVED_ORIGINAL_M3'),capacity_comparison='native B32 versus VideoSeal256; no prefix or replicated-message equalization')


def compact(result,output):
    base=result.get('saved',{}).get('result.json',result)
    try:ev=read(Path(output)/'evaluation.json')
    except Exception:ev={}
    return dict(status=result['status'],saved_status=base.get('status'),reference_status=ev.get('reference_status'),denominators=DENOMINATORS,
        physical_counts=base.get('physical_counts'),observed_internal=base.get('observed_internal'),summary=ev.get('summary'),stop=base.get('stop'),output=str(output))
