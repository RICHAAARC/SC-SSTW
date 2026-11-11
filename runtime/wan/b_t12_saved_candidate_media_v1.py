"""One saved a -> original causal head -> unchanged media/readers. No solver."""
from pathlib import Path
import copy
import json
import subprocess
import shutil
import time
import numpy as np
import torch
from main.tube_state import b_framewise_qim_v1 as p
from .b_framewise_qim_v1_operator import replay_head,validate_head
from .b_terminal_three_group_backend_v1 import WanBackend
from .vae import quantize_rgb8_no_codec


def array(path,shape,dtype):
    value=np.load(path,mmap_mode='r',allow_pickle=False)
    if value.shape!=shape or value.dtype!=dtype: raise ValueError('invalid array geometry/dtype: '+str(path))
    return value


def command(args,input_bytes=None):
    result=subprocess.run(args,input=input_bytes,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode: raise RuntimeError(f'codec exit {result.returncode}: '+result.stderr.decode(errors='replace')[-4000:])
    return result.stdout


def prediction_file(candidate,output):
    """Join private transport fragments only; ndarray validation follows normally."""
    candidate,output=Path(candidate),Path(output);source=candidate/'r_predicted_packed.npy'
    if source.is_file(): return source
    output.mkdir(parents=True,exist_ok=True);target=output/'r_predicted_packed.npy'
    temporary=target.with_suffix('.npy.partial')
    with temporary.open('wb') as destination:
        for index in range(2):
            with (candidate/f'r_predicted_packed.npy.part{index:03}').open('rb') as fragment:
                shutil.copyfileobj(fragment,destination,1024*1024)
    temporary.replace(target)
    return target


class Backend:
    def __init__(self,model_id,load_receipt):
        self.wan=WanBackend(model_id);self.loaded=False;self.load_receipt=load_receipt

    def load(self):
        if self.loaded: return
        if getattr(self,'load_failure',None): raise RuntimeError('shared model load unavailable: '+self.load_failure)
        token=self.load_receipt(False,None);tick=time.monotonic()
        try:
            self.wan.load_vae();self.loaded=True
            self.load_receipt(True,dict(token=token,seconds=time.monotonic()-tick,status='RETURNED'))
        except BaseException as error:
            self.load_failure=f'{type(error).__name__}: {error}'
            self.load_receipt(True,dict(token=token,seconds=time.monotonic()-tick,status='FAILED',error=str(error)));raise

    def prepare_payload(self):
        """Resolve the shared model dependency before reserving an E41 forward."""
        self.load()

    def native(self,candidate,host,head_path,output,event,dump):
        from .b_joint_time_packet_a_write_cpu_v1 import load_head
        from main.tube_state.b_framewise_qim_v1_solver import budget_metrics
        self.load();candidate,host,output=map(Path,(candidate,host,output));output.mkdir(parents=True,exist_ok=True)
        data=load_head(head_path)
        conv=copy.deepcopy(self.wan.vae.decoder.conv_out)
        conv.load_state_dict({'weight':data['weight'],'bias':data['bias']})
        conv=conv.to(device=self.wan.device,dtype=torch.float32).eval().requires_grad_(False);validate_head(conv)
        channels=conv.in_channels
        a=torch.load(candidate/'coefficients.pt',map_location='cpu',weights_only=True)
        if not isinstance(a,torch.Tensor) or a.dtype!=torch.float64 or tuple(a.shape)!=(181,10,channels,7,9) or not bool(torch.isfinite(a).all()):
            raise ValueError('finite saved FP64 coefficients required')
        predicted=array(prediction_file(candidate,output),(181,10,3,58,74),np.float64)
        base=array(host/'BASE_preclip.npy',(181,320,512,3),np.float32)
        active=(torch.arange(181)[:,None]>=torch.arange(5)[None,:]).repeat(1,2);mask=p.support(active)
        def features():
            for start in (0,*range(1,181,4)):
                h=torch.load(host/'features'/f'{start:03}.pt',map_location='cpu',weights_only=True)
                if not isinstance(h,torch.Tensor) or h.shape!=(1,channels,1 if start==0 else 4,320,512) or h.dtype!=torch.float32 or not bool(torch.isfinite(h).all()):
                    raise ValueError(f'invalid original feature {start:03}')
                yield h.to(self.wan.device)
        result=np.lib.format.open_memmap(output/'preclip.npy',mode='w+',dtype=np.float32,shape=base.shape)
        rows=[];cursor=0
        for start,value in replay_head(conv,features(),a,active,event):
            value=value.detach().cpu();n=len(value)
            original=torch.from_numpy(np.array(base[start:start+n],copy=True));prediction=torch.from_numpy(np.array(predicted[start:start+n],copy=True))
            if start!=cursor or not bool(torch.isfinite(value).all() and torch.isfinite(original).all() and torch.isfinite(prediction).all()):
                raise ValueError('incomplete/nonfinite original native values')
            delta=value.double()-original.double();packed=p.pack(delta)
            metrics=budget_metrics(packed,mask[start:start+n]);full=torch.zeros_like(delta,dtype=torch.bool)
            for tile,(y,x) in enumerate(p.ROIS):
                h=min(58,321-y);full[:,y-1:y-1+h,x-1:x+73]=mask[start:start+n,tile,:,:h].permute(0,2,3,1)
            outside=float(torch.where(full,0,delta).abs().max())
            metrics.update(outside_full_canvas=outside,excess=max(metrics['excess'],outside),
                prediction_max=float((packed-prediction).abs().max()),start=start,length=n)
            for k in range(n):
                rows.append(dict(frame=start+k,rms=metrics['rms'][k],linf=float(delta[k].abs().max()),
                    whole_mse=float(delta[k].square().mean()),outside=float(torch.where(full[k],0,delta[k]).abs().max())))
            dump(output/f'chunk_{start:03}.json',metrics)
            result[start:start+n]=value.numpy();result.flush();cursor+=n
        if cursor!=181: raise ValueError('native181 incomplete')
        del conv,result
        worst=max(max(0,r['rms']-p.RMS,r['linf']-p.LINF,r['outside']) for r in rows)
        return dict(status='FINITE_COMPLETE',frames=rows,actual_budget='PASS' if worst<=1e-7 else 'FAIL',
            maximum_budget_excess=worst,tolerances=dict(linf=p.LINF,rms=p.RMS,rgb=1e-7),
            optimality='UNPROVEN',cpu_budget_failure_retained=True,scientific_pass=None)

    def quantize(self,path):
        pre=array(path,(181,320,512,3),np.float32);out=np.empty(pre.shape,np.uint8)
        for start in range(0,181,4):
            block=torch.from_numpy(np.array(pre[start:start+4],copy=True))
            if not bool(torch.isfinite(block).all()): raise ValueError('nonfinite native RGB')
            out[start:start+4]=quantize_rgb8_no_codec(block).numpy()
        return out

    def encode(self,pixels,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): raise FileExistsError('refusing existing media output: '+str(path))
        if pixels.dtype!=np.uint8 or pixels.ndim!=4 or pixels.shape[1:]!=(320,512,3): raise ValueError('actual uint8 RGB required')
        command(['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-s','512x320','-r','8',
            '-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-n',str(path)],pixels.tobytes())
        return dict(file=str(path),frames=len(pixels))

    def probe(self,path,n):
        from main.tube_state import b_native_y_diagnostic_v1 as y
        value=json.loads(command(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=width,height,r_frame_rate,nb_frames,pix_fmt,codec_name,color_range,color_space,color_transfer,color_primaries,bits_per_raw_sample,bit_rate:format=size,bit_rate','-of','json',str(path)]))
        streams=value.get('streams',[])
        if len(streams)!=1: raise ValueError('one video stream required')
        s=streams[0]
        if (s.get('width'),s.get('height'),s.get('r_frame_rate'),int(s.get('nb_frames',-1)))!=(512,320,'8/1',n) or s.get('codec_name')!='h264' or s.get('pix_fmt')!='yuv420p':
            raise ValueError('original H264 yuv420p geometry required')
        return dict(stream=s,profile=y.profile(s),format=value.get('format',{}))

    def decode(self,path,n,kind):
        if kind=='Y':
            from .b_native_y_diagnostic_v1 import decode_command
            args=decode_command(path);shape=(n,320,512)
        else:
            args=['ffmpeg','-v','error','-threads','1','-i',str(path),'-map','0:v:0','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'];shape=(n,320,512,3)
        raw=command(args)
        if len(raw)!=int(np.prod(shape)): raise ValueError('decoded media byte count mismatch')
        return np.frombuffer(raw,np.uint8).reshape(shape)

    def read(self,y):
        from main.tube_state.b_native_y_diagnostic_v1 import read_y
        return read_y(y)

    def hard(self,raw):
        from main.tube_state.b_joint_time_packet_a_receiver_v1 import diagnostics
        return diagnostics(raw['soft'],raw['mask'],'NEW')

    def relative(self,packet,uid):
        from main.tube_state.b_joint_time_packet_a_relative_v1 import diagnose
        return diagnose(packet,uid)

    def encode_view(self,pixels,start):
        self.load()
        if pixels.dtype!=np.uint8 or pixels.shape[1:]!=(320,512,3) or not 0<=start<=len(pixels)-41:
            raise ValueError('received uint8 E41 geometry required')
        with torch.inference_mode(): return self.wan.encode_view(torch.from_numpy(np.array(pixels[start:start+41],copy=True)))

    def project(self,latent):
        from .b_relative_time_payload_backend_v1 import plain
        return plain(self.wan.payload_project(latent,list(range(1,11)),['R0','R1']))

    def bits(self,projection,uid,key):
        return self.wan.payload_decode(projection,{'key00':'watermark','key01':'watermark-wrong'}[key],uid,key)

    def close(self):
        peak=int(torch.cuda.max_memory_allocated()) if self.wan.device=='cuda' else None
        self.wan.release_vae()
        return dict(device=self.wan.device,peak_cuda_allocated_bytes=peak,resource_hard_caps=None)
