"""User-run physical adapter. Import does not load models or execute media."""
from pathlib import Path
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
from main.tube_state import b_local_vae_code_v1 as code
from main.tube_state import b_local_vae_head_v1 as writer


def finite(value,name):
    if not bool(torch.isfinite(value).all()):raise FloatingPointError('nonfinite '+name)
    return value


class Backend:
    def __init__(self,cfg,scratch,event):
        self.cfg=cfg;self.scratch=Path(scratch).expanduser().resolve();self.scratch.mkdir(parents=True,exist_ok=True)
        self.event=event;self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.wan=None;self.video=None;self.original_cwd=None;self.head=None
    def call(self,phase,kind,fn,frames=0):
        self.event(phase,kind,False,frames)
        value=fn()
        self.event(phase,kind,True,frames)
        return value
    def load_vae(self):
        from runtime.wan.b_rank3_four_source_confirmation_backend_v1 import WanBackend
        self.wan=WanBackend(self.cfg['model_id']);receipt=self.wan.load_vae()
        self.wan.vae.eval().requires_grad_(False)
        return receipt
    def release_vae(self):
        if self.wan is not None:self.wan.release_vae();self.wan=None
        gc.collect()
        if self.device=='cuda':torch.cuda.empty_cache()
    def load_video(self):
        vendor=self.scratch/'vendor/videoseal'
        if not (vendor/'videoseal/utils/cfg.py').is_file():raise FileNotFoundError('VideoSeal source unavailable: '+str(vendor))
        self.original_cwd=Path.cwd();os.chdir(vendor);sys.path.insert(0,str(vendor))
        import videoseal
        self.video=videoseal.load('videoseal_1.0').float().eval().requires_grad_(False).to(self.device)
        return dict(card='videoseal_1.0',device=self.device,dtype='float32',teacher_image_mode=True,
            extractor_input_gradient_preserved=True,actual_img_size=int(self.video.img_size))
    def release_video(self):
        self.video=None
        if self.original_cwd is not None:os.chdir(self.original_cwd);self.original_cwd=None
        gc.collect()
        if self.device=='cuda':torch.cuda.empty_cache()
    def probe(self,path,length,phase):
        def execute():
            p=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                'stream=width,height,nb_frames,r_frame_rate,codec_name,pix_fmt','-of','json',str(path)],capture_output=True,text=True,check=True)
            s=json.loads(p.stdout)['streams'][0]
            if (int(s['width']),int(s['height']),int(s.get('nb_frames',-1)),s['r_frame_rate'],s['codec_name'],s['pix_fmt'])!=(512,320,length,'8/1','h264','yuv420p'):
                raise ValueError('actual H264 stream shape/fps/length mismatch: '+str(s))
            return s
        return self.call(phase,'codec_probe',execute,length)
    def decode_media(self,path,length,phase):
        def execute():
            p=subprocess.run(['ffmpeg','-v','error','-threads','1','-noautorotate','-i',str(path),'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True,check=True)
            if len(p.stdout)!=length*320*512*3:raise ValueError('incomplete decoded RGB24')
            return torch.from_numpy(np.frombuffer(p.stdout,dtype=np.uint8).copy().reshape(length,320,512,3))
        return self.call(phase,'codec_decode',execute,length)
    def encode_media(self,pixels,path,phase):
        from runtime.wan.io import encode_rgb8
        if pixels.dtype!=torch.uint8 or tuple(pixels.shape[1:])!=(320,512,3):raise ValueError('same-chain RGB8 required')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        return self.call(phase,'codec_encode',lambda:encode_rgb8(pixels,path,fps=8,crf=18),len(pixels))
    def quantize(self,canvas):
        from runtime.wan.vae import quantize_rgb8_no_codec
        return quantize_rgb8_no_codec(canvas)
    def roundtrip(self,canvas,path,phase,probe=True):
        pixels=self.quantize(canvas)
        self.encode_media(pixels,path,phase)
        if probe:self.probe(path,len(pixels),phase)
        return self.decode_media(path,len(pixels),phase)
    def prepare_cache(self,path,source):
        from runtime.wan.vae import reencode_rgb24_readback,decode_normalized_latent
        self.probe(path,181,'preparation');pixels=self.decode_media(path,181,'preparation')
        z=self.call('preparation','E181',lambda:reencode_rgb24_readback(self.wan.vae,pixels.float()/255),181)
        channels=int(self.wan.vae.decoder.conv_out.in_channels)
        folder=self.scratch/'features'/source;folder.mkdir(parents=True,exist_ok=True)
        features=np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype=np.float16,shape=(181,channels,256,256))
        preclip=np.lib.format.open_memmap(folder/'preclip.npy',mode='w+',dtype=np.float32,shape=(181,3,320,512))
        def capture(start,feature,base):
            cached=finite(feature.detach().cpu().to(torch.float16),'FP16 cached features')
            features[start:start+len(feature)]=cached.numpy()
            preclip[start:start+len(base)]=base.detach().cpu().float().numpy()
        with writer.VAEHeadHook(self.wan.vae,capture=capture) as hook:
            rgb=self.call('preparation','D181',lambda:decode_normalized_latent(self.wan.vae,z),181)
        features.flush();preclip.flush()
        # Real cached preclip clamp must reproduce the returned unmodified decode.
        expected=torch.from_numpy(np.array(preclip,copy=True)).permute(0,2,3,1).clamp(0,1)
        if not torch.equal(expected,rgb.detach().cpu()):raise ValueError('captured preclip does not reproduce native OFF')
        return dict(source=source,channels=channels,feature_shape=list(features.shape),preclip_shape=list(preclip.shape),
            feature_dtype='float16',preclip_dtype='float32',feature_bytes=(folder/'features.npy').stat().st_size,
            preclip_bytes=(folder/'preclip.npy').stat().st_size,decode_chunks=hook.chunks,
            preclip_is_before_native_clamp=True,cache=str(folder),clamped_base_equals_OFF=True)
    def initialize_head(self,channels):
        torch.manual_seed(self.cfg['training']['initialization_seed'])
        if self.device=='cuda':torch.cuda.manual_seed_all(self.cfg['training']['initialization_seed'])
        self.head=writer.LocalHead(channels).float().to(self.device)
        return self.head
    def teacher(self,roi,messages):
        with torch.no_grad():
            result=self.call('training','teacher_ROI',lambda:self.video.embed(roi,msgs=messages,is_video=False,
                lowres_attenuation=False,interpolation=dict(mode='bilinear',align_corners=False,antialias=True)),len(roi))['imgs_w']
        return finite(result,'teacher image')
    def detector(self,roi,phase):
        x=roi
        if tuple(x.shape[-2:])!=(self.video.img_size,self.video.img_size):
            x=torch.nn.functional.interpolate(x,size=(self.video.img_size,self.video.img_size),mode='bilinear',align_corners=False,antialias=False)
        # Direct frozen detector, not no_grad-decorated VideoSeal.detect.
        logits=self.call(phase,'student_ROI' if phase=='training' else 'extractor_ROI',lambda:self.video.detector(x),len(x))
        if logits.shape[:2]!=(len(x),257) or any(v!=1 for v in logits.shape[2:]):raise ValueError('VideoSeal native257 logits required')
        return finite(logits.reshape(len(x),257)[:,1:],'VideoSeal logits')
    def train_step(self,record,cache,optimizer):
        start=record['start'];length=8
        f=torch.from_numpy(np.array(cache['features'][start:start+length],copy=True)).to(self.device,dtype=torch.float32)
        p=torch.from_numpy(np.array(cache['preclip'][start:start+length],copy=True)).to(self.device,dtype=torch.float32)
        message=record['message'];key=record['key'];indices=range(start,start+8)
        native=torch.tensor([code.native_message(message,key,t) for t in indices],device=self.device,dtype=torch.float32)
        condition=native[:,:96];base=p.clamp(0,1)
        target=writer.teacher_target(p,self.teacher(base[:,:,32:288,:256],native))
        optimizer.zero_grad(set_to_none=True)
        student=writer.compose(p,writer.bounded_delta(self.head,f,condition))
        zero_equal=None
        if record['step']==0:
            zero_equal=torch.equal(student,base)
            if not zero_equal:raise ValueError('zero initialized writer differs from OFF cached base')
        extractor_canvas=student
        if record['step']%2:
            path=self.scratch/'train-media'/f'step-{record["step"]}.mp4'
            first=self.roundtrip(student.permute(0,2,3,1),path,'training',probe=False)
            second=self.roundtrip(first.float()/255,path.with_name(path.stem+'-second.mp4'),'training',probe=False)
            decoded=second.permute(0,3,1,2).to(self.device,dtype=torch.float32)/255
            extractor_canvas=student+(decoded-student).detach()
            path.unlink();path.with_name(path.stem+'-second.mp4').unlink()
        logits=self.detector(extractor_canvas[:,:,32:288,:256],'training')[:,:96]
        loss,parts=writer.fixed_loss(logits,condition,student,target,base);finite(loss,'training loss')
        loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in self.head.parameters()):raise FloatingPointError('nonfinite head gradient')
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in self.head.parameters()):raise FloatingPointError('nonfinite head weights')
        return dict(step=record['step'],loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in parts.items()},
            zero_initialized_equal_OFF=zero_equal,augmentation='CLEAN' if record['step']%2==0 else 'TWO_REAL_H264_ST',
            cuda_allocated=torch.cuda.memory_allocated() if self.device=='cuda' else None,
            cuda_peak=torch.cuda.max_memory_allocated() if self.device=='cuda' else None)
    def generate(self,source):
        from runtime.wan.b_rank3_four_source_confirmation_backend_v1 import WanBackend
        generator=WanBackend(self.cfg['model_id'])
        try:
            generator.generation_load(source,self.cfg)
            traces=[]
            terminal,receipt=generator.generate('OFF',{},lambda kind,done:self.event('validation',kind,done,0),traces.append)
            path=self.scratch/'terminals'/(source['id']+'.pt');path.parent.mkdir(parents=True,exist_ok=True);torch.save(terminal,path)
            return dict(path=str(path),receipt=receipt,traces=traces)
        finally:generator.release_generation()
    def native_decode(self,terminal,message=None,key=None):
        from runtime.wan.vae import decode_normalized_latent
        z=torch.load(terminal,map_location=self.device,weights_only=True)
        conditions=None if message is None else lambda start,length:[code.condition(message,key,t) for t in range(start,start+length)]
        with writer.VAEHeadHook(self.wan.vae,head=self.head if message is not None else None,conditions=conditions):
            rgb=self.call('validation','D181',lambda:decode_normalized_latent(self.wan.vae,z),181)
        return rgb.detach().cpu().float()
    def extract(self,pixels,roi):
        y0,y1,x0,x1=writer.L if roi=='L' else writer.R
        results=[];batch=self.cfg['receiver']['extract_batch']
        with torch.no_grad():
            for start in range(0,len(pixels),batch):
                crop=pixels[start:start+batch,y0:y1,x0:x1].permute(0,3,1,2).to(self.device,dtype=torch.float32)/255
                results.append(self.detector(crop,'validation').cpu().numpy())
        return np.concatenate(results)
