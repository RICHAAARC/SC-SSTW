"""CPU media, fixed RGB/Y projections and complete-domain blind saving."""
from pathlib import Path
import json
import numpy as np
import torch
from main.tube_state import b_framewise_qim_v1 as rgb_method
from main.tube_state import b_native_y_diagnostic_v1 as y_method
from main.tube_state import b_joint_time_packet_a_receiver_v1 as receiver
from .b_framewise_qim_time_recovery_v1 import Backend as Codec,command
from .b_native_y_diagnostic_v1 import decode_command
from .vae import quantize_rgb8_no_codec


class Backend(Codec):
    def probe(self,path,frames):
        value=json.loads(command(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=width,height,r_frame_rate,nb_frames,pix_fmt,codec_name,color_range,color_space,color_transfer,color_primaries,bits_per_raw_sample,bit_rate:format=size,bit_rate',
            '-of','json',str(path)],self.resources))
        streams=value.get('streams',[])
        if len(streams)!=1:raise ValueError('one actual video stream required')
        s=streams[0]
        if (s.get('width'),s.get('height'),s.get('r_frame_rate'),int(s.get('nb_frames',-1)))!=(512,320,'8/1',frames):
            raise ValueError('fixed media geometry/rate/length mismatch')
        if s.get('codec_name')!='h264' or s.get('pix_fmt') not in ('yuv420p','yuvj420p'):
            raise ValueError('actual H264 planar 8-bit 420 required')
        return dict(stream=s,format=value.get('format',{}),profile=y_method.profile(s),bytes=Path(path).stat().st_size)

    def decode_y(self,path,frames,probe):
        data=command(decode_command(path),self.resources)
        if len(data)!=frames*320*512:raise ValueError('native Y byte count mismatch')
        return np.frombuffer(data,np.uint8).reshape(frames,320,512)

    def quantize(self,preclip):
        if preclip.dtype!=np.float32 or preclip.shape!=(181,320,512,3):raise ValueError('native FP32 full181 preclip required')
        pixels=np.empty(preclip.shape,np.uint8)
        for start in range(0,181,4):
            self.resources.check();value=torch.from_numpy(np.array(preclip[start:start+4],copy=True))
            if not bool(torch.isfinite(value).all()):raise ValueError('nonfinite native preclip')
            pixels[start:start+4]=quantize_rgb8_no_codec(value).numpy()
        return pixels

    def rgb_raw(self,values,layer):
        if values.shape!=(181,320,512,3):raise ValueError('full181 RGB observation required')
        expected=np.uint8 if layer=='firstRGB' else np.float32
        if values.dtype!=expected:raise ValueError('actual RGB layer dtype mismatch')
        result={k:[] for k in ('q','d0','d1','soft','mask')}
        for start in range(0,181,4):
            self.resources.check();value=torch.from_numpy(np.array(values[start:start+4],copy=True))
            if not bool(torch.isfinite(value).all()):raise ValueError('nonfinite RGB input')
            if layer=='clip':value=value.clamp(0,1)
            elif layer=='rgb8':value=quantize_rgb8_no_codec(value).double()/255
            elif layer=='firstRGB':value=value.double()/255
            elif layer!='preclip':raise ValueError('unknown fixed RGB layer')
            for k,v in rgb_method.read_rgb(value).items():result[k].append(v.numpy())
        return {k:np.concatenate(v) for k,v in result.items()}

    def receive(self,raw_path,output,coding,frames,dump,*,native_y=None):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        with self.resources.receiving(600):
            if native_y is not None:
                y=np.load(native_y,mmap_mode='r',allow_pickle=False)
                if y.shape!=(frames,320,512):raise ValueError('native Y length mismatch')
                raw=y_method.read_y(y,self.resources.check)
            else:
                with np.load(raw_path,allow_pickle=False) as data:raw={k:data[k] for k in ('q','d0','d1','soft','mask')}
            if raw['soft'].shape!=(frames,5,2,63):raise ValueError('fixed received raw frame geometry')
            np.savez(output/'readout.npz',**raw)
            diagnostic=receiver.diagnostics(raw['soft'],raw['mask'],coding,self.resources.check)
            dump(output/'packets.json',diagnostic)
            startup=receiver.startup_scores(raw['soft'],raw['mask'],coding)
            np.savez(output/'startup.npz',**startup)
            chunks=[];folder=output/'scores';folder.mkdir()
            for lo,hi,values in receiver.complete_scores(raw['soft'],raw['mask'],coding,self.resources.check):
                name=f'{lo:05}-{hi-1:05}.npy';np.save(folder/name,values,allow_pickle=False)
                chunks.append(dict(start=lo,stop=hi,file='scores/'+name,shape=list(values.shape),dtype='float64'))
                self.resources.check()
                dump(output/'scores_progress.json',dict(chunks=chunks,complete=False))
                if len(chunks)==1 or len(chunks)%8==0 or hi==92672:
                    print(f'[cpu_media] {output}: saved score states [{lo}, {hi}); chunks={len(chunks)}',flush=True)
            result=dict(status='BLIND_SAVED',coding=coding,frames=frames,chunks=chunks,
                        state_chunk=receiver.STATE_CHUNK,mature_domain=[4,92671],epsilon=receiver.EPSILON,
                        no_soft_observation_frames=np.flatnonzero(~receiver.hinge_terms(raw['soft'],raw['mask'])['support'].any(axis=(1,2))).tolist(),
                        startup='separate numerator, denominator, active mask and partial scores; not ranked with mature states',
                        formal_output=None,state_evaluations=frames*92668,startup_state_evaluations=frames*4)
            dump(output/'scores.json',result);self.resources.check();return result
