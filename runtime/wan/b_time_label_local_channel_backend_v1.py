"""Card106 physical adapter, loaded lazily only after persisted USER attempt."""
import math,subprocess
from pathlib import Path
import torch
from runtime.wan.local_joint_positive_projection_margin_v1 import WanBackend as MediaBackend
from runtime.wan import b_time_label_tensor_v1 as tensor
from main.tube_state import b_real_four_phase_v1 as payload

class WanBackend(MediaBackend):
    def load_vae(self):
        self.blocks={'encoder':None,'decoder':None};self.hooks=[]
        return self.load()
    def release_vae(self):
        for hook in self.hooks:hook.remove()
        self.hooks=[];self.vae=None
        import gc
        gc.collect()
        if self.device=='cuda':torch.cuda.empty_cache()
    def load_native(self,path):
        return tensor.validate(torch.load(path,map_location='cpu',weights_only=True),46)
    def load_E(self,path):
        # Already normalized latent, no /255, denorm, encode or gain fit.
        return tensor.validate(torch.load(path,map_location='cpu',weights_only=True),11)
    def load_pixels(self,path,length):
        return self.validate_pixels(torch.load(path,map_location='cpu',weights_only=True),length)
    def load_tensor(self,path):return torch.load(path,map_location='cpu',weights_only=True)
    def save_tensor(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);torch.save(value,path)
    def validate_pixels(self,value,length):
        if not isinstance(value,torch.Tensor) or tuple(value.shape)!=(length,320,512,3) or value.dtype!=torch.uint8:
            raise ValueError(f'actual uint8[{length},320,512,3] required')
        return value.detach().cpu()
    def construct(self,source,key):return tensor.construct(source,key)
    def project_native(self,z,key):return tensor.project_rows(z,key,native=True)
    def project_local(self,z,key):return tensor.project_rows(z,key,native=False)
    def crop45(self,pixels,origin):
        if pixels.ndim!=4 or pixels.shape[1:]!=(320,512,3) or pixels.dtype!=torch.uint8:raise ValueError('actual source pixels required')
        return self.validate_pixels(pixels[origin:origin+45].clone(),45)
    def prepare(self,received45,phase):
        self.validate_pixels(received45,45)
        if phase not in range(4):raise ValueError('phase0..3')
        return received45[phase:phase+41].clone()
    def encode_local(self,pixels):
        from runtime.wan.vae import reencode_rgb24_readback
        self.validate_pixels(pixels,41)
        return tensor.validate(reencode_rgb24_readback(self.vae,pixels.float()/255).detach().cpu().float(),11)
    def decode_PT(self,z):
        # Shared decode performs z_norm*std+mean exactly once internally.
        tensor.validate(z,46)
        return self.decode(z)
    def edit(self,pixels,private):
        self.validate_pixels(pixels,181);a,b,d=(private[k] for k in ('a','b','d'))
        selected=pixels[a:a+121+d]
        return self.validate_pixels(torch.cat([selected[:b],selected[b+d:]],dim=0).clone(),121)
    def read_media(self,path,probe,length):
        import numpy as np
        if (probe.get('codec_name'),probe.get('r_frame_rate'),probe.get('pix_fmt'),int(probe.get('nb_frames',-1)))!=('h264','8/1','yuv420p',length):
            raise ValueError('actual fixed H264/fps8/yuv420p/frame-count receipt mismatch')
        proc=subprocess.run(['ffmpeg','-v','error','-threads','1','-noautorotate','-i',str(path),
            '-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-'],check=True,capture_output=True)
        if len(proc.stdout)!=length*320*512*3:raise ValueError('decoded complete RGB byte count mismatch')
        value=torch.from_numpy(np.frombuffer(proc.stdout,dtype=np.uint8).copy().reshape(length,320,512,3))
        return self.validate_pixels(value,length)
    def spectrum(self,z):return payload.spectrum(z)
    def votes(self,spectrum,key):return payload.read_spectrum(spectrum,key)
    def preclip(self,value):
        from runtime.wan.c73_identity_quality_v1 import preclip_record
        return preclip_record(value)
    def quality(self,left,right,source_frames,meaning):
        from runtime.wan.local_joint_positive_projection_margin_quality_v1 import _frame,_stats,_combine,nominal_mask
        if tuple(left.shape)!=tuple(right.shape) or len(left)!=len(source_frames) or tuple(left.shape[1:])!=(320,512,3):
            raise ValueError('matching actual full/cross RGB and post source map required')
        frames=[];adjacent=[];previous=previous_mask=None
        for t,source_frame in enumerate(source_frames):
            a,b=_frame(left[t]),_frame(right[t])
            if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):raise ValueError('nonfinite paired quality data')
            delta=b-a;mask=nominal_mask(source_frame,320,512)
            frames.append(dict(frame=t,source_frame=source_frame,whole=_stats(delta),outside_nominal_support=_stats(delta[~mask])))
            if previous is not None:
                residual=delta-previous
                adjacent.append(dict(frame_before=t-1,frame_after=t,source_frame_before=source_frames[t-1],source_frame_after=source_frame,
                    spans_deletion_seam=source_frame!=source_frames[t-1]+1,whole=_stats(residual),
                    outside_union_nominal_support=_stats(residual[~(mask|previous_mask)])))
            previous,previous_mask=delta,mask
        whole=_combine([x['whole'] for x in frames])
        return dict(status='SCORED',meaning=meaning,frame_denominator=len(frames),adjacent_denominator=len(adjacent),
            frames=frames,adjacent=adjacent,whole=whole,
            outside_nominal_support=_combine([x['outside_nominal_support'] for x in frames]),
            adjacent_residual=_combine([x['whole'] for x in adjacent]),
            adjacent_outside_union_nominal_support=_combine([x['outside_union_nominal_support'] for x in adjacent]),
            mask_is_actual_causal_support=False,quality_pass=None,
            psnr_db_range1=-20*math.log10(whole['rms']) if whole['rms'] else None)
