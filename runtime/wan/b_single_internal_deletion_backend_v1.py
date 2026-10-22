"""Card103 actual media/E41 adapter. Imported only after persisted user attempt."""
import json,math,subprocess
from pathlib import Path
import torch
from runtime.wan.local_joint_positive_projection_margin_v1 import WanBackend as MediaBackend
from main.tube_state import b_real_four_phase_v1 as physical
from main.tube_state import b_legacy_position_margin_v1 as carrier

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
    def load_source(self,path):
        x=torch.load(path,map_location='cpu',weights_only=True)
        return self.validate(x,181)
    def validate(self,x,length):
        if not isinstance(x,torch.Tensor) or tuple(x.shape)!=(length,320,512,3) or x.dtype!=torch.uint8:
            raise ValueError(f'actual uint8[{length},320,512,3] RGB required')
        return x.detach().cpu()
    def save_tensor(self,path,x):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);torch.save(x,path)
    def load_tensor(self,path):
        return torch.load(path,map_location='cpu',weights_only=True)
    def edit(self,pixels,private):
        self.validate(pixels,181);a,b,d=(private[k] for k in ('a','b','d'))
        segment=pixels[a:a+121+d]
        result=torch.cat([segment[:b],segment[b+d:]],dim=0).clone()
        return self.validate(result,121)
    def read_mp4(self,path,probe):
        import numpy as np
        if int(probe.get('nb_frames',-1))!=121 or probe.get('r_frame_rate')!='8/1' or probe.get('codec_name')!='h264':
            raise ValueError('actual121-frame8fps H264 stream required')
        proc=subprocess.run(['ffmpeg','-v','error','-threads','1','-noautorotate','-i',str(path),
            '-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-'],check=True,capture_output=True)
        if len(proc.stdout)!=121*320*512*3:raise ValueError('decoded RGB length/geometry mismatch')
        return self.validate(torch.from_numpy(np.frombuffer(proc.stdout,dtype=np.uint8).copy().reshape(121,320,512,3)),121)
    def prepare(self,received,anchor,phase):
        self.validate(received,121)
        if anchor not in (0,76) or phase not in range(4):raise ValueError('fixed received-only anchor/phase')
        return received[anchor+phase:anchor+phase+41].clone()
    def encode(self,pixels):
        from runtime.wan.vae import reencode_rgb24_readback
        self.validate(pixels,41)
        z=reencode_rgb24_readback(self.vae,pixels.float()/255).detach().cpu().float()
        physical.validate_local(z);return z
    def book(self,key):return carrier.codebook(key)
    def observe(self,z,book):return physical.observe(z,book)
    def spectrum(self,z):return physical.spectrum(z)
    def votes(self,spectrum,key):return physical.read_spectrum(spectrum,key)
    def score(self,banks,model):
        from main.tube_state import b_background_normalized_match_v1 as score
        return score.score_family(banks,model)
    def independent(self,family):
        from main.tube_state import b_background_normalized_match_v1 as old
        return old.decide(family,dict(status='FROZEN',tau_N=0.5368688672209747))
    def quality(self,left,right,source_frames,meaning):
        from runtime.wan.local_joint_positive_projection_margin_quality_v1 import _frame,_stats,_combine,nominal_mask
        self.validate(left,121);self.validate(right,121)
        if len(source_frames)!=121:raise ValueError('post-only actual source frame map required')
        from runtime.wan.b_single_internal_deletion_evaluation_v1 import quality_coordinates
        coordinates,pair_coordinates=quality_coordinates(source_frames)
        frames=[];adjacent=[];previous=previous_mask=None
        for t,source_frame in enumerate(source_frames):
            delta=_frame(right[t])-_frame(left[t])
            mask=nominal_mask(source_frame,320,512)
            frames.append(dict(coordinates[t],whole=_stats(delta),outside_nominal_support=_stats(delta[~mask])))
            if previous is not None:
                residual=delta-previous
                adjacent.append(dict(pair_coordinates[t-1],whole=_stats(residual),
                    outside_union_nominal_support=_stats(residual[~(mask|previous_mask)])))
            previous,previous_mask=delta,mask
        whole=_combine([r['whole'] for r in frames])
        return dict(status='SCORED',meaning=meaning,frames=frames,adjacent=adjacent,frame_denominator=121,adjacent_denominator=120,
            whole=whole,outside_nominal_support=_combine([r['outside_nominal_support'] for r in frames]),
            adjacent_residual=_combine([r['whole'] for r in adjacent]),
            adjacent_outside_union_nominal_support=_combine([r['outside_union_nominal_support'] for r in adjacent]),
            psnr_db_range1=None if not whole['rms'] else -20*math.log10(whole['rms']),
            mask_is_actual_causal_support=False,mask_meaning='post-only source-frame nominal ROIs, not actual causal support',
            quality_pass=None)
