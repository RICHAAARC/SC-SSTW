"""Three official weight loads, four writers, one pixels-only FP32 extraction interface."""
import sys
import time
from .native_baseline_models_v3 import NativeAdapterV3
from .native_baseline_models_v1 import working_directory,finite_tensor,quantize


class NativeAdapterV4(NativeAdapterV3):
    def load(self):
        if self.method=='wam':return super().load()
        if self.method not in ('pixelseal','videoseal'):raise ValueError('unsupported native backend')
        import torch
        started=time.monotonic();repo=self.vendor/'videoseal';sys.path.insert(0,str(repo))
        if self.device=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable for this user-run GPU batch')
            torch.cuda.reset_peak_memory_stats()
        card='pixelseal' if self.method=='pixelseal' else 'videoseal_1.0'
        with working_directory(repo):
            import videoseal
            self.model=videoseal.load(card).float().eval().to(self.device)
        expected=dict(chunk_size=32,step_size=8 if self.method=='pixelseal' else 4,scaling_w=.2,scaling_i=1.,img_size=256,video_mode='repeat',clamp=True)
        actual={key:getattr(self.model,key,None) for key in expected}
        actual.update({key:getattr(getattr(self.model,'blender',None),key,None) for key in ('scaling_w','scaling_i')})
        if actual!=expected or self.model.attenuation is None:raise ValueError(f'actual {card} method configuration differs: {actual}')
        def hook(kind):
            def observed(module,args):
                self.counts[kind+'_calls']+=1;self.counts[kind+'_frames']+=int(args[0].shape[0])
            return observed
        self.handles=[self.model.embedder.register_forward_pre_hook(hook('embedder')),
                      self.model.detector.register_forward_pre_hook(hook('detector'))]
        return dict(card=card,weight_url='https://dl.fbaipublicfiles.com/videoseal/'+('pixelseal/checkpoint.pth' if self.method=='pixelseal' else 'y_256b_img.pth'),
            native_bits=256,parameters=actual,device=self.device,precision='FP32',autocast=False,seconds=time.monotonic()-started)

    def embed_batch(self,rgb,message,*,method,start):
        import torch
        if method=='wam':return self.embed(rgb,message,[80,128,240,384])
        if self.method not in ('pixelseal','videoseal'):raise ValueError('wrong physical backend for Seal writer')
        is_video=method!='frameindex'
        if is_video:
            expected=181 if method=='pixelseal' else 128 if start==0 else 53
            if len(rgb)!=expected or (method=='videoseal' and start not in (0,128)):
                raise ValueError('native anchor/chunk boundaries must be preserved')
        elif not 1<=len(rgb)<=8:raise ValueError('FrameIndex requires real B8/tail')
        with torch.inference_mode():
            x=self.tensor(rgb)
            msgs=torch.tensor([message] if is_video else message,dtype=torch.float32,device=self.device)
            if msgs.shape!=(1 if is_video else len(rgb),256):raise ValueError('full256 writer messages required')
            result=self.model.embed(x,msgs=msgs,is_video=is_video,lowres_attenuation=False,
                interpolation=dict(mode='bilinear',align_corners=False,antialias=True))['imgs_w']
            finite_tensor(result,x.shape)
            return quantize(result)

    def detect(self,rgb):
        """No query, source, attack mapping, truth, target ROI or original image arguments."""
        if self.method=='wam':return super().detect(rgb)
        import torch
        import numpy as np
        if not 1<=len(rgb)<=32:raise ValueError('Seal reader requires real B32/tail')
        with torch.inference_mode():
            x=self.tensor(rgb)
            preds=finite_tensor(self.model.detect(x,is_video=False,
                interpolation=dict(mode='bilinear',align_corners=False,antialias=True))['preds'],(len(rgb),257))
            return dict(logits=preds.float().cpu().numpy(),bits=(preds[:,1:]>0).to(torch.uint8).cpu().numpy(),
                        valid=np.ones(len(rgb),dtype=bool))
