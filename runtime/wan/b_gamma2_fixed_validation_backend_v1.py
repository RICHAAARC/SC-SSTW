"""Thin gamma2 adapter over the successful three-group VAE/media path."""
import torch
from runtime.wan.b_terminal_three_group_backend_v1 import WanBackend as Base
from main.tube_state import b_legacy_position_margin_v1 as state


def counted(event,fn):
    event('started')
    try:value=fn()
    except BaseException:
        event('failed');raise
    event('returned');return value


class CountedVAE:
    def __init__(self,vae,event):self.vae,self.event=vae,event
    def __getattr__(self,name):return getattr(self.vae,name)
    def encode(self,*args,**kwargs):return counted(self.event,lambda:self.vae.encode(*args,**kwargs))
    def decode(self,*args,**kwargs):return counted(self.event,lambda:self.vae.decode(*args,**kwargs))


class WanBackend(Base):
    def preflight(self,source):
        import json
        from runtime.wan.b_gamma2_fixed_validation_v1 import HOSTS
        rows=[]
        for h,_,_ in HOSTS:
            z=self.load_tensor(source/'private'/h/'P0_terminal.pt')
            state.geometry.validate_latent(z)
            if z.dtype!=torch.float32 or not bool(torch.isfinite(z).all()):raise ValueError('finite saved FP32 P0 required: '+h)
            native=json.loads((source/'private'/h/'native/P0_Q.json').read_text())
            q=native.get('roi_q')
            if native.get('status')!='SCORE_OBSERVED' or not isinstance(q,list) or len(q)!=44 or any(not isinstance(row,list) or len(row)!=4 or any(not isinstance(v,list) or len(v)!=44 for v in row) for row in q):raise ValueError('native Q44x4x44 required: '+h)
            import math
            if not all(type(v) in (int,float) and math.isfinite(v) for row in q for roi in row for v in roi):raise ValueError('finite native Q required: '+h)
            rows.append(dict(source=h,shape=list(z.shape),dtype=str(z.dtype),native_Q_shape=[44,4,44]));del z
        return dict(status='READABLE_VALIDATED',sources=rows,model_loaded=False)
    def construct_saved(self,terminal_path,native_path):
        import json
        z=self.load_terminal(terminal_path)
        if z.dtype!=torch.float32 or not bool(torch.isfinite(z).all()):raise ValueError('finite saved FP32 P0 required')
        native=json.loads(native_path.read_text())
        book=state.codebook('watermark')
        changed,receipt=state.construct(z,book,2.0,native_observation=native)
        post=state.observe(changed,book)
        audit=super().native_audit(z,changed,post)
        for row in audit['units']:row['shortage']=max(0.,2.-row['q'])
        audit.update(gamma=2.,native_Q=post)
        if any(audit[k]!=0 for k in ('outside_support_max','payload_channels_max','guard_max')):raise ValueError('gamma2 altered protected coordinates')
        return changed,receipt,audit
    def decode(self,terminal,event):
        original=self.vae;self.vae=CountedVAE(original,event)
        try:return super().decode(terminal)
        finally:self.vae=original
    def encode_view(self,pixels,event):
        original=self.vae;self.vae=CountedVAE(original,event)
        try:return super().encode_view(pixels)
        finally:self.vae=original
    def save_mp4(self,pixels,path,event):
        path.parent.mkdir(parents=True,exist_ok=True)
        return counted(event,lambda:super(WanBackend,self).save_mp4(pixels,path))
    def probe(self,path,event):return counted(event,lambda:super(WanBackend,self).probe(path))
    def read_media(self,path,probe,length,event):return counted(event,lambda:super(WanBackend,self).read_media(path,probe,length))
    def release_vae(self):
        super().release_vae();return dict(status='RELEASED')
