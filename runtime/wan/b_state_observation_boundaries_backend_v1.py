"""VAE-only physical adapter for the fixed saved-input boundary diagnostic."""
from pathlib import Path
import gc
import torch
from runtime.wan.vae import reencode_rgb24_readback
from main.tube_state import b_legacy_position_margin_v1 as state

DEFAULT_MODEL_ID='Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
DEFAULT_VAE_CACHE=Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Cache/models/Wan2.1-T2V-1.3B-Diffusers')


def select_vae_source(model_id,cache_base=DEFAULT_VAE_CACHE):
    """Same successful Card102/three-group local-cache selection; no fallback load."""
    explicit=Path(model_id)
    if explicit.is_dir():
        return dict(location=str(explicit),subfolder='vae' if (explicit/'vae').is_dir() else None,origin='explicit_local')
    if model_id==DEFAULT_MODEL_ID:
        base=Path(cache_base);candidates=[base/'available',base]
        if base.is_dir():candidates.extend(sorted(p for p in base.iterdir() if p.is_dir()))
        for path in candidates:
            vae=path/'vae'
            if (vae/'config.json').is_file() and (any(vae.glob('*.safetensors')) or any(vae.glob('*.bin'))):
                return dict(location=str(path),subfolder='vae',origin='shared_drive_cache')
    return dict(location=model_id,subfolder='vae',origin='model_id')


def prepare_pixels(value,layer,start):
    if not isinstance(value,torch.Tensor) or tuple(value.shape)!=(181,320,512,3):
        raise ValueError('saved full RGB must be [181,320,512,3]')
    if layer=='C':
        if value.dtype!=torch.uint8:raise ValueError('C requires saved uint8 RGB8')
        return value[start:start+41].float()/255.0
    if layer not in ('A','B'):raise ValueError('only A/B/C may encode')
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):raise ValueError('finite saved floating RGB required')
    if float(value.min())<0 or float(value.max())>1:raise ValueError('saved float RGB outside [0,1]; no repair/clamp')
    return value if layer=='A' else value[start:start+41]


def project_q40(z,book,s,full):
    expected=(1,16,46 if full else 11,40,64)
    if not isinstance(z,torch.Tensor) or tuple(z.shape)!=expected or z.dtype!=torch.float32 or not bool(torch.isfinite(z).all()):
        raise ValueError('finite normalized FP32 latent with adopted geometry required')
    # Identical channels, ROI flatten and signed directions to the existing Q;
    # only the predeclared 40 coordinates are needed in this post-truth task.
    value=z.detach().cpu().double();values=[]
    for i in range(10):
        canonical=s+i; time=canonical if full else i+1
        for roi,(y0,y1,x0,x1) in enumerate(state.ROIS):
            patch=value[0,4:16,time,y0:y1,x0:x1].reshape(-1)
            direction=book['directions'][roi,canonical-1].reshape(-1)
            values.append(float((patch@direction)*book['signs'][roi,canonical-1]))
    return values


class CountedVAE:
    """Count the actual encode invocation, after preparation/cache clearing."""
    def __init__(self,vae,event):self.vae,self.event=vae,event
    def __getattr__(self,name):return getattr(self.vae,name)
    def encode(self,*args,**kwargs):
        self.event('started')
        try:value=self.vae.encode(*args,**kwargs)
        except BaseException:
            self.event('failed');raise
        self.event('returned');return value


class BoundaryBackend:
    def __init__(self,model_id,key='watermark'):
        self.model_id=model_id;self.key=key
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.vae=None;self.book=None;self.hooks=[];self.blocks={'encoder':None,'decoder':None}
    def load(self):
        from diffusers import AutoencoderKLWan
        selected=select_vae_source(self.model_id)
        kwargs=dict(torch_dtype=torch.float32)
        if selected['subfolder']:kwargs['subfolder']=selected['subfolder']
        self.vae=AutoencoderKLWan.from_pretrained(selected['location'],**kwargs)
        self.vae=self.vae.to(self.device).eval().requires_grad_(False)
        for name in self.blocks:
            module=getattr(self.vae,name,None)
            if module is not None and callable(getattr(module,'register_forward_pre_hook',None)):
                self.blocks[name]=0
                def count(_module,_args,label=name):self.blocks[label]+=1
                self.hooks.append(module.register_forward_pre_hook(count))
        self.book=state.codebook(self.key)
        return dict(device=self.device,dtype='float32',model_selection=selected,posterior='mode',cache_reset_each_call=True)
    def encode_q40(self,row,event):
        value=torch.load(row['input_path'],map_location='cpu',weights_only=True)
        pixels=prepare_pixels(value,row['layer'],row['source_start'])
        try:
            z=reencode_rgb24_readback(CountedVAE(self.vae,event),pixels).detach().cpu().float()
            return project_q40(z,self.book,row['s'],row['layer']=='A')
        finally:
            del pixels,value
    def release(self):
        for hook in self.hooks:hook.remove()
        self.hooks=[];self.vae=None;self.book=None
        gc.collect()
        if self.device=='cuda':torch.cuda.empty_cache()
