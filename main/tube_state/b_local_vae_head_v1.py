"""Small conditional output head; original VAE parameters/caches remain untouched."""
import math
import torch
from torch import nn

L=(32,288,0,256);R=(32,288,256,512)
RGB_LIMIT=8/255

def raised_cosine(size=256,width=16,*,device=None,dtype=torch.float32):
    if 2*width>size or width<1:raise ValueError('valid cosine boundary required')
    v=torch.ones(size,device=device,dtype=dtype)
    ramp=.5*(1-torch.cos(torch.arange(width,device=device,dtype=dtype)*math.pi/width))
    v[:width]=ramp;v[-width:]=ramp.flip(0)
    return v[:,None]*v[None,:]

class LocalHead(nn.Module):
    def __init__(self,channels):
        super().__init__()
        if type(channels) is not int or channels<=0:raise ValueError('actual conv_out input channels required')
        self.channels=channels
        self.conv1=nn.Conv2d(channels,64,3,padding=1)
        self.conv2=nn.Conv2d(64,64,3,padding=1)
        self.conv3=nn.Conv2d(64,3,3,padding=1)
        self.film1=nn.Linear(96,128);self.film2=nn.Linear(96,128)
        nn.init.zeros_(self.conv3.weight);nn.init.zeros_(self.conv3.bias)
    def forward(self,features,condition):
        if features.ndim!=4 or features.shape[1]!=self.channels or condition.shape!=(len(features),96):raise ValueError('head feature/96-bit condition shape')
        condition=condition.to(dtype=features.dtype)*2-1
        x=self.conv1(features)
        scale,shift=self.film1(condition).chunk(2,dim=1)
        x=torch.nn.functional.silu(x*(1+scale[:,:,None,None])+shift[:,:,None,None])
        x=self.conv2(x);scale,shift=self.film2(condition).chunk(2,dim=1)
        return self.conv3(torch.nn.functional.silu(x*(1+scale[:,:,None,None])+shift[:,:,None,None]))

def bounded_delta(head,features,condition):
    if tuple(features.shape[-2:])!=(256,256):raise ValueError('fixed L ROI feature shape required')
    raw=head(features.detach().float(),condition.float())
    return RGB_LIMIT*torch.tanh(raw)*raised_cosine(device=raw.device)[None,None]

def compose(preclip,delta):
    if preclip.ndim!=4 or tuple(preclip.shape[1:])!=(3,320,512) or tuple(delta.shape)!=(len(preclip),3,256,256):raise ValueError('full canvas and fixed L delta required')
    full=torch.nn.functional.pad(delta,(0,256,32,32))
    return (preclip+full).clamp(0,1)

def teacher_target(preclip,teacher_rgb):
    base=preclip.clamp(0,1);roi=base[:,:,32:288,:256]
    if teacher_rgb.shape!=roi.shape:raise ValueError('teacher image ROI dimensions')
    delta=(teacher_rgb-roi).clamp(-RGB_LIMIT,RGB_LIMIT)*raised_cosine(device=preclip.device)[None,None]
    return compose(preclip,delta)

def fixed_loss(logits96,bits96,student,target,base):
    if logits96.shape!=bits96.shape or logits96.shape[-1]!=96:raise ValueError('BCE only96 effective bits')
    s=student[:,:,32:288,:256];t=target[:,:,32:288,:256];b=base[:,:,32:288,:256]
    rs=s-b;rt=t-b
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits96,bits96)
    distillation=(rs-rt).square().mean()/(rt.square().mean()+1e-6)
    image_mse=(s-b).square().mean()
    return bce+distillation+100*image_mse,dict(BCE96=bce,distillation=distillation,ROI_MSE=image_mse)

class VAEHeadHook:
    """Forward hook after untouched causal conv_out, before decoder's final clamp.

    The detached input is norm_out+SiLU output. No input or cache is mutated.
    Caller creates a fresh context for each complete181 decode.
    """
    def __init__(self,vae,head=None,conditions=None,capture=None):
        self.conv=vae.decoder.conv_out;self.channels=int(self.conv.in_channels)
        if int(self.conv.out_channels)!=3:raise ValueError('RGB decoder conv_out required')
        if head is not None and head.channels!=self.channels:raise ValueError('head does not match actual decoder features')
        self.head=head;self.conditions=conditions;self.capture=capture;self.frame=0;self.chunks=[];self.handle=None
    def __enter__(self):
        self.frame=0;self.chunks=[]
        self.handle=self.conv.register_forward_hook(self._hook)
        return self
    def _hook(self,module,args,output):
        feature=args[0].detach();start=self.frame
        if feature.ndim!=5 or feature.shape[0]!=1 or tuple(feature.shape[-2:])!=(320,512):raise ValueError('full native decoder feature canvas required')
        length=int(output.shape[2]);expected=1 if start==0 else 4
        if length!=expected or feature.shape[2]!=length or start+length>181:raise ValueError('native first1 then4 decode chunks required')
        roi=feature[0,:,:,32:288,:256].permute(1,0,2,3).contiguous()
        preclip=output.detach()[0].permute(1,0,2,3).float()/2.0+0.5
        if not bool(torch.isfinite(roi).all() and torch.isfinite(preclip).all()):raise ValueError('nonfinite decoder features/preclip')
        if self.capture is not None:self.capture(start,roi,preclip)
        self.frame+=length;self.chunks.append(length)
        if self.head is None:return output
        cond=torch.as_tensor(self.conditions(start,length),device=roi.device,dtype=torch.float32)
        delta=bounded_delta(self.head,roi,cond)
        if not bool(torch.isfinite(delta).all()):raise ValueError('nonfinite native writer residual')
        native=torch.nn.functional.pad(2*delta,(0,256,32,32)).permute(1,0,2,3).unsqueeze(0)
        return output+native.to(output.dtype)
    def __exit__(self,kind,error,traceback):
        self.handle.remove()
        if kind is None and self.frame!=181:raise ValueError('incomplete181 native decode')
