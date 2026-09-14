"""Public image-only statistic. RGB code values, not linear-light luminance."""
import math

def basis(length,np):
    if type(length) is not int or length<2:raise ValueError('dimension >=2 required')
    h=np.cos(math.pi*(np.arange(length,dtype=np.float64)+.5)/length)
    h=h-h.mean();return h/np.sqrt(np.mean(h*h))

def read_rgb(rgb,np):
    x=np.asarray(rgb)
    if x.ndim!=4 or x.shape[0]==0 or x.shape[-1]!=3 or min(x.shape[1:3])<2:raise ValueError('T,H,W,3 required')
    if not np.isfinite(x).all() or x.min()<0 or x.max()>1:raise ValueError('finite RGB code [0,1] required')
    y=x.astype(np.float64)@np.asarray([.2126,.7152,.0722])
    hx=basis(x.shape[2],np);hy=basis(x.shape[1],np)
    return np.stack(((y*hx[None,None,:]).mean((1,2)),(y*hy[None,:,None]).mean((1,2))),axis=-1)

def read_rgb_differentiable(rgb,torch):
    if rgb.ndim!=4 or rgb.shape[0]==0 or rgb.shape[-1]!=3 or min(rgb.shape[1:3])<2:raise ValueError('T,H,W,3 required')
    if not bool(torch.isfinite(rgb).all()) or bool((rgb<0).any()) or bool((rgb>1).any()):raise ValueError('RGB range')
    if not rgb.is_floating_point():raise ValueError('floating RGB required')
    def h(n):
        v=torch.cos(math.pi*(torch.arange(n,device=rgb.device,dtype=rgb.dtype)+.5)/n);v=v-v.mean();return v/v.square().mean().sqrt()
    y=(rgb*rgb.new_tensor([.2126,.7152,.0722])).sum(-1)
    return torch.stack(((y*h(rgb.shape[2])[None,None,:]).mean((1,2)),(y*h(rgb.shape[1])[None,:,None]).mean((1,2))),-1)
