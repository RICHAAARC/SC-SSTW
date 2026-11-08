"""Three matched stages, full181 RGB quality and actual B16 LPIPS checkpoints."""
from pathlib import Path
import time
from .native_baseline_storage_v1 import write_json,read_json
from .native_baseline_storage_v4 import SpaceUnavailable


class LPIPS:
    def __init__(self,device='cuda'):self.device=device;self.model=None;self.forwards=0;self.handle=None
    def load(self):
        import lpips
        self.model=lpips.LPIPS(net='alex').float().eval().to(self.device)
        def hook(module,args):self.forwards+=1
        self.handle=self.model.net.register_forward_pre_hook(hook)
        return dict(net='alex',precision='FP32',autocast=False)
    def pair(self,left,right):
        import torch
        import numpy as np
        def tensor(rgb):return torch.from_numpy(rgb.copy()).permute(0,3,1,2).float().to(self.device)/127.5-1
        with torch.inference_mode():value=self.model(tensor(left),tensor(right)).reshape(-1).float().cpu().numpy()
        if value.shape!=(len(left),) or not np.isfinite(value).all():raise ValueError('invalid actual LPIPS output')
        return dict(lpips=value)
    def close(self):
        if self.handle:self.handle.remove()
        self.model=None


def cpu_quality(left,right):
    import numpy as np
    from skimage.metrics import structural_similarity
    if left.shape!=right.shape or left.shape!=(181,320,512,3) or left.dtype!=np.uint8 or right.dtype!=np.uint8:
        raise ValueError('full matched181 RGB8 frames required for quality')
    rows=[];previous=None;flicker=[]
    for i,(a,b) in enumerate(zip(left,right)):
        x=a.astype(np.float64)/255;y=b.astype(np.float64)/255;residual=y-x
        mse=float(np.square(residual).mean())
        rows.append(dict(frame=i,mse=mse,psnr_db=None if mse==0 else float(-10*np.log10(mse)),
            psnr_positive_infinity=mse==0,ssim=float(structural_similarity(x,y,channel_axis=-1,
                data_range=1.,gaussian_weights=True,sigma=1.5,win_size=11,use_sample_covariance=False))))
        if previous is not None:flicker.append(dict(edge=i-1,residual_difference_rms=float(np.sqrt(np.square(residual-previous).mean()))))
        previous=residual
    return dict(frames=rows,edges=flicker,planned_frames=181,planned_edges=180,
                scope='matched stage quality; residual difference proxy, not motion-compensated tLPIPS')


def measure(store,stage,left,right,model):
    from main.baselines.public_fpr001_v4 import batches
    import numpy as np
    path=store.folder/'quality'/f'{stage}.json'
    value=read_json(path) if path.is_file() else cpu_quality(left,right)
    write_json(path,value)
    losses=[];failures=[]
    for start,n in batches(181,16):
        key=f'quality/{stage}/lpips/{start}'
        if model is None and not store.reusable(key):
            failures.append(dict(start=start,frames=n,error='LPIPS loader unavailable; no physical call charged'))
            continue
        try:
            result=store.perform(key,'lpips',
                lambda s=start,k=n:model.pair(left[s:s+k],right[s:s+k]),arrays=True,
                validate=lambda x,k=n:validate_lpips(x,k))
            losses.extend(dict(frame=start+i,value=float(x)) for i,x in enumerate(result['lpips']))
        except SpaceUnavailable:raise
        except Exception as error:failures.append(dict(start=start,frames=n,error=str(error)))
    value.update(lpips=losses,lpips_failures=failures,lpips_observed_frames=len(losses),
                 lpips_mean=float(np.mean([r['value'] for r in losses])) if len(losses)==181 else None)
    write_json(path,value);return value


def validate_lpips(value,n):
    import numpy as np
    if value['lpips'].shape!=(n,) or not np.isfinite(value['lpips']).all():raise ValueError('invalid LPIPS batch')
