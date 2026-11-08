"""v3 real AVI normalization plus the adopted shared 40-view media transformations."""
from fractions import Fraction
import io
import json
from pathlib import Path
from .native_baseline_media_v3 import MediaV3


def rgb8(value):
    import numpy as np
    if not np.isfinite(value).all():raise ValueError('non-finite media transform')
    return np.rint(np.clip(value,0,255)).astype(np.uint8)


def edit_temporal(rgb,mapping,donor=None):
    import numpy as np
    output=np.empty((len(mapping),*rgb.shape[1:]),dtype=np.uint8)
    for i,row in enumerate(mapping):
        if row['kind']=='target':output[i]=rgb[row['index']]
        elif row['kind']=='foreign':
            if donor is None:raise FileNotFoundError('fixed DEV donor G1 unavailable')
            output[i]=donor[row['index']]
        elif row['kind']=='synthetic':
            a,b=row['parents'];output[i]=rgb8((rgb[a].astype(np.float64)+rgb[b].astype(np.float64))/2)
        else:raise ValueError('invalid editor mapping kind')
    return output


def spatial(rgb,operation,noise_seed):
    import numpy as np
    import torch
    from torch.nn import functional as F
    if operation in ('S00','S01','S07'):return rgb.copy()
    output=np.empty_like(rgb);random=np.random.Generator(np.random.PCG64(np.random.SeedSequence(noise_seed)))
    for i,frame in enumerate(rgb):
        if operation=='S05':output[i]=rgb8(frame.astype(np.float64)*.8);continue
        if operation=='S06':output[i]=rgb8(frame.astype(np.float64)+random.normal(0,2,frame.shape));continue
        x=torch.from_numpy(frame.copy()).permute(2,0,1).unsqueeze(0).float()
        if operation=='S02':x=F.interpolate(x[:,:,32:288,52:460],size=(320,512),mode='bilinear',align_corners=False,antialias=False)
        elif operation=='S03':
            x=F.interpolate(x,size=(160,256),mode='bilinear',align_corners=False,antialias=False)
            x=F.interpolate(x,size=(320,512),mode='bilinear',align_corners=False,antialias=False)
        elif operation=='S04':
            g=torch.arange(-3,4,dtype=torch.float32);g=torch.exp(-g.square()/2);g=g/g.sum()
            kernel=(g[:,None]*g[None,:])[None,None].repeat(3,1,1,1)
            x=F.conv2d(F.pad(x,(3,3,3,3),mode='reflect'),kernel,groups=3)
        else:raise ValueError('unknown fixed spatial operation')
        output[i]=rgb8(x.squeeze(0).permute(1,2,0).numpy())
    return output


class MediaV4(MediaV3):
    def encode(self,rgb,path,fps=25,*,codec='h264',crf=18):
        import numpy as np
        if rgb.dtype!=np.uint8 or rgb.ndim!=4 or rgb.shape[1:]!=(320,512,3):raise ValueError('actual RGB8 geometry required')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        args=['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s','512x320',
              '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264' if codec=='h264' else 'libx265',
              '-preset','medium','-crf',str(crf),'-pix_fmt','yuv420p','-g','250','-bf','3','-threads','1']
        if codec=='h265':args+=['-x265-params','pools=none:frame-threads=1:keyint=250:bframes=3']
        self.command(args+[str(path)],data=rgb.tobytes())
        return dict(path=str(path),bytes=path.stat().st_size,codec=codec,crf=crf,preset='medium',gop=250,bframes=3,threads=1)

    def probe(self,path,shape,fps=25,*,codec='h264'):
        s=json.loads(self.command(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames','-of','json',str(path)]))['streams'][0]
        if (s['codec_name']!=('hevc' if codec=='h265' else 'h264') or s['pix_fmt']!='yuv420p'
                or [s['height'],s['width']]!=list(shape[1:3]) or Fraction(s['r_frame_rate'])!=fps or int(s['nb_frames'])!=shape[0]):
            raise ValueError('actual codec/shape/frame-rate/length differs from plan')
        return s

    def jpeg_encode(self,frame):
        import numpy as np
        from PIL import Image
        output=io.BytesIO();Image.fromarray(frame).save(output,format='JPEG',quality=75,subsampling=2)
        return np.frombuffer(output.getvalue(),dtype=np.uint8)

    def jpeg_decode(self,data):
        import numpy as np
        from PIL import Image
        with Image.open(io.BytesIO(data.tobytes())) as image:result=np.asarray(image.convert('RGB')).copy()
        if result.shape!=(320,512,3):raise ValueError('JPEG decoded geometry mismatch')
        return result
