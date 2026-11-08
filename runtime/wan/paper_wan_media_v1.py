"""Use the C4 H264 command, including option order, for the new paired comparison."""
from pathlib import Path
import subprocess
from .native_baseline_media_v4 import MediaV4


class CohortMedia(MediaV4):
    def probe(self,path,*args,**kwargs):
        from .paper_recovery_v1 import selected_media
        return super().probe(selected_media(path),*args,**kwargs)
    def decode(self,path,*args,**kwargs):
        from .paper_recovery_v1 import selected_media
        return super().decode(selected_media(path),*args,**kwargs)
    def command(self,args,*,data=None):
        return subprocess.run(args,input=data,check=True,capture_output=True,timeout=300).stdout
    def encode(self,rgb,path,fps=8,**kwargs):
        import numpy as np
        if fps!=8 or rgb.dtype!=np.uint8 or rgb.ndim!=4 or rgb.shape[1:]!=(320,512,3):
            raise ValueError('C4 uint8 RGB/8fps geometry required')
        from .paper_recovery_v1 import target,publish_media
        logical=Path(path);path=target(logical);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError('refusing existing media output: '+str(path))
        self.command(['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-s','512x320','-r','8',
            '-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-n',str(path)],data=rgb.tobytes())
        publish_media(logical,path)
        return dict(path=str(path),bytes=path.stat().st_size,codec='h264',crf=18,fps=8)
