"""CPU-only adapters for the two original readers and two existing-media decodes."""
from pathlib import Path
import json
import time
import numpy as np
from .b_framewise_qim_time_recovery_v1 import Resources,ResourceExceeded,command
from .b_time_layer_diagnostic_v1 import Backend as SavedRawBackend,RawLimitExceeded as LocalLimitExceeded


class Backend:
    def __init__(self):self.resources=Resources();self.started=time.monotonic();self.reader_started=None
    def check(self):
        self.resources.check()
        now=time.monotonic()
        if now-self.started>6300:raise ResourceExceeded('whole diagnostic 6300s limit observed')
        if self.reader_started is not None and now-self.reader_started>600:raise LocalLimitExceeded('reader 600s limit observed')
    def codec(self,args):
        try:return command(args,self)
        except ResourceExceeded as error:
            if str(error)=='codec 300 second deadline exceeded':raise LocalLimitExceeded(str(error)) from error
            raise
    def probe(self,path):
        value=json.loads(self.codec(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames,pix_fmt,codec_name','-of','json',str(path)]))
        streams=value.get('streams',[])
        if len(streams)!=1:raise ValueError('one video stream required')
        s=streams[0]
        if s.get('width')!=512 or s.get('height')!=320 or s.get('r_frame_rate')!='8/1' or int(s.get('nb_frames',-1))!=181:raise ValueError('fixed 181/320/512/8 media geometry mismatch')
        return s
    def validate_rgb(self,value):
        if value.dtype!=np.uint8 or value.shape!=(181,320,512,3):raise ValueError('uint8 RGB [181,320,512,3] required')
        return value
    def load_rgb(self,path):return self.validate_rgb(np.load(path,allow_pickle=False,mmap_mode='r'))
    def decode(self,path):
        data=self.codec(['ffmpeg','-v','error','-threads','1','-i',str(path),'-map','0:v:0','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
        if len(data)!=181*320*512*3:raise ValueError('decoded RGB byte count mismatch')
        return self.validate_rgb(np.frombuffer(data,dtype=np.uint8).reshape(181,320,512,3))
    def old_raw(self,path):return SavedRawBackend.load_raw(self,path,181)
    def read(self,rgb,reader):
        self.validate_rgb(rgb);self.reader_started=time.monotonic()
        try:
            from main.tube_state.b_media_reader_candidate_diagnostic_v1 import observation_trace
            chunks={k:[] for k in ('q','d0','d1','soft','mask')}
            if reader=='torch':
                import torch
                from main.tube_state.b_framewise_qim_v1 import read_rgb
                threads=torch.get_num_threads();torch.set_num_threads(1)
                try:
                    with torch.no_grad():
                        for frame in rgb:
                            self.check()
                            # Original Torch interface takes normalized RGB; normalize exactly once.
                            raw=read_rgb(torch.from_numpy(np.array(frame[None],copy=True)).to(dtype=torch.float64,device='cpu')/255)
                            for key in chunks:chunks[key].append(raw[key].detach().cpu().numpy())
                finally:torch.set_num_threads(threads)
            elif reader=='numpy':
                from main.tube_state.b_framewise_qim_time_recovery_v1 import read_rgb
                for frame in rgb:
                    self.check();raw=read_rgb(frame[None],self.check)
                    for key in chunks:chunks[key].append(raw[key])
            else:raise ValueError('fixed original reader required')
            value={k:np.concatenate(v) for k,v in chunks.items()}
            if any(not np.isfinite(v).all() for k,v in value.items() if k!='mask'):raise ValueError('nonfinite original reader result')
            result=observation_trace(value);self.check();return result
        finally:self.reader_started=None
    def view(self,folder):
        folder=Path(folder);package=dict(compat={},errors={})
        for name in ('compatibility_F.json','compatibility_R0.json','compatibility_R1.json','saved_public.json'):
            try:
                value=json.loads((folder/name).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
                if not isinstance(value,dict):raise ValueError('saved public object required')
                if name=='saved_public.json':package['saved']=value
                else:package['compat'][name[len('compatibility_'):-len('.json')]]=value
            except (OSError,ValueError) as error:package['errors'][name]=str(error)
        return package
    def close(self):return dict(self.resources.close(),wall_seconds=time.monotonic()-self.started,enforcement='RSS watcher; checks between chunks and after blocking calls; no hard interruption guarantee')
