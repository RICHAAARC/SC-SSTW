"""CPU-only media adapter and actual RSS/wall-clock resource limits."""
from contextlib import contextmanager
from pathlib import Path
import json
import subprocess
import threading
import time
import numpy as np
import psutil
from main.tube_state import b_framewise_qim_time_recovery_v1 as method


class ResourceExceeded(RuntimeError): pass


class Resources:
    def __init__(self):
        self.peak_rss=0;self.limit=8*1024**3;self.error=None;self.deadline=None
        self.receive_seconds=0.;self.stop=threading.Event();self.process=psutil.Process()
        self.thread=threading.Thread(target=self._watch,daemon=True);self.thread.start()

    def _watch(self):
        while not self.stop.is_set():
            try:
                rss=self.process.memory_info().rss
                for child in self.process.children(recursive=True):
                    try: rss+=child.memory_info().rss
                    except psutil.NoSuchProcess: pass
                self.peak_rss=max(self.peak_rss,rss)
                if rss>self.limit: self.error='peak resident memory exceeds 8 GiB'
            except psutil.NoSuchProcess: pass
            self.stop.wait(.05)

    def check(self):
        if self.error: raise ResourceExceeded(self.error)
        if self.deadline is not None and time.monotonic()>self.deadline:
            raise ResourceExceeded('fixed receiver wall-clock deadline exceeded')

    @contextmanager
    def receiving(self,seconds):
        start=time.monotonic(); self.deadline=start+min(seconds,6300-self.receive_seconds)
        try:
            self.check();yield;self.check()
        finally:
            self.receive_seconds+=time.monotonic()-start;self.deadline=None

    def close(self):
        self.stop.set();self.thread.join(timeout=1)
        return dict(peak_resident_bytes=self.peak_rss,peak_limit_bytes=self.limit,
                    receiver_seconds=self.receive_seconds,receiver_total_limit_seconds=6300,error=self.error)


def command(argv,resources,input_bytes=None):
    """Every real codec call: 300 seconds, terminate/30 seconds/kill."""
    resources.check(); start=time.monotonic()
    proc=subprocess.Popen(argv,stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        pending=input_bytes
        while True:
            resources.check()
            if time.monotonic()-start>=300: raise ResourceExceeded('codec 300 second deadline exceeded')
            try:
                out,err=proc.communicate(input=pending,timeout=.1);break
            except subprocess.TimeoutExpired: pending=None
        resources.check()
        if proc.returncode: raise RuntimeError(f'codec exit {proc.returncode}: '+err.decode(errors='replace')[-4000:])
        return out
    except BaseException:
        proc.terminate()
        try: proc.communicate(timeout=30)
        except subprocess.TimeoutExpired: proc.kill();proc.communicate()
        raise


class Backend:
    def __init__(self): self.resources=Resources()

    def probe(self,path,frames):
        value=json.loads(command(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=width,height,r_frame_rate,nb_frames,pix_fmt,codec_name','-of','json',str(path)],self.resources))
        streams=value.get('streams',[])
        if len(streams)!=1: raise ValueError('one video stream required')
        s=streams[0]
        if s.get('width')!=512 or s.get('height')!=320 or s.get('r_frame_rate')!='8/1' or int(s.get('nb_frames',-1))!=frames:
            raise ValueError('media geometry/rate/frame-count mismatch: '+str(s))
        return s

    def decode(self,path,frames):
        out=command(['ffmpeg','-v','error','-threads','1','-i',str(path),'-map','0:v:0','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],self.resources)
        if len(out)!=frames*320*512*3: raise ValueError('decoded RGB byte count mismatch')
        return np.frombuffer(out,dtype=np.uint8).reshape(frames,320,512,3)

    def encode(self,rgb,path):
        if rgb.dtype!=np.uint8 or rgb.ndim!=4 or rgb.shape[1:]!=(320,512,3): raise ValueError('uint8 RGB required')
        command(['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-s','512x320','-r','8',
                 '-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-n',str(path)],self.resources,rgb.tobytes())

    def codebook(self):
        with self.resources.receiving(300): return method.codebook(self.resources.check)

    def read(self,rgb): return method.read_rgb(rgb,self.resources.check)

    def receiving_view(self): return self.resources.receiving(600)

    def receive(self,raw,table):
        return method.receive(raw,table,self.resources.check)

    def close(self): return self.resources.close()
