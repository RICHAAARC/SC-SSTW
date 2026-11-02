"""Native plane extraction, without RGB or automatic filter conversion."""
from pathlib import Path
import json
import time
import numpy as np
from .b_media_reader_candidate_diagnostic_v1 import Backend as ResourceBackend
from main.tube_state import b_native_y_diagnostic_v1 as method
from main.tube_state import b_framewise_qim_time_recovery_v1 as hard


def decode_command(path):
    return ['ffmpeg','-v','error','-noauto_conversion_filters','-threads','1','-i',str(path),
            '-map','0:v:0','-vsync','0','-vf','extractplanes=y','-f','rawvideo','-pix_fmt','gray','pipe:1']


class Backend(ResourceBackend):
    def probe(self,path,frames):
        value=json.loads(self.codec(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=width,height,r_frame_rate,nb_frames,pix_fmt,codec_name,color_range,color_space,color_transfer,color_primaries,bits_per_raw_sample',
            '-of','json',str(path)]))
        streams=value.get('streams',[])
        if len(streams)!=1:raise ValueError('one video stream required')
        s=streams[0]
        # Return metadata even when geometry/format is unusable; decode reports that failure.
        return dict(stream=s,profile=method.profile(s),expected_frames=frames)
    def decode(self,path,frames,probe):
        s=probe['stream']
        if s.get('width')!=512 or s.get('height')!=320 or s.get('r_frame_rate')!='8/1' or int(s.get('nb_frames',-1))!=frames:
            raise ValueError('fixed media geometry/rate/frame count mismatch')
        if s.get('pix_fmt') not in ('yuv420p','yuvj420p'):raise ValueError('native planar 8-bit 420 Y required')
        data=self.codec(decode_command(path))
        if len(data)!=frames*320*512:raise ValueError('decoded native Y byte count mismatch')
        return np.frombuffer(data,dtype=np.uint8).reshape(frames,320,512)
    def read(self,y):
        self.reader_started=time.monotonic()
        try:
            value=method.read_y(y,self.check);self.check();return value
        finally:self.reader_started=None
    def codebook(self):return hard.codebook(self.check)
    def compatible(self,raw,table):
        value,merged,known=hard.compatibility(raw['soft'],raw['mask'],table,self.check)
        method.validate_labels(value,len(raw['soft']))
        if value['comparisons']>len(raw['soft'])*hard.DOMAIN:raise ValueError('fixed comparison upper bound exceeded')
        return value
    def public(self,path,frames):
        value=json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
        if not isinstance(value,dict):raise ValueError('saved public object required')
        # Baseline-only allowlist. Never passed to read/compatibility/path.
        result={k:value[k] for k in ('semantic.json','blind.json','submission.json','candidates.json','errors') if k in value}
        if 'semantic.json' in result:
            result['semantic.json']={'hard':result['semantic.json'].get('hard')}
        return result
