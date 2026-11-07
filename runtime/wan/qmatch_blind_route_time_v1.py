"""Original native-Y/hard/F42/relative calls with passive branch accounting."""
from contextlib import contextmanager
from pathlib import Path
import time

RESOURCE_LIMITS = dict(process_tree_rss_bytes=8*1024**3, codec_call_seconds=300,
    hard_receive_seconds=600, relative_seconds=600, receiving_total_seconds=6300,
    attempt_wall_seconds=7200, output_disk_bytes=2*1024**3)


@contextmanager
def replace(module, name, value):
    original=getattr(module,name);setattr(module,name,value)
    try:yield
    finally:setattr(module,name,original)


class Backend:
    def __init__(self, output, dump):
        from .b_joint_time_relative_physical_v1 import Backend as Original
        self.output=Path(output);self.started=time.monotonic();self.last_disk_check=0.
        self.disk_bytes=0;self.dump=dump
        self.original=Original({},output,self.output/'unused_scratch',lambda *args:None,dump)
        medium=self.original.media_backend()
        self.original_check=medium.resources.check
        medium.resources.check=self.check
        self.internal={}

    def check(self):
        from .b_framewise_qim_time_recovery_v1 import ResourceExceeded
        self.original_check()
        now=time.monotonic()
        if now-self.started>RESOURCE_LIMITS['attempt_wall_seconds']:
            raise ResourceExceeded('fixed blind-route attempt wall limit exceeded')
        if now-self.last_disk_check>=1:
            self.disk_bytes=sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())
            self.last_disk_check=now
        if self.disk_bytes>RESOURCE_LIMITS['output_disk_bytes']:
            raise ResourceExceeded('fixed blind-route output disk limit exceeded')

    def probe(self, path, frames):
        return self.original.probe(path,frames)

    def decode_y(self, path, frames, probe):
        return self.original.decode_y(path,frames,probe)

    def hard_receive(self, y, folder, media_id):
        from main.tube_state import b_joint_time_packet_a_v1 as packet
        from main.tube_state import b_joint_time_packet_a_receiver_v1 as receiver
        counts=self.internal.setdefault(media_id,{})
        counts.update(role_packets_started=0,role_packets_returned=0,bch_started=0,bch_returned=0)
        original_decode,original_observation=packet.decode_raw24,receiver._observation
        def decode(bits):
            self.check()
            if sum(c.get('bch_started',0) for c in self.internal.values())>=getattr(self,'bch_limit',21150):
                raise RuntimeError('fixed BCH call ceiling consumed')
            counts['bch_started']+=1
            value=original_decode(bits);counts['bch_returned']+=1
            return value
        def observation(*args,**kwargs):
            counts['role_packets_started']+=1
            value=original_observation(*args,**kwargs);counts['role_packets_returned']+=1
            return value
        try:
            with replace(packet,'decode_raw24',decode),replace(receiver,'_observation',observation):
                value=self.original.hard_receive(y,folder)
            if (value['logical_counts']['raw24_decode']!=counts['bch_returned']
                    or value['logical_counts']['planned_role_packets']!=counts['role_packets_returned']):
                raise RuntimeError('returned original packet branch accounting mismatch')
            return value
        finally:self.dump(self.output/'internal_calls.json',self.internal)

    def relative(self, packet, media_id):
        from main.tube_state import b_joint_time_packet_a_relative_v1 as relative
        fcounts,rcounts=relative.f42.counters(),relative.counters()
        self.internal.setdefault(media_id,{}).update(f42=fcounts,relative=rcounts)
        try:
            with self.original.media_backend().resources.receiving(600):
                with replace(relative.f42,'counters',lambda:fcounts),replace(relative,'counters',lambda:rcounts):
                    return relative.diagnose(packet,media_id,self.check)
        finally:self.dump(self.output/'internal_calls.json',self.internal)

    def close(self):
        return dict(original=self.original.close(),limits=RESOURCE_LIMITS,
                    observed_output_bytes=self.disk_bytes,wall_seconds=time.monotonic()-self.started)
