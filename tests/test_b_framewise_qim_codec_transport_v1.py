"""Codec transport regression: artificial bytes/video and controlled child limits."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import threading
import time
import numpy as np
import pytest
from runtime.wan import b_framewise_qim_time_recovery_v1 as media
from runtime.wan.b_minimal_l_v1 import Backend

pytestmark=pytest.mark.unit


class Guard:
    def __init__(self): self.checks=0;self.started=time.monotonic()
    def check(self):
        self.checks+=1
        if time.monotonic()-self.started>5: raise media.ResourceExceeded('engineering safety bound')


def test_delayed_reader_receives_full_large_input_and_closes_stdin(monkeypatch):
    payload=bytes(range(256))*4096
    children=[];real_popen=subprocess.Popen
    class Process(real_popen):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.exchanges=0;children.append(self)
        def communicate(self,*args,**kwargs):
            self.exchanges+=1
            assert 'timeout' not in kwargs
            return super().communicate(*args,**kwargs)
    monkeypatch.setattr(media.subprocess,'Popen',Process)
    program=('import sys,time; time.sleep(.25); '
             'sys.stderr.buffer.write(b"e"*262144); sys.stderr.buffer.flush(); '
             'data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data)')
    guard=Guard();out=media.command([sys.executable,'-c',program],guard,payload)
    assert out==payload and guard.checks>=4
    assert len(children)==1 and children[0].exchanges==1 and children[0].poll()==0
    assert all(stream.closed for stream in (children[0].stdin,children[0].stdout,children[0].stderr))
    assert not any(t.name=='video-wm-codec-io' for t in threading.enumerate())


@pytest.mark.parametrize('failure',['deadline','rss','interrupt'])
def test_limits_terminate_kill_and_join_actual_stalled_child(tmp_path,monkeypatch,failure):
    ready=tmp_path/'ready';actions=[];children=[];real_popen=subprocess.Popen;clock=time.monotonic
    class Process(real_popen):
        def __init__(self,*args,**kwargs): super().__init__(*args,**kwargs);children.append(self)
        def terminate(self): actions.append('terminate');return super().terminate()
        def kill(self): actions.append('kill');return super().kill()
        def wait(self,timeout=None):
            if timeout==30:
                actions.append(('requested_grace',30))
                return super().wait(timeout=.15)  # short controlled engineering grace
            return super().wait(timeout=timeout)
    monkeypatch.setattr(media.subprocess,'Popen',Process)
    if failure=='deadline':
        monkeypatch.setattr(media,'time',SimpleNamespace(monotonic=lambda:clock()+(301 if ready.exists() else 0)))
    if failure=='rss':
        resources=media.Resources()
        resources.limit=resources.process.memory_info().rss+16*1024**2
        resources.deadline=clock()+5  # test safety bound; actual RSS failure must fire first
    else:
        class ResourceGuard(Guard):
            def check(self):
                super().check()
                if failure=='interrupt' and ready.exists(): raise KeyboardInterrupt('controlled cancellation')
        resources=ResourceGuard()
    program=('import signal,time; from pathlib import Path; '
             'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
             f'Path({str(ready)!r}).write_text("ready"); '
             'allocation=bytearray(64*1024**2); time.sleep(10)')
    expected=KeyboardInterrupt if failure=='interrupt' else media.ResourceExceeded
    message={'deadline':'300 second','rss':'peak resident memory','interrupt':'controlled cancellation'}[failure]
    started=clock()
    try:
        with pytest.raises(expected,match=message):
            media.command([sys.executable,'-c',program],resources,b'x'*1048576)
    finally:
        if failure=='rss': resources.close()
    assert clock()-started<3
    assert actions==['terminate',('requested_grace',30),'kill']
    assert len(children)==1 and children[0].poll()==-9
    assert all(stream.closed for stream in (children[0].stdin,children[0].stdout,children[0].stderr))
    assert not any(t.name=='video-wm-codec-io' for t in threading.enumerate())


def test_actual_ffmpeg_artificial_video_roundtrip_and_no_overwrite(tmp_path):
    frames=12;rgb=np.empty((frames,320,512,3),dtype=np.uint8)
    y,x=np.indices((320,512))
    for frame in range(frames):
        rgb[frame,:,:,0]=(x+17*frame)%256
        rgb[frame,:,:,1]=(y+11*frame)%256
        rgb[frame,:,:,2]=(x//2+y//2+23*frame)%256
    path=tmp_path/'artificial.mp4';backend=Backend()
    try:
        backend.encode(rgb,path)
        info=backend.probe(path,frames);decoded=backend.decode(path,frames)
        assert info['codec_name']=='h264' and info['pix_fmt']=='yuv420p' and info['r_frame_rate']=='8/1'
        assert decoded.shape==rgb.shape and decoded.dtype==np.uint8
        assert not np.array_equal(decoded[0],decoded[-1])
        before=path.read_bytes()
        with pytest.raises(FileExistsError,match='existing media output'): backend.encode(rgb,path)
        assert path.read_bytes()==before
    finally: report=backend.close()
    assert report['error'] is None and report['codec_seconds']>0 and report['receiver_seconds']==0
    assert not any(t.name=='video-wm-codec-io' for t in threading.enumerate())


def test_nonzero_child_exit_keeps_error_and_closes_io(monkeypatch):
    children=[];real_popen=subprocess.Popen
    class Process(real_popen):
        def __init__(self,*args,**kwargs): super().__init__(*args,**kwargs);children.append(self)
    monkeypatch.setattr(media.subprocess,'Popen',Process)
    with pytest.raises(RuntimeError,match='codec exit 7: artificial failure'):
        media.command([sys.executable,'-c','import sys;sys.stderr.write("artificial failure");sys.exit(7)'],Guard())
    assert children[0].poll()==7 and children[0].stdout.closed and children[0].stderr.closed
    assert not any(t.name=='video-wm-codec-io' for t in threading.enumerate())
