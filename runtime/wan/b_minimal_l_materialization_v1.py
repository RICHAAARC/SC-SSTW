"""Frozen e02 time-only A/B adapters using the exercised VAE/head paths."""
from pathlib import Path
import time
import numpy as np
import torch
from main.tube_state import b_framewise_qim_v1 as p
from main.tube_state import b_framewise_qim_saved_candidate_v1 as checks
from main.tube_state.b_minimal_l_v1 import replica_major_q0,validate_cached_targets
from main.tube_state.b_framewise_qim_v1_solver import solve
from .b_framewise_qim_v1_operator import HeadOperator
from .b_framewise_qim_v1_backend import Backend as OriginalBackend
from .b_framewise_qim_saved_candidate_v1_backend import Backend as SavedBackend
from .b_framewise_qim_time_recovery_v1 import Backend as BoundedMedia
from .vae import quantize_rgb8_no_codec


class CodecGuard:
    # The shared command implements actual per-call deadline/terminate/kill.
    # The CPU receiver's 8 GiB RSS cap does not apply to GPU capture feature storage.
    def check(self): pass


class TimeOnlyMedia:
    def media(self, preclip, arm):
        rgb=torch.from_numpy(np.array(np.load(preclip,mmap_mode='r',allow_pickle=False),copy=True))
        pixels=quantize_rgb8_no_codec(rgb).cpu().numpy()
        adapter=BoundedMedia.__new__(BoundedMedia); adapter.resources=CodecGuard()
        path=self.output/(arm+'.mp4')
        def timed(kind,fn):
            start=time.monotonic()
            try: return self.call(kind,fn)
            finally:
                if not hasattr(self,'codec_times'): self.codec_times=[]
                self.codec_times.append(dict(kind=kind,seconds=time.monotonic()-start,limit_seconds=300,terminate_seconds=30))
        timed('codec_encode',lambda:adapter.encode(pixels,path))
        timed('codec_probe',lambda:adapter.probe(path,181))
        received=timed('codec_decode',lambda:adapter.decode(path,181))
        saved=self.output/(arm+'_h264.npy');np.save(saved,received,allow_pickle=False)
        return str(saved)

    def identity(self,*args): raise RuntimeError('time-only package: E41 is not planned')


class CaptureBackend(TimeOnlyMedia, OriginalBackend):
    def capture(self,terminal):
        original=self.base.load_vae
        def counted_load():
            self.loaded=True  # release a partial actual model load on failure
            return self.call('model_load',original)
        self.base.load_vae=counted_load
        try: return super().capture(terminal)
        finally: self.base.load_vae=original

    def solve_cached(self,baseline,raw_q,progress):
        self.base.vae.to('cpu')
        if self.base.device=='cuda': torch.cuda.empty_cache()
        self.operator=HeadOperator(self.base.vae.decoder.conv_out.weight,self.active,self.event)
        q0=torch.from_numpy(replica_major_q0(raw_q))
        target=self.call('targets',lambda:p.nearest(q0,self.bits))
        np.save(self.output/'fixed_targets.npy',target.numpy(),allow_pickle=False)
        answer=solve(self.operator,(target-q0)/255,self.active[:,:,None].expand(-1,-1,63),
                     self.operator.mask,progress=progress)
        a=answer.pop('coefficients',None)
        valid=(isinstance(a,torch.Tensor) and a.dtype==torch.float64 and a.device.type=='cpu'
               and tuple(a.shape)==self.operator.input_shape and bool(torch.isfinite(a).all()))
        answer['finite_candidate_saved']=bool(valid)
        if valid: torch.save(a,self.output/'coefficients.pt')
        return answer

    def solve(self,*args): raise RuntimeError('q0 must come from the saved first BASE/preclip read')


class CandidateBackend(TimeOnlyMedia, SavedBackend):
    def check_candidate(self):
        def base_projection():
            packed=torch.empty(self.operator.output_shape,dtype=torch.float64)
            for start in range(0,len(self.active),4):
                block=torch.from_numpy(np.array(self.baseline[start:start+4],copy=True))
                if not bool(torch.isfinite(block).all()): raise ValueError('nonfinite saved BASE')
                packed[start:start+len(block)]=p.pack(block.double())
            return 255*p.project(packed)
        q0=self.call('BASE_constraint_P',base_projection)
        if self.targets.shape!=q0.shape or not bool(torch.isfinite(self.targets).all()):
            raise ValueError('finite saved fixed targets matching BASE required')
        # A constructed targets once. B consumes that saved target; no nearest reconstruction.
        validate_cached_targets(q0.numpy(),self.targets.numpy(),self.bits.numpy())
        self.prediction=self.operator.forward(self.coefficients)
        projected=self.call('Aa_constraint_P',lambda:p.project(self.prediction))
        return checks.direct_constraints(q0,projected,self.targets,self.prediction,self.operator.mask,self.active)
