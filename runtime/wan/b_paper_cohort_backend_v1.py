"""Actual OFF/P0/capture and received-only adapters; model loading stays lazy."""
from pathlib import Path
from .b_joint_time_relative_physical_v1 import Backend as Original
from .qmatch_blind_route_time_v1 import Backend as Blind


class GenerationBackend(Original):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.generation_peak_active=False;self.generation_uses_cuda=False;self.generation_peaks=[]

    def generation_peak(self,phase):
        import torch
        row=dict(phase=phase,peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()) if self.generation_uses_cuda else None,
                 cuda_measured=self.generation_uses_cuda)
        self.generation_peaks.append(row)
        from .paper_checkpoint_v1 import resource_record
        resource_record(self.output,'generation_resources',dict(scope='generation load plus both OFF/P0 trajectories; before capture reset',
            measurements=self.generation_peaks,generation_complete=any(r['phase']=='P0_RETURNED' for r in self.generation_peaks)))

    def generation_load(self):
        import torch
        self.generation_uses_cuda=bool(torch.cuda.is_available())
        if self.generation_uses_cuda: torch.cuda.reset_peak_memory_stats()
        self.generation_peak_active=True
        try:
            result=super().generation_load()
            path=self.scratch/'generation_context.pt'
            self.scratch.mkdir(parents=True,exist_ok=True)
            pipe,initial,prompt,negative,dtype=self.native.wan.context
            if path.is_file():
                saved=torch.load(path,map_location='cpu',weights_only=True)
                self.native.wan.context=(pipe,saved['initial'].to(initial.device),saved['prompt'].to(prompt.device),saved['negative'].to(negative.device),dtype)
            else:torch.save(dict(initial=initial.detach().cpu(),prompt=prompt.detach().cpu(),negative=negative.detach().cpu()),path)
            return result
        finally: self.generation_peak('LOAD_FINISHED_OR_FAILED')

    def vae_load(self):
        if self.generation_peak_active:
            self.generation_peak('BEFORE_CAPTURE_RESET');self.generation_peak_active=False
        return super().vae_load()

    def generate_arm(self,arm):
        import torch
        traces=[]
        def record(row):
            traces.append(row);self.dump(self.output/f'private/{arm}_trace.json',traces)
        z,receipt=self.native.wan.generate(arm,self.config['writer'],self.event,record)
        self.scratch.mkdir(parents=True,exist_ok=True)
        torch.save(z,self.scratch/f'{arm}_terminal.pt')
        self.dump(self.output/f'private/{arm}_generation.json',receipt)
        if self.generation_peak_active: self.generation_peak(arm+'_RETURNED')
        if arm=='P0':
            self.dump(self.output/'private/writer.json',self.config['writer'])
            self.native.wan.release_generation();self.generation_loaded=False
        return receipt

    def decode_off(self):
        import numpy as np
        import torch
        from .vae import decode_normalized_latent
        z=torch.load(self.scratch/'OFF_terminal.pt',map_location='cpu',weights_only=True)
        if z.dtype!=torch.float32 or tuple(z.shape)!=(1,16,46,40,64) or not bool(torch.isfinite(z).all()):
            raise ValueError('finite original OFF FP32 terminal required')
        self.event('D181',False)
        with torch.inference_mode(): rgb=decode_normalized_latent(self.native.wan.vae,z.to(self.native.wan.device))
        self.event('D181',True)
        # Ordinary decoder output; no baseline head capture hook is installed here.
        np.save(self.scratch/'OFF_RGB.npy',rgb.detach().cpu().float().numpy(),allow_pickle=False)
        return dict(frames=len(rgb),ordinary_D181=True,head_capture=False)

    def capture(self):
        return self.native.capture(self.scratch/'P0_terminal.pt',self.scratch/'host',self.event)

    def target(self):
        from main.tube_state import b_joint_time_packet_a_v1 as packet
        from .qmatch_blind_route_time_v1 import replace
        original=packet.encode_raw24
        def encode(message):
            self.event('writer_BCH_encode',False)
            result=original(message)
            self.event('writer_BCH_encode',True)
            return result
        with replace(packet,'encode_raw24',encode): return super().target()

    def restore_solver(self):
        import numpy as np
        import torch
        from .b_joint_time_packet_a_write_cpu_v1 import Backend as Solver,load_head
        from .b_framewise_qim_v1_operator import HeadOperator
        from .b_paper_cohort_records_v1 import read
        folder=self.output/'cpu'
        self.cpu=Solver(load_head(self.output/'private/head.pt'),folder,self.event,self.dump)
        if (folder/'q0.npy').is_file():self.cpu.q0=torch.from_numpy(np.load(folder/'q0.npy',allow_pickle=False))
        if (folder/'schedule.npz').is_file():
            with np.load(folder/'schedule.npz',allow_pickle=False) as rows:self.cpu.fields={k:torch.from_numpy(rows[k].copy()) for k in rows.files}
            self.cpu.targets=torch.from_numpy(np.load(folder/'fixed_targets.npy',allow_pickle=False))
            self.cpu.operator=HeadOperator(self.cpu.head['weight'],self.cpu.fields['active'],self.event)
            self.cpu.operator.calls={key:self.event.rows[key]['started'] for key in ('K','KT')}
        if (folder/'solver.json').is_file():
            self.cpu.answer=read(folder/'solver.json');self.cpu.eligible=self.cpu.answer.get('native_handoff_eligible') is True
            if self.cpu.eligible:self.cpu.coefficients=torch.load(folder/'coefficients.pt',map_location='cpu',weights_only=True)

    def preclip(self,arm):
        import numpy as np
        relative={'OFF':'OFF_RGB.npy','P0':'host/BASE_preclip.npy','NEW':'new/NEW_preclip.npy'}[arm]
        return np.load(self.scratch/relative,mmap_mode='r',allow_pickle=False)

    def close(self):
        if self.generation_peak_active:
            self.generation_peak('GENERATION_CLOSED_BEFORE_CAPTURE');self.generation_peak_active=False
        return super().close()


class TimeBackend(Blind):
    """One receiver resource object for hard AND relative; no old whole-attempt gates."""
    def __init__(self,config,output,scratch,event,dump):
        self.output=Path(output);self.dump=dump;self.bch_limit=36990
        from .b_paper_cohort_records_v1 import read
        path=self.output/'internal_calls.json'
        self.internal=read(path) if path.is_file() else {}
        state=read(self.output/'state.json') if (self.output/'state.json').is_file() else {}
        if not isinstance(self.internal,dict):raise ValueError('invalid saved internal receiver counts')
        for key,row in state.items():
            if not key.startswith('time/'):continue
            fields=('bch_started','bch_returned','role_packets_started','role_packets_returned') if key.endswith('/hard_receive') else ('f42','relative') if key.endswith('/relative_compare') else ()
            complete=row['status']=='RETURNED' or (self.output/'checkpoints/returns'/(key+'.json')).is_file()
            if fields and complete and not all(field in self.internal.get(key.split('/')[1],{}) for field in fields):
                raise ValueError('saved receiver internal consumption unavailable; cannot reset to zero')
        self.original=GenerationBackend(config,output,scratch,event,dump)
        medium=self.original.media_backend()
        medium.resources.receiving_total_seconds=9450
        self.original_check=medium.resources.check
    def check(self): self.original_check()
    def preclip(self,arm): return self.original.preclip(arm)
    def quantize(self,value): return self.original.quantize(value)
    def encode(self,pixels,path): return self.original.encode(pixels,path)
    def decode(self,path,frames): return self.original.decode(path,frames)
    def close(self): return self.original.close()


def identity_backend(model_id):
    from .b_relative_time_payload_backend_v1 import Backend
    return Backend(model_id)
