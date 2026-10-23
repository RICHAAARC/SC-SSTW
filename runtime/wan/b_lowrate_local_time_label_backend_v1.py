"""Lazy physical adapter for future user Run-all; fixed E45 is an actual pixel encode."""
from collections import Counter
from pathlib import Path
from runtime.wan.b_lfg_posthoc_backend_v1 import LFGBackend
from main.tube_state import b_lowrate_local_time_label_v1 as core
from runtime.wan import b_time_label_tensor_v1 as tensor

class Backend(LFGBackend):
    def initialize_PN(self):return core.initialize()
    def construct_tag(self,z,key):
        import torch
        source=tensor.validate(z,46);before=source.clone();marked,receipt=core.construct(source[0].numpy(),key)
        if not torch.equal(source,before):raise ValueError('source modified in-place')
        return tensor.validate(torch.from_numpy(marked[None].copy()),46),receipt
    def tag_lll(self,z,local=False):return core.lll(tensor.validate(z,12 if local else 46)[0].numpy())
    def tag_scores(self,low,key,rows):return core.observe(low,key,rows)
    def crop_received(self,pixels,origin):
        self.validate_pixels(pixels,181)
        if type(origin) is not int or not 0<=origin<=129:raise ValueError('private prep origin0..129')
        return self.validate_pixels(pixels[origin:origin+52].clone(),52)
    def prepare_local(self,received,phase):
        self.validate_pixels(received,52)
        if type(phase) is not int or phase not in range(8):raise ValueError('public phase0..7')
        return self.validate_pixels(received[phase:phase+45].clone(),45)
    def encode_E45(self,pixels):
        from runtime.wan.vae import reencode_rgb24_readback
        self.validate_pixels(pixels,45)
        return tensor.validate(reencode_rgb24_readback(self.vae,pixels.float()/255).detach().cpu().float(),12)
    def local_payload_spectrum(self,z):
        import torch
        tensor.validate(z,12)
        return torch.fft.fft2(z[0,:4,1:12].float(),dim=(-2,-1),norm='ortho').real
    def local_payload_votes(self,spectrum,key):
        import torch
        from main.tube_state.b_legacy_position_margin_v1 import grow
        if tuple(spectrum.shape)!=(4,11,40,64) or not bool(torch.isfinite(spectrum).all()):raise ValueError('finite E45 spectrum[4,11,40,64]')
        coords=grow.coordinates(key);x=spectrum[:,:,[c[0] for c in coords],[c[1] for c in coords]]
        signed=(2*(x>0).to(torch.int8)-1).cpu().tolist();zeros=(x==0).to(torch.int8).cpu().tolist();cells=[]
        for t in range(11):
            row=[]
            for b in range(8):
                seq=[int(signed[c][t][i]>0) for c in range(4) for i in range(b,240,8)]
                row.append(dict(slot=b,decoded=Counter(seq).most_common(1)[0][0],ones=sum(seq),count=120,tie=sum(seq)*2==120,zero_coefficients=sum(zeros[c][t][i] for c in range(4) for i in range(b,240,8))))
            cells.append(dict(local_row=t+1,bits=row))
        return dict(status='LOCAL_RAW_PAYLOAD',local_cells=cells,signed_votes=signed,zero_coefficient_mask=zeros,raw_bit_slots=88,truth_inputs=False,message_inputs=False,aggregate32=None)
