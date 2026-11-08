"""Actual saved-media E41/DCT2 adapter. No writer, time reread or truth inputs."""
import gc
import numpy as np
import torch
from .b_terminal_three_group_backend_v1 import WanBackend
from .b_framewise_qim_time_recovery_v1 import Backend as MediaBackend


class Backend:
    def __init__(self, model_id):
        self.media = MediaBackend()
        self.wan = WanBackend(model_id)

    def check(self): self.media.resources.check()

    def load(self):
        if self.wan.device == 'cuda': torch.cuda.reset_peak_memory_stats()
        return self.wan.load_vae()

    def probe(self, path, length):
        value = self.media.probe(path, length)
        if value.get('codec_name') != 'h264' or value.get('pix_fmt') != 'yuv420p':
            raise ValueError('saved second-generation H264 yuv420p received media required')
        return value

    def decode(self, path, length):
        return self.media.decode(path, length)

    def encode(self, pixels, start, length):
        if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.shape != (length,320,512,3):
            raise ValueError('actual received uint8 RGB geometry required')
        if type(start) is not int or not 0 <= start <= length-41:
            raise ValueError('received E41 bounds required')
        # Actual received order; never invoke old fixed-151 prepare_window.
        window = torch.from_numpy(pixels[start:start+41].copy())
        with torch.inference_mode():
            return self.wan.encode_view(window)

    def project(self, latent):
        return self.wan.payload_project(latent, list(range(1,11)), ['R0','R1'])

    def bits(self, projection, view_id):
        return self.wan.payload_decode(projection, 'watermark', view_id, 'key00')

    def close(self):
        peak = int(torch.cuda.max_memory_allocated()) if self.wan.device == 'cuda' else None
        try: self.wan.release_vae()
        finally: resources = self.media.close()
        gc.collect()
        return dict(resources, peak_cuda_allocated_bytes=peak, device=self.wan.device)


def flatten_cache(decoded):
    """Use the original DCT2 bit records; no new votes, labels or group assignment."""
    result = {}
    for region in ('R0','R1'):
        for row in decoded['regions'][region]['local_cells']:
            result.setdefault(row['evidence_id'], {})[decoded['key']] = {str(b['local_bit']):b for b in row['bits']}
    return result
