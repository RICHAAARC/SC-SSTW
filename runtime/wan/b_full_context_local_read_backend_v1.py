"""181-frame encode-only adapter; same shared Wan normalization, no new media operation."""
from runtime.wan.b_time_label_local_channel_backend_v1 import WanBackend
from runtime.wan import b_time_label_tensor_v1 as tensor
from main.tube_state import b_time_label_local_channel_v1 as method
class FullBackend(WanBackend):
    def load_received(self,path):return self.load_pixels(path,181)
    def prepare_full(self,pixels):
        self.validate_pixels(pixels,181)
        return pixels.float()/255
    def encode_full(self,rgb):
        from runtime.wan.vae import reencode_rgb24_readback
        if tuple(rgb.shape)!=(181,320,512,3):raise ValueError('181-frame RGB input required')
        return tensor.validate(reencode_rgb24_readback(self.vae,rgb).detach().cpu().float(),46)
    def read_raw(self,z,key):
        return method.raw_table(tensor.project_rows(z,key,native=True),key,44)
