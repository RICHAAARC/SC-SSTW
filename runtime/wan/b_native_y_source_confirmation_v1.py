"""Thin native-Y adapter with source RGB fixture construction and one codebook."""
from pathlib import Path
import numpy as np
from .b_native_y_reuse_v1 import Backend as ReuseBackend
from .b_native_y_conditional_v1 import flatten_codebook
from .b_framewise_qim_time_recovery_v1 import Backend as RGBBackend


class Backend(ReuseBackend):
    def __init__(self):
        super().__init__()
        self.rgb = RGBBackend.__new__(RGBBackend)
        self.rgb.resources = self  # Same RSS/6300s check and codec deadline, no second watcher.

    def source_probe(self, path):
        return self.rgb.probe(path, 181)

    def source_decode(self, path):
        return self.rgb.decode(path, 181)

    def encode_view(self, pixels, path):
        return self.rgb.encode(pixels, path)

    def make_codebook(self, path):
        buckets = super().codebook()
        saved = {name: value for lane, (js, words) in enumerate(buckets)
                 for name, value in ((f'j_{lane}', js), (f'word_{lane}', words))}
        np.savez(Path(path), **saved)
        return buckets, flatten_codebook(saved, self.check)
