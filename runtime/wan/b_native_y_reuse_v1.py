"""Original native-Y and conditional CPU adapters; no experiment imports."""
import numpy as np
from main.tube_state import b_native_y_diagnostic_v1 as method
from .b_native_y_diagnostic_v1 import Backend as NativeBackend
from .b_native_y_conditional_v1 import Backend as ConditionalBackend, flatten_codebook, validate_hard, load_json
from .b_time_layer_diagnostic_v1 import RawLimitExceeded
import time


class Backend(NativeBackend):
    C, L, submit = ConditionalBackend.C, ConditionalBackend.L, ConditionalBackend.submit
    begin_view, end_view = ConditionalBackend.begin_view, ConditionalBackend.end_view

    def __init__(self):
        super().__init__()
        self.view_started = None

    def check(self):
        super().check()
        if self.view_started is not None and time.monotonic()-self.view_started > 600:
            raise RawLimitExceeded('whole view 600s limit observed')

    def saved_codebook(self, path):
        with np.load(path, allow_pickle=False) as saved:
            flat = flatten_codebook(saved, self.check)
            buckets = [(saved[f'j_{i}'], saved[f'word_{i}']) for i in range(5)]
        return buckets, flat

    def H(self, compatible, frames):
        hard = method.path(compatible, frames)
        return hard, validate_hard(hard, frames)

    def historical(self, folder):
        result = dict(errors={})
        for key, name in (('hard', 'blind.json'), ('candidates', 'candidates.json'), ('preservation', 'formal_preservation.json')):
            try: result[key] = load_json(folder/name)
            except (OSError, ValueError, TypeError) as error: result['errors'][name] = str(error)
        return result
