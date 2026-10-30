"""New-media CPU adapter; one shared codebook, inherited bounded codec."""
import time
from main.tube_state import b_minimal_l_v1 as method
from main.tube_state import b_soft_time_evidence_v1 as soft
from .b_framewise_qim_time_recovery_v1 import Backend as MediaBackend


class Backend(MediaBackend):
    def __init__(self):
        super().__init__()
        self.started = time.monotonic()
        self.codec_seconds = 0.

    def _codec(self, fn, *args):
        start = time.monotonic()
        try: return fn(*args)
        finally: self.codec_seconds += time.monotonic()-start

    def probe(self, *args): return self._codec(super().probe, *args)
    def decode(self, *args): return self._codec(super().decode, *args)
    def encode(self, *args): return self._codec(super().encode, *args)

    def codebook(self):
        with self.resources.receiving(300): return method.shared_codebook(self.resources.check)

    def receive(self, raw, table): return super().receive(raw, table[0])

    def C(self, raw, decision, table):
        graph = soft.graph_input(decision, len(raw['soft']))
        return soft.costs(raw['soft'], raw['mask'], graph, table[1], self.resources.check)

    def L(self, cost):
        self.resources.check()
        return soft.local_rank(cost)

    def close(self):
        result = super().close()
        return dict(result, codec_seconds=self.codec_seconds, backend_wall_seconds=time.monotonic()-self.started,
                    codec_call_limit_seconds=300, codec_terminate_seconds=30,
                    receiver_time_excludes_codec=True)
