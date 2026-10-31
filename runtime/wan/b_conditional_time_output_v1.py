"""Original fused hard/C/L and bounded media adapter, plus conditional submission."""
from contextlib import contextmanager
from main.tube_state import b_conditional_time_output_v1 as method
from main.tube_state.b_minimal_l_v1 import candidates
from .b_minimal_l_v1 import Backend as OriginalBackend
from .b_framewise_qim_time_recovery_v1 import Resources


class ConditionalResources(Resources):
    @contextmanager
    def receiving(self, seconds):
        with super().receiving(min(seconds, 3300-self.receive_seconds)):
            yield

    def close(self):
        return dict(super().close(), receiver_total_limit_seconds=3300)


class Backend(OriginalBackend):
    def __init__(self):
        # Preserve the original codec implementation; install the narrower total clock.
        import time
        self.resources = ConditionalResources()
        self.started = time.monotonic()
        self.codec_seconds = 0.

    def semantic(self, hard, local):
        self.resources.check()
        return method.submit(hard, local)

    def candidates(self, hard, cost, local):
        return candidates(hard, cost, local)
