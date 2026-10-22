"""Card101 physical adapter; unchanged source/media/Q/votes from accepted Card99."""
from runtime.wan.b_absolute_confirmation_backend_v1 import WanBackend as PreviousBackend


class WanBackend(PreviousBackend):
    def score(self,*args,**kwargs):
        raise RuntimeError('legacy F/V-min scorer is not part of Card101 runtime')
