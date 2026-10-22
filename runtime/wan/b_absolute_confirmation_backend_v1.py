"""Card99 real adapter reuses successful Card94 physical calls, never imported by fake checks."""
from runtime.wan.b_fixed_detection_confidence_backend_v1 import WanBackend as PreviousBackend
from main.tube_state import b_phase_centered_saved_v1 as arithmetic


class WanBackend(PreviousBackend):
    def reduce_candidate(self,votes,source_start,counters):
        # Exactly one unique-max candidate. Original c/r/k encounter order,
        # strict >0 and Counter tie semantics are unchanged in reduce_all.
        table=dict(candidates=[arithmetic.candidate(source_start)])
        arithmetic.reduce_all(votes,table,counters)
        return table['candidates'][0]
