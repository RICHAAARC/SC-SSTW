"""Saved-Y IO only; original C/L/conditional pure functions remain unchanged."""
from pathlib import Path
import json
import time
import numpy as np
from main.tube_state import b_soft_time_evidence_v1 as soft
from main.tube_state import b_minimal_l_v1 as minimal
from main.tube_state import b_conditional_time_output_v1 as conditional
from main.tube_state.b_framewise_qim_time_recovery_v1 import DOMAIN
from .b_framewise_qim_time_recovery_v1 import Resources, ResourceExceeded
from .b_time_layer_diagnostic_v1 import RawLimitExceeded


def load_json(path):
    value = json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError('saved JSON object required')
    return value


def validate_raw(raw, frames):
    shape = (frames, 5, 2, 63)
    for key in ('q', 'd0', 'd1', 'soft'):
        value = raw[key]
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError(key + ': finite FP64 [R,5,2,63] required')
    if raw['mask'].shape != shape or raw['mask'].dtype != np.bool_:
        raise ValueError('original boolean mask [R,5,2,63] required')
    q = raw['q']
    with np.errstate(over='raise', invalid='raise'):
        for bit, key in enumerate(('d0', 'd1')):
            low = np.floor((q-16*bit)/32)*32+16*bit
            expected = np.abs(q-np.where(q-low <= 16, low, low+32))
            if not np.allclose(raw[key], expected, atol=1e-12, rtol=1e-12):
                raise ValueError(key + ': original 32/16 lattice distance mismatch')
        expected = (raw['d0']**2-raw['d1']**2)/256
        if not np.allclose(raw['soft'], expected, atol=1e-12, rtol=1e-12):
            raise ValueError('saved soft unit mismatch')
    return raw  # Never replace soft, erase zero soft, fuse, or convert dtype.


def flatten_codebook(saved, check=lambda: None):
    flat = np.empty(DOMAIN, dtype=np.uint64)
    covered = np.zeros(DOMAIN, dtype=bool)
    for lane in range(5):
        check()
        js, words = saved['j_'+str(lane)], saved['word_'+str(lane)]
        if (js.ndim != 1 or words.shape != js.shape or js.dtype.kind not in 'iu'
                or words.dtype.kind not in 'iu' or np.any(js < 0) or np.any(js >= DOMAIN)
                or np.any(js % 5 != lane) or np.any(words < 0) or np.any(words > (1 << 63)-1)):
            raise ValueError('saved codebook requires legal integer j/lane/63-bit words')
        indexes = js.astype(np.int64)
        if len(np.unique(indexes)) != len(indexes) or covered[indexes].any():
            raise ValueError('duplicate public j')
        flat[indexes] = words
        covered[indexes] = True
    if not covered.all():
        raise ValueError('saved codebook must cover j=0..92671 once')
    return flat


def validate_hard(hard, frames):
    graph = soft.graph_input(hard, frames)
    # Reuse submit's complete hard semantic validator without running a submission
    # or constructing candidates. Undefined placeholders cannot add outputs.
    empty = dict(qualification=hard['qualification'], epsilon=soft.EPSILON, nodes=[
        dict(received_frame=r, model_status='NO_SOFT_OBSERVATION', L_status='NO_SOFT_OBSERVATION',
             best_count=0, model_index=None, candidate_index=None) for r in range(frames)])
    conditional._validate(hard, empty)
    return graph


class Backend:
    def __init__(self):
        self.resources = Resources()
        self.started = time.monotonic()
        self.view_started = None

    def check(self):
        self.resources.check()
        now = time.monotonic()
        if now-self.started > 6300:
            raise ResourceExceeded('whole saved-Y conditional attempt exceeds 6300 seconds')
        if self.view_started is not None and now-self.view_started > 600:
            raise RawLimitExceeded('saved-Y view exceeds 600 seconds')

    def begin_view(self):
        self.view_started = time.monotonic()

    def end_view(self):
        self.view_started = None

    def codebook(self, path):
        with np.load(path, allow_pickle=False) as saved:
            return flatten_codebook(saved, self.check)

    def raw(self, path, frames):
        with np.load(path, allow_pickle=False) as saved:
            raw = {key: saved[key] for key in ('q', 'd0', 'd1', 'soft', 'mask')}
        return validate_raw(raw, frames)

    def hard(self, path, frames):
        value = load_json(path)
        graph = validate_hard(value, frames)
        return value, graph

    def C(self, raw, graph, table):
        return soft.costs(raw['soft'], raw['mask'], graph, table, self.check)

    def L(self, graph, cost):
        local = soft.local_rank(cost)
        return local, minimal.candidates(graph, cost, local)

    def submit(self, hard, candidates):
        return conditional.submit(hard, candidates)

    def public(self, path):
        value = load_json(path)
        result = {k: value[k] for k in ('semantic.json', 'blind.json', 'submission.json', 'errors') if k in value}
        if isinstance(result.get('semantic.json'), dict):
            result['semantic.json'] = {'hard': result['semantic.json'].get('hard')}
        return result

    def close(self):
        return dict(self.resources.close(), wall_seconds=time.monotonic()-self.started,
                    enforcement='RSS watcher; checks between calls and C chunks; after blocking calls; no hard interruption guarantee')
