"""Thin actual c01 E41 adapter with two real keys and plain canonical projections."""
from .b_time_payload_cdc32_v1 import Backend as Previous, flatten_cache

KEYS = dict(key00='watermark', key01='watermark-wrong')


def plain(value):
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if hasattr(value, 'tolist'): return value.tolist()
    return value


class Backend(Previous):
    def project(self, latent):
        return plain(super().project(latent))

    def bits(self, projection, view_id, key_id):
        return self.wan.payload_decode(projection, KEYS[key_id], view_id, key_id)

    def close(self):
        result = super().close()
        # No receiving() context was entered; the old observer's 6300 is not a limit here.
        result.pop('receiver_total_limit_seconds', None)
        result['codec_per_call_seconds'] = 300
        result['time_receiver_context_entered'] = False
        return result
