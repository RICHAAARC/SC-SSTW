"""Exact restricted causal head and its transpose. No model loading."""
import torch
from torch.nn import functional as F
from main.tube_state import b_framewise_qim_v1 as p


class HeadOperator:
    def __init__(self, weight, active, event=None):
        w = weight.detach().cpu().double()
        if w.ndim != 5 or w.shape[0] != 3 or tuple(w.shape[2:]) != (3, 3, 3) or not bool(torch.isfinite(w).all()):
            raise ValueError('finite RGB causal 3x3x3 head weights required')
        if active.ndim != 2 or active.shape[1] != 10 or active.dtype != torch.bool: raise ValueError('active source T by 10 tiles')
        self.active = active.cpu(); self.mask = p.support(self.active)
        self.frames, self.channels = len(active), w.shape[1]
        self.input_shape = (self.frames, 10, self.channels, 7, 9)
        self.output_shape = (self.frames, 10, 3, 58, 74)
        self.event = event; self.calls = dict(K=0, KT=0)
        self.bound = .5*sum(float(torch.linalg.matrix_norm(w[:, :, t, y, x], 2)) for t in range(3) for y in range(3) for x in range(3))
        self.kernel = torch.zeros((3, self.channels, 3, 10, 10), dtype=torch.float64)
        atom = p.basis()
        for lag in range(3):
            for y in range(3):
                for x in range(3):
                    self.kernel[lag, :, :, 2-y:10-y, 2-x:10-x] += .5*w[:, :, 2-lag, y, x].T[:, :, None, None]*atom

    def _count(self, kind, done):
        if not done:
            if self.calls[kind] >= 4201: raise RuntimeError('head operator fixed call budget exhausted')
            self.calls[kind] += 1
        if self.event: self.event(kind, done)

    def forward(self, a):
        self._count('K', False)
        if tuple(a.shape) != self.input_shape: raise ValueError('coefficient geometry')
        out = a.new_zeros(self.output_shape)
        for lag in range(3):
            for start in range(0, self.frames-lag, 4):
                stop = min(start+4, self.frames-lag)
                coeff = a[start:stop]*self.active[start:stop, :, None, None, None]
                values = F.conv_transpose2d(coeff.reshape(-1, self.channels, 7, 9), self.kernel[lag], stride=8)
                out[start+lag:stop+lag] += values.reshape(stop-start, 10, 3, 58, 74)
        out *= self.mask
        self._count('K', True)
        return out

    def adjoint(self, r):
        self._count('KT', False)
        if tuple(r.shape) != self.output_shape: raise ValueError('residual geometry')
        out = r.new_zeros(self.input_shape)
        for lag in range(3):
            for start in range(0, self.frames-lag, 4):
                stop = min(start+4, self.frames-lag)
                values = r[start+lag:stop+lag]*self.mask[start+lag:stop+lag]
                coeff = F.conv2d(values.reshape(-1, 3, 58, 74), self.kernel[lag], stride=8)
                out[start:stop] += coeff.reshape(stop-start, 10, self.channels, 7, 9)
        out *= self.active[:, :, None, None, None]
        self._count('KT', True)
        return out


def feature_delta(a, active, *, dtype, device):
    """Materialize only one native chunk, with all actual channels."""
    a = a.to(device=device, dtype=dtype)*active.to(device)[:, :, None, None, None]
    values = torch.einsum('ntcij,yx->ntciyjx', a, p.basis(dtype, device)).reshape(len(a), 10, a.shape[2], 56, 72)
    out = a.new_zeros((1, a.shape[2], len(a), 320, 512))
    for tile, (y, x) in enumerate(p.ROIS): out[0, :, :, y:y+56, x:x+72] = values[:, tile].transpose(0, 1)
    return out


def validate_head(conv):
    if (getattr(conv, '_padding', None) != (1, 1, 1, 1, 2, 0)
            or tuple(conv.stride) != (1, 1, 1) or tuple(conv.dilation) != (1, 1, 1)
            or conv.groups != 1 or tuple(conv.weight.shape[2:]) != (3, 3, 3)
            or conv.out_channels != 3 or conv.weight.dtype != torch.float32):
        raise ValueError('actual FP32 stride1 RGB causal head contract unsupported')


def replay_head(conv, chunks, coefficients, active, event=None):
    """Same global a, streaming modified history. Ignores pre-cloned baseline tail."""
    validate_head(conv); tail = None; cursor = 0
    for feature in chunks:
        length = feature.shape[2]
        if length != (1 if cursor == 0 else 4) or cursor+length > len(coefficients): raise ValueError('native first1 then4 chunk schedule')
        if feature.dtype != torch.float32 or not bool(torch.isfinite(feature).all()): raise ValueError('finite FP32 captured features required')
        changed = feature+feature_delta(coefficients[cursor:cursor+length], active[cursor:cursor+length], dtype=feature.dtype, device=feature.device)
        if event: event('head_replay', False)
        with torch.inference_mode(): output = conv(changed, tail)
        if event: event('head_replay', True)
        # Keep the modified causal history, including the first native singleton.
        history = changed if tail is None else torch.cat((tail, changed), dim=2)
        tail = history[:, :, -2:].clone()
        yield cursor, output[0].permute(1, 2, 3, 0).float()/2+.5
        cursor += length
    if cursor != len(coefficients): raise ValueError('incomplete native replay')
