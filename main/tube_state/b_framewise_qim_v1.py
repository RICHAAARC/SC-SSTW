"""Adopted RGB-time protocol. Pure algebra/geometry, no models or IO."""
from functools import lru_cache
import math
import torch

FRAMES, HEIGHT, WIDTH = 181, 320, 512
LUMA = (.299, .587, .114)
RMS, LINF = 2/255, 8/255
ROIS = tuple((200+64*replica, 16+96*lane) for replica in range(2) for lane in range(5))


def basis(dtype=torch.float64, device=None):
    x = torch.arange(8, dtype=dtype, device=device)+.5
    return .25*torch.cos(math.pi*x[:, None]/8)*torch.cos(2*math.pi*x[None, :]/8)


def _mul(a, b):
    out = 0
    while b:
        if b & 1: out ^= a
        b >>= 1; a <<= 1
        if a & 64: a ^= 67
    return out


@lru_cache(None)
def _powers():
    out = [1]
    for _ in range(62): out.append(_mul(out[-1], 2))
    return tuple(out)


def _div(a, b):
    if b == 0: raise ZeroDivisionError('GF64 zero divisor')
    return 0 if a == 0 else _powers()[(_powers().index(a)-_powers().index(b)) % 63]


@lru_cache(None)
def generator():
    roots = set()
    for root in range(1, 15):
        n = root
        while n not in roots:
            roots.add(n); n = 2*n % 63
    poly = [1]
    for n in sorted(roots):
        updated = [0]*(len(poly)+1)
        for k, c in enumerate(poly):
            updated[k] ^= _mul(c, _powers()[n]); updated[k+1] ^= c
        poly = updated
    if len(poly) != 40 or any(x not in (0, 1) for x in poly):
        raise ArithmeticError('BCH63/24 generator construction')
    return sum(x << k for k, x in enumerate(poly))


def encode(j):
    if type(j) is not int or not 0 <= j <= 92671: raise ValueError('global j must be 0..92671')
    word = j << 39; rem = word; g = generator()
    while rem.bit_length() >= 40: rem ^= g << (rem.bit_length()-40)
    word ^= rem
    return [(word >> k) & 1 for k in range(62, -1, -1)]


def syndromes(bits):
    if len(bits) != 63 or any(type(x) is not int or x not in (0, 1) for x in bits):
        raise ValueError('63 integer binary bits required')
    out = []
    for i in range(1, 15):
        value = 0
        for k, bit in enumerate(bits):
            if bit: value ^= _powers()[((62-k)*i) % 63]
        out.append(value)
    return out


def decode(bits):
    """Bounded t=7 hard BCH diagnostic, not presence/source authentication."""
    s = syndromes(bits); c = [1]+[0]*14; old = c.copy(); degree = 0; shift = 1; last = 1
    for n in range(14):
        discrepancy = s[n]
        for k in range(1, degree+1): discrepancy ^= _mul(c[k], s[n-k])
        if not discrepancy: shift += 1; continue
        saved = c.copy(); factor = _div(discrepancy, last)
        for k in range(15-shift): c[k+shift] ^= _mul(factor, old[k])
        if 2*degree <= n: degree, old, last, shift = n+1-degree, saved, discrepancy, 1
        else: shift += 1
    errors = []
    for power in range(63):
        value = 0
        for k in range(degree+1): value ^= _mul(c[k], _powers()[(-power*k) % 63])
        if value == 0: errors.append(62-power)
    if degree > 7 or len(errors) != degree: return dict(status='UNRESOLVED_CODE', j=None)
    corrected = bits.copy()
    for k in errors: corrected[k] ^= 1
    if any(syndromes(corrected)): return dict(status='UNRESOLVED_CODE', j=None)
    j = sum(corrected[k] << (23-k) for k in range(24))
    if j > 92671: return dict(status='OUT_OF_DOMAIN', j=None, corrected_j=j)
    return dict(status='BCH_CANDIDATE_UNCALIBRATED', j=j, corrected_positions=errors)


def nearest(q, bit):
    """Fixed class center, ties go to the numerically smaller center."""
    lower = torch.floor((q-16*bit)/32)*32+16*bit
    return torch.where(q-lower <= 16, lower, lower+32)


def distances(q):
    d0 = (q-nearest(q, 0)).abs(); d1 = (q-nearest(q, 1)).abs()
    return d0, d1, (d0.square()-d1.square())/256


def schedule(frames=FRAMES, start=0):
    if type(frames) is not int or type(start) is not int or frames <= 0 or start < 0 or start+frames > 92672:
        raise ValueError('finite source interval within global domain required')
    js = torch.full((frames, 10), -1, dtype=torch.int64)
    bits = torch.zeros((frames, 10, 63), dtype=torch.float64)
    for t in range(frames):
        n = start+t
        for j in range(max(0, n-4), n+1):
            for replica in range(2):
                tile = replica*5+j % 5; js[t, tile] = j
                bits[t, tile] = torch.tensor(encode(j))
    return js, bits


def support(active):
    """Allowed-support union through current/prior two LOCAL frames; native reset."""
    union = active.clone()
    for lag in (1, 2): union[lag:] |= active[:-lag]
    mask = union[:, :, None, None, None].expand(-1, -1, 3, 58, 74).clone()
    mask[:, 5:, :, 57, :] = False  # bottom canvas boundary, not a spare budget row
    return mask


def pack(rgb):
    if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (320, 512, 3): raise ValueError('RGB [T,320,512,3]')
    out = rgb.new_zeros((len(rgb), 10, 3, 58, 74))
    for tile, (y, x) in enumerate(ROIS):
        h = min(58, 320-(y-1))
        out[:, tile, :, :h] = rgb[:, y-1:y-1+h, x-1:x+73].permute(0, 3, 1, 2)
    return out


def project(r):
    """Normalized RGB luma DCT; independent nonoverlapping unit atoms."""
    core = r[..., 1:57, 1:73].reshape(*r.shape[:3], 7, 8, 9, 8)
    return torch.einsum('ntciyjx,yx,c->ntij', core,
                        basis(r.dtype, r.device), r.new_tensor(LUMA)).reshape(len(r), 10, 63)


def project_transpose(q):
    values = q.reshape(len(q), 10, 7, 9)
    core = torch.einsum('ntij,yx,c->ntciyjx', values, basis(q.dtype, q.device), q.new_tensor(LUMA))
    out = q.new_zeros((len(q), 10, 3, 58, 74))
    out[..., 1:57, 1:73] = core.reshape(len(q), 10, 3, 56, 72)
    return out


def read_rgb(rgb):
    """Received-only reader: all fixed ROIs; no writer time, message or active mask."""
    q = (255*project(pack(rgb.double()))).reshape(len(rgb), 2, 5, 63).transpose(1, 2)
    mask = torch.isfinite(q)
    d0, d1, soft = distances(q)
    return dict(q=q, d0=d0, d1=d1, soft=soft, mask=mask)


def merge_replicas(soft, mask):
    if soft.shape != mask.shape or soft.ndim != 4 or tuple(soft.shape[1:]) != (5, 2, 63):
        raise ValueError('received-frame/lane/replica/bit arrays required')
    valid = mask & torch.isfinite(soft)
    merged = torch.where(valid, soft, 0).sum(2); observed = valid.any(2)
    decisions = []
    for frame in range(len(soft)):
        for lane in range(5):
            score = merged[frame, lane]; known = observed[frame, lane] & (score != 0)
            value = decode((score > 0).int().tolist()) if bool(known.all()) else dict(status='ERASURE', j=None)
            decisions.append(dict(received_frame=frame, lane=lane, **value))
    return dict(soft=merged, mask=observed, candidates=decisions)
