"""Received-only erasure compatibility and exact nondecreasing interval paths.

Public domain and layout only. No source length, arm, editing plan or truth input.
"""
import numpy as np

DOMAIN = 92672
BLOCK = 1024
POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def generator():
    # The original BCH63/24 generator, over GF(64), x^6+x+1.
    def mul(a, b):
        out = 0
        while b:
            if b & 1: out ^= a
            b >>= 1; a <<= 1
            if a & 64: a ^= 67
        return out
    powers = [1]
    for _ in range(62): powers.append(mul(powers[-1], 2))
    roots = set()
    for root in range(1, 15):
        n = root
        while n not in roots:
            roots.add(n); n = 2*n % 63
    poly = [1]
    for n in sorted(roots):
        updated = [0]*(len(poly)+1)
        for k, c in enumerate(poly):
            updated[k] ^= mul(c, powers[n]); updated[k+1] ^= c
        poly = updated
    if len(poly) != 40 or any(x not in (0, 1) for x in poly):
        raise ArithmeticError('invalid BCH generator')
    return sum(x << k for k, x in enumerate(poly))


def words(js):
    js = np.asarray(js, dtype=np.uint64)
    if np.any(js >= DOMAIN): raise ValueError('public domain exceeded')
    code = js << np.uint64(39); rem = code.copy(); g = generator()
    for power in range(55, 38, -1):
        rem ^= ((rem >> np.uint64(power)) & np.uint64(1))*np.uint64(g << (power-39))
    return code ^ rem


def codebook(check=lambda: None):
    buckets = []
    for lane in range(5):
        js = np.arange(lane, DOMAIN, 5, dtype=np.uint64)
        parts = []
        for start in range(0, len(js), BLOCK):
            check(); parts.append(words(js[start:start+BLOCK]))
        buckets.append((js, np.concatenate(parts)))
    return buckets


def read_rgb(rgb, check=lambda: None):
    """Same ROI/DCT/QIM arithmetic as read_rgb, uint8 normalized exactly once.

    Work one received frame at a time; never allocate a full float64 movie.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or tuple(rgb.shape[1:]) != (320, 512, 3) or not len(rgb):
        raise ValueError('uint8 received RGB [R,320,512,3] required')
    x = np.arange(8, dtype=np.float64)+.5
    basis = .25*np.cos(np.pi*x[:, None]/8)*np.cos(2*np.pi*x[None, :]/8)
    q = np.empty((len(rgb), 5, 2, 63), dtype=np.float64)
    for r, frame in enumerate(rgb):
        check()
        for replica in range(2):
            for lane in range(5):
                y, x0 = 200+64*replica, 16+96*lane
                core = (frame[y:y+56, x0:x0+72].astype(np.float64)/255).reshape(7, 8, 9, 8, 3)
                q[r, lane, replica] = (255*np.einsum('iyjxc,yx,c->ij', core, basis, [.299,.587,.114])).reshape(63)
    distances = []
    for bit in (0, 1):
        low = np.floor((q-16*bit)/32)*32+16*bit
        center = np.where(q-low <= 16, low, low+32)
        distances.append(np.abs(q-center))
    d0, d1 = distances
    return dict(q=q, d0=d0, d1=d1, soft=(d0*d0-d1*d1)/256, mask=np.isfinite(q))


def fuse(soft, mask):
    soft, mask = np.asarray(soft), np.asarray(mask)
    if soft.ndim != 4 or soft.shape[1:] != (5, 2, 63) or mask.shape != soft.shape or mask.dtype != np.bool_:
        raise ValueError('received frame/lane/replica/bit soft and boolean mask required')
    valid = mask & np.isfinite(soft)
    merged = np.where(valid, soft, 0).sum(axis=2)
    known = valid.any(axis=2) & (merged != 0) & np.isfinite(merged)
    return merged, known


def compatibility(soft, mask, table, check=lambda: None):
    merged, known = fuse(soft, mask)
    shifts = np.arange(62, -1, -1, dtype=np.uint64)
    weights = np.uint64(1) << shifts
    labels = []; compared = 0
    for r in range(len(merged)):
        frame = []
        for lane, (js, codes) in enumerate(table):
            check(); u = int(63-known[r, lane].sum()); hits = []
            observed = np.sum(weights*known[r, lane], dtype=np.uint64)
            hard = np.sum(weights*(merged[r, lane] > 0), dtype=np.uint64)
            if u <= 14:
                for start in range(0, len(codes), BLOCK):
                    check(); chunk = codes[start:start+BLOCK]
                    xor = (chunk ^ hard) & observed
                    errors = POPCOUNT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
                    hits.extend(int(j) for j in js[start:start+BLOCK][2*errors+u <= 14])
                    compared += len(chunk)
            frame.append(dict(lane=lane, labels=hits, erasures=u,
                              status='COMPATIBLE' if len(hits)==1 else 'IMPLEMENTATION_UNRESOLVED' if len(hits)>1 else 'NO_COMPATIBLE_CODE'))
        labels.append(frame)
    return dict(labels=labels, comparisons=compared), merged, known


def interval_paths(labels):
    local = []; intervals = []; evidence = []
    anomaly = False
    for frame in labels:
        js = [j for row in frame for j in row['labels']]
        bad = any(len(row['labels']) > 1 for row in frame)
        lo, hi = (max(js), min(DOMAIN-1, min(js)+4)) if js else (0, DOMAIN-1)
        conflict = bool(js and lo > hi)
        status = 'IMPLEMENTATION_UNRESOLVED' if bad else 'LOCAL_CONFLICT' if conflict else 'LOCAL_LABEL' if js else 'OBSERVATION_UNSUPPORTED'
        local.append(dict(status=status, labels=js, intersection=[lo,hi] if js else None))
        intervals.append((0,DOMAIN-1) if conflict or bad else (lo,hi))
        if not conflict and not bad:
            evidence += [(j, row['lane']) for row in frame for j in row['labels']]
        anomaly |= bad
    if not intervals: raise ValueError('at least one received frame required')
    lower = np.maximum.accumulate([x[0] for x in intervals]).tolist()
    upper = np.minimum.accumulate([x[1] for x in intervals][::-1])[::-1].tolist()
    feasible = all(l <= u for l,u in zip(lower,upper))
    supported = feasible and not anomaly and len({j for j,l in evidence}) >= 2 and len({l for j,l in evidence}) >= 2
    nodes = []
    for r, item in enumerate(local):
        unresolved = item['status'] in ('LOCAL_CONFLICT','IMPLEMENTATION_UNRESOLVED') or anomaly
        exact = feasible and lower[r]==upper[r] and not unresolved
        nodes.append(dict(received_frame=r, **item, interval=[lower[r],upper[r]] if feasible else None,
                          raw_index=lower[r] if exact else None,
                          output_index=lower[r] if exact and supported else None,
                          inference='INFERRED_NO_LOCAL_LABEL' if exact and not item['labels'] else 'LOCAL_CONSTRAINED' if exact else 'UNRESOLVED'))
    relations = []
    for r in range(len(nodes)-1):
        blocked = any(nodes[i]['status'] in ('LOCAL_CONFLICT','IMPLEMENTATION_UNRESOLVED') for i in (r,r+1)) or anomaly
        delta = [max(0,lower[r+1]-upper[r]),upper[r+1]-lower[r]] if feasible else None
        a,b = nodes[r]['raw_index'],nodes[r+1]['raw_index']
        exact = feasible and not blocked and a is not None and b is not None
        kind = None if not feasible or blocked else 'REPEAT' if delta==[0,0] else 'NORMAL' if delta==[1,1] else 'GAP' if delta[0]>1 else 'AMBIGUOUS'
        relations.append(dict(received_left=r, delta=delta, kind=kind,
                              status='LOCAL_CONFLICT_UNRESOLVED' if blocked else 'EXACT' if exact else 'AMBIGUOUS' if feasible else 'NO_COMPATIBLE_PATH',
                              raw_endpoints=[a,b] if exact else None,
                              output_endpoints=[a,b] if exact and supported else None,
                              missing_indices=list(range(a+1,b)) if exact and b>a+1 else None,
                              repeated_index=a if exact and a==b else None,
                              output_missing_indices=list(range(a+1,b)) if exact and supported and b>a+1 else None,
                              output_repeated_index=a if exact and supported and a==b else None))
    return dict(graph='NONCONFLICT_OBSERVATIONS_ALL_NONDECREASING_PATHS',
                path_status='NO_COMPATIBLE_PATH' if not feasible else 'PARTIAL_PATH_WITH_LOCAL_CONFLICT' if any(x['status']=='LOCAL_CONFLICT' for x in local) else 'COMPATIBLE_PATH',
                qualification='IMPLEMENTATION_UNRESOLVED' if anomaly else 'SUPPORTED_RELATION_SET' if supported else 'INSUFFICIENT_MULTILANE_EVIDENCE',
                nodes=nodes, relations=relations)


def receive(raw, table, check=lambda: None):
    compatible, merged, known = compatibility(raw['soft'],raw['mask'],table,check)
    result = interval_paths(compatible['labels']); result.update(compatible)
    return result, merged, known
