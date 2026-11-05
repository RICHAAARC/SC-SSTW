"""Adopted natural-video, given-message protocol. No I/O, model or truth lookup."""
from collections import Counter
import math
import re
from main.baselines.native_paper_temporal_v2 import CHANNELS, CONDITIONS, METHODS, edit_maps

EXPERIMENT = "native-wam-pixelseal-paper-fpr1-v3"
SHAPE = [181, 320, 512, 3]
QUERY_SEED = 202611052001
BOOTSTRAP_SEED = 202611052002


def validate_config(c):
    expected = dict(experiment=EXPERIMENT, shape=SHAPE, fps=25, crop_yxyx=[80, 128, 240, 384],
                    methods=list(METHODS), channels=list(CHANNELS), conditions=list(CONDITIONS),
                    wam_batch=8, shard_size=10, cal_sources=500, test_sources=500,
                    query_seed=QUERY_SEED, split_seed=20261105, edit_seed_base=20261105,
                    bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=2000, alpha=0.01)
    for k, v in expected.items():
        if c.get(k) != v:
            raise ValueError(f"Adopted method setting {k} requires {v!r}; got {c.get(k)!r}")
    return c


def eligible_header(row):
    try:
        return (int(row['header_frames']) >= 181 and float(row['fps']) == 25
                and int(row['width']) > 0 and int(row['height']) > 0)
    except (TypeError, ValueError, KeyError):
        return False


def select_sources(eligible):
    """One eligible clip per group, before watermark outputs. Pure metadata helper."""
    import numpy as np
    selected = {}
    for s in sorted(eligible, key=lambda x: (x['action'], int(x['group']), x['clip'])):
        if eligible_header(s):
            selected.setdefault((s['action'], int(s['group'])), s)
    rows = [selected[k] for k in sorted(selected)]
    if len(rows) < 1000:
        raise ValueError('PENDING_INPUT: fewer than 1000 eligible groups')
    order = np.random.Generator(np.random.PCG64(20261105)).permutation(len(rows))[:1000]
    return [dict(rows[i], ordinal=j+1, split='CAL' if j < 500 else 'TEST') for j, i in enumerate(order)]


def validate_sources(rows):
    if len(rows) != 1000:
        raise ValueError('PENDING_INPUT: the real roster must contain 1000 sources')
    ids, groups = set(), set()
    for i, row in enumerate(rows, 1):
        if (row['ordinal'] != i or row['split'] != ('CAL' if i <= 500 else 'TEST')
                or not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']) or not eligible_header(row)):
            raise ValueError(f'Invalid actual source/format/ordinal: {row}')
        if not row['url'].startswith('https://'):
            raise ValueError('Source URL must be an HTTPS video URL')
        key = row['action'], str(row['group'])
        if row['id'] in ids or key in groups:
            raise ValueError('Duplicate source or action/group across the fixed roster')
        if row['id'].startswith('b-rank3r-m'):
            raise ValueError('Development source is not a held-out source')
        ids.add(row['id']); groups.add(key)
    return rows


def queries(sources):
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(QUERY_SEED))
    result = {}
    for source in sources:
        for method, n in METHODS.items():
            q = rng.integers(0, 2, size=n, dtype=np.uint8)
            wrong = None
            if source['split'] == 'TEST':
                mask = rng.integers(0, 2, size=n, dtype=np.uint8)
                while not mask.any():  # Uniform nonzero mask; no model/truth selection.
                    mask = rng.integers(0, 2, size=n, dtype=np.uint8)
                wrong = (q ^ mask).tolist()
            result[f"{method}/{source['id']}"] = dict(query=q.tolist(), wrong_query=wrong)
    return result


def source_maps(source):
    # Preserve the original global ordinal edit seed; no shard-local reset.
    c = dict(sources=[source], fps=25, edit_seed_base=20261105+100*(source['ordinal']-1))
    return edit_maps(c)[source['id']]


def tasks(source, method):
    return [dict(id=f'{arm}/{channel}/{condition}', source=source['id'], split=source['split'],
                 action=source['action'], method=method, arm=arm, channel=channel,
                 condition=condition, frames=n, nbits=METHODS[method])
            for arm in (['OFF'] if source['split'] == 'CAL' else ['OFF', 'WM'])
            for channel in CHANNELS for condition, n in CONDITIONS.items()]


def operations(source, method=None):
    rows = []
    def add(key, kind):
        rows.append(dict(id=key, kind=kind, status='NOT_STARTED'))
    if method is None:
        for key in ('download', 'source_probe', 'normalize'):
            add(key, key)
        for name in ['G1', *CONDITIONS]:
            for op in ('encode', 'probe'):
                add(f'{op}/OFF/{name}', op)
        add('decode/OFF/G1', 'decode')
    else:
        add('load', 'load')
        for name in ['G1', *CONDITIONS]:
            add(f'decode/OFF/{name}', 'decode')
        if source['split'] == 'TEST':
            add('normalize', 'normalize')
            for i in (range(0, 181, 8) if method == 'wam' else [0]):
                add(f'embed/{i}', 'embed')
            for name in ['G1', *CONDITIONS]:
                for op in ('encode', 'probe', 'decode'):
                    add(f'{op}/WM/{name}', op)
            for stage in ['precodec', 'G1', *CONDITIONS]:
                add(f'quality/{stage}', 'quality_'+('G2' if stage in CONDITIONS else stage))
        for t in tasks(source, method):
            for i in (range(0, t['frames'], 8) if method == 'wam' else [0]):
                add(f"detect/{t['id']}/{i}", 'detect')
    return rows


def budget():
    return dict(source_prepare_units=1000, source_method_units=2000, shards=100,
                read_tasks=66000, frame_rows=10050000, model_load_upper=2000,
                wam_detect_api=645000, wam_detector_forward=645000, wam_detector_frames=5025000,
                wam_embed_api=11500, wam_embedder_forward=11500, wam_embedder_frames=90500,
                pixelseal_detect_api=33000, pixelseal_detector_forward=180000,
                pixelseal_detector_frames=5025000, pixelseal_embed_api=500,
                pixelseal_embedder_forward=500, pixelseal_embedder_frames=11500,
                encode=24000, probe=24000, decode=37000, normalize=2000,
                source_probe=1000, download=1000, quality_precodec=1000, quality_G1=1000, quality_G2=11000)


def view_score(method, bits, valid, query, *, complete, video_bits=None):
    import numpy as np
    if not complete:
        return None
    q = np.asarray(query)
    if q.shape != (METHODS[method],) or not np.isin(q, [0, 1]).all():
        raise ValueError('Invalid single full-length query')
    if method == 'pixelseal':
        b = np.asarray(video_bits)
        if b.shape != q.shape or not np.isin(b, [0, 1]).all():
            raise ValueError('Invalid native video message')
        return int(np.count_nonzero(b == q))
    b, v = np.asarray(bits), np.asarray(valid, dtype=bool)
    if b.shape != (len(v), 32) or not np.isin(b[v], [0, 1]).all():
        raise ValueError('Invalid native WAM messages')
    return int((b[v] == q).sum(axis=1).max()) if v.any() else -1


def calibrate(source_view_scores, nbits):
    if len(source_view_scores) != 500 or any(len(v) != 22 or any(s is None for s in v)
                                             for v in source_view_scores):
        return dict(status='CALIBRATION_INCOMPLETE', tau=None, planned_sources=500)
    z = [max(v) for v in source_view_scores]
    if any(s < -1 or s > nbits or not isinstance(s, int) for s in z):
        raise ValueError('Invalid calibration score')
    tau = sorted(z)[495]
    return dict(status='CALIBRATED', tau=tau, order=496, planned_sources=500,
                source_max_scores=z, ties_at_tau=z.count(tau), strict_comparison='score>tau',
                degeneration='ANY_NONEMPTY' if tau == -1 else 'ALL_REJECT' if tau == nbits else None,
                nominal_family_alpha=0.01, marginal_bound=5/501)


def family_event(events):
    if any(x is True for x in events):
        return True
    return False if len(events) == 22 and all(x is False for x in events) else None


def rate_summary(events):
    """Fixed denominator bounds + conservative exact CI envelope for missing events."""
    from scipy.stats import beta
    n = len(events); k = sum(x is True for x in events); u = sum(x is None for x in events)
    if not n:
        raise ValueError('Empty fixed denominator')
    def cp(count, total):
        if not total:
            return None
        return [0.0 if count == 0 else float(beta.ppf(.025, count, total-count+1)),
                1.0 if count == total else float(beta.ppf(.975, count+1, total-count))]
    interval = [cp(k, n)[0], cp(k+u, n)[1]]
    upper = 1.0 if k+u == n else float(beta.ppf(.95, k+u+1, n-k-u))
    return dict(planned=n, positive=k, negative=n-k-u, unknown=u, completed=n-u,
                coverage=(n-u)/n, rate_lower=k/n, rate_upper=(k+u)/n,
                conservative_cp95=interval, conservative_one_sided95_upper=upper,
                complete_subset_cp95=cp(k, n-u), complete_subset_denominator=n-u)


def action_sensitivity(events, actions):
    import numpy as np
    if len(events) != len(actions) or not events:
        raise ValueError('Action sensitivity requires aligned sources')
    groups = sorted(set(actions)); counts = []
    for g in groups:
        e = [e for e, a in zip(events, actions) if a == g]
        counts.append((len(e), sum(x is True for x in e), sum(x is None for x in e)))
    a = np.asarray(counts); rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    total = a[rng.integers(0, len(groups), size=(2000, len(groups)))].sum(axis=1)
    lo, hi = total[:, 1]/total[:, 0], (total[:, 1]+total[:, 2])/total[:, 0]
    return dict(groups=len(groups), replicates=2000, seed=BOOTSTRAP_SEED,
                lower_rate_percentiles=np.quantile(lo, [.025, .975]).tolist(),
                upper_rate_percentiles=np.quantile(hi, [.025, .975]).tolist(),
                interpretation='action-cluster sensitivity; zero-event bootstrap does not replace exact upper bound')
