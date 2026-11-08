"""Card74 C73 synchronization identity; no payload writing or phase certificate."""
from __future__ import annotations

import hashlib
import hmac
import torch
from main.tube_state import local_segmented_terminal_sync_v1 as legacy

LATENT_SHAPE = legacy.LATENT_SHAPE
VIDEO_SHAPE = legacy.VIDEO_SHAPE
ROIS = legacy.ROIS
KEYS = ("watermark", "watermark-wrong")
ARMS = ("P0", "S_base", "S_ID")
A_ID = 0.5
TAU = 1 / 16
DIMENSION = 64
IDENTITIES = 46
MASK_DOMAIN = "B-GroupPhase64-v1"


def codebook(key):
    """Only key/public geometry; basis selection preserves the legacy first22."""
    if not isinstance(key, str) or not key:
        raise ValueError("nonempty watermark key required")
    def digest(text):
        return hmac.new(key.encode(), text.encode("ascii"), hashlib.sha256).digest()
    basis = [(c, a, b) for c in range(4, 16) for a in range(8)
             for b in range(8) if (a, b) != (0, 0)]
    directions = torch.zeros((4, 64, 12, 8, 8), dtype=torch.float64)
    selected, legacy_selected = [], []
    for roi in range(4):
        order = sorted(range(756), key=lambda i: (
            digest(f"{legacy.DOMAIN}|roi={roi}|basis={i}|perm"), i))
        legacy_selected.append(order[:22])
        selected.append(order[22:86])
        for j, index in enumerate(selected[-1]):
            channel, a, b = basis[index]
            polarity = 1 if digest(f"{legacy.DOMAIN}|roi={roi}|basis={index}|sign")[0] & 1 == 0 else -1
            for y in range(8):
                for x in range(8):
                    directions[roi, j, channel-4, y, x] = polarity * (
                        1-2*(((a & y).bit_count()+(b & x).bit_count()) % 2)) / 8
    mask = [1 if digest(f"{MASK_DOMAIN}|mask|{j}")[0] & 1 == 0 else -1 for j in range(64)]
    words = torch.tensor([[mask[j]*(1-2*((t & j).bit_count() % 2))/8
                           for j in range(64)] for t in range(46)], dtype=torch.float64)
    return dict(key=key, directions=directions, words=words, selected_basis=selected,
                legacy_selected_basis=legacy_selected, mask_domain=MASK_DOMAIN)


def validate_book(book):
    if tuple(book["directions"].shape) != (4, 64, 12, 8, 8) or tuple(book["words"].shape) != (46, 64):
        raise ValueError("C73 requires 64 single-latent directions and 46 identity rows")
    if not bool(torch.isfinite(book["directions"]).all() and torch.isfinite(book["words"]).all()):
        raise ValueError("nonfinite C73 book")


def project_times(latent, times, book):
    validate_book(book)
    if not isinstance(latent, torch.Tensor) or latent.ndim != 5 or tuple(latent.shape[:2]) != (1, 16) or tuple(latent.shape[-2:]) != (40, 64):
        raise ValueError("normalized [1,16,T,40,64] latent required")
    if not latent.is_floating_point() or not bool(torch.isfinite(latent).all()):
        raise ValueError("finite normalized latent required")
    if not times or any(type(t) is not int or not 0 <= t < latent.shape[2] for t in times):
        raise ValueError("valid public latent time indices required")
    z = latent.detach().cpu().double()
    roi = torch.empty((len(times), 4, 64), dtype=torch.float64)
    for r, (y0, y1, x0, x1) in enumerate(ROIS):
        block = z[0, 4:16, times, y0:y1, x0:x1].permute(1, 0, 2, 3).reshape(len(times), -1)
        roi[:, r] = block @ book["directions"][r].reshape(64, -1).T
    pooled = roi.mean(1)
    if not bool(torch.isfinite(roi).all() and torch.isfinite(pooled).all()):
        raise ValueError("nonfinite C73 projection")
    return roi, pooled


def classify(y, words):
    """No truth, score maximization, phase choice, gain fit, or epsilon."""
    y = torch.as_tensor(y, dtype=torch.float64)
    if y.shape != (64,) or not bool(torch.isfinite(y).all()):
        raise ValueError("finite 64-coordinate observation required")
    e = float(y.square().sum())
    rows, members = [], []
    for t, c in enumerate(words):
        s = float(y @ c)
        rho = None if e == 0 else float((y-s*c).square().sum()) / e
        if rho is not None and not __import__('math').isfinite(rho):
            raise ValueError("nonfinite residual")
        status = ("UNDEFINED_ZERO_ENERGY" if e == 0 else
                  "BORDERLINE_UNRESOLVED" if rho == TAU else
                  "CODE_MEMBER" if s > 0 and rho < TAU else "REJECTED")
        if status == "CODE_MEMBER":
            members.append(t)
        rows.append(dict(identity=t, s=s, rho=rho, status=status))
    return dict(status="OBSERVED", y=y.tolist(), energy=e, identities=rows,
                members=members, phase_status="NOT_CERTIFIED", truth_used=False,
                message_used=False)


def observe(latent, book, *, local=False):
    if local:
        if tuple(latent.shape) != (1, 16, 3, 40, 64):
            raise ValueError("nine-frame reset E must yield [1,16,3,40,64]")
        times = [2]
    else:
        legacy.validate_latent(latent)
        times = list(range(46))
    roi, pooled = project_times(latent, times, book)
    vectors = []
    for i, y in enumerate(pooled):
        row = classify(y, book["words"])
        row.update(latent_time=times[i], roi_scores=roi[i].tolist())
        vectors.append(row)
    return dict(status="OBSERVED", pooled_scores=pooled.tolist(), vectors=vectors,
                local_reset=local, truth_used=False, message_used=False)


def unavailable(status, *, local=False, detail=None):
    times = [2] if local else list(range(46))
    return dict(status=status, detail=detail, pooled_scores=None,
                truth_used=False, message_used=False, local_reset=local,
                vectors=[dict(status=status, latent_time=t, y=None, roi_scores=None,
                              energy=None, members=[], phase_status="NOT_CERTIFIED",
                              identities=[dict(identity=j, s=None, rho=None, status=status)
                                          for j in range(46)]) for t in times])


def construct(source, host_observation, book):
    legacy.validate_latent(source)
    if source.dtype != torch.float32:
        raise ValueError("saved S_base must be FP32")
    validate_book(book)
    a = torch.tensor(host_observation.get("pooled_scores"), dtype=torch.float64)
    if a.shape != (46, 64) or not bool(torch.isfinite(a).all()):
        raise ValueError("finite native S_base pooled coordinates required")
    target = A_ID * book["words"]
    eta = target-a
    value = source.detach().cpu().double().clone()
    for r, (y0, y1, x0, x1) in enumerate(ROIS):
        correction = (eta @ book["directions"][r].reshape(64, -1)).reshape(46, 12, 8, 8).permute(1, 0, 2, 3)
        value[0, 4:16, :, y0:y1, x0:x1] += correction
    result = value.float()  # sole candidate FP32 materialization
    legacy.validate_latent(result)
    actual_delta = result.double()-source.double()
    # The actual unselected spatial coordinates may contain FP32 rounding, but
    # no coordinates outside these four rectangles/channels are assigned.
    outside = actual_delta.clone()
    for y0, y1, x0, x1 in ROIS:
        outside[0, 4:16, :, y0:y1, x0:x1] = 0
    receipt = dict(status="CONSTRUCTED", A_ID=A_ID, net_bits=32,
                   fp32_materializations=1, norm_cap=None, payload_writes=0,
                   continuous_distance_squared=float(4*eta.square().sum()),
                   target_energy=float(4*target.square().sum()),
                   host_removal_energy=float(4*a.square().sum()),
                   cross_energy=float(-8*(target*a).sum()),
                   actual_fp32_distance_squared=float(actual_delta.square().sum()),
                   fp32_materialization_error_linf=float((result.double()-value).abs().max()),
                   outside_written_rectangles_linf=float(outside.abs().max()),
                   payload_channels_linf=float(actual_delta[:, :4].abs().max()),
                   times=[dict(latent_time=t, host=a[t].tolist(), eta=eta[t].tolist(),
                               continuous_distance_squared=float(4*eta[t].square().sum()))
                          for t in range(46)])
    return result, receipt


def native_audit(before, after, book, old_before, old_after):
    target = A_ID*book["words"]
    pooled = torch.tensor(after["pooled_scores"], dtype=torch.float64)
    drift = {}
    for key in ("key00", "key01"):
        b = torch.tensor(old_before[key]["pooled_scores"], dtype=torch.float64)
        a = torch.tensor(old_after[key]["pooled_scores"], dtype=torch.float64)
        d = a-b
        drift[key] = dict(max_abs=float(d.abs().max()), values=d.tolist(),
                          scope="correct-key native preservation" if key == "key00" else "wrong-key change control; not constrained to zero")
    return dict(status="MEASURED", target_coordinate_shortfall=(pooled-target).tolist(),
                max_abs_target_error=float((pooled-target).abs().max()), legacy22_drift=drift,
                native_correct_members=[v["members"] for v in after["vectors"]],
                media_efficacy=None)


def prepare_local(received, a):
    """Only received frames/public start; no source map or identity input."""
    if not isinstance(received, torch.Tensor) or received.ndim != 4 or received.shape[-1] != 3 or received.dtype != torch.uint8:
        raise ValueError("received RGB24 uint8 required")
    if type(a) is not int or a < 0 or a+4 > len(received):
        raise ValueError("four actual received target frames required")
    indices = [max(0, i) for i in range(a-5, a+4)]
    real_context = [i >= 0 for i in range(a-5, a)]
    return received[indices].clone(), dict(received_indices=indices,
        context_real_mask=real_context, target_real_mask=[True]*4,
        source_truth_used=False, target_latent_time=2)
