"""Card51 fixed carrier extension and pilot-only blind local payload protection."""
import hashlib
import hmac
import math

import torch

from main.tube_state import local_joint_positive_projection_margin_v1 as base
from main.tube_state import local_payload_secded_39_32_v1 as ecc

LATENT_SHAPE, VIDEO_SHAPE, ROIS = base.LATENT_SHAPE, base.VIDEO_SHAPE, base.ROIS
STATES, KEYS, DOMAIN = base.STATES, base.KEYS, base.DOMAIN
GAMMA_S, GAMMA_P = base.GAMMA_S, base.GAMMA_P
ARMS, LAYERS = ("P", "J0", "J_ECC"), ("terminal", "mp4")
validate_latent = base.validate_latent


def public_codebook(key):
    public = base.public_codebook(key)
    def digest(text):
        return hmac.new(key.encode("utf-8"), text.encode("ascii"), hashlib.sha256).digest()
    def sign(text):
        return 1 if digest(text)[0] & 1 == 0 else -1
    basis = [(c,a,b) for c in range(4,16) for a in range(8) for b in range(8) if (a,b) != (0,0)]
    directions = torch.zeros((4,61,12,2,8,8), dtype=torch.float64)
    directions[:, :54] = public["directions"]
    selected = []
    for r in range(4):
        indices = sorted(range(756), key=lambda i:(digest(f"{DOMAIN}|roi={r}|basis={i}|perm"),i))[:61]
        selected.append(indices)
        for j in range(54,61):
            i = indices[j]
            c,a,b = basis[i]
            polarity = sign(f"{DOMAIN}|roi={r}|basis={i}|sign")
            for y in range(8):
                for x in range(8):
                    value = polarity*(1-2*(((a&y).bit_count()+(b&x).bit_count())%2))/math.sqrt(128)
                    directions[r,j,c-4,0,y,x] = value
                    directions[r,j,c-4,1,y,x] = -value
    public.update(directions=directions, selected_basis=selected,
        parity_permutations=[sorted(range(7),key=lambda a:(digest(f"{DOMAIN}|state={k}|parity={a}|ecc39-perm"),a)) for k in STATES],
        parity_masks=[[sign(f"{DOMAIN}|state={k}|parity={j}|ecc39-sign") for j in range(7)] for k in STATES])
    return public


def observe(received, key, public):
    """Only projections: returned scalars are persisted before path selection."""
    validate_latent(received)
    if public["key"] != key or tuple(public["directions"].shape) != (4,61,12,2,8,8):
        raise ValueError("public key or direction geometry mismatch")
    z = received.detach().to(device="cpu",dtype=torch.float64)
    scores = torch.empty((22,4,61),dtype=torch.float64)
    for pos in STATES:
        for r,(y0,y1,x0,x1) in enumerate(ROIS):
            block = z[0,4:16,1+2*pos:3+2*pos,y0:y1,x0:x1].reshape(-1)
            scores[pos,r] = public["directions"][r].reshape(61,-1) @ block
    pooled = scores.mean(dim=1)
    if not bool(torch.isfinite(pooled).all()):
        raise ValueError("nonfinite local scores")
    return dict(status="READ",truth_used=False,roi_scores=scores.tolist(),pooled_scores=pooled.tolist(),
        public_permutations=public["permutations"],public_masks=public["masks"],
        public_parity_permutations=public["parity_permutations"],public_parity_masks=public["parity_masks"],
        geometry="complete clean public grid; observation indices are not decoded states")


def select_pilot_path(pilot_scores):
    if len(pilot_scores) != 22 or any(len(row) != 22 for row in pilot_scores):
        raise ValueError("expected 22x22 pilot matrix")
    if any(type(x) not in (int,float) or not math.isfinite(x) for row in pilot_scores for x in row):
        raise ValueError("finite numeric pilot scores required")
    candidates = []
    for initial in range(22):
        path = [(initial+k)%22 for k in range(22)]
        terms = [pilot_scores[k][state] for k,state in enumerate(path)]
        candidates.append(dict(initial_state=initial,path=path,pilot_terms=terms,score=math.fsum(terms)))
    ranked = sorted((c["score"] for c in candidates),reverse=True)
    winners = [c for c in candidates if c["score"] == ranked[0]]
    unique = len(winners) == 1
    return dict(status="READ" if unique else "UNRESOLVED",candidates=candidates,unique=unique,
        selected_initial_state=winners[0]["initial_state"] if unique else None,
        selected_path=winners[0]["path"] if unique else None,
        optimal_initial_states=[c["initial_state"] for c in winners],optimal_paths=[c["path"] for c in winners],
        best_score=ranked[0],second_best_score=ranked[1],best_minus_second=ranked[0]-ranked[1],
        path_scores=22,pilot_terms_reduced=484,truth_used=False,
        arithmetic="math.fsum of22 original pilot scores, equal weights, exact==ties")


def despread(raw, selection):
    """No true state or message; reconstruct canonical code coordinates 0..38."""
    path = selection.get("selected_path")
    positions = []
    for k in STATES:
        state = None if path is None else path[k]
        soft = [None]*39
        if state is not None:
            row = raw["pooled_scores"][k]
            for l,b in enumerate(raw["public_permutations"][state]):
                soft[ecc.DATA_COORDINATES[b]] = raw["public_masks"][state][l]*row[22+l]
            for j,p in enumerate(raw["public_parity_permutations"][state]):
                soft[ecc.PARITY_COORDINATES[p]] = raw["public_parity_masks"][state][j]*row[54+j]
        bits = [None if x is None or x == 0 else int(x > 0) for x in soft]
        positions.append(dict(position=k,inferred_state=state,status="READ" if state is not None else "UNRESOLVED",
            codeword_soft=soft,codeword_bits=bits,data_soft=[soft[i] for i in ecc.DATA_COORDINATES],
            data_bits=[bits[i] for i in ecc.DATA_COORDINATES],parity_bits=[bits[i] for i in ecc.PARITY_COORDINATES]))
    return dict(status="READ",truth_used=False,positions=positions)


def aggregate(soft_by_position):
    if len(soft_by_position) != 22 or any(len(row) != 32 for row in soft_by_position):
        raise ValueError("expected22x32 soft values")
    rows=[]
    for b in range(32):
        support=[k for k in STATES if soft_by_position[k][b] is not None]
        value=math.fsum(soft_by_position[k][b] for k in support)
        if not math.isfinite(value):
            raise ValueError("nonfinite aggregate")
        decoded=None if not support or value == 0 else int(value>0)
        rows.append(dict(bit=b,decoded=decoded,status="READ" if decoded is not None else "UNRESOLVED",
            soft_sum=value,fixed_denominator_mean=value/22,support_positions=support,support_count=len(support),
            erased_positions=[k for k in STATES if k not in support],position_denominator=22))
    return rows


def ecc_soft(raw_position, decoded):
    return [(2*b-1)*abs(raw_position["data_soft"][i]) if b is not None else None
            for i,b in enumerate(decoded["message"])]


def write_projection(source_p, host_scores, codeword, public):
    validate_latent(source_p)
    ecc._bits(codeword,39)
    if source_p.dtype != torch.float32:
        raise ValueError("P must be the saved FP32 source")
    if host_scores.get("status") != "READ":
        raise ValueError("available P host scores required")
    host=torch.tensor(host_scores["pooled_scores"],dtype=torch.float64)
    if host.shape != (22,61) or not bool(torch.isfinite(host).all()):
        raise ValueError("invalid61-direction host scores")
    data=[codeword[i] for i in ecc.DATA_COORDINATES]
    parity=[codeword[i] for i in ecc.PARITY_COORDINATES]
    out=source_p.detach().cpu().double().clone()
    rows, coefficients=[],[]
    for k in STATES:
        state,active=base.project_state(host[k,:22],k)
        signs=torch.tensor(
            [(2*data[b]-1)*public["masks"][k][l] for l,b in enumerate(public["permutations"][k])]+
            [(2*parity[p]-1)*public["parity_masks"][k][j] for j,p in enumerate(public["parity_permutations"][k])],dtype=torch.float64)
        payload_delta=signs*(GAMMA_P-signs*host[k,22:]).clamp_min(0)
        delta=torch.cat((state-host[k,:22],payload_delta))
        coefficients.append(delta)
        rows.append(dict(position=k,active_state_competitors=active,state_score_correction=delta[:22].tolist(),
            data_score_correction=delta[22:54].tolist(),parity_score_correction=delta[54:].tolist()))
        for r,(y0,y1,x0,x1) in enumerate(ROIS):
            out[0,4:16,1+2*k:3+2*k,y0:y1,x0:x1]+=(delta@public["directions"][r].reshape(61,-1)).reshape(12,2,8,8)
    candidate=out.float()  # the sole candidate materialization, never add to FP32 J0.
    validate_latent(candidate)
    coeff=torch.stack(coefficients)
    return candidate,dict(status="CONSTRUCTED",rows=rows,source="same saved FP32 P; not frozen J0",
        continuous_distance_squared_display=float(4*coeff.square().sum()),
        original54_continuous_distance_squared_display=float(4*coeff[:,:54].square().sum()),
        added_parity_continuous_distance_squared_display=float(4*coeff[:,54:].square().sum()),
        fp32_materializations=1,norm_cap=None,
        rounding="one FP64 computation then FP32; no ULP adjustment, retry or amplitude change")


def unavailable_local(status, detail=None):
    symbols=dict(status=status,truth_used=False,positions=[dict(position=k,inferred_state=None,status=status,
        codeword_soft=[None]*39,codeword_bits=[None]*39,data_soft=[None]*32,data_bits=[None]*32,parity_bits=[None]*7) for k in STATES])
    return dict(status=status,truth_used=False,detail=detail,raw_scores=dict(status=status),selection=dict(status=status,selected_path=None),
        symbols=symbols,ecc_blocks=[dict(position=k,status=status,message=[None]*32,returned_codeword=None,rule_applied=False,
            flipped_coordinate=None,syndrome=None,syndrome_reductions=0) for k in STATES],
        raw_aggregate=[dict(bit=b,decoded=None,status=status) for b in range(32)],
        ecc_aggregate=[dict(bit=b,decoded=None,status=status) for b in range(32)])
