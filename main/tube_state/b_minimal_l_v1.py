"""L-only candidates on the unchanged hard graph. No new formal output."""
import numpy as np
from . import b_framewise_qim_time_recovery_v1 as hard
from . import b_soft_time_evidence_v1 as soft


def shared_codebook(check=lambda: None):
    buckets = hard.codebook(check)
    flat = np.empty(hard.DOMAIN, dtype=np.uint64)
    for js, words in buckets:
        check()
        flat[js.astype(np.int64)] = words
    return buckets, flat


def candidates(graph, cost, local):
    supported = graph['qualification'] == 'SUPPORTED_RELATION_SET'
    nodes = []
    for r, (a, b) in enumerate(zip(cost['offsets'][:-1], cost['offsets'][1:])):
        count = int(local['best'][a:b].sum())
        defined = cost['states'][r] == 'DEFINED'
        index = int(cost['bounds'][r, 0]+np.flatnonzero(local['best'][a:b])[0]) if defined and count == 1 else None
        nodes.append(dict(received_frame=r, model_status=cost['states'][r], L_status=local['status'][r],
                          best_count=count, model_index=index, candidate_index=index if supported else None))
    return dict(qualification=graph['qualification'], nodes=nodes, epsilon=soft.EPSILON,
                formal_new_determined=0, new_relation_outputs=0, joint_path=None,
                claim_ceiling='Conditional node candidates only; old hard formal and relations remain separate')


def replica_major_q0(q):
    q = np.asarray(q)
    if q.shape != (181, 5, 2, 63) or q.dtype != np.float64 or not np.isfinite(q).all():
        raise ValueError('finite FP64 BASE/preclip q [181,5,2,63] required')
    return q.transpose(0, 2, 1, 3).reshape(181, 10, 63).copy()


def validate_cached_targets(q0, targets, bits):
    """Check class/nearest/lower-tie membership without constructing another target."""
    q0,targets,bits=map(np.asarray,(q0,targets,bits))
    if (q0.shape!=targets.shape or bits.shape!=targets.shape or q0.dtype!=np.float64 or targets.dtype!=np.float64
            or not np.isfinite(q0).all() or not np.isfinite(targets).all() or not np.isin(bits,(0,1)).all()):
        raise ValueError('finite saved target geometry and binary class required')
    cells=(targets-16*bits)/32
    displacement=q0-targets
    if not np.equal(cells,np.floor(cells)).all() or not ((displacement>-16)&(displacement<=16)).all():
        raise ValueError('saved target violates fixed class/nearest/lower-tie semantics; not reconstructed')
