"""Conditional outputs from frozen original fused L; no truth or new scoring."""
from copy import deepcopy
from . import b_soft_time_evidence_v1 as soft
from .b_framewise_qim_time_recovery_v1 import DOMAIN


def event(a, b):
    delta = b-a
    if delta < 0:
        raise ValueError('inverse endpoints')
    return dict(output_endpoints=[a, b], delta=delta,
                kind='REPEAT' if delta == 0 else 'NORMAL' if delta == 1 else 'GAP',
                output_missing_indices=list(range(a+1, b)) if delta > 1 else None,
                output_repeated_index=a if delta == 0 else None)


def extend(bounds, fixed):
    """Unweighted integer feasibility only. Bounds never create outputs."""
    work = [list(pair) for pair in bounds]
    for r, value in fixed.items():
        if not work[r][0] <= value <= work[r][1]:
            return False, work
        work[r] = [value, value]
    for r in range(1, len(work)):
        work[r][0] = max(work[r][0], work[r-1][0])
    for r in range(len(work)-2, -1, -1):
        work[r][1] = min(work[r][1], work[r+1][1])
    return all(lo <= hi for lo, hi in work), work


def prune_candidates(H, S):
    """Freeze after H filtering, then remove every inversion participant once."""
    SH = {r for r, n in S.items() if any((h < r and v > n) or (h > r and v < n) for h, v in H.items())}
    T = {r: n for r, n in S.items() if r not in SH}
    conflicts = set()
    for r, n in T.items():
        for t, m in T.items():
            if r < t and n > m:
                conflicts.update((r, t))
    return SH, T, conflicts, {r: n for r, n in T.items() if r not in conflicts}


def _preserve_old_nodes(hard):
    """Retain readable old node outputs independently of L/relation eligibility."""
    if not isinstance(hard, dict) or not isinstance(hard.get('nodes'), list):
        return [], [dict(error='old nodes are unavailable or not a list')]
    rows, errors = [], []
    for r, node in enumerate(hard['nodes']):
        try:
            if not isinstance(node, dict) or type(node.get('received_frame')) is not int or node['received_frame'] != r:
                raise ValueError('invalid old node coordinate')
            if node['status'] not in ('LOCAL_LABEL', 'LOCAL_CONFLICT', 'OBSERVATION_UNSUPPORTED', 'IMPLEMENTATION_UNRESOLVED'):
                raise ValueError('invalid old observation status')
            interval, value = node['interval'], node['output_index']
            if interval is None:
                if hard.get('path_status') != 'NO_COMPATIBLE_PATH':
                    raise ValueError('missing original interval')
            elif not isinstance(interval, list) or len(interval) != 2 or any(type(n) is not int for n in interval) or not 0 <= interval[0] <= interval[1] < DOMAIN:
                raise ValueError('invalid original interval')
            if value is not None and (type(value) is not int or not 0 <= value < DOMAIN or interval != [value, value] or node['status'] == 'LOCAL_CONFLICT'):
                raise ValueError('invalid old exact node output')
            rows.append(dict(received_frame=r, interval=deepcopy(interval), output_index=value,
                             origin='OLD' if value is not None else 'UNRESOLVED'))
        except (ValueError, KeyError, TypeError) as error:
            message = f'{type(error).__name__}: {error}'
            errors.append(dict(row_position=r, error=message))
            # Preserve the slot and the raw original separately, without inventing a coordinate/index.
            rows.append(dict(received_frame=None, interval=None, output_index=None,
                             origin='INVALID_OLD_NODE', error=message))
    return rows, errors


def _validate(hard, local):
    nodes = hard['nodes']
    graph = soft.graph_input(hard, len(nodes))
    if len(hard['relations']) != len(nodes)-1:
        raise ValueError('complete old formal relation rows required')
    if local['qualification'] != hard['qualification'] or local['epsilon'] != soft.EPSILON:
        raise ValueError('original qualification and epsilon=1e-12 required')
    if len(local['nodes']) != len(nodes):
        raise ValueError('complete original fused L rows required')
    H = {}
    for r, (node, candidate) in enumerate(zip(nodes, local['nodes'])):
        if type(node['received_frame']) is not int or type(candidate['received_frame']) is not int or candidate['received_frame'] != r:
            raise ValueError('integer received coordinates required')
        value = node['output_index']
        if value is not None:
            if type(value) is not int or not 0 <= value < DOMAIN or node['interval'] != [value, value] or node['status'] == 'LOCAL_CONFLICT':
                raise ValueError('old exact node contradicts original interval/status')
            H[r] = value
        for key in ('model_status', 'L_status', 'best_count', 'model_index', 'candidate_index'):
            if key not in candidate:
                raise ValueError('incomplete original L row: '+key)
        if type(candidate['best_count']) is not int or candidate['best_count'] < 0:
            raise ValueError('invalid original best count')
        states = ('DEFINED', 'NO_COMPATIBLE_PATH', 'OLD_GRAPH_ABNORMAL', 'LOCAL_CONFLICT', 'STARTUP_MODEL_UNDEFINED', 'NO_SOFT_OBSERVATION')
        if candidate['model_status'] not in states:
            raise ValueError('unknown original model status')
        defined = candidate['model_status'] == 'DEFINED'
        expected = ('SOFT_MODEL_BEST' if candidate['best_count'] == 1 else 'MODEL_TIE') if defined else candidate['model_status']
        if candidate['L_status'] != expected:
            raise ValueError('original L status/count contradiction')
        for key in ('model_index', 'candidate_index'):
            n = candidate[key]
            if n is not None and (type(n) is not int or not 0 <= n < DOMAIN):
                raise ValueError('illegal original candidate index')
        if defined and candidate['best_count'] == 1:
            n = candidate['model_index']
            if n is None or node['interval'] is None or not node['interval'][0] <= n <= node['interval'][1]:
                raise ValueError('unique original L candidate outside original N')
            if node['interval'][0] < 4 or node['status'] == 'LOCAL_CONFLICT':
                raise ValueError('DEFINED contradicts startup/conflict state')
            expected_index = n if hard['qualification'] == 'SUPPORTED_RELATION_SET' else None
            if candidate['candidate_index'] != expected_index:
                raise ValueError('candidate/model/qualification contradiction')
        elif candidate['model_index'] is not None or candidate['candidate_index'] is not None:
            raise ValueError('nonunique or undefined model has a candidate')
    if any(a > b for a, b in zip(H.values(), list(H.values())[1:])):
        raise ValueError('old H has an inversion')
    for r, relation in enumerate(hard['relations']):
        if type(relation['received_left']) is not int or relation['received_left'] != r:
            raise ValueError('old relation coordinate mismatch')
        blocked = graph['abnormal'] or any(hard['nodes'][i]['status'] in ('LOCAL_CONFLICT', 'IMPLEMENTATION_UNRESOLVED') for i in (r, r+1))
        feasible = graph['path_status'] != 'NO_COMPATIBLE_PATH'
        delta = relation['delta']
        if feasible:
            if not isinstance(delta, list) or len(delta) != 2 or any(type(n) is not int for n in delta) or not 0 <= delta[0] <= delta[1] < DOMAIN:
                raise ValueError('illegal old relation delta interval')
            left, right = hard['nodes'][r]['interval'], hard['nodes'][r+1]['interval']
            if delta != [max(0, right[0]-left[1]), right[1]-left[0]]:
                raise ValueError('old relation delta contradicts original intervals')
        elif delta is not None:
            raise ValueError('infeasible old relation has a delta interval')
        expected_kind = (None if blocked or not feasible else 'REPEAT' if delta == [0, 0]
                         else 'NORMAL' if delta == [1, 1] else 'GAP' if delta[0] > 1 else 'AMBIGUOUS')
        if relation['kind'] != expected_kind:
            raise ValueError('old relation kind contradicts original graph')
        status = relation['status']
        if (status not in ('EXACT', 'AMBIGUOUS', 'NO_COMPATIBLE_PATH', 'LOCAL_CONFLICT_UNRESOLVED')
                or (status == 'LOCAL_CONFLICT_UNRESOLVED') != bool(blocked)
                or (not blocked and (status == 'NO_COMPATIBLE_PATH') != (not feasible))):
            raise ValueError('old relation status contradicts adjacent public nodes/path')
        endpoints = relation['output_endpoints']
        if endpoints is None:
            if relation['output_missing_indices'] is not None or relation['output_repeated_index'] is not None:
                raise ValueError('old relation event output has no exact endpoints')
            continue  # A nonempty old kind is not an exact old relation.
        if blocked or not isinstance(endpoints, list) or len(endpoints) != 2 or any(type(n) is not int for n in endpoints) or status != 'EXACT':
            raise ValueError('invalid old exact relation endpoints/status')
        if r not in H or r+1 not in H or endpoints != [H[r], H[r+1]]:
            raise ValueError('old exact relation contradicts H')
        expected = event(*endpoints)
        for key in ('kind', 'output_missing_indices', 'output_repeated_index'):
            if relation[key] != expected[key]:
                raise ValueError('old exact relation event contradiction')
        if relation['delta'] != [expected['delta'], expected['delta']]:
            raise ValueError('old exact relation delta contradiction')
    return graph, H


def submit(hard, local):
    """Consume complete received-side semantics, preserving originals even on failure."""
    out = dict(status='VIEW_UNRESOLVED', old_hard=deepcopy(hard), original_L=deepcopy(local),
               new_nodes=[], new_relations=[], final_nodes=[], final_relations=deepcopy(hard.get('relations', [])) if isinstance(hard, dict) else [],
               S=[], S_H=[], T=[], inversion_participants=[], proposed=[], attrition=[], error=None)
    try:
        out['final_nodes'], out['old_node_errors'] = _preserve_old_nodes(hard)
        if out['old_node_errors']:
            raise ValueError('unreadable or illegal old node rows; see old_node_errors')
        graph, H = _validate(hard, local)
        bounds = [[int(x) for x in pair] for pair in graph['bounds']]
        if graph['abnormal'] or graph['path_status'] == 'NO_COMPATIBLE_PATH' or graph['qualification'] != 'SUPPORTED_RELATION_SET':
            out.update(status='OLD_GRAPH_UNSUPPORTED', error='original qualification/path/graph does not permit additions')
            return out
        S = {r: row['candidate_index'] for r, row in enumerate(local['nodes'])
             if r not in H and row['model_status'] == 'DEFINED' and row['best_count'] == 1}
        SH, T, conflicts, proposed = prune_candidates(H, S)
        out.update(S=list(S), S_H=sorted(SH), T=list(T), inversion_participants=sorted(conflicts), proposed=list(proposed))
        for r in range(len(bounds)):
            reason = ('OLD' if r in H else 'NOT_ELIGIBLE' if r not in S else 'H_INVERSION' if r in SH
                      else 'T_INVERSION_PARTICIPANT' if r in conflicts else 'PROPOSED')
            out['attrition'].append(dict(received_frame=r, reason=reason))
        feasible, propagated = extend(bounds, H | proposed)
        out['extension_check'] = dict(feasible=feasible, bounds=propagated)
        if not feasible:
            out.update(status='NOT_EXTENDABLE', error='frozen H and proposed nodes cannot extend in original intervals')
            return out
        for r, n in proposed.items():
            row = out['final_nodes'][r]
            row.update(output_index=n, origin='NEW')
            out['new_nodes'].append(deepcopy(row))
        final = H | proposed
        for r, original in enumerate(hard['relations']):
            if original['output_endpoints'] is None and r in final and r+1 in final:
                row = dict(received_left=r, status='CONDITIONAL_EXACT', **event(final[r], final[r+1]))
                out['new_relations'].append(row)
                out['final_relations'][r] = deepcopy(row)
        out['status'] = 'CONDITIONAL_OUTPUTS_SAVED'
    except (ValueError, KeyError, TypeError, IndexError) as error:
        out.update(status='SEMANTIC_UNRESOLVED', error=f'{type(error).__name__}: {error}')
    return out
