"""Frozen relative singletons to existing FULL-cell/CDC32 semantics; no IO."""
from copy import deepcopy
from . import b_time_payload_cdc32_v1 as previous
from . import b_cdc32_aggregation_v1 as cdc

SCOPES = dict(B_NEW='RELATIVE_TIME_CONDITIONAL', U_NEW='RECEIVED_CLOCK_FOUR_PHASE',
              W_NEW='FIXED_PLUS_ONE_TIME_SHIFT', K_NEW='RELATIVE_TIME_CONDITIONAL',
              P_BASE='PAIRED_GIVEN_TIME')


def saved_relative(value, length):
    """Copy only received coordinates, conditional singletons and their origins."""
    try:
        nodes = value['conditional_nodes']
        if not isinstance(nodes, list) or len(nodes) != length:
            raise ValueError('complete relative conditional nodes required')
        mapping, origins = [], []
        for r, node in enumerate(nodes):
            if not all(k in node for k in ('conditional_candidates','conditional_position','status')):
                raise ValueError('explicit candidate/position/status fields required')
            if type(node.get('received_frame')) is not int or node['received_frame'] != r:
                raise ValueError('ordered received coordinates required')
            candidates, position, status = (node.get(k) for k in
                                             ('conditional_candidates', 'conditional_position', 'status'))
            if candidates is not None and (not isinstance(candidates, list) or
                    any(type(n) is not int or not 0 <= n < 92672 for n in candidates) or
                    len(set(candidates)) != len(candidates)):
                raise ValueError('invalid conditional candidate list')
            if position is not None:
                if (type(position) is not int or not 0 <= position < 92672 or candidates != [position]
                        or status not in ('UNCHANGED_SINGLETON', 'CONDITIONAL_STRICT_MINIMUM')):
                    raise ValueError('position must be an explicit saved conditional singleton')
            elif status in ('UNCHANGED_SINGLETON', 'CONDITIONAL_STRICT_MINIMUM'):
                raise ValueError('saved singleton lacks position')
            if status not in ('UNCHANGED_SINGLETON','CONDITIONAL_STRICT_MINIMUM','UNCHANGED_EMPTY',
                              'ORIGINAL_UNRESOLVED','COMPARISON_UNRESOLVED','COMPARISON_TIE'):
                raise ValueError('known saved conditional status required')
            if status == 'UNCHANGED_EMPTY' and candidates != []:
                raise ValueError('empty origin must retain its empty candidates')
            mapping.append(position)
            origins.append(dict(received_frame=r, status=status, conditional_position=position))
        return dict(status='READY', mapping=mapping, origins=origins,
                    formal_output=None, presence_claim=False)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        return dict(status='SEMANTIC_UNRESOLVED', mapping=[None]*length,
                    error=str(value.get('input_error', error)) if isinstance(value,dict) else str(error),
                    origins=[], formal_output=None, presence_claim=False)


def shifted(time):
    result = deepcopy(time)
    result['mapping'] = [n+1 if n is not None and 0 <= n+1 <= 180 else None for n in time['mapping']]
    result['control_transform'] = 'n+1; outside 0..180 -> None; no wrap'
    return result


def plans(input_id, length, time, strategy, *, borrowed_from=None):
    """All strategies use the same four-phase bank and first earliest-FULL rule."""
    if strategy not in SCOPES or len(time['mapping']) != length:
        raise ValueError('fixed strategy and complete received mapping required')
    result = []
    for query in range(3):
        # B selects all four phases in the original planner. No old p0-U or W rotation.
        plan = previous._plan(input_id, length, query, time, 'B')
        plan.update(arm=strategy, mapping_scope=SCOPES[strategy], key_id='key01' if strategy == 'K_NEW' else 'key00',
                    borrowed_from=borrowed_from, control_transform=time.get('control_transform'))
        for group in plan['groups']:
            for ref in group['references']:
                ref.update(candidate_id=strategy, borrowed_from=borrowed_from,
                           control_transform=time.get('control_transform'))
        result.append(plan)
    return cdc.reconcile_plans(result)


def new_input(input_id, length, relative):
    time = saved_relative(relative, length)
    baseline = dict(status='READY', mapping=list(range(length)), source='PUBLIC_RECEIVED_CLOCK')
    b = plans(input_id, length, time, 'B_NEW')
    k = deepcopy(b)
    for plan in k:
        plan.update(arm='K_NEW', key_id='key01')
        for group in plan['groups']:
            for ref in group['references']:
                ref['candidate_id'] = 'K_NEW'
    return dict(time=time, bank=previous.bank(input_id, length), strategies=dict(
        B_NEW=b, U_NEW=plans(input_id, length, baseline, 'U_NEW'),
        W_NEW=plans(input_id, length, shifted(time), 'W_NEW'), K_NEW=k))


def paired_input(input_id, length, new_frozen, new_input_id):
    time = deepcopy(new_frozen['time'])
    return dict(time=time, bank=previous.bank(input_id, length), strategies=dict(
        P_BASE=plans(input_id, length, time, 'P_BASE', borrowed_from=new_input_id)))
