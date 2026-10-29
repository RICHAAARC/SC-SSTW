"""Post-only saved observations. Missing measurements never count as lost support."""
from pathlib import Path
import time
from .b_framewise_qim_saved_candidate_v1_records import load, dump, LAYERS, denominators


def identity_rows(directory, arm, state):
    rows = []
    for start in (0, 40):
        for phase in range(4):
            key = f'{arm}/identity/{start}/{phase}'
            receipt = state.get(key, {}) if isinstance(state, dict) else {}
            read_status = receipt.get('status') if isinstance(receipt, dict) else None
            saved, error = None, None
            if read_status == 'RETURNED':
                try: saved = load(Path(directory)/'identity'/(key+'.json'))
                except (OSError, ValueError) as exc: error = str(exc)
            for region in ('R0', 'R1'):
                for local_row in range(1, 11):
                    first = start+phase+1+4*(local_row-1)
                    eligible = first % 4 == 1
                    for bit in range(8):
                        item = None
                        if saved is not None:
                            try: item = saved['regions'][region]['local_cells'][local_row-1]['bits'][bit]
                            except (KeyError, IndexError, TypeError): pass
                        status = item.get('status') if isinstance(item, dict) else None
                        value = item.get('value') if isinstance(item, dict) else None
                        supported = status == 'AVAILABLE' and type(value) is int and value in (0, 1)
                        measured = read_status == 'RETURNED' and (supported or (status in ('ERASED','UNSUPPORTED') and value is None))
                        rows.append(dict(arm=arm, start=start, phase=phase, region=region, local_row=local_row, bit=bit,
                            read_status=read_status, slot_status=status, measured=measured,
                            value=value if measured and supported else None,
                            measurement_status=('AVAILABLE' if supported else status) if measured else 'UNMEASURED',
                            eligible=eligible, group=min(3, (first//4)//11) if eligible else None,
                            error=error if error else None))
    return rows


def support_comparison(base, qim, truth):
    def one(pairs, eligible_only):
        selected = [(a, b) for a, b in pairs if not eligible_only or a['eligible']]
        comparable = [(a, b) for a, b in selected if a['measured'] and b['measured']]
        common = [(a, b) for a, b in comparable if a['value'] is not None and b['value'] is not None and a['eligible']]
        observed = len(comparable)
        errors = {arm:sum(row['value'] != truth[row['group']*8+row['bit']] for pair in common for row in [pair[index]])
                  if common and truth is not None else None for index, arm in enumerate(('BASE','QIM'))}
        return dict(fixed_slots=len(selected), comparable_measured_slots=observed,
            unmeasured_pairs=len(selected)-observed,
            base_unmeasured=sum(not a['measured'] for a, b in selected),
            qim_unmeasured=sum(not b['measured'] for a, b in selected),
            both_supported=sum(a['value'] is not None and b['value'] is not None for a, b in comparable) if observed else None,
            added_support=sum(a['value'] is None and b['value'] is not None for a, b in comparable) if observed else None,
            lost_support=sum(a['value'] is not None and b['value'] is None for a, b in comparable) if observed else None,
            both_without_support=sum(a['value'] is None and b['value'] is None for a, b in comparable) if observed else None,
            common_available_eligible_slots=len(common), common_errors=errors,
            common_ber={arm:errors[arm]/len(common) if errors[arm] is not None else None for arm in errors})
    if len(base) != 1280 or len(qim) != 1280: raise ValueError('fixed identity row coverage required')
    keys = ('start','phase','region','local_row','bit')
    if any(tuple(a[k] for k in keys) != tuple(b[k] for k in keys) for a, b in zip(base, qim)):
        raise ValueError('fixed identity coordinates must align')
    pairs = list(zip(base, qim))
    return dict(local_1280=one(pairs, False), eligible_320=one(pairs, True),
                rule='support changes require both reads RETURNED and AVAILABLE/ERASED/UNSUPPORTED; BER additionally requires both AVAILABLE and eligible')


def time_rows(output, state, arm='QIM'):
    import numpy as np
    from main.tube_state import b_framewise_qim_v1 as p
    js, bits = p.schedule()
    truth = bits.reshape(181, 2, 5, 63).transpose(1, 2).numpy()
    active = (js >= 0).reshape(181, 2, 5).transpose(1, 2).numpy()[..., None]
    result = []
    for layer in LAYERS:
        key = arm+'/'+layer
        receipt = state.get(key, {}) if isinstance(state, dict) else {}
        read_status = receipt.get('status') if isinstance(receipt, dict) else None
        row = dict(key=key, read_status=read_status, fixed_slots=114030,
                   active_slots=112770, status='UNMEASURED', observed_active_slots=None,
                   erased_active_slots=None, bit_errors=None, wrong_origin_candidate_count=None)
        if read_status == 'RETURNED':
            try:
                with np.load(Path(output)/'raw'/(key+'.npz'), allow_pickle=False) as raw:
                    soft, mask = raw['soft'], raw['mask']
                    if soft.shape != (181, 5, 2, 63) or mask.shape != soft.shape or mask.dtype != np.bool_:
                        raise ValueError('saved raw shape/mask')
                    known = mask & np.isfinite(soft) & (soft != 0) & active
                    count = int(known.sum())
                    row.update(status='MEASURED', observed_active_slots=count,
                        erased_active_slots=112770-count,
                        bit_errors=int(((soft > 0) != truth)[known].sum()) if count else None)
                candidates = load(Path(output)/'reads'/(key+'.json'))['candidates']
                row['wrong_origin_candidate_count'] = sum(c['j'] is not None and c['j'] != int(js[c['received_frame'], c['lane']]) for c in candidates)
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                row.update(status='POST_UNRESOLVED', error=str(error), observed_active_slots=None,
                           erased_active_slots=None, bit_errors=None, wrong_origin_candidate_count=None)
        result.append(row)
    return result


def post_evaluate(output, saved, truth_source, state):
    output, saved = Path(output), Path(saved)
    try: base_state = load(saved/'state.json')
    except (OSError, ValueError): base_state = {}
    base = identity_rows(saved, 'BASE', base_state)
    qim = identity_rows(output, 'QIM', state)
    # Called only after all fixed reads have returned, failed or been terminalized.
    dump(output/'post_truth_access.json', dict(unix=time.time(), meaning='after fixed QIM read termination'))
    truth, truth_error = None, None
    try:
        truth = load(Path(truth_source)/'protocol.json')['config']['writer']['bits']
        if not isinstance(truth, list) or len(truth) != 32 or any(type(v) is not int or v not in (0, 1) for v in truth):
            raise ValueError('32 saved binary identity truth bits required')
    except (OSError, ValueError, KeyError, TypeError) as error:
        truth, truth_error = None, f'{type(error).__name__}: {error}'
    layers = time_rows(output, state)
    base_layers = time_rows(saved, base_state, 'BASE')
    time_comparison = [dict(layer=layer, status='SAVED_READS_AVAILABLE' if a['status']==b['status']=='MEASURED' else 'COMPARISON_MISSING',
                            base=a, qim=b, new_base_physical_calls=0)
                       for layer, a, b in zip(LAYERS, base_layers, layers)]
    legacy = None
    try: legacy = load(saved/'evaluation.json').get('identity_comparison', {}).get('lost_support')
    except (OSError, ValueError, AttributeError): pass
    return dict(status='POST_RECORDED' if truth_error is None and all(r['status'] != 'POST_UNRESOLVED' for r in layers) else 'POST_UNRESOLVED',
                denominators=denominators(), time=layers, saved_base_time_comparison=time_comparison,
                identity=dict(BASE=base, QIM=qim),
                support=support_comparison(base, qim, truth), truth_status='AVAILABLE' if truth is not None else 'UNRESOLVED',
                truth_error=truth_error, legacy_saved_lost_support=legacy,
                legacy_interpretation='old non-returned QIM observations are unmeasured, not demonstrated damage; old files unchanged',
                claim_ceiling='known-position saved-candidate diagnostics; no optimality, presence, blind-sync, authentication or quality PASS')
