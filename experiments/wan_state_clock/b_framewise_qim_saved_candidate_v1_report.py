"""Optional standard-library saved-candidate report. Missing/corrupt fields remain unknown."""
from .b_framewise_qim_saved_candidate_v1_records import recover, BUDGETS, denominators, LAYERS


def recovery_summary(snapshot):
    def count(value, limit): return type(value) is int and 0 <= value <= limit
    saved, rows = snapshot.get('saved', {}), snapshot.get('rows', {})
    calls, states = {}, {}
    for key, limit in BUDGETS.items():
        row = saved.get('counts', {}).get(key, {})
        a, b = (row.get('started'), row.get('returned')) if isinstance(row, dict) else (None, None)
        ok = count(a, limit) and count(b, a)
        calls[key] = dict(planned=limit, started=a if ok else None, returned=b if ok else None)
    for row in rows.values():
        status = row.get('status', 'UNRESOLVED_RECEIPT'); states[status] = states.get(status, 0)+1
    evaluation = saved.get('evaluation', {})
    layers = []
    source_rows = evaluation.get('time', [])
    if not isinstance(source_rows, list): source_rows = []
    for layer in LAYERS:
        selected = [row for row in source_rows if isinstance(row, dict) and row.get('key') == 'QIM/'+layer]
        row = selected[0] if len(selected) == 1 else {}
        known = row.get('observed_active_slots'); errors = row.get('bit_errors')
        ok = (row.get('status') == 'MEASURED' and row.get('read_status') == 'RETURNED'
              and row.get('fixed_slots') == 114030 and row.get('active_slots') == 112770
              and count(known, 112770) and row.get('erased_active_slots') == 112770-known
              and (errors is None if known == 0 else count(errors, known))
              and count(row.get('wrong_origin_candidate_count'), 905))
        layers.append(dict(key='QIM/'+layer, fixed_slots=114030, active_slots=112770,
            read_status=rows.get('QIM/'+layer, {}).get('status', 'UNRESOLVED_RECEIPT'),
            measurement='MEASURED' if ok else 'UNMEASURED_OR_POST_UNRESOLVED',
            observed_active_slots=known if ok else None, bit_errors=errors if ok else None,
            erased_active_slots=row.get('erased_active_slots') if ok else None))
    support = evaluation.get('support')
    support_ok = isinstance(support, dict)
    identity = evaluation.get('identity', {})
    for arm in ('BASE', 'QIM'):
        items = identity.get(arm, []) if isinstance(identity, dict) else []
        valid = isinstance(items, list) and len(items) == 1280
        coordinates = set()
        if valid:
            for row in items:
                if not isinstance(row, dict): valid = False; break
                coordinate = tuple(row.get(k) for k in ('start','phase','region','local_row','bit'))
                start, phase, region, local_row, bit = coordinate
                if (type(start) is not int or start not in (0,40) or type(phase) is not int or phase not in range(4)
                        or region not in ('R0','R1') or type(local_row) is not int or local_row not in range(1,11)
                        or type(bit) is not int or bit not in range(8) or type(row.get('measured')) is not bool): valid=False; break
                if row['measured']:
                    if (row.get('read_status') != 'RETURNED' or row.get('measurement_status') not in ('AVAILABLE','ERASED','UNSUPPORTED')
                            or (not count(row.get('value'), 1) if row['measurement_status']=='AVAILABLE' else row.get('value') is not None)):
                        valid=False; break
                elif row.get('value') is not None: valid=False; break
                eligible = phase == 0
                if row.get('eligible') is not eligible or row.get('group') != (min(3, ((start+phase+1+4*(local_row-1))//4)//11) if eligible else None):
                    valid=False; break
                coordinates.add(coordinate)
        support_ok = support_ok and valid and len(coordinates) == 1280
    for key, size in (('local_1280',1280), ('eligible_320',320)):
        row = support.get(key, {}) if isinstance(support, dict) else {}
        common = row.get('comparable_measured_slots') if isinstance(row, dict) else None
        valid = isinstance(row, dict) and row.get('fixed_slots') == size and count(common, size)
        if valid:
            valid = (row.get('unmeasured_pairs') == size-common
                     and all(count(row.get(k), size) for k in ('base_unmeasured','qim_unmeasured'))
                     and all(row.get(k) is None if common == 0 else count(row.get(k), common)
                             for k in ('both_supported','added_support','lost_support','both_without_support')))
        if valid and common:
            valid = sum(row[k] for k in ('both_supported','added_support','lost_support','both_without_support')) == common
        available = row.get('common_available_eligible_slots') if isinstance(row, dict) else None
        valid = valid and count(available, min(320, common) if type(common) is int else 0)
        if valid:
            for arm in ('BASE', 'QIM'):
                errors = row.get('common_errors', {}).get(arm) if isinstance(row.get('common_errors'), dict) else 'INVALID'
                ber = row.get('common_ber', {}).get(arm) if isinstance(row.get('common_ber'), dict) else 'INVALID'
                if available and evaluation.get('truth_status') == 'AVAILABLE':
                    valid = valid and count(errors, available) and type(ber) in (int, float) and ber == errors/available
                else: valid = valid and errors is None and ber is None
        support_ok = support_ok and valid
    final = saved.get('result', {})
    unresolved = (any(status in states for status in ('STARTED','UNRESOLVED_RECEIPT','UNRESOLVED_INTERRUPTED'))
                  or any(row['started'] is None for row in calls.values()))
    final_status = final.get('status')
    if not isinstance(final_status, str) or final_status not in ('COMPLETED_FIXED_RECORDS','STOPPED_FIXED_ATTEMPT'):
        final_status = 'UNRESOLVED_FINAL_RECORD'
    status = 'UNRESOLVED_RECORDS' if unresolved else final_status
    if status == 'COMPLETED_FIXED_RECORDS' and (not support_ok or evaluation.get('status') != 'POST_RECORDED'
            or any(row.get('status') != 'RETURNED' for row in rows.values())
            or any(row['measurement'] != 'MEASURED' for row in layers)): status='UNRESOLVED_POST_RECORD'
    return dict(status=status, saved_result_status=final.get('status'), stop=final.get('stop'),
                state_counts=states, denominators=denominators(), calls=calls,
                linear=saved.get('linear'), native_status=saved.get('native', {}).get('status', 'UNMEASURED'),
                time=layers, support=support if support_ok else None,
                truth_status=evaluation.get('truth_status', 'UNRESOLVED'),
                base_time_comparison=[dict(layer=row.get('layer'), status=row.get('status')) for row in evaluation.get('saved_base_time_comparison', []) if isinstance(row, dict)]
                    if isinstance(evaluation.get('saved_base_time_comparison'), list) else None,
                legacy_saved_lost_support=evaluation.get('legacy_saved_lost_support'),
                solver=dict(status='NOT_RUN_SAVED_CANDIDATE', iterations=None), optimality='UNPROVEN',
                read_errors=snapshot.get('read_errors', {}), physical_calls_this_invocation=0)


def display(output):
    summary = recovery_summary(recover(output))
    for key, value in summary.items(): print(key, value)
    print('旧 lost_support 中未执行的 QIM 读量属于未测，不能解释成真实损伤。')
    print('固定保存候选诊断；原求解未决与最优性未证保持，不据此宣布媒体/盲同步/身份认证/画质 PASS。')
    return summary
