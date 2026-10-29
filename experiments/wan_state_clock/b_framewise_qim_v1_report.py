"""Optional standard-library presentation of saved records; no experiment dependency."""
from .b_framewise_qim_v1_records import (
    ARMS, LAYERS, BUDGETS, denominators, recover)


def recovery_summary(snapshot):
    """Summarize saved receipts only; no raw arrays, scoring, truth or scientific imports."""
    def count(value, maximum):
        return type(value) is int and 0 <= value <= maximum
    def time_row_valid(row):
        status = row.get('status')
        if not isinstance(status,str) or status not in ('RETURNED','NOT_STARTED','FAILED','INTERRUPTED'): return False
        if row.get('fixed_slots') != 114030: return False
        if status != 'RETURNED': return True
        fields = ('active_slots','observed_active_slots','bit_errors','erased_active_slots')
        if any(not count(row.get(k),114030) for k in fields): return False
        return (row['bit_errors'] <= row['observed_active_slots'] <= row['active_slots']
                and row['erased_active_slots']==row['active_slots']-row['observed_active_slots']
                and count(row.get('wrong_origin_candidate_count'),905))
    def identity_row_valid(row):
        required = ('start','phase','region','local_row','bit','value','status','common_complete_cell_eligible','group','correct')
        if any(k not in row for k in required): return False
        if (type(row['start']) is not int or row['start'] not in (0,40)
                or type(row['phase']) is not int or row['phase'] not in range(4)
                or row['region'] not in ('R0','R1')
                or type(row['local_row']) is not int or row['local_row'] not in range(1,11)
                or type(row['bit']) is not int or row['bit'] not in range(8)): return False
        status = row['status']
        if not isinstance(status,str) or status not in ('AVAILABLE','NOT_STARTED','NOT_ENTERED','FAILED','INTERRUPTED','UNRESOLVED','UNSUPPORTED'): return False
        available = status=='AVAILABLE'
        if (available and not count(row['value'],1)) or (not available and row['value'] is not None): return False
        full = row['common_complete_cell_eligible']
        if type(full) is not bool: return False
        if (full and not count(row['group'],3)) or (not full and row['group'] is not None): return False
        return type(row['correct']) is bool if full and available else row['correct'] is None
    def comparison_valid(row):
        if not isinstance(row,dict) or row.get('fixed_paired_slots')!=1280: return False
        if any(not count(row.get(k),1280) for k in ('common_complete_observed_slots','added_support','lost_support')): return False
        common = row['common_complete_observed_slots']
        for arm in ('base','qim'):
            errors, ber = row.get(arm+'_common_errors'), row.get(arm+'_common_ber')
            if not count(errors,common): return False
            if common==0:
                if ber is not None: return False
            elif type(ber) not in (int,float) or not 0 <= ber <= 1 or ber != errors/common: return False
        return True
    saved = snapshot.get('saved',{})
    result = saved.get('result',{}); evaluation = saved.get('evaluation',{})
    rows = snapshot.get('rows',{})
    counts = saved.get('counts',{})
    calls = {}
    for kind, planned in BUDGETS.items():
        row = counts.get(kind,{})
        started = row.get('started') if isinstance(row,dict) else None
        returned = row.get('returned') if isinstance(row,dict) else None
        valid = type(started) is int and type(returned) is int and 0 <= returned <= started <= planned
        calls[kind] = dict(planned=planned,started=started if valid else None,
            returned=returned if valid else None,status='RECORDED' if valid else 'UNRESOLVED_RECEIPT')
    state_counts = {}
    for row in rows.values():
        status = row.get('status','UNRESOLVED_RECEIPT')
        state_counts[status] = state_counts.get(status,0)+1
    layer_rows = evaluation.get('time',[])
    if not isinstance(layer_rows,list): layer_rows = []
    layers = []
    for arm in ARMS:
        for layer in LAYERS:
            key = f'{arm}/{layer}'
            matches = [x for x in layer_rows if isinstance(x,dict) and x.get('key')==key]
            candidate = matches[0] if len(matches)==1 else {}
            row = candidate if time_row_valid(candidate) else {}
            layers.append(dict(key=key,fixed_slots=181*5*2*63,read_status=rows.get(key,{}).get('status','UNRESOLVED_RECEIPT'),
                evaluation_status=row.get('status','POST_UNRESOLVED'),
                **{field:row.get(field) if row.get('status')=='RETURNED' else None for field in ('active_slots','observed_active_slots','bit_errors','erased_active_slots','wrong_origin_candidate_count')}))
    identity_rows = evaluation.get('identity',[])
    if not isinstance(identity_rows,list): identity_rows = []
    identity = []
    for arm in ARMS:
        selected = [x for x in identity_rows if isinstance(x,dict) and x.get('arm')==arm]
        complete = len(selected)==1280 and all(identity_row_valid(x) for x in selected)
        if complete:
            coordinates = {(x['start'],x['phase'],x['region'],x['local_row'],x['bit']) for x in selected}
            complete = len(coordinates)==1280
        known = [x for x in selected if x.get('status')=='AVAILABLE' and type(x.get('value')) is int and x['value'] in (0,1)]
        scored = sum(type(x.get('correct')) is bool for x in selected) if complete else None
        identity.append(dict(arm=arm,fixed_slots=1280,recorded_rows=len(selected),
            observed_bits=len(known) if complete else None,
            erased_or_unread_bits=1280-len(known) if complete else None,
            full_cell_scored_bits=scored,
            full_cell_errors=sum(x.get('correct') is False for x in selected) if scored else None,
            status='SAVED_ROWS' if complete else 'POST_UNRESOLVED'))
    comparison = evaluation.get('identity_comparison')
    comparison_ok = comparison_valid(comparison)
    unresolved = any(x in state_counts for x in ('UNRESOLVED_RECEIPT','UNRESOLVED_INTERRUPTED','STARTED'))
    final_status = result.get('status')
    status = ('UNRESOLVED_RECORDS' if unresolved or any(c['status']!='RECORDED' for c in calls.values())
              else final_status if final_status in ('COMPLETED_FIXED_RECORDS','STOPPED_FIXED_ATTEMPT') else 'UNRESOLVED_FINAL_RECORD')
    if status=='COMPLETED_FIXED_RECORDS' and any(row.get('status')!='RETURNED' for row in rows.values()):
        status='UNRESOLVED_RECORDS'
    if status=='COMPLETED_FIXED_RECORDS' and (result.get('evaluation_status')!='POST_RECORDED'
            or evaluation.get('status')=='POST_UNRESOLVED'
            or any(row['evaluation_status']=='POST_UNRESOLVED' for row in layers)
            or any(row['status']=='POST_UNRESOLVED' for row in identity)
            or not comparison_ok):
        status='UNRESOLVED_POST_RECORD'
    solver = saved.get('solver',saved.get('solver_progress',{}))
    metrics = solver.get('metrics',{})
    if not isinstance(metrics,dict): metrics = {}
    budget = metrics.get('budget',{})
    if not isinstance(budget,dict): budget = {}
    solver_summary = {k:solver.get(k) for k in ('status','reason','iterations','estimated_bytes')}
    solver_summary['metrics'] = {k:metrics.get(k) for k in ('split','target','a_stationarity','r_stationarity')}
    solver_summary['budget'] = {k:budget.get(k) for k in ('linf','outside','excess')}
    return dict(status=status,saved_result_status=final_status,stop=result.get('stop'),
        evaluation_status=result.get('evaluation_status','POST_UNRESOLVED'),denominators=denominators(),
        state_counts=state_counts,calls=calls,
        solver=solver_summary,
        native_status=saved.get('native',{}).get('status','UNRESOLVED'),time_layers=layers,identity=identity,
        identity_comparison=comparison if comparison_ok else None,read_errors=snapshot.get('read_errors',{}),
        physical_calls_this_invocation=0,claim_ceiling='saved-record display only; no media/identity/synchronization/quality scientific PASS')


def display(output):
    summary = recovery_summary(recover(output))
    print('恢复状态:', summary['status'], '保存的最终状态:', summary['saved_result_status'])
    print('停止原因:', summary['stop'], '后评:', summary['evaluation_status'])
    print('固定分母:', summary['denominators'], '逐项状态:', summary['state_counts'])
    print('调用记录（None = 未知）:')
    for name, row in summary['calls'].items(): print(name, row)
    print('求解:', summary['solver'], '原生物化:', summary['native_status'])
    print('逐层保存读量（不是重评分；None = 未知）:')
    for row in summary['time_layers']: print(row)
    print('身份支持/错误/擦除:')
    for row in summary['identity']: print(row)
    print('共同完整cell比较:', summary['identity_comparison'])
    for name, error in summary['read_errors'].items(): print('记录不可用:', name, error)
    print('这里只显示已有记录，不据此推断媒体/身份/盲同步/画质的科学 PASS。')
    print('输出目录:', output)
    return summary
