"""Post-only fixed-denominator accounting; receives truth after blind readback."""
from collections import Counter
from pathlib import Path

from runtime.wan import b_terminal_three_group_v1 as run


def _load(path, default):
    try:
        return run.read(path)
    except (OSError, ValueError):
        return default


def support(ref, frames, group):
    received = ref.get('nominal_received_cell', [])
    mapped = [frames[x] for x in received] if len(received) == 4 and all(type(x) is int and 0 <= x < len(frames) for x in received) else []
    t = (mapped[0]+3)//4 if mapped else None
    full = t in range(1, 45) and mapped == list(range(4*t-3, 4*t+1))
    same_group = full and (t-1)//11 == group
    exact = full and t == ref.get('canonical_cell')
    return dict(evidence_id=ref.get('evidence_id'), received_frames=received,
                actual_source_frames=mapped, actual_cell=t if full else None,
                actual_full=full, actual_correct_group=same_group,
                exact_claimed_cell=exact, claimed_cell=ref.get('canonical_cell'))


def assess(decision, bits, positive, frames):
    rows = []
    supplied = decision.get('slots', [])
    for i in range(32):
        item = supplied[i] if i < len(supplied) else run.empty_slots()['slots'][i]
        value = item.get('value')
        emitted = type(value) is int and value in (0, 1)
        refs = [support(ref, frames, i//8) for ref in item.get('selected_refs', [])]
        any_group = any(r['actual_correct_group'] for r in refs)
        any_exact = any(r['actual_correct_group'] and r['exact_claimed_cell'] for r in refs)
        all_exact = bool(refs) and all(r['actual_correct_group'] and r['exact_claimed_cell'] for r in refs)
        rows.append(dict(slot=i, value=value, status=item.get('status', 'NOT_ENTERED'),
                         cause_set=item.get('cause_set', []), emitted=emitted,
                         literal_correct=(value == bits[i]) if emitted else None,
                         positive=positive, actual_support=refs,
                         any_full_correct_group=any_group, any_exact_reference=any_exact,
                         all_references_exact=all_exact,
                         null_output=emitted and not positive,
                         false_attribution=emitted and (not positive or not any_exact)))
    return dict(slots=rows, counts=dict(
        planned=32, emitted=sum(r['emitted'] for r in rows),
        literal_errors=sum(r['emitted'] and not r['literal_correct'] for r in rows),
        supported_correct=sum(r['emitted'] and positive and r['literal_correct'] and r['any_exact_reference'] for r in rows),
        null_outputs=sum(r['null_output'] for r in rows),
        false_attribution=sum(r['false_attribution'] for r in rows),
        technical_unread=sum(any(word in r['status'] for word in ('TECHNICAL', 'UNRESOLVED', 'UNSUPPORTED', 'NOT_ENTERED')) for r in rows),
        conflict_slots=sum('CONFLICT' in r['status'] for r in rows),
        no_full_correct_group=sum(r['emitted'] and not r['any_full_correct_group'] for r in rows),
        not_all_references_exact=sum(r['emitted'] and not r['all_references_exact'] for r in rows)))


def compare(left, right):
    a, b = left['slots'], right['slots']
    common = [i for i in range(32) if a[i]['emitted'] and b[i]['emitted']]
    return dict(common_slots=common, common_count=len(common),
                left_common_errors=sum(not a[i]['literal_correct'] for i in common),
                right_common_errors=sum(not b[i]['literal_correct'] for i in common),
                added_slots=[i for i in range(32) if b[i]['emitted'] and not a[i]['emitted']],
                lost_slots=[i for i in range(32) if a[i]['emitted'] and not b[i]['emitted']],
                added_supported_correct=[i for i in range(32) if not a[i]['emitted'] and b[i]['emitted']
                    and b[i]['positive'] and b[i]['literal_correct'] and b[i]['any_exact_reference']])


def geometry(frames):
    present = set(frames)
    cells = []
    nominal = set()
    for w in run.WINDOWS.values():
        p = (-frames[w]) % 4
        for r in range(1, 11):
            mapped = frames[w+p+4*r-3:w+p+4*r+1]
            if len(mapped) == 4:
                t = (mapped[0]+3)//4
                if mapped == list(range(4*t-3, 4*t+1)) and t in range(1, 45):
                    nominal.add(t)
    for t in range(1, 45):
        overlap = len(present.intersection(range(4*t-3, 4*t+1)))
        cells.append(dict(cell=t, group=(t-1)//11, present_frames=overlap,
                          status='FULL' if overlap == 4 else 'ABSENT' if overlap == 0 else 'PARTIAL',
                          nominal_true_phase_observed=t in nominal))
    return dict(cells=cells, counts=dict(Counter(c['status'] for c in cells)),
                full_not_nominally_observed=[c['cell'] for c in cells if c['status']=='FULL' and not c['nominal_true_phase_observed']])


def path_assessment(plan, host, fixture):
    d, seam = run.FIXTURES[fixture]
    gap = 'CONT' if not d else 'LEFT_GAP' if seam == 49 else 'RIGHT_GAP'
    answer = {}
    for mode in ('E2', 'M3'):
        result = plan.get('paths', {}).get(mode, {})
        rows = result.get('paths', [])
        true = next((r for r in rows if (r['a'], r['d'], r['gap']) == (host['a'], d, gap)), None)
        compatible = result.get('compatible_path_ids')
        answer[mode] = dict(status=result.get('status', 'NOT_ENTERED'),
                            true_class_retained=true.get('compatibility') if true else None,
                            compatible_class_count=result.get('compatible_class_count'),
                            compatible_seam_hypothesis_count=result.get('compatible_seam_hypothesis_count'),
                            incorrect_compatible_class_count=sum(r.get('compatibility') is True and r is not true for r in rows) if compatible is not None else None,
                            exact_seam_certified=False)
    a, b = (plan.get('paths', {}).get(m, {}).get('compatible_path_ids') for m in ('E2', 'M3'))
    answer['P3_subset_P2_when_defined'] = set(b) <= set(a) if a is not None and b is not None else None
    return answer


def _bits(record, region):
    return {int(row['local_row']): {int(b['local_bit']): b.get('value') for b in row['bits']}
            for row in record.get('regions', {}).get(region, {}).get('local_cells', [])}


def pair_rows(before, after, rows, bits, groups):
    pairs = []
    for region in ('R0', 'R1'):
        a, b = _bits(before, region), _bits(after, region)
        for row in rows:
            for bit in range(8):
                x, y = a.get(row, {}).get(bit), b.get(row, {}).get(bit)
                truth = bits[8*groups[row]+bit] if groups[row] in range(4) else None
                pairs.append(dict(region=region, row=row, bit=bit, before=x, after=y,
                                  expected=truth, paired=x in (0, 1) and y in (0, 1),
                                  before_error=None if x is None or truth is None else x != truth,
                                  after_error=None if y is None or truth is None else y != truth))
    return pairs


def evaluate(cfg, out, backend, journal):
    out = Path(out)
    if not (out/'blind_complete.json').is_file():
        raise RuntimeError('blind results must be persisted and read back before evaluation')
    bits = cfg['writer']['bits']
    tasks, quality, media_pairs, native_pairs, bundle_rows = {}, {}, [], [], []
    for host in cfg['sources']:
        h = host['id']
        for pair in ('OFF_to_P0', 'P0_to_PM'):
            left, right = pair.split('_to_')
            for scope, layers in [('full', ('preclip', 'rgb8', 'readback'))]+[(f, ('rgb8', 'readback')) for f in run.FIXTURES]:
                mapping = list(range(181)) if scope == 'full' else run.source_frames(host, scope)
                for layer in layers:
                    slot = f'{h}/quality/{scope}/{pair}/{layer}'
                    op = 'D' if layer == 'preclip' else 'quantize' if layer == 'rgb8' and scope == 'full' else 'edit' if layer == 'rgb8' else 'decode'
                    deps = [f'{h}/{arm}/{scope}/{op}' for arm in (left, right)]
                    quality_path = out/'quality'/(slot+'.json')
                    quality[slot] = journal.record(slot, lambda: backend.quality(
                        backend.load_tensor(out/'media'/h/left/scope/(layer+'.pt')),
                        backend.load_tensor(out/'media'/h/right/scope/(layer+'.pt')), mapping, pair),
                        quality_path, deps)
                    if quality[slot] is None:
                        reason = journal.data[slot]['status']
                        quality[slot] = dict(status=reason, frame_denominator=len(mapping), adjacent_denominator=len(mapping)-1,
                            cause=dict(journal.data[slot]),
                            frames=[dict(frame=t, source_frame=f, status=reason, whole=None) for t,f in enumerate(mapping)],
                            adjacent=[dict(frame_before=t, frame_after=t+1, status=reason, whole=None) for t in range(len(mapping)-1)],
                            quality_pass=None)
                        run.dump(quality_path, quality[slot])
        if host['role'] != 'eval':
            continue
        native = {}
        for arm in ('P0', 'PM'):
            slot = f'{h}/native/{arm}'
            dependency = h+'/construct' if arm == 'PM' else h+'/generate/P0'
            projected = journal.call(slot+'/DCT', lambda: backend.payload_project(
                backend.load_tensor(out/'private'/h/(arm+'_terminal.pt')), list(range(1, 45)), ('R0', 'R1')), [dependency])
            for key, secret in run.KEYS.items():
                native[arm, key] = journal.record(slot+'/'+key+'/bits', lambda key=key, secret=secret: backend.payload_decode(
                    projected, secret, f'native-{h}-{arm}', key), out/'private'/h/'native'/arm/(key+'_bits.json'), [slot+'/DCT']) or {}
        native_pairs.extend(dict(source=h, **r) for r in pair_rows(native['P0', 'key00'], native['PM', 'key00'],
                                range(1, 45), bits, {r: (r-1)//11 for r in range(1, 45)}))
        for fixture in run.FIXTURES:
            frames = run.source_frames(host, fixture)
            for window, w in run.WINDOWS.items():
                p = (-frames[w]) % 4
                records = {arm: _load(out/'observations'/h/arm/fixture/window/f'p{p}'/'key00/bits.json', {}) for arm in ('P0', 'PM')}
                groups = {r: ((frames[w+p+4*r-3]+3)//4-1)//11 for r in range(1, 11)}
                media_pairs.extend(dict(source=h, fixture=fixture, window=window, true_phase=p, **r)
                                   for r in pair_rows(records['P0'], records['PM'], range(1, 11), bits, groups))
            for arm in run.ARMS:
                for key in run.KEYS:
                    q = run.task_id(h, arm, fixture, key)
                    plan = _load(out/'blind'/q/'plan.json', {})
                    decision = _load(out/'blind'/q/'decision.json', {})
                    positive = arm == 'PM' and key == 'key00'
                    local = {r: {w: assess(decision.get('local', {}).get(r, {}).get(w, {}), bits, positive, frames)
                                 for w in run.WINDOWS} for r in ('R0', 'R1', 'CDC')}
                    aggregate = {name: assess(decision.get('aggregates', {}).get(name, {}), bits, positive, frames) for name in run.READERS}
                    tasks[q] = dict(positive=positive, local=local, aggregates=aggregate,
                                    paths=path_assessment(plan, host, fixture), geometry=geometry(frames),
                                    M_vs_E=compare(aggregate['CDC_E'], aggregate['CDC_M']),
                                    CDC_vs_union=compare(aggregate['UNION_M'], aggregate['CDC_M']),
                                    invariance=decision.get('aggregates', {}).get('invariance'))
    for i in range(72):
        private = _load(out/'private/bundles'/f'{i:02}.json', {})
        value = _load(out/'bundles'/f'{i:02}/result.json', {})
        outputs = value.get('outputs', {})
        bundle_rows.append(dict(index=i, private_relation=private, status=value.get('status', 'NOT_ENTERED'),
                                outputs={name: dict(planned=32,
                                    emitted=sum(r.get('value') in (0, 1) for r in outputs.get(name, {}).get('slots', [])),
                                    literal_errors=sum(r.get('value') in (0, 1) and r['value'] != bits[r['slot']] for r in outputs.get(name, {}).get('slots', [])),
                                    source_authenticated=outputs.get(name, {}).get('source_authenticated'),
                                    counts=outputs.get(name, {}).get('counts')) for name in ('SEP_A', 'SEP_B', 'JOIN_CONDITIONAL')},
                                conditional_only=True, scientific_positive=False))
    def pair_summary(rows):
        paired = [r for r in rows if r['paired']]
        return dict(planned=len(rows), paired=len(paired), missing=len(rows)-len(paired),
                    before_errors=sum(r['before_error'] is True for r in paired),
                    after_errors=sum(r['after_error'] is True for r in paired), rows=rows)
    totals = {}
    for name in run.READERS:
        for scope in ('positive', 'null'):
            records = [t['aggregates'][name]['counts'] for t in tasks.values() if t['positive'] == (scope == 'positive')]
            totals[name+'/'+scope] = {k: sum(r[k] for r in records) for k in records[0]} if records else {}
    return dict(status='POST_RECORDED', denominators=run.denominators(), tasks=tasks, totals=totals,
                quality=quality, media_P0_PM_pair=pair_summary(media_pairs), native_P0_PM_pair=pair_summary(native_pairs),
                bundles=bundle_rows, scientific_pass=None, quality_pass=None, CFAR=False,
                population_FPR=None, source_authentication=False,
                evidence_ceiling='two eval sources, fixed full-grid 151 protocol; conditional local payload/path/aggregation only')
