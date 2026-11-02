"""Post-only n01 time/control/payload report; no source selection or feedback."""
from pathlib import Path
from collections import Counter
import json
from . import b_native_y_source_confirmation_v1_records as rec


def private_message(source, output):
    rec.require_whole_blind(output)
    path = Path(source)/'protocol.json'
    config = rec.load(path)['config']
    writer, generation = config['writer'], config['generation']
    bits = writer['bits']
    if not isinstance(bits, list) or len(bits) != 32 or any(type(b) is not int or b not in (0, 1) for b in bits):
        raise ValueError('original writer.bits must be 32 binary integers; no default message')
    if writer.get('key') != 'watermark':
        raise ValueError('original DCT2 key must be watermark')
    for key, expected in dict(height=320, width=512, frames=181, fps=8).items():
        if type(generation.get(key)) not in (int, float) or generation[key] != expected:
            raise ValueError('original generation geometry mismatch: '+key)
    sources = config['sources']
    if not isinstance(sources, list) or any(not isinstance(s, dict) for s in sources):
        raise ValueError('original sources list required')
    matched = [s for s in sources if s.get('id') == rec.SOURCE_ID]
    if len(matched) != 1 or matched[0].get('seed') != 2026102501:
        raise ValueError('fixed original n01 source and seed required')
    value = dict(message_bits=bits, message_source=str(path)+'#config.writer.bits', source_record=matched[0],
                 generation=generation, key=writer['key'], management_fields_are_reader_inputs=False)
    rec.dump(Path(output)/'private/post_fixture.json', value)
    return value


def evaluate_time(output, rows, truth_path, *, whole_output):
    rec.require_whole_blind(whole_output)
    rec.require_time_blind(output, rows)
    from .b_native_y_conditional_v1_evaluation import node_status, edge_status
    from .b_conditional_time_output_v1_evaluation import evaluate_view
    output = Path(output)
    truth_error = None
    try:
        truth = rec.load(truth_path)['views']
        for view, n in zip(rec.VIEWS, rec.LENGTHS):
            mapping = truth[view]
            if not isinstance(mapping, list) or len(mapping) != n or any(type(t) is not int or not 0 <= t < 181 for t in mapping):
                raise ValueError('complete fixed truth mapping required: '+view)
    except (OSError, ValueError, KeyError, TypeError) as error:
        truth, truth_error = {}, str(error)
    complete = not truth_error and all(r['status'] == 'RETURNED' for r in rows.values())
    resources = rec.load(output/'resources.json')
    complete &= not bool(resources.get('completion_errors') or resources.get('error'))
    totals, views, bad, structural, profiles, events = Counter(), {}, [], [], {}, []
    folder = output/'post'
    folder.mkdir(exist_ok=True)
    known = ('CORRECT', 'WRONG', 'UNRESOLVED')
    def at(values, i):
        return values[i] if isinstance(values, list) and i < len(values) else None
    with (folder/'nodes.jsonl').open('w') as nodes_file, (folder/'edges.jsonl').open('w') as edges_file:
        for unit in rec.units():
            uid, arm, view, n = (unit[k] for k in ('id', 'arm', 'view', 'frames'))
            path = output/unit['folder']
            loaded, load_errors = {}, {}
            for op, key, name in [('probe', 'probe', 'probe.json'), ('F_compatibility', 'F', 'compatibility_F.json'),
                                  ('H', 'hard', 'Y_H.json'), ('C', 'C', 'C.json'), ('L', 'L', 'candidates.json'),
                                  ('submit', 'submission', 'submission.json')]:
                if rows[uid+'/'+op]['started'] == 0:
                    continue
                try:
                    loaded[key] = rec.load(path/name)
                    if not isinstance(loaded[key], dict):
                        raise ValueError('object required')
                except (OSError, ValueError, TypeError) as error:
                    loaded.pop(key, None); load_errors[key] = str(error)
            h, final = loaded.get('hard', {}), loaded.get('submission', {})
            mapping = truth.get(view, [None]*n)
            profiles[uid] = loaded.get('probe', {}).get('profile')
            local = Counter()
            preservation, issues = None, []
            if all(k in loaded for k in ('hard', 'L', 'submission')):
                try:
                    compared = evaluate_view(dict(hard=h, L=loaded['L']), final, mapping, n)
                    preservation = compared['preservation']
                    issues = compared['structural_errors'] + ([] if preservation else ['original H/L not preserved'])
                    structural.extend(dict(unit=uid, error=e) for e in issues)
                except (ValueError, TypeError, KeyError, IndexError) as error:
                    issues.append(str(error))
                    structural.append(dict(unit=uid, error=str(error)))
            legal_final = preservation is True and not issues
            for layer, op, graph in [('H', 'H', h), ('final', 'submit', final)]:
                node_set = graph.get('nodes' if layer == 'H' else 'final_nodes')
                edge_set = graph.get('relations' if layer == 'H' else 'final_relations')
                missing = 'NOT_EXECUTED' if rows[uid+'/'+op]['started'] == 0 else 'UNKNOWN'
                complete &= isinstance(node_set, list) and len(node_set) == n
                complete &= isinstance(edge_set, list) and len(edge_set) == n-1
                for i, t in enumerate(mapping):
                    row = at(node_set, i)
                    geometry_status = node_status(row, t, i, missing)
                    status = 'FALSE_ACCEPT' if arm == 'BASE' and geometry_status in ('CORRECT','WRONG') else geometry_status
                    local[layer+'/nodes/'+status] += 1
                    if type(t) is int and t >= 4:
                        local[layer+'/nonstartup_nodes/'+status] += 1
                    if status not in (known if arm == 'QIM' else ('FALSE_ACCEPT','UNRESOLVED')): complete = False
                    if (layer == 'H' or legal_final) and status in ('WRONG','FALSE_ACCEPT'):
                        bad.append(dict(unit=uid, layer=layer, kind='BASE_EXACT_NODE_ACCEPT' if arm == 'BASE' else 'TIME_POSITION', coordinate=i))
                    nodes_file.write(json.dumps(dict(unit=uid, arm=arm, view=view, layer=layer, received_frame=i,
                        truth=t if arm == 'QIM' else None, status=status, geometric_match_status=geometry_status,
                        legal_final=legal_final if layer == 'final' else None, saved=row), allow_nan=False)+'\n')
                for i in range(n-1):
                    a, b = mapping[i:i+2]
                    row = at(edge_set, i)
                    result = edge_status(row, a, b, i, missing)
                    geometry_status = result['status']
                    status = 'FALSE_ACCEPT' if arm == 'BASE' and geometry_status in ('CORRECT','WRONG') else geometry_status
                    result['status'] = status
                    local[layer+'/edges/'+status] += 1
                    if type(a) is int and type(b) is int and a >= 4 and b >= 4:
                        local[layer+'/nonstartup_edges/'+status] += 1
                    if status not in (known if arm == 'QIM' else ('FALSE_ACCEPT','UNRESOLVED')): complete = False
                    if (layer == 'H' or legal_final) and status in ('WRONG','FALSE_ACCEPT'):
                        bad.append(dict(unit=uid, layer=layer, kind='BASE_EXACT_EDGE_ACCEPT' if arm == 'BASE' else 'TIME_EXACT_EDGE', coordinate=i))
                    item = dict(unit=uid, arm=arm, view=view, layer=layer, received_left=i,
                                truth_endpoints=[a, b] if arm == 'QIM' else None, status=status, detail=result,
                                geometric_match_status=geometry_status, legal_final=legal_final if layer == 'final' else None, saved=row)
                    edges_file.write(json.dumps(item, allow_nan=False)+'\n')
                    if arm == 'QIM' and type(a) is int and type(b) is int and b-a != 1:
                        events.append(item)
            qualification = h.get('qualification')
            false_accept = arm == 'BASE' and qualification == 'SUPPORTED_RELATION_SET'
            if false_accept: bad.append(dict(unit=uid, kind='BASE_SUPPORTED_RELATION_SET', exact_outputs_required=False))
            code_hits = sum(len(lane.get('labels', [])) for frame in loaded.get('F', {}).get('labels', []) for lane in frame)
            cost = loaded.get('C', {})
            complete &= final.get('status') in ('CONDITIONAL_OUTPUTS_SAVED','OLD_GRAPH_UNSUPPORTED') and legal_final and not load_errors
            views[uid] = dict(arm=arm, view=view, counts=dict(local), qualification=qualification,
                BASE_qualification_false_accept=false_accept, BASE_is_identity_present_time_absent=arm == 'BASE',
                local_codeword_hits=code_hits if 'F' in loaded else None,
                exact_nodes=sum(r.get('output_index') is not None for r in final.get('final_nodes', [])),
                exact_edges=sum(r.get('output_endpoints') is not None for r in final.get('final_relations', [])),
                NEW_L_nodes=len(final.get('new_nodes', [])), NEW_L_edges=len(final.get('new_relations', [])),
                C_defined_rows=cost.get('defined_rows'), C_actual_j_scores=cost.get('comparisons'),
                C_candidate_slots=cost.get('candidate_slots'), C_state_counts=cost.get('state_counts'),
                preservation=preservation, legal_final=legal_final, structural_errors=issues, load_errors=load_errors, profile=profiles[uid])
            totals.update({arm+'/'+k: v for k, v in local.items()})
    qevents = [e for e in events if e['arm'] == 'QIM' and e['layer'] == 'final']
    exact_events = len(qevents) == 3 and all(e['status'] == 'CORRECT' for e in qevents)
    coverage = dict(nonstartup_edges_correct=totals['QIM/final/nonstartup_edges/CORRECT'],
                    nonstartup_edges_denominator=sum(sum(a >= 4 and b >= 4 for a,b in zip(m,m[1:])) for m in rec.mappings().values()),
                    nonstartup_correct=totals['QIM/final/nonstartup_nodes/CORRECT'],
                    nonstartup_denominator=sum(sum(t >= 4 for t in m) for m in rec.mappings().values()),
                    full_nodes=822, unresolved=totals['QIM/final/nodes/UNRESOLVED'])
    complete = bool(complete and not structural)
    return dict(status='COUNTEREXAMPLE' if bad else 'EVIDENCE_INCOMPLETE' if not complete else 'FINITE_TIME_EVIDENCE',
        evidence_complete=complete, counterexamples=bad, counts=dict(totals), views=views, structural_errors=structural,
        profiles=profiles, true_events=events, QIM_true_events_exact=exact_events, QIM_coverage=coverage,
        truth_error=truth_error, denominators=rec.denominators()['time'], scientific_pass=None,
        claim_ceiling='one fixed new-current-mechanism source; BASE finite control, not general FPR or source authentication',
        RGB_comparison_run=False, Y_benefit_claim=False)


def summarize(time_evaluation, payload_evaluation, stage_results, errors):
    from .b_new_source_chain_confirmation_v1_evaluation import effect_report
    timing, payload = time_evaluation or {}, payload_evaluation or {}
    bad = timing.get('counterexamples', []) + [dict(kind='PAYLOAD', **r) for r in payload.get('B_counterexamples', [])]
    per_view = {v: payload.get('views', {}).get(v, {}).get('arms', {}).get('B', {}) for v in rec.VIEWS}
    full = all(r.get('full32_supported') is True and r.get('correct_supported') == 32 for r in per_view.values())
    material_errors = {stage: {key:value for key,value in stage_results.get(stage, {}).get('errors', {}).items()
                       if not (stage == 'B' and key == 'native')} for stage in ('A','B')}
    material_errors = {stage:value for stage,value in material_errors.items() if value}
    complete = bool(timing.get('evidence_complete') and payload.get('evidence_complete') and not errors and not material_errors)
    coverage = timing.get('QIM_coverage', {})
    # Coverage is reported separately; the adopted plan adds no new coverage threshold.
    time_target = bool(timing.get('QIM_true_events_exact'))
    status = ('COUNTEREXAMPLE' if bad else 'EVIDENCE_INCOMPLETE' if not complete else
              'FINITE_SINGLE_SOURCE_CONDITIONAL_READBACK' if full and time_target else 'COVERAGE_TARGET_NOT_MET')
    return dict(status=status, evidence_complete=complete, counterexamples=bad,
        time=dict(status=timing.get('status'), counts=timing.get('counts'), QIM_true_events_exact=timing.get('QIM_true_events_exact'),
                  QIM_coverage=coverage, structural_errors=timing.get('structural_errors'), profiles=timing.get('profiles')),
        payload_transfer=dict(full_five_view_target_observed=full, correct_supported=sum(r.get('correct_supported', 0) for r in per_view.values()),
            fixed_B_slots=160, per_view={v: {k: r.get(k) for k in ('correct_supported','full32_supported','output_count','group_support')} for v,r in per_view.items()}),
        synchronization_effect=effect_report(payload_evaluation),
        materialization=dict(native_status=stage_results.get('B', {}).get('native_status'),
            errors={k: stage_results.get(k, {}).get('errors', {}) for k in ('A','B')},
            technical_errors=material_errors, exact_native_target_failure_retained=True, solver_optimality_proven=False, quality_proven=False),
        source_scope='fixed n01; previously used by RANK3/RANK3R, held out only for current mechanism within inspected evidence',
        BASE='identity present; no time writer; not wholly unwatermarked null', RGB_comparison_run=False, Y_benefit_claim=False,
        source_authenticated=False, scientific_pass=None, denominators=rec.denominators(), errors=dict(errors),
        not_evaluated=['general FPR', 'source authentication', 'cross-message reliability', 'long-video recovery'],
        post_physical_observations_added=0)
