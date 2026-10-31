"""Post-only literal/geometry/context evaluation; never used by the receiver."""
from .b_time_payload_cdc32_v1_records import load, VIEWS, LENGTHS, ARMS, denominators


def _reference(ref, mapping, raw, slot):
    received=ref.get('nominal_received_cell',[]);support=ref.get('encoder_input_support',[])
    def actual(indices):
        return [mapping[r] for r in indices] if all(type(r) is int and 0<=r<len(mapping) for r in indices) else None
    frames=actual(received);actual_context=actual(support);t=ref.get('canonical_cell')
    exact=type(t) is int and len(received)==4 and frames==list(range(4*t-3,4*t+1))
    true_t=(frames[0]+3)//4 if frames and len(frames)==4 else None
    true_full=true_t in range(1,45) and frames==list(range(4*true_t-3,4*true_t+1))
    group=bool(true_full and (true_t-1)//11==slot//8)
    unit=raw.get(ref.get('view_id'),{})
    physical=(unit.get('status')=='RETURNED' and len(support)==41 and
              support==unit.get('encoder_input_support') and support==list(range(support[0],support[0]+41)))
    visible=ref.get('context',{}).get('visible_mapping',[])
    context_wrong=None if actual_context is None or len(visible)!=41 else sum(a is not None and a!=b for a,b in zip(visible,actual_context))
    actual_kind=('UNKNOWN' if actual_context is None else 'CONTIGUOUS' if all(b-a==1 for a,b in zip(actual_context,actual_context[1:])) else 'MIXED')
    return dict(evidence_id=ref.get('evidence_id'),declared_cell=t,true_cell=true_t if true_full else None,
        four_frame_exact=exact,group_exact=group,actual_physical_support=physical,
        correct_geometry_support=bool(exact and group and physical),
        encoder_received_support=support,actual_encoder_source_mapping=actual_context,
        declared_context_status=ref.get('context',{}).get('status'),actual_context_kind=actual_kind,
        context_visible_wrong=context_wrong,context_unknown=sum(n is None for n in visible),
        native_181_context_equivalent=False)


def compare(b, control):
    common=[i for i in range(32) if b[i]['literal_correct'] is not None and control[i]['literal_correct'] is not None]
    def ber(rows):return sum(rows[i]['literal_correct'] is False for i in common)/len(common) if common else None
    return dict(common_output_slots=common,common_count=len(common),B_common_BER=ber(b),control_common_BER=ber(control),
        wrong_to_correct=[i for i in common if control[i]['literal_correct'] is False and b[i]['literal_correct'] is True],
        correct_to_wrong=[i for i in common if control[i]['literal_correct'] is True and b[i]['literal_correct'] is False],
        added_correct_support=[i for i in range(32) if b[i]['correct_supported'] and not control[i]['correct_supported']],
        lost_correct_support=[i for i in range(32) if control[i]['correct_supported'] and not b[i]['correct_supported']],
        coverage_is_not_common_BER=True,actual_evidence_shared=True)


def evaluate(output,truth_path):
    from pathlib import Path
    terminal=load(Path(output)/'blind_terminal.json')
    if terminal.get('status')!='ALL_BLIND_SLOTS_TERMINAL':raise ValueError('all blind slots must be saved before truth')
    resources_path=Path(output)/'resources.json'
    completion=load(resources_path) if resources_path.is_file() else {}
    truth=load(truth_path)['views']
    # Original same-source writer fixture. Created only after the blind terminal read.
    message=[int(b) for byte in bytes.fromhex('A6D39C5E') for b in f'{byte:08b}']
    views={};bad=[];supported=0
    technical=bool(terminal.get('errors') or completion.get('completion_errors')) or any(r['status']!='RETURNED' for r in terminal['rows'].values())
    technical |= any(p['status']!='READY' for view in VIEWS for p in terminal['plans'][view]['B'])
    for view,length in zip(VIEWS,LENGTHS):
        mapping=truth[view]
        if not isinstance(mapping,list) or len(mapping)!=length or any(type(n) is not int or not 0<=n<181 for n in mapping):
            raise ValueError('complete post-only same-source mapping required: '+view)
        arms={}
        for arm in ARMS:
            slots=terminal['final'][view][arm]['slots']
            if len(slots)!=32 or [s['slot'] for s in slots]!=list(range(32)):raise ValueError('fixed ordered32 slots required')
            rows=[]
            for s in slots:
                i=s['slot'];value=s['value']
                if value is not None and (type(value) is not int or value not in (0,1)):raise ValueError('illegal output bit')
                selected=[_reference(r,mapping,terminal['raw_bank'],i) for r in s['selected_refs']]
                refs=[_reference(r,mapping,terminal['raw_bank'],i) for r in s.get('supporting_refs',[])]
                literal=None if value is None else value==message[i]
                wrong=bool(value is not None and any(not r['four_frame_exact'] or not r['group_exact'] for r in refs))
                unsupported=bool(value is not None and (not refs or any(not r['actual_physical_support'] for r in refs)))
                correct=literal is True and bool(refs) and all(r['correct_geometry_support'] for r in refs)
                row=dict(slot=i,value=value,status=s['status'],cause_set=s.get('cause_set',[]),literal_correct=literal,
                    wrong_attribution=wrong,unsupported_output=unsupported,correct_supported=correct,
                    selected_refs=selected,supporting_refs=refs,missing_refs=s.get('missing_refs',[]),conflict_refs=s.get('conflict_refs',[]))
                if arm=='B' and (literal is False or wrong or unsupported):bad.append(dict(view=view,slot=i,literal_error=literal is False,wrong_attribution=wrong,unsupported_output=unsupported))
                if arm=='B' and correct:supported+=1
                if s['status'] in ('TECHNICAL_UNREAD','PARTIAL_EVIDENCE_READ','NOT_ENTERED'):technical=True
                rows.append(row)
            arms[arm]=dict(rows=rows,output_count=sum(r['value'] is not None for r in rows),
                literal_errors=sum(r['literal_correct'] is False for r in rows),correct_supported=sum(r['correct_supported'] for r in rows),
                full32_supported=all(r['correct_supported'] for r in rows),
                group_support=[any(r['correct_supported'] for r in rows[8*g:8*g+8]) for g in range(4)])
        views[view]=dict(arms=arms,B_vs_W=compare(arms['B']['rows'],arms['W']['rows']),B_vs_U=compare(arms['B']['rows'],arms['U']['rows']))
    status='COUNTEREXAMPLE' if bad else 'EVIDENCE_INCOMPLETE' if technical else 'FINITE_CONDITIONAL_READBACK' if supported else 'NO_SUPPORTED_READBACK'
    return dict(status=status,evidence_complete=not technical,views=views,B_counterexamples=bad,B_correct_supported=supported,
        denominators=denominators(),post_evaluations=1,physical_observations_added=0,scientific_pass=None,
        message_source='original same-source fixed writer A6D39C5E',
        limits=['same-chain development; source already seen','no matched pre-QIM identity baseline or no-loss claim',
                'W support-zero alone is definitional, not synchronization benefit','same literal quality with different labels is not improved decoding',
                'four groups repeat within 11 cells; no within-group position authentication',
                'native target failures and unproved optimality retained','no null/FPR/physical identity/long-video/speed/interpolation conclusion'])
