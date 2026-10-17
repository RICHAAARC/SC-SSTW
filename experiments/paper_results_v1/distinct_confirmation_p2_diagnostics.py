"""Posthoc P2 diagnostics; never a selector input or extra payload arm."""
import numpy as np
from main.tube_state.video_trajectory_temporal_edit_receiver_v1 import decode_visible_span


def rowmax_diagnostic(q):
    maximum=q.max(axis=1)
    ties=[np.flatnonzero(row==top).tolist() for row,top in zip(q,maximum)]
    tied=[i for i,v in enumerate(ties) if len(v)!=1]
    if tied:return dict(status='UNRESOLVED_ROW_TIE',tie_positions=tied,tied_source_columns={str(i):ties[i] for i in tied},path=None)
    path=[v[0] for v in ties];backward=[i for i in range(1,len(path)) if path[i]<path[i-1]]
    if backward:return dict(status='ROWMAX_NONMONOTONE',path=path,nonmonotone_positions=backward,operation=None)
    operation=decode_visible_span(path)
    return dict(status=operation['status'],path=path,operation=operation,tie_positions=[],nonmonotone_positions=[])


def equality_deltas(array):
    """Exact classes and all-pair differences; no near-equality cutoff."""
    reps=[];classes=[];maximum=0.;squares=0.;elements=0
    for i,row in enumerate(array):
        group=next((g for g,j in enumerate(reps) if np.array_equal(row,array[j])),None)
        if group is None:group=len(reps);reps.append(i)
        classes.append(group)
        for j in range(i):
            delta=row.astype(np.float64)-array[j].astype(np.float64)
            maximum=max(maximum,float(np.max(np.abs(delta))));squares+=float(np.sum(delta*delta));elements+=delta.size
    return dict(exact_class_count=len(reps),class_indices=classes,class_sizes=[classes.count(g) for g in range(len(reps))],
                pairwise_max_abs=maximum,pairwise_rms=float(np.sqrt(squares/elements)) if elements else 0.)


def matched_delta(left,right):
    if left is None or right is None:return dict(status='MISSING')
    if left.shape!=right.shape:raise ValueError('fixed media pair changed geometry')
    maximum=0.;squares=0.;equal=0
    for a,b in zip(left,right):
        delta=a.astype(np.float64)-b.astype(np.float64)
        maximum=max(maximum,float(np.max(np.abs(delta))));squares+=float(np.sum(delta*delta));equal+=np.array_equal(a,b)
    return dict(status='COMPARED',exact_equal_frames=int(equal),frames=len(left),max_abs=maximum,rms=float(np.sqrt(squares/left.size)))


def diagnose(store):
    from experiments.paper_results_v1 import distinct_confirmation_p2 as run
    if not run.blind_plans_complete(store) or 'blind_plans_committed' not in store.data['phases']:return
    paths=[];repeats=[];pairs=[];media_pairs=[]
    for case in run.cases():
        cid=case['case_id']
        for view in run.VIEWS:
            truth=run.view_indices(view);file=store.output/'observations'/cid/(view+'.npz')
            record=dict(case_id=cid,view=view,status='MISSING',planned_positions=len(truth))
            if file.exists():
                with np.load(file,allow_pickle=False) as z:q=z['DISTINCT_REF_U'];ref=z['REF_U'];reps=z['representative_indices'];classes=z['class_indices']
                diag=rowmax_diagnostic(q);candidate=run.saved_plan(store,f'plan/{cid}/{view}/DISTINCT_REF_U')
                u_path=candidate['plan']['estimated_path'] if candidate and candidate['plan'] else None
                row_path=diag['path'];u_exact=u_path==truth;row_exact=row_path==truth
                record.update(status='DIAGNOSED',rowmax=diag,u_status=candidate['status'] if candidate else 'MISSING',
                              u_path_exact=u_exact,rowmax_path_exact=row_exact,same_path=u_path is not None and u_path==row_path,
                              observable_U_path_contribution=u_exact and not row_exact,
                              background_representatives=len(reps),class_sizes=np.bincount(classes).tolist())
                rgb=run.received_view(store,cid,view);y=run.received_view(store,cid,view,latent=True)
                for source in sorted(set(truth)):
                    indices=[i for i,t in enumerate(truth) if t==source]
                    if len(indices)<2:continue
                    r=dict(case_id=cid,view=view,source_position=source,received_positions=indices)
                    for label,arr in [('RGB',rgb),('Y',y),('Q_ref',ref)]:
                        r[label]=equality_deltas(arr[indices]) if arr is not None else dict(status='MISSING')
                    available=all('exact_class_count' in r[k] for k in ('RGB','Y','Q_ref'))
                    r['media_approximate_boundary']=('MISSING' if not available else 'NOT_EXERCISED_EXACT' if all(r[k]['exact_class_count']==1 for k in ('RGB','Y','Q_ref')) else 'LAYER_DIFFERENCES_OBSERVED') if view in run.MEDIA_VIEWS else 'EXACT_INDEXING_REFERENCE'
                    repeats.append(r)
            else:
                for source in sorted(set(truth)):
                    indices=[i for i,t in enumerate(truth) if t==source]
                    if len(indices)>1:
                        repeats.append(dict(case_id=cid,view=view,source_position=source,received_positions=indices,
                            RGB=dict(status='MISSING'),Y=dict(status='MISSING'),Q_ref=dict(status='MISSING'),media_approximate_boundary='MISSING'))
            paths.append(record)
            rows=[r for r in store.data['rows'] if r['case_id']==cid and r['view']==view]
            reads={r['receiver']:run.saved_read(store,'read/'+r['slot_id']) for r in rows}
            candidate=reads['DISTINCT_REF_U'];truth_plan=run.method.segment_plan(truth)
            present=[i for i in range(32) if truth_plan['segments'][i//8]['cells']]
            for arm in ('RAW_U','CENTERED_U','REF_U','UNALIGNED','ORACLE'):
                old=reads[arm];a=old['decoded_bits'] if old else [None]*32;b=candidate['decoded_bits'] if candidate else [None]*32
                common=[i for i in present if a[i] is not None and b[i] is not None]
                ea=sum(a[i]!=run.BITS[i] for i in common);eb=sum(b[i]!=run.BITS[i] for i in common)
                pairs.append(dict(case_id=cid,view=view,baseline=arm,common_bits=len(common),baseline_common_wrong=ea,candidate_common_wrong=eb,
                    baseline_common_BER=ea/len(common) if common else None,candidate_common_BER=eb/len(common) if common else None,
                    baseline_coverage=sum(a[i] is not None for i in present),candidate_coverage=sum(b[i] is not None for i in present),
                    baseline_unread=sum(a[i] is None for i in present),candidate_unread=sum(b[i] is None for i in present),
                    structural_missing_bits=32-len(present),baseline_false_attribution=sum(a[i] is not None for i in range(32) if i not in present),
                    candidate_false_attribution=sum(b[i] is not None for i in range(32) if i not in present)))
        for media,base in [('SHORT17_MEDIA','SHORT17'),('SHORT17_HOLD_MEDIA','SHORT17_HOLD_EXACT')]:
            item=dict(case_id=cid,primary_view=base,secondary_view=media,
                      meaning='matched primary/secondary codec difference, not a receiver effect')
            for label,latent in [('RGB',False),('Y',True)]:
                item[label]=matched_delta(run.received_view(store,cid,base,latent=latent),run.received_view(store,cid,media,latent=latent))
            refs=[]
            for view in (base,media):
                path=store.output/'observations'/cid/(view+'.npz')
                if path.exists():
                    with np.load(path,allow_pickle=False) as z:refs.append(z['REF_U'])
                else:refs.append(None)
            item['Q_ref']=matched_delta(*refs);media_pairs.append(item)
    run.write_json(store.output/'rowmax_diagnostics.json',paths);run.write_json(store.output/'repeat_layer_diagnostics.json',repeats)
    run.write_json(store.output/'paired_rows.json',pairs);run.write_json(store.output/'media_pair_diagnostics.json',media_pairs)
