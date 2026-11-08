"""Twelve-view tables with finite CAL sanity, fixed TEST and read-only DEV appendix."""
from pathlib import Path
from main.baselines import public_compact12_v1 as p
from main.baselines.native_paper_fpr1_v3 import action_sensitivity
from main.tube_state.paper_suite_v1 import rate_summary as source_rate
from .native_baseline_storage_v1 import read_json,write_json
from .native_baseline_storage_v4 import events
from .native_baseline_evaluation_v4 import missing_post,post_path,csv_rows,rate_record


def saved(path,default):
    try:
        value=read_json(path) if Path(path).is_file() else default
        if not isinstance(value,type(default)):raise ValueError('saved JSON has wrong container type')
        return value
    except (OSError,ValueError,TypeError) as error:return dict(default,read_error=str(error)) if isinstance(default,dict) else default


def rows_for(root,source,method,arm,source_roots=None):
    rows=[]
    for view in p.views():
        fallback=missing_post(source,method,arm,view)
        roots=(source_roots or {}).get(source['id'],[root])
        paths=[post_path(r,source['id'],method,arm,view['id']) for r in roots]
        path=next((f for f in paths if f.is_file()),paths[0])
        row=saved(path,fallback)
        try:
            if any(not isinstance(row.get(k),dict) for k in ('payload','time','ownership')):raise ValueError('post record fields invalid')
            if any(row.get(k)!=fallback[k] for k in ('source_id','split','method','arm','view')):raise ValueError('post row does not describe requested slot')
            if type(row['ownership'].get('complete')) is not bool or row['ownership'].get('decision') not in (True,False,None):raise ValueError('invalid decision fields')
        except (ValueError,TypeError) as error:row=dict(fallback,read_error=str(error))
        rows.append(row)
    return rows


def quality_rows(root,sources,source_roots=None):
    rows=[]
    for s in sources:
        if s['split']=='CAL':continue
        for m in p.METHODS:
            for stage in ('precodec','G1','G2-identity'):
                paths=[Path(r)/'units/quality'/s['id']/m/'quality'/(stage+'.json') for r in (source_roots or {}).get(s['id'],[root])]
                path=next((f for f in paths if f.is_file()),paths[0]);value=saved(path,{})
                frames=value.get('frames',[]);edges=value.get('edges',[])
                try:
                    import math
                    if not isinstance(frames,list) or not isinstance(edges,list):raise ValueError('quality rows must be lists')
                    if len(frames) not in (0,181) or len(edges) not in (0,180):raise ValueError('invalid quality frame/edge counts')
                    for row in frames:
                        if not isinstance(row,dict) or any(not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) for k in ('mse','ssim')):raise ValueError('invalid quality numeric fields')
                        if row.get('psnr_db') is not None and (not isinstance(row['psnr_db'],(int,float)) or not math.isfinite(row['psnr_db'])):raise ValueError('invalid PSNR')
                    for row in edges:
                        if not isinstance(row,dict) or not isinstance(row.get('residual_difference_rms'),(int,float)) or not math.isfinite(row['residual_difference_rms']):raise ValueError('invalid quality edges')
                    n=value.get('lpips_observed_frames',0)
                    if type(n) is not int or not 0<=n<=181:raise ValueError('invalid LPIPS observed count')
                except (ValueError,TypeError) as error:
                    value=dict(read_error=str(error));frames=[];edges=[]
                finite=[r['psnr_db'] for r in frames if r.get('psnr_db') is not None]
                rows.append(dict(source_id=s['id'],split=s['split'],method=m,stage=stage,
                    planned_frames=181,planned_edges=180,observed_frames=len(frames),observed_edges=len(edges),
                    lpips_observed_frames=value.get('lpips_observed_frames',0),lpips_mean=value.get('lpips_mean'),
                    mse_mean=sum(r['mse'] for r in frames)/181 if len(frames)==181 else None,
                    ssim_mean=sum(r['ssim'] for r in frames)/181 if len(frames)==181 else None,
                    psnr_positive_infinity_frames=sum(r.get('psnr_positive_infinity',False) for r in frames),
                    psnr_db_finite_mean=sum(finite)/len(finite) if finite else None,
                    residual_difference_rms_mean=sum(r['residual_difference_rms'] for r in edges)/180 if len(edges)==180 else None,
                    saved_path=str(path),read_error=value.get('read_error'),imported_from=value.get('imported_from')))
    return rows


def build_tables(output,sources,roster,queries,budget,old,source_roots=None):
    output=Path(output);directory=output/'tables';directory.mkdir(parents=True,exist_ok=True)
    payload=[];ownership=[];time_rows=[];incomplete=0
    for method in p.METHODS:
        for split in ('CAL','TEST'):
            selected=[s for s in sources if s['split']==split]
            for arm in ('OFF',) if split=='CAL' else ('OFF',method):
                groups=[rows_for(output,s,method,arm,source_roots) for s in selected]
                for rows in groups:
                    for row in rows:
                        meta={k:row[k] for k in ('source_id','split','method','arm','view')}
                        payload.append(dict(meta,**row['payload']));time_rows.append(dict(meta,**row['time']))
                        incomplete+=not row['ownership']['complete']
                for verification in ('query',) if arm=='OFF' else ('query','wrong_query'):
                    ordered=[[r.get('ownership' if verification=='query' else 'wrong_query') or dict(decision=None,complete=False) for r in rows] for rows in groups]
                    flags=[queries[method+'/'+s['id']].get('wrong_query_collision') for s in selected]
                    collisions=[verification=='wrong_query' and flag is True for flag in flags]
                    if verification=='wrong_query':
                        ordered=[rows if type(flag) is bool else [dict(decision=None,complete=False) for _ in rows] for rows,flag in zip(ordered,flags)]
                    channels=[(v['id'],[i],False) for i,v in enumerate(p.views())]
                    channels += [('FAMILY12',list(range(12)),False),('ALL12_ROBUST',list(range(12)),True)]
                    for channel,indices,robust in channels:
                        decisions=[];complete=[]
                        for rows in ordered:
                            group=[rows[i] for i in indices];complete.append(all(r['complete'] for r in group))
                            decisions.append(group[0]['decision'] if len(indices)==1 else p.family_event(group,robust))
                        eligible=[i for i,c in enumerate(collisions) if not c]
                        summary=rate_record([decisions[i] for i in eligible],[complete[i] for i in eligible]) if eligible else {}
                        # 'planned' remains the frozen roster count; collision exclusion is explicit, not missing work.
                        summary.update(planned=len(selected),eligible=len(eligible),query_collisions=sum(collisions),not_applicable=sum(collisions))
                        summary.update(source_rate(decisions,collisions));summary.pop('scope',None)
                        row=dict(method=method,split=split,arm=arm,verification=verification,channel=channel,**summary,
                            scope='CAL is null sanity without fitted thresholds; TEST independent held-out sources',
                            interval_assumption='source iid reference only; action bootstrap is sensitivity')
                        if split=='TEST':row['action_cluster_bootstrap']=action_sensitivity([decisions[i] for i in eligible],[selected[i]['action'] for i in eligible]) if eligible else None
                        ownership.append(row)
    quality=quality_rows(output,sources,source_roots)
    dev=[s for s in roster if s['split']=='DEV'];devrows=[]
    for source in dev:
        for method in p.METHODS:
            for arm in ('OFF',method):
                for row in rows_for(old,source,method,arm):devrows.append(dict(row,reference_only=True,new_physical_calls=0))
    for name,rows in [('A_payload',payload),('B_ownership',ownership),('C_time',time_rows),('D_quality',quality),
                      ('DEV_saved_reference',devrows),('DEV_quality_reference',quality_rows(old,dev))]:
        write_json(directory/(name+'.json'),rows);csv_rows(directory/(name+'.csv'),rows)
    resources=budget.snapshot();resources.update(import_receipts=list(events(output/'imports.jsonl')),
        imported_work_is_not_new_inference=True,model_resources=list(events(output/'model_resources.jsonl')),
        quality_resources=list(events(output/'quality_resources.jsonl')),model_loads=list(events(output/'model_loads.jsonl')))
    write_json(directory/'D_resources.json',resources)
    complete_quality=sum(r['observed_frames']==181 and r['lpips_observed_frames']==181 for r in quality)
    bounds=p.nominal_bounds()
    result=dict(status='COMPACT_MEASUREMENT_RETURNED' if not incomplete and complete_quality==len(quality) else 'INCOMPLETE_RETAINED',
        experiment=p.EXPERIMENT,logical_views=len(payload),execution_incomplete_views=int(incomplete),
        quality_planned_pairs=len(quality)*181,quality_complete_stages=complete_quality,
        nominal_random_query_bounds=bounds,four_method_per_source_random_query_union_upper=sum(v['upper'] for v in bounds.values()),
        union_scope='one OFF source, four random method queries, twelve views; not the 500-source package or fixed-message population',
        old_v4=str(old),old_v4_completion_assessed=False,old_v4_status_untouched=True,dev_reference_only=True,
        roster={split:sum(s['split']==split for s in sources) for split in ('CAL','TEST')},global_roster=dict(CAL=32,TEST=500),view_ids=list(p.VIEW_IDS),scientific_success='NOT_ADJUDICATED')
    write_json(output/'result.json',result)
    (output/'REPORT.md').write_text('# Compact public baseline measurement\n\n'+
        'CAL32 independent null sanity; TEST500; four methods; twelve fixed views. No threshold fitting. '+
        'FAMILY12 is any acceptance; ALL12_ROBUST is all twelve accepted and execution complete. '+
        'Random-query theory, observed source rates, missing observations and quality are separate.\n\n'+
        'Only query/wrong_query and saved DEV references are summarized; no new cross_source/DEV-donor revalidation is performed. Old cross-source tables remain in the old package. '+
        'DEV is an existing-reference appendix only. The original full40 package and all its missing/failed rows remain unchanged. '+
        'Not measured here: mild deletion, point repeat, reorder, foreign/synthetic inserts, slow resampling, H265, half resize, blur, brightness, noise and JPEG. '+
        'Only T00 has paired G1/G2, so codec effects cannot be isolated for every other attack.\n')
    return result
