"""Four fixed-denominator result tables from already persisted native observations."""
from collections import Counter, OrderedDict
import csv
import json
from pathlib import Path
import time

from main.baselines.public_fpr001_v4 import METHODS, backend, views, verify, family_event, all_robust, nominal_bounds
from main.baselines.public_metrics_v4 import evaluate_view
from main.baselines.native_paper_fpr1_v3 import rate_summary, action_sensitivity
from .native_baseline_storage_v1 import read_json, write_json
from .native_baseline_storage_v4 import UnitStore, ArrayLog, append_json, events as event_stream
from .native_baseline_io_v4 import saved_raw, empty_raw


def post_path(output,sid,method,arm,view):return Path(output)/'post'/sid/method/arm/(view+'.json')


def write_post(output,source,method,arm,view,raw,epoch,queries,plan,container_id='post'):
    mapping=plan['maps'].get(view['operation'],plan['maps']['T00'])
    result,arrays=evaluate_view(method,arm,raw,queries[method+'/'+source['id']],mapping,queries)
    result.update(source_id=source['id'],method=method,arm=arm,view=view['id'],split=source['split'],
        epoch=epoch['id'] if epoch else None,observed_frames=int(raw['observed'].sum()),planned_frames=view['frames'])
    path=post_path(output,source['id'],method,arm,view['id']);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file():
        old=read_json(path)
        if {k:v for k,v in old.items() if k!='time_arrays'}==result:return old
        append_json(path.parent/'superseded_post.jsonl',old)
    if arrays:
        # Per-unit append container avoids a file per received node or edge.
        log=path.parents[1]/('time-'+container_id+'.bin')
        ArrayLog(log).append(view['id'],arrays,epoch=result['epoch'])
        result['time_arrays']=str(log.relative_to(Path(output)))
    write_json(path,result);return result


def missing_post(source,method,arm,view):
    return dict(source_id=source['id'],split=source['split'],method=method,arm=arm,view=view['id'],epoch=None,
        observed_frames=0,planned_frames=view['frames'],ownership=dict(decision=None,complete=False,accepted_from_partial=False),
        wrong_query=dict(decision=None,complete=False) if arm!='OFF' else None,
        payload=dict(planned_frames=view['frames'],planned_bits=view['frames']*METHODS[method],status='NOT_EVALUATED'),
        time=dict(status='NOT_EVALUATED' if method=='frameindex' else 'N/A',planned_nodes=view['frames'],planned_edges=view['frames']-1))


def rows_for(output,source,method,arm,queries=None,plan=None):
    output_rows=[]
    for view in views():
        path=post_path(output,source['id'],method,arm,view['id'])
        try:
            if path.is_file():row=read_json(path)
            elif queries is not None:
                mapping=plan['maps'].get(view['operation'],plan['maps']['T00'])
                metric,_=evaluate_view(method,arm,empty_raw(view['frames'],backend(method)),queries[method+'/'+source['id']],mapping,queries)
                row=dict(missing_post(source,method,arm,view),**metric)
            else:row=missing_post(source,method,arm,view)
        except (OSError,ValueError) as error:
            row=missing_post(source,method,arm,view);row['read_error']=str(error)
        output_rows.append(row)
    return output_rows


def cross_source(output,sources,queries,plans,budget):
    """Reuse DEV donor WM raw arrays only. This function cannot import/load a model."""
    cache=OrderedDict();records=[]
    for method in METHODS:
        for source in sources:
            if source['split']=='CAL':continue
            sid=source['id'];donor=plans[sid]['donor'];key=(method,donor)
            if key not in cache:
                unit='native/'+donor+'/'+backend(method)+'/'+method
                cache[key]=UnitStore(output,unit,budget)
                if len(cache)>8:cache.popitem(last=False)
            store=cache[key];cache.move_to_end(key)
            for view in views():
                try:
                    raw,epoch=saved_raw(store,view['id'],view['frames'],backend(method))
                    verified=verify(method,raw,queries[method+'/'+sid]['query'],planned_frames=view['frames'])
                    record=dict(source_id=sid,donor_id=donor,split=source['split'],method=method,view=view['id'],
                        decision=verified['decision'],complete=verified['complete'],epoch=epoch['id'] if epoch else None)
                except (OSError,ValueError) as error:
                    record=dict(source_id=sid,donor_id=donor,split=source['split'],method=method,view=view['id'],decision=None,complete=False,error=str(error))
                records.append(record)
    write_json(Path(output)/'tables/cross_source.json',dict(rows=records,logical_comparisons=len(records),
        distinct_donors=len(set(r['donor_id'] for r in records)),physical_inference_calls=0,
        interpretation='conditional on fixed DEV donor bank; comparisons are not independent sources'))
    return records


def csv_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def refresh_post(output,sources,queries,plans,budget):
    """Finish interrupted post-processing from committed arrays without a physical call."""
    for index,source in enumerate(sources,1):
        if getattr(budget,'progress',None) and (index==1 or index%8==0):budget.progress('FINAL_POST_NATIVE_READ',source=source['id'],processed=index,total=len(sources))
        for physical in ('wam','pixelseal','videoseal'):
            arms=['OFF']+([] if source['split']=='CAL' else ['pixelseal','frameindex'] if physical=='pixelseal' else [physical])
            for arm in arms:
                unit='native/'+source['id']+'/'+physical+'/'+arm
                if not (Path(output)/'units'/unit).is_dir():continue
                store=UnitStore(output,unit,budget)
                methods=('pixelseal','frameindex') if physical=='pixelseal' and arm=='OFF' else (physical if arm=='OFF' else arm,)
                actual_complete=True
                for view in views():
                    try:raw,epoch=saved_raw(store,view['id'],view['frames'],physical)
                    except (OSError,ValueError,KeyError,TypeError) as error:
                        raw,epoch=empty_raw(view['frames'],physical),None
                        append_json(store.folder/'post_read_errors.jsonl',dict(view=view['id'],error=str(error),runtime_id=budget.runtime_id))
                    write_json(store.folder/'views'/view['id']/'status.json',dict(epoch=epoch,planned=view['frames'],
                        observed=int(raw['observed'].sum()),complete=bool(raw['observed'].all()),record_errors=store.record_errors))
                    actual_complete=actual_complete and bool(raw['observed'].all())
                    for method in methods:write_post(output,source,method,arm,view,raw,epoch,queries,plans[source['id']],budget.runtime_id)
                write_json(store.folder/'scheduling.json',dict(complete=actual_complete,views=40,basis='actual final post read'))


def rate_record(decisions,complete):
    """Known decisions and execution-complete cases are different subsets for partial OR witnesses."""
    if len(decisions)!=len(complete):raise ValueError('unaligned completion flags')
    result=rate_summary(decisions)
    result['decision_known']=result.pop('completed');result['decision_coverage']=result.pop('coverage')
    result['known_decision_subset_denominator']=result.pop('complete_subset_denominator')
    result['known_decision_subset_cp95']=result.pop('complete_subset_cp95')
    completed=[d for d,c in zip(decisions,complete) if c]
    subset=rate_summary(completed) if completed else None
    result.update(execution_complete=len(completed),execution_incomplete=len(decisions)-len(completed),
        execution_coverage=len(completed)/len(decisions),complete_subset_denominator=len(completed),
        complete_subset_positive=sum(d is True for d in completed),complete_subset_negative=sum(d is False for d in completed),
        complete_subset_unknown=sum(d is None for d in completed),complete_subset_cp95=subset['conservative_cp95'] if subset else None,
        complete_subset_one_sided95_upper=subset['conservative_one_sided95_upper'] if subset else None)
    return result


def build_tables(output,sources,queries,plans,budget):
    output=Path(output);directory=output/'tables';directory.mkdir(parents=True,exist_ok=True)
    # Completed historical post records are trusted; current tasks save their own post before advancing.
    cross_path=directory/'cross_source.json'
    cross=read_json(cross_path)['rows'] if cross_path.is_file() else cross_source(output,sources,queries,plans,budget)
    cross_index={(r['method'],r['source_id'],r['view']):r for r in cross}
    table_b=[];table_a=[];table_c=[];incomplete=0;logical_views=0
    for method in METHODS:
        for split in ('DEV','CAL','TEST'):
            selected=[s for s in sources if s['split']==split]
            if not selected:continue
            for arm in ('OFF',method) if split!='CAL' else ('OFF',):
                by_source={s['id']:rows_for(output,s,method,arm,queries,plans[s['id']]) for s in selected}
                for source in selected:
                    for row in by_source[source['id']]:
                        meta={k:row[k] for k in ('source_id','split','method','arm','view')}
                        table_a.append(dict(meta,**row['payload']));table_c.append(dict(meta,**row['time']))
                        incomplete+=int(not row['ownership']['complete']);logical_views+=1
                for verification in ('query','wrong_query','cross_source') if arm!='OFF' else ('query',):
                    def decision_rows(source):
                        result=[]
                        for row in by_source[source['id']]:
                            if verification=='cross_source':value=cross_index[(method,source['id'],row['view'])]
                            else:value=row['ownership' if verification=='query' else 'wrong_query']
                            result.append(value or dict(decision=None,complete=False))
                        return result
                    ordered=[decision_rows(s) for s in selected]
                    # Individual views, two media generations, and the full40 OR/all40 robustness.
                    channels=[(v['id'],[i],False) for i,v in enumerate(views())]
                    channels += [(g,[i for i,v in enumerate(views()) if v['generation']==g],False) for g in ('G1','G2')]
                    channels += [('FAMILY40',list(range(40)),False),('ALL40_ROBUST',list(range(40)),True)]
                    for channel,indices,robust in channels:
                        events=[];complete=[]
                        for rows in ordered:
                            group=[rows[i] for i in indices];complete.append(all(r['complete'] for r in group))
                            if len(indices)==1:event=group[0]['decision']
                            elif robust:event=all_robust(group)
                            elif len(indices)==40:event=family_event(group)
                            else:event=True if any(r['decision'] is True for r in group) else False if all(r['complete'] and r['decision'] is False for r in group) else None
                            events.append(event)
                        summary=dict(method=method,split=split,arm=arm,verification=verification,channel=channel,
                            **rate_record(events,complete))
                        if split=='TEST':summary['action_cluster_bootstrap']=action_sensitivity(events,[s['action'] for s in selected])
                        if verification=='cross_source':summary.update(logical_comparisons=len(indices)*len(selected),distinct_donors=len(set(plans[s['id']]['donor'] for s in selected)))
                        table_b.append(summary)
    quality=[]
    for method in METHODS:
        for source in sources:
            if source['split']=='CAL':continue
            folder=output/'units'/'quality'/source['id']/method/'quality'
            for stage in ('precodec','G1','G2-identity'):
                path=folder/(stage+'.json')
                try:value=read_json(path) if path.is_file() else dict(status='NOT_STARTED',planned_frames=181,planned_edges=180,lpips_observed_frames=0)
                except (OSError,ValueError) as error:value=dict(status='UNREADABLE',error=str(error),planned_frames=181,planned_edges=180,lpips_observed_frames=0)
                frames=value.get('frames',[]);edges=value.get('edges',[])
                value=dict(source_id=source['id'],method=method,split=source['split'],stage=stage,
                    planned_frames=181,planned_edges=180,observed_frames=len(frames),observed_edges=len(edges),
                    lpips_observed_frames=value.get('lpips_observed_frames',0),lpips_mean=value.get('lpips_mean'),
                    mse_mean=sum(r['mse'] for r in frames)/181 if len(frames)==181 else None,
                    ssim_mean=sum(r['ssim'] for r in frames)/181 if len(frames)==181 else None,
                    psnr_positive_infinity_frames=sum(r['psnr_positive_infinity'] for r in frames),
                    psnr_db_finite_mean=sum(r['psnr_db'] for r in frames if r['psnr_db'] is not None)/sum(r['psnr_db'] is not None for r in frames) if any(r['psnr_db'] is not None for r in frames) else None,
                    residual_difference_rms_mean=sum(r['residual_difference_rms'] for r in edges)/180 if len(edges)==180 else None,
                    detailed_path=str(path.relative_to(output)))
                quality.append(value)
    for name,rows in (('A_payload',table_a),('B_ownership',table_b),('C_time',table_c),('D_quality',quality)):
        write_json(directory/(name+'.json'),rows);csv_rows(directory/(name+'.csv'),rows)
    resources=budget.snapshot()
    resources.update(model_resources=list(event_stream(output/'model_resources.jsonl')),
        quality_resources=list(event_stream(output/'quality_resources.jsonl')),model_loads=list(event_stream(output/'model_loads.jsonl')))
    write_json(directory/'D_resources.json',resources)
    result=dict(status='ALL_NATIVE_OBSERVED' if incomplete==0 else 'INCOMPLETE_RETAINED',
        logical_views=logical_views,execution_incomplete_views=incomplete,nominal_random_query_bounds=nominal_bounds(),
        quality_planned_pairs=len(quality)*181,quality_complete_stages=sum(q['observed_frames']==181 and q['observed_edges']==180 for q in quality),
        lpips_policy='not requested; historical observations and missingness retained, not a completion requirement',
        scientific_success='NOT_ADJUDICATED',tables=['A_payload','B_ownership','C_time','D_quality','D_resources'],
        next_run='resume uncommitted work from checkpoint; infrastructure costs recorded without quota; fixed scientific tasks and thresholds unchanged')
    write_json(output/'result.json',result)
    (output/'REPORT.md').write_text('# Public native and FrameIndex v4\n\n'+
        f"Native logical views: {logical_views}; execution incomplete: {incomplete}.\n\n"+
        'A: payload and coverage; B: query verification, fixed denominators and uncertainty; C: time attribution; D: quality and actual costs.\n\n'+
        'The 0.1% statement is per method, one uniformly random query and the fixed40 family. It is not empirical certification of a natural-video population, nor a combined four-method bound. OFF, wrong-query and reused DEV-donor comparisons are separate.\n\n'+
        'A partial positive WAM/FrameIndex witness is TRUE with incomplete execution. A missing Seal frame makes its native video verification UNKNOWN. Native methods have N/A time outputs. Failed and unexecuted slots remain in the tables.\n',encoding='utf-8')
    return result
