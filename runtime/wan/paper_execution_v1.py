"""Two source pipelines, four public packs and one final CPU writer."""
from collections import Counter
from pathlib import Path
import time
import uuid
from main.tube_state import paper_execution_v1 as schedule
from main.tube_state import paper_suite_v1 as p
from main.baselines import public_compact12_v1 as public
from . import paper_suite_records_v1 as rec
from .paper_checkpoint_v1 import Boundary,BoundaryPause
from .native_baseline_storage_v4 import runtime_session,events

OUTPUT=Path('/content/drive/MyDrive/Video-WM_Result/paper_execution_v1')


def check_full_owner(previous_runtime_stopped=False,output=OUTPUT):
    from .native_baseline_colab_v4 import check_previous_worker
    check_previous_worker(Path(output)/'owners/full-suite',previous_runtime_stopped)


def active_worker_check(previous_runtime_stopped=False,output=OUTPUT,public_root=public.OUTPUT):
    from .native_baseline_colab_v4 import check_previous_worker
    for path in [Path(output)/'workers'/('main'+str(i)) for i in range(2)]+[Path(public_root)/'shards'/('part'+str(i)) for i in range(4)]:
        check_previous_worker(path,previous_runtime_stopped)


def terminal(root,sid):
    value=rec.optional(Path(root)/sid/'attempt01/result.json',{}) or {}
    return value.get('status') in ('COMPLETED','STOPPED_FIXED_ATTEMPT')


def main_worker(shard,max_sources=1,seconds=4*3600,previous_runtime_stopped=False,cohort_only=False,
                output=OUTPUT,cohort_root=p.COHORT_ROOT,wan_root=p.WAN_ROOT,vendor_factory=None,
                cohort_call=None,wan_call=None,cohort_prepare=None):
    from .b_paper_cohort_resume_v1 import run_source as cohort_run
    from .paper_wan_comparison_v1 import run_source as wan_run
    from .native_baseline_colab_v4 import check_previous_worker
    check_previous_worker(Path(output)/'owners/finalizer',previous_runtime_stopped)
    source_ids=schedule.main_sources(shard);output=Path(output);worker=output/'workers'/('main'+str(shard))
    mode='cohort' if cohort_only else 'cohort_and_wan'
    boundary=Boundary(max_sources,seconds);results=[];vendor=None;cohort_prepared=False
    with runtime_session(worker,previous_runtime_stopped=previous_runtime_stopped) as owner:
        try:
            for index,sid in enumerate(source_ids):
                if (worker/'legacy_unresolved'/(sid+'.json')).is_file():continue
                if terminal(cohort_root,sid) and (cohort_only or terminal(wan_root,sid)):continue
                boundary.before()
                rec.write(worker/'cursor.json',dict(source_index=index,source=sid,mode=mode,session=owner['id']))
                remaining=lambda:None if boundary.deadline is None else max(.001,boundary.deadline-time.monotonic())
                if not terminal(cohort_root,sid):
                    if cohort_prepare is not None and not cohort_prepared:cohort_prepare();cohort_prepared=True
                    result=(cohort_call or cohort_run)(sid,output_root=cohort_root,previous_runtime_stopped=previous_runtime_stopped,seconds=remaining())
                    results.append(dict(source=sid,component='cohort',result=result))
                    if result.get('status')=='PAUSED_BOUNDARY':raise BoundaryPause('source stage paused')
                    if not terminal(cohort_root,sid):
                        rec.write(worker/'legacy_unresolved'/(sid+'.json'),dict(source=sid,status='LEGACY_UNRESOLVED',
                            note='old consumed nonterminal package not restarted; fixed missing rows retained'))
                        boundary.returned();continue
                if not cohort_only and not terminal(wan_root,sid):
                    boundary.before()
                    if vendor is None and vendor_factory:vendor=vendor_factory()
                    result=(wan_call or wan_run)(sid,cohort_root,wan_root,vendor,
                        previous_runtime_stopped=previous_runtime_stopped,seconds=remaining())
                    results.append(dict(source=sid,component='wan',result=result))
                    if result.get('status')=='PAUSED_BOUNDARY':raise BoundaryPause('paired baseline paused')
                    if not terminal(wan_root,sid):
                        rec.write(worker/'legacy_unresolved'/(sid+'.json'),dict(source=sid,component='wan',status='LEGACY_UNRESOLVED',
                            note='old nonterminal evidence retained; no attempt reset'))
                boundary.returned();rec.write(worker/'cursor.json',dict(next_source_index=index+1,mode=mode,session=owner['id']))
            result=dict(status='WORKER_TASKS_RETURNED',shard=shard,mode=mode,results=results)
        except BoundaryPause as error:result=dict(status='PAUSED_BOUNDARY',shard=shard,mode=mode,reason=str(error),results=results)
        rec.write(worker/'sessions'/owner['id']/'result.json',result);rec.write(worker/'progress.json',result);return result


class PublicCost:
    def __init__(self,parent,sources):self.parent=Path(parent);self.sources=sources
    def snapshot(self):
        from .native_baseline_storage_v4 import Budget
        total_base=Counter();total_extra=Counter();parts=[]
        try:inherited=schedule.inherited_spend(self.sources,list(events(self.parent/'physical_calls.jsonl')))
        except Exception as error:
            return dict(base=None,recovery=None,parts=[],limits=public.physical_budget(),counts_complete=False,
                status='UNKNOWN_CONSUMPTION',error=str(error),physical_calls_this_report=0)
        for shard in range(4):
            folder=self.parent/'shards'/('part'+str(shard));saved=rec.optional(folder/'inherited_budget.json',{}) or {}
            try:
                budget=Budget(folder,'read-only',limits=saved.get('limits') or public.physical_budget())
                previous=(saved.get('limits') or {}).get('inherited',inherited[shard])
                base=Counter(previous['base'])+budget.base;extra=Counter(previous['recovery'])+budget.extra
                total_base.update(base);total_extra.update(extra)
                parts.append(dict(shard=shard,base=dict(base),recovery=dict(extra),status='READ_ONLY_COUNTS'))
            except Exception as error:parts.append(dict(shard=shard,status='UNKNOWN',error=str(error)))
        return dict(base=dict(total_base),recovery=dict(total_extra),parts=parts,limits=public.physical_budget(),
            counts_complete=all(v['status']=='READ_ONLY_COUNTS' for v in parts),physical_calls_this_report=0)


def collect_public(destination,parent=public.OUTPUT,old=public.OLD_OUTPUT):
    from .native_baseline_compact_tables_v1 import build_tables
    from .native_baselines_compact_v1 import DEFAULT_CONFIG
    parent=Path(parent);old=Path(old);config=rec.read(DEFAULT_CONFIG)
    path=next((f for f in (parent/'roster.json',old/'sources.json') if f.is_file()),DEFAULT_CONFIG.parent/config['source_list'])
    roster=rec.read(path)
    if isinstance(roster,dict):roster=roster['sources']
    sources=public.selected_sources(roster);source_roots={};queries={}
    fallback=next((f for f in (parent/'queries.json',old/'queries.json') if f.is_file()),None)
    if fallback:queries.update(rec.read(fallback))
    for shard in range(4):
        folder=parent/'shards'/('part'+str(shard));q=rec.optional(folder/'queries.json',{}) or {}
        for pack in schedule.owned_packs(sources,shard):
            for source in pack['sources']:
                sid=source['id'];source_roots[sid]=[folder,parent,old]
                for method in public.METHODS:
                    key=method+'/'+sid
                    if key in q:queries[key]=q[key]
                    queries.setdefault(key,dict(wrong_query_collision=None))
    result=build_tables(destination,sources,roster,queries,PublicCost(parent,sources),old,source_roots)
    resource_path=Path(destination)/'tables/D_resources.json';resources=rec.read(resource_path)
    for field,filename in [('model_resources','model_resources.jsonl'),('quality_resources','quality_resources.jsonl'),('model_loads','model_loads.jsonl'),('import_receipts','imports.jsonl')]:
        roots=[parent]+[parent/'shards'/('part'+str(i)) for i in range(4)]
        resources[field]=[dict(row,record_root=str(root)) for root in roots for row in events(root/filename)]
    resources['old_v4_cost_scope']='separate prior protocol; referenced without counting as new compact calls'
    rec.write(resource_path,resources);return result


def finalize(previous_runtime_stopped=False,output=OUTPUT,cohort_root=p.COHORT_ROOT,wan_root=p.WAN_ROOT,query_root=p.QUERY_ROOT,
             public_root=public.OUTPUT):
    from .paper_query_statistics_v1 import run as queries
    from .paper_suite_tables_v1 import build
    output=Path(output)
    with runtime_session(output/'owners/finalizer',previous_runtime_stopped=previous_runtime_stopped):
        active_worker_check(previous_runtime_stopped,output,public_root)
        query_result=queries(cohort_root,query_root)
        destination=output/'final_reports'/uuid.uuid4().hex;destination.mkdir(parents=True)
        public_result=collect_public(destination/'public',public_root)
        report=build(cohort_root,wan_root,query_root,destination/'public',destination/'main')
        # This is the only parallel-suite writer of the shared latest index.
        rec.write(Path(p.REPORT_ROOT)/'latest_report.json',report)
        complete=all(row.get('status') in ('COMPLETED','STOPPED_FIXED_ATTEMPT') for row in report['components']) and query_result.get('status') in ('COMPLETED','READ_ONLY') and public_result.get('status')=='COMPACT_MEASUREMENT_RETURNED'
        value=dict(status='FINAL_REPORT_SAVED' if complete else 'PARTIAL_REPORT_SAVED',queries=query_result,public=public_result,report=report,
            output=str(destination),new_model_calls=0,new_codec_calls=0)
        rec.write(output/'latest_final.json',value);return value
