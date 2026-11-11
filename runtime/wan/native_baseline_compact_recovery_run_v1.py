"""CPU recovery or user-run targeted physical continuation, then global tables."""
import argparse
import json
from pathlib import Path
import signal
import time
from main.baselines import public_compact12_v1 as p
from .native_baseline_storage_v1 import read_json, write_json
from .native_baseline_storage_v4 import append_json, runtime_session
from .native_baseline_colab_v4 import check_previous_worker
from .native_baseline_compact_recovery_v1 import Context, OUTPUT, cpu_recover


def run(parent=p.OUTPUT,output=OUTPUT,cache=p.CACHE,old=p.OLD_OUTPUT,old_cache=p.OLD_CACHE,
        cpu_only=False,tables=True,previous_runtime_stopped=True,vendor_factory=None,
        tables_only=False,local_work=None):
    ctx=Context(parent,output,cache,old,old_cache)
    # The new repair owns its own runtime, while all original shards remain read-only.
    for shard in range(4):check_previous_worker(ctx.parent/'shards'/f'part{shard}',previous_runtime_stopped)
    tick=time.monotonic()
    with runtime_session(ctx.output,previous_runtime_stopped=previous_runtime_stopped):
        from .native_baseline_compact_local_tables_v1 import collect_local
        if tables_only:
            # Saved records only: no recovery, media reconstruction or model import.
            return collect_local(ctx,local_work)['delivery']
        before=cpu_recover(ctx);physical=[]
        if not cpu_only and (before['logical_remaining'] or before['quality_statuses'].get('PAIRED_MATERIAL_UNAVAILABLE',0)):
            from .native_baseline_compact_recovery_gpu_v1 import run_physical
            if vendor_factory is None:
                from .paper_suite_worker_v1 import prepare_vendor
                vendor_factory=prepare_vendor
            physical=run_physical(ctx,before,vendor_factory,previous_runtime_stopped)
        after=cpu_recover(ctx) if physical else before
        combined=collect_local(ctx,local_work) if tables else None
        value=dict(status='REPAIR_COMPLETE' if not after['logical_remaining'] and not after['quality_statuses'].get('PAIRED_MATERIAL_UNAVAILABLE',0) else 'REPAIR_INCOMPLETE_RETAINED',
            output=str(ctx.output),cpu_only=cpu_only,before=dict(logical_remaining=before['logical_remaining'],quality_statuses=before['quality_statuses']),
            after=dict(logical_remaining=after['logical_remaining'],quality_statuses=after['quality_statuses']),
            global_result=combined,physical_events=len(physical),wall_seconds=time.monotonic()-tick,
            old_results_unchanged=True,scientific_success='NOT_ADJUDICATED')
        path=ctx.output/'result.json'
        if path.is_file():append_json(ctx.output/'result_history.jsonl',read_json(path))
        write_json(path,value);return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name,default in [('parent',p.OUTPUT),('output',OUTPUT),('cache',p.CACHE),('old',p.OLD_OUTPUT),('old-cache',p.OLD_CACHE)]:
        parser.add_argument('--'+name,type=Path,default=default)
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--no-tables',action='store_true',help='For a partial local evidence mirror, omit unavailable full-cohort tables')
    parser.add_argument('--tables-only',action='store_true',help='Aggregate saved records locally; no model, codec or raw-array replay')
    parser.add_argument('--local-work',type=Path,help='Local staging directory outside Drive and formal result roots')
    parser.add_argument('--previous-runtime-stopped',action='store_true')
    args=parser.parse_args(argv)
    def stop(signum,frame):raise KeyboardInterrupt('runtime interrupted')
    signal.signal(signal.SIGTERM,stop)
    try:
        value=run(args.parent,args.output,args.cache,args.old,args.old_cache,args.cpu_only,
            not args.no_tables,args.previous_runtime_stopped,tables_only=args.tables_only,local_work=args.local_work)
        print(json.dumps(value,ensure_ascii=False),flush=True);return 0
    except KeyboardInterrupt:return 130


if __name__=='__main__':raise SystemExit(main())
