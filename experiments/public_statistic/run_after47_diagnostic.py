"""One supervised after47 diagnostic. Import performs no GPU/model work."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from experiments.public_statistic.run_phase2 import install_lifetime,stop_group,process_tree_rss,RunCancelled
from runtime.public_statistic.phase2_execution import ResourceGuard,load_wan,validate_config,write

BUDGET=dict(transformer_calls=116,vae_calls=5,backward_calls=1,mp4=0,saved_frames=0,wall_seconds=1800)
REPLAY=dict(transformer_block=120,vae_chunk=13)


def stopped(out,reason):
    p=Path(out)/'result.json'
    r=json.loads(p.read_text()) if p.exists() else dict(science_denominator=0)
    r.update(status='FATAL_STOP',error=reason);write(p,r)


def worker(config,reference,output):
    import torch
    import numpy as np
    from runtime.public_statistic.after47_diagnostic import diagnose
    c=json.loads(Path(config).read_text());validate_config(c)
    out=Path(output);report=dict(status='RUNNING',science_denominator=0,automatic_retries=0,active='LOAD_MODEL',reference=str(reference),budget=BUDGET,recomputation_budget=REPLAY)
    def persist():write(out/'result.json',report)
    persist();guard=ResourceGuard(BUDGET,torch=torch,persist=lambda counts:write(out/'progress.json',dict(attempted=guard.counts,completed=guard.completed,current=guard.events[-1] if guard.events else None)))
    adapter=None;start=time.monotonic()
    try:
        # Fixed diagnosis of the existing after47 point, not a parameter search.
        if c['generation']['steps']!=50 or c['feedback']['after_indices']!=[47,48]:raise ValueError('after47 entry requires the 50-step source configuration')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 computation required')
        raw=np.load(reference,allow_pickle=False)
        expected=(c['generation']['frames'],c['generation']['height'],c['generation']['width'],3)
        if raw.shape!=expected or not np.isfinite(raw).all():raise ValueError('OFF1 reference shape/nonfinite')
        c['checkpoint_recomputation']=REPLAY.copy();write(out/'config.json',c)
        write(out/'runtime.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),allocator_cap_gib=None,device_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory))
        torch.cuda.reset_peak_memory_stats()
        adapter,z,state=load_wan(c,guard,torch=torch,record=lambda v:write(out/'loaded_model.json',v))
        reference_rgb=torch.from_numpy(raw).to(device=z.device,dtype=torch.float32);del raw
        report['active']='REBUILD_STEP_0_TO_47';persist()
        with torch.no_grad():
            while state.next_index<47:z,state=adapter.advance(z,state)
            r0=float(z.square().mean().sqrt())
            z,state=adapter.advance(z,state)
        report['prefix_rms_before_step47']=r0
        diagnose(adapter,z,state,reference_rgb,learning_rate=c['feedback']['learning_rate'],step_rms=c['feedback']['per_update_fraction']*r0,amplitude=c['feedback']['amplitude'],report=report,persist=persist,save=lambda data:torch.save(data,out/'after47_state_and_gradient.pt'))
    except BaseException as exc:
        report.update(status='FATAL_STOP',error=repr(exc));raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-start,counts=dict(attempted=guard.counts,completed=guard.completed))
        if torch.cuda.is_available():report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if adapter is not None and adapter.checkpoint_ledger is not None:write(out/'recomputation.json',adapter.checkpoint_ledger.summary())
        persist()


def supervise(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    write(out/'result.json',dict(status='RUNNING',science_denominator=0,active='LAUNCHER_START',automatic_retries=0,budget=BUDGET,recomputation_budget=REPLAY))
    command=[sys.executable,'-m','experiments.public_statistic.run_after47_diagnostic','--worker','--parent-pid',str(os.getpid()),'--config',str(Path(a.config).resolve()),'--reference',str(Path(a.reference).resolve()),'--output',str(out.resolve())]
    child=None;reason=None;start=time.monotonic();memory=dict(peak_tree_rss_bytes=None)
    try:
        with (out/'execution.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                if time.monotonic()-start>=BUDGET['wall_seconds']:reason='WALL_BUDGET';break
                try:memory['peak_tree_rss_bytes']=max(memory['peak_tree_rss_bytes'] or 0,process_tree_rss(child.pid))
                except Exception as exc:memory['error']=repr(exc)
                time.sleep(.2)
            if reason:stop_group(child);stopped(out,reason)
            elif child.returncode:
                stop_group(child)
                # Keep the worker traceback/result; only supply a missing failure record.
                r=json.loads((out/'result.json').read_text())
                if r['status']=='RUNNING':stopped(out,'WORKER_FAILURE')
    except BaseException:
        reason='LAUNCHER_INTERRUPTED'
        if child is not None:stop_group(child)
        stopped(out,reason);raise
    finally:write(out/'execution_exit.json',dict(command=command,returncode=child.returncode if child else None,stop=reason,elapsed_seconds=time.monotonic()-start,automatic_retries=0,host_memory_diagnostic=memory))
    return child.returncode if child.returncode else (1 if reason else 0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--parent-pid',type=int);a=p.parse_args()
    try:
        install_lifetime(a.parent_pid if a.parent_pid is not None else os.getppid())
        if a.worker:worker(a.config,a.reference,a.output)
        else:sys.exit(supervise(a))
    except RunCancelled as exc:
        if Path(a.output).exists():stopped(a.output,str(exc))
        if a.worker and os.getpgrp()==os.getpid():os.killpg(os.getpid(),signal.SIGKILL)
        sys.exit(130)

if __name__=='__main__':main()
