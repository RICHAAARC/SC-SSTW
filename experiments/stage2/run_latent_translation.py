"""One-process-group wall limit. No implicit GPU retry."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:
        from runtime.stage2.wan_translation import run
        run(a.config,a.output);return
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('retain prior output; this protocol does not rerun')
    config=json.loads(Path(a.config).read_text());start=time.monotonic()
    log=out.parent/(out.name+'.execution.log');exit_path=out.parent/(out.name+'.execution_exit.json')
    cmd=[sys.executable,'-m','experiments.stage2.run_latent_translation','--worker','--config',str(Path(a.config).resolve()),'--output',str(out.resolve())]
    timed_out=False
    with log.open('x') as stream:
        proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=config['engineering_budgets']['gpu_wall_seconds'])
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
    record=dict(command=cmd,exit_code=code,timed_out=timed_out,elapsed_seconds=time.monotonic()-start,
                fixed_arms=list(config['intervention']['arms']),log=str(log),automatic_retries=0)
    exit_path.write_text(json.dumps(record,indent=2)+'\n')
    if timed_out:
        states={k:dict(status='NOT_COMPLETED_TIMEOUT') for k in config['intervention']['arms']}
        if (out/'arm_status.json').exists():states.update(json.loads((out/'arm_status.json').read_text()))
        record['retained_arm_status']=states
        exit_path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if code:sys.exit(124 if timed_out else 1)
if __name__=='__main__':main()
