"""CPU Phase1 launcher; this executes real pixel media only after user approval."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:
        from runtime.public_statistic.phase1 import run
        run(a.config,a.output);return
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);log=out.parent/(out.name+'.log');record=out.parent/(out.name+'.exit.json')
    if any(x.exists() for x in (out,log,record)):raise FileExistsError('preserve previous outputs')
    c=json.loads(Path(a.config).read_text())
    from runtime.public_statistic.phase1 import validate_config
    validate_config(c)
    cmd=[sys.executable,'-m','experiments.public_statistic.run_phase1','--worker','--config',str(Path(a.config).resolve()),'--output',str(out.resolve())];start=time.monotonic();timeout=False
    with log.open('x') as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=c['wall_seconds'])
        except subprocess.TimeoutExpired:timeout=True;os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
    record.write_text(json.dumps(dict(command=cmd,exit_code=code,timed_out=timeout,elapsed=time.monotonic()-start,fixed_inputs=c['inputs'],fixed_arms=c['arms'],fixed_rows=192,automatic_retries=0),indent=2)+'\n')
    if timeout and (out/'status.json').exists():
        states=json.loads((out/'status.json').read_text())
        for arms in states.values():
            for state in arms.values():
                if state['status']=='RUNNING':state.update(previous_status='RUNNING',status='NOT_COMPLETED_TIMEOUT')
        (out/'status_at_timeout.json').write_text(json.dumps(states,indent=2)+'\n')
    if code:raise SystemExit(124 if timeout else 1)
if __name__=='__main__':main()
