"""Direct user entry; workers are process boundaries, not method modes."""
import argparse
import json
import os
import signal
import subprocess
import sys
from experiments.paper_results_v1.segmented_payload_v1 import Backend,OUTPUT_DEFAULT,PHASES,run_phase


def main():
    parser=argparse.ArgumentParser(description='Fixed four-source segmented 32-bit payload use demonstration')
    parser.add_argument('--output',default=OUTPUT_DEFAULT)
    parser.add_argument('--wan-model',default='/content/drive/MyDrive/Video-WM/Paper-Results-V1-Cache/models/Wan2.1-T2V-1.3B-Diffusers/available')
    parser.add_argument('--framewise-model',default='/content/drive/MyDrive/Video-WM/Paper-Results-V1-Cache/models/sd-vae-ft-mse/available')
    parser.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    parser.add_argument('--worker',choices=('init','report',*PHASES),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        backend=Backend(args.wan_model,args.framewise_model,args.device)
        store=run_phase(args.output,args.worker,backend)
        print(json.dumps(dict(phase=args.worker,status=store.data['phases'][args.worker]),ensure_ascii=False))
        return
    run_phase(args.output,'init')
    codes={}
    for phase in PHASES:
        command=[sys.executable,'-u','-m',__spec__.name,'--output',args.output,'--wan-model',args.wan_model,
                 '--framewise-model',args.framewise_model,'--device',args.device,'--worker',phase]
        print('Starting phase:',phase,flush=True)
        # Inherit the notebook CLI process group so its interruption reaches workers too.
        child=subprocess.Popen(command)
        try:codes[phase]=child.wait()
        except BaseException:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            run_phase(args.output,'report');raise
        # A failed unit is retained; independent unstarted units in later phases may continue.
    store=run_phase(args.output,'report')
    store.data['last_worker_exit_codes']=codes;store.save()
    print('Fixed 36-row results:',store.output/'row_results.json',flush=True)
    if any(codes.values()):raise SystemExit('One or more phases failed; fixed rows and concrete errors were retained.')


if __name__=='__main__':main()
