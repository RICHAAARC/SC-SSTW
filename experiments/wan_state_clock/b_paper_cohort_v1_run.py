"""Frozen TEST-A/TEST-B Colab worker and batch CLI."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('configs')/'b_paper_cohort_v1.json')
    parser.add_argument('--batch',choices=['TEST-A','TEST-B'])
    parser.add_argument('--worker-stage',choices=['prepare','solve','replay','receive','identity'])
    parser.add_argument('--output',type=Path);parser.add_argument('--scratch',type=Path)
    args=parser.parse_args()
    if args.worker_stage:
        if args.output is None or args.scratch is None: parser.error('worker requires output/scratch')
        from runtime.wan.b_paper_cohort_worker_v1 import worker
        result=worker(args.worker_stage,args.output,args.scratch)
    else:
        from runtime.wan.b_paper_cohort_runner_v1 import run_batch
        config=json.loads(args.config.read_text())
        result=run_batch(args.batch or config['batch'],config['output_root'],config['scratch_root'])
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=1))
    return 1 if result['status']=='STOPPED_FIXED_ATTEMPT' else 0


if __name__=='__main__': raise SystemExit(main())
