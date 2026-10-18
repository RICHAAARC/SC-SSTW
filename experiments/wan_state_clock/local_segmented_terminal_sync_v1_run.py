"""B local draft: one saved segmented source, terminal state only, clean35."""
import argparse,json
from pathlib import Path
from runtime.wan.local_segmented_terminal_sync_v1 import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/local_segmented_terminal_sync_v1.json')
    args=p.parse_args();c=json.loads(args.config.read_text())
    out=run(c['source_terminal'],c['source_native'],c['output'],c['truth_record'],c['model_id'])
    print(json.dumps(dict(status=out['status'],output=c['output'],counts=out.get('counts')),indent=2))
    return 0 if out['status'] in ('EXECUTION_COMPLETE','EXISTING_ATTEMPT_NOT_REEXECUTED') else 1

if __name__=='__main__':raise SystemExit(main())
