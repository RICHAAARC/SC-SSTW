"""Card101 one fixed user-run normalized Q matching task. Existing attempt reads saved records."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from runtime.wan.b_background_normalized_confirmation_v1 import run
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=None)
    args=parser.parse_args();print(json.dumps(run(args.output),ensure_ascii=False,indent=2,allow_nan=False))
