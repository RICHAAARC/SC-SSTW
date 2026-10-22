"""Card101 one fixed seen-Q CPU development; existing attempt is read-only."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from runtime.wan.b_background_normalized_cpu_v1 import run_saved
if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run_saved(),ensure_ascii=False,indent=2,allow_nan=False))
