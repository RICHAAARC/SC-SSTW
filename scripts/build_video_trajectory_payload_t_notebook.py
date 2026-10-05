"""Build the fixed adopted T notebook."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.build_video_trajectory_payload_gt_notebooks import build

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-sha")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    print(build("T",args.source_sha,args.output))
