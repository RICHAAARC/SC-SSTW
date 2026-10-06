"""Build the explicit-input video_trajectory_payload_framewise_sync_m05_v1 notebook; no real execution."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.notebook_common import build as build_notebook
OUTPUT=ROOT/'notebooks/video_trajectory_payload_framewise_sync_m05_v1_colab.ipynb'
def build(source_sha=None,output=OUTPUT):
    return build_notebook('video_trajectory_payload_framewise_sync_m05_v1','experiments.wan_state_clock.video_trajectory_payload_framewise_sync_m05_v1_run','experiments/wan_state_clock/configs/video_trajectory_payload_framewise_sync_m05_v1.json',source_sha=source_sha,output=output,media=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-sha');p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();print(build(a.source_sha,a.output))
