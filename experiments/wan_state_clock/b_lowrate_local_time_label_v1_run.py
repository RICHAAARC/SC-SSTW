"""Fixed local-tag user CLI, independently callable from editable source."""
import argparse,json
from runtime.wan.b_lowrate_local_time_label_v1 import run
from main.tube_state.b_lowrate_local_time_label_plan_v1 import SOURCE,OUTPUT

def main():
    parser=argparse.ArgumentParser(description='Fixed low-rate local time label development package; GPU user-run only')
    parser.add_argument('--source',default=SOURCE);parser.add_argument('--output',default=OUTPUT)
    args=parser.parse_args();print(json.dumps(run(args.source,args.output),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
