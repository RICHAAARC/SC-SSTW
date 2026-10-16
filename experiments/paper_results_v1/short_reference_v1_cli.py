"""Run the fixed saved-Y CPU comparison from a plain source directory."""
import argparse
import json
from experiments.paper_results_v1 import short_reference_v1 as task


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--source-run',default=task.SOURCE_DEFAULT)
    args=parser.parse_args()
    store=task.run(args.output,args.source_run)
    print(json.dumps(task.report(store),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
