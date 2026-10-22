"""Fixed user full-context E4 entry; no execution on import."""
import argparse,json
from runtime.wan.b_full_context_local_read_v1 import run,SOURCE,OUTPUT,REFERENCE
def main():
    p=argparse.ArgumentParser(description='Fixed positive-channel full E4 with historical local reference.')
    p.add_argument('--source',default=SOURCE);p.add_argument('--output',default=OUTPUT);p.add_argument('--reference-file',default=str(REFERENCE))
    a=p.parse_args();print(json.dumps(run(a.source,a.output,a.reference_file),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
