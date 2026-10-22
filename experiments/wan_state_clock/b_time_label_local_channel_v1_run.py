"""Card106 direct user CLI. Real execution is never entered on import."""
import argparse,json
from runtime.wan.b_time_label_local_channel_v1 import run,DEFAULT_SOURCE,DEFAULT_DELETION,DEFAULT_OUTPUT,CONFIG
def main():
    p=argparse.ArgumentParser(description='Fixed adopted seen-local time-label channel, one user physical attempt.')
    p.add_argument('--source-original',default=DEFAULT_SOURCE)
    p.add_argument('--source-deletion',default=DEFAULT_DELETION)
    p.add_argument('--output',default=DEFAULT_OUTPUT)
    p.add_argument('--config',default=str(CONFIG))
    a=p.parse_args();print(json.dumps(run(a.source_original,a.source_deletion,a.output,a.config),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
