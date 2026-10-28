"""Complete CLI; notebook reserves once before calling the internal worker."""
import argparse
import json
import signal
from pathlib import Path

def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(Path(__file__).parent/'configs/b_local_vae_head_v1.json'))
    p.add_argument('--local-output');p.add_argument('--mirror');p.add_argument('--inputs')
    return p

def _notebook_reserved_main():
    from runtime.wan.b_local_vae_head_v1 import run
    args=parser().parse_args();cfg=json.loads(Path(args.config).read_text())
    if not args.inputs or not args.local_output or not args.mirror:raise ValueError('reserved worker needs concrete local output, mirror and downloaded inputs')
    install_termination_handler()
    result=run(cfg,args.local_output,json.loads(Path(args.inputs).read_text()),mirror=args.mirror)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result.get('status') in ('COMPLETED_FIXED_ATTEMPT','READ_ONLY_EXISTING_ATTEMPT') else 1

def install_termination_handler():
    def interrupted(signum,frame):raise InterruptedError('worker terminated by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)

def main():
    from runtime.wan import b_local_vae_head_storage_v1 as storage
    from runtime.wan.b_local_vae_head_v1 import run
    args=parser().parse_args();cfg=json.loads(Path(args.config).read_text())
    # Outside Colab, standard ADC may be supplied. No model work occurs before storage check.
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    credentials,_=google.auth.default(scopes=['https://www.googleapis.com/auth/drive'])
    client=storage.Drive(AuthorizedSession(credentials));mirror=args.mirror or cfg['output']
    if not storage.preflight(client,mirror)['can_start']:
        print(json.dumps(storage.read_existing(client,mirror),ensure_ascii=False));return 0
    from runtime.wan.b_local_vae_head_setup_v1 import prepare_vendor,check_dependencies
    source_root=Path(__file__).resolve().parents[2]
    vendor=prepare_vendor(cfg['scratch']);check_dependencies(source_root,vendor)
    inputs=json.loads(Path(args.inputs).read_text()) if args.inputs else storage.download_training(client,cfg,cfg['scratch'])
    receipt=storage.reserve(client,mirror)
    if receipt['status']!='START_RECEIPT_CLOUD_READ_BACK':raise RuntimeError('once-attempt reservation not established')
    local=args.local_output or str(Path(cfg['scratch'])/'results')
    install_termination_handler()
    result=run(cfg,local,inputs,mirror=mirror)
    saved=storage.sync_results(client,local,mirror)
    print(json.dumps(dict(result=result,storage=saved),ensure_ascii=False));return 0 if result['status']=='COMPLETED_FIXED_ATTEMPT' else 1

if __name__=='__main__':raise SystemExit(main())
