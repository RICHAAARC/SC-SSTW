"""Runtime-local complete-paper stage entry. GPU execution belongs to user Colab."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from main.tube_state import paper_suite_v1 as p
from . import paper_suite_records_v1 as rec

ROOT=Path(__file__).resolve().parents[2]
WORK=Path('/content/video-wm-paper-suite-v1')


def prepare_vendor():
    from .native_baseline_colab_v4 import source_download
    from .native_baseline_dependencies_v4 import _install_dependencies
    import urllib.request
    vendor=WORK/'vendor';vendor.mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HOME=str(WORK/'weights/huggingface'),TORCH_HOME=str(WORK/'weights/torch'))
    source_download('facebookresearch/videoseal','main',vendor/'videoseal')
    source_download('facebookresearch/watermark-anything','main',vendor/'watermark-anything')
    _install_dependencies(ROOT,vendor)
    weight=vendor/'watermark-anything/checkpoints/wam_mit.pth';weight.parent.mkdir(parents=True,exist_ok=True)
    if not weight.is_file():
        pending=weight.with_suffix('.download')
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth',pending)
        os.replace(pending,weight)
    return vendor


def run(stage,previous_runtime_stopped=False):
    if stage=='cohort':
        from .b_paper_cohort_colab_v1 import run_colab
        rows=[]
        for batch in ('TEST-A','TEST-B'):
            row=run_colab(batch,source_directory=ROOT,source_ref='main');rows.append(row)
            if row.get('interrupted') or row['status'] in ('BATCH_INTERRUPTED','PREPARATION_FAILED'):break
        return dict(status='INTERRUPTED' if rows[-1].get('interrupted') else rows[-1]['status'],batches=rows)
    if stage=='wan':
        from .paper_wan_comparison_v1 import run_all
        # Do not download models for an entirely consumed or not-ready panel.
        pending=[sid for sid in p.SOURCES if (Path(p.COHORT_ROOT)/sid/'attempt01/result.json').is_file()
                 and not (Path(p.WAN_ROOT)/sid/'attempt01/attempt.json').exists()]
        vendor=prepare_vendor() if pending else WORK/'vendor'
        return run_all(vendor=vendor)
    if stage=='queries':
        from .paper_query_statistics_v1 import run
        return run()
    if stage=='public_compact':
        from .native_baseline_compact_colab_v1 import run_colab
        return run_colab(source_directory=ROOT,source_ref='main',previous_runtime_stopped=previous_runtime_stopped)
    if stage=='public':
        from .native_baseline_colab_v4 import check_previous_worker,run_colab
        # Check before the old bootstrap can write a setup failure into an active package.
        try:check_previous_worker(p.PUBLIC_ROOT,previous_runtime_stopped)
        except RuntimeError as error:return dict(status='EXTERNAL_RUNTIME_ACTIVE',error=str(error),output=p.PUBLIC_ROOT)
        return run_colab(source_directory=ROOT,source_ref='main',previous_runtime_stopped=previous_runtime_stopped)
    if stage=='tables':
        from .paper_suite_tables_v1 import build
        return build()
    raise ValueError('unknown stage')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['cohort','wan','queries','public','public_compact','tables'])
    parser.add_argument('--previous-runtime-stopped',action='store_true')
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args(argv)
    def stop(signum,frame):raise KeyboardInterrupt('user runtime stopped')
    signal.signal(signal.SIGTERM,stop)
    started=time.monotonic()
    try:
        result=run(args.stage,args.previous_runtime_stopped)
        rec.write(args.receipt,dict(result,wall_seconds=time.monotonic()-started))
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return 130 if result.get('status') in ('BATCH_INTERRUPTED','INTERRUPTED') else 0
    except BaseException as error:
        interrupted=isinstance(error,(KeyboardInterrupt,InterruptedError))
        rec.write(args.receipt,dict(status='INTERRUPTED' if interrupted else 'STAGE_FAILED',
            error=f'{type(error).__name__}: {error}',wall_seconds=time.monotonic()-started))
        return 130 if interrupted else 1


if __name__=='__main__':raise SystemExit(main())
