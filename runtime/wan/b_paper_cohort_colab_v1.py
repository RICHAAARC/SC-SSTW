"""Thin stdlib Colab bootstrap. All consumed sources are readable offline."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import urllib.request

REPOSITORY='RICHAAARC/SC-SSTW'
SOURCE_REF='main'
OUTPUT_ROOT=Path('/content/drive/MyDrive/Video-WM_Result/main_method/paper_cohort_v1')
WORK=Path('/content/video-wm-paper-cohort-v1')


def source_text(path,source_directory,source_ref):
    if source_directory: return (Path(source_directory)/path).read_text()
    return urllib.request.urlopen(f'https://raw.githubusercontent.com/{REPOSITORY}/{source_ref}/{path}',timeout=120).read().decode('utf-8')


def standalone(path,source_directory,source_ref):
    module=types.ModuleType(Path(path).stem)
    exec(compile(source_text(path,source_directory,source_ref),path,'exec'),module.__dict__)
    return module


def ids(batch):
    if batch not in ('TEST-A','TEST-B'): raise ValueError('fixed TEST-A or TEST-B required')
    return [f'paper-joint-t{i:02d}' for i in (range(1,5) if batch=='TEST-A' else range(5,13))]


def show_results(batch='TEST-A',output_root=OUTPUT_ROOT):
    root=Path(output_root);print('正式保留目录:',root)
    for sid in ids(batch):
        path=root/sid/'attempt01';row=dict(source=sid,status='NOT_STARTED')
        for name in ('result.json','evaluation_summary.json','post_status.json','preparation_failure.json'):
            if (path/name).is_file():
                try:
                    value=json.loads((path/name).read_text())
                    if not isinstance(value,dict): raise ValueError('saved record must be a JSON object')
                    row[name]={k:value[k] for k in ('status','failure','error','comparisons','claims','B_correct_supported_bounds',
                        'U_correct_supported_bounds','delta_bounds','NEW_node_statuses','NEW_q0_key00_MATCH','B_wrong_attribution_observed') if k in value}
                except (OSError,ValueError) as error: row[name]=dict(read_error=str(error))
        if path.exists(): row['status']='CONSUMED'
        print(json.dumps(row,ensure_ascii=False))
    for name in (batch+'_summary.json','ALL_summary.json'):
        if (root/name).is_file():
            try:
                value=json.loads((root/name).read_text())
                print(name,json.dumps({k:value[k] for k in ('source_denominator','delta','descriptive_targets','gain_claim')},ensure_ascii=False))
            except (OSError,ValueError,KeyError,TypeError) as error:
                print(name,'已有汇总不可读；不补后评:',str(error))


def prepare_source(source_directory,source_ref,output):
    if source_directory: source=Path(source_directory).resolve()
    else:
        from urllib.parse import quote
        helper=standalone('runtime/wan/b_colab_support_v1.py',None,source_ref)
        WORK.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix='available-source-',dir=WORK));archive=stage/'source.zip'
        urllib.request.urlretrieve(f'https://codeload.github.com/{REPOSITORY}/zip/refs/heads/{quote(source_ref,safe="/")}',archive)
        helper.safe_extract(archive,stage/'unpacked')
        choices=[f for f in (stage/'unpacked').iterdir() if (f/'runtime/wan/b_paper_cohort_runner_v1.py').is_file()]
        if len(choices)!=1: raise ValueError('one readable cohort source directory required')
        source=choices[0]
    sys.path.insert(0,str(source))
    os.environ.update(HF_HOME=str(WORK/'cache/huggingface'),TORCH_HOME=str(WORK/'cache/torch'))
    # Reuse the tested r02 actual-import installer; it does not instantiate a model.
    from runtime.wan.b_colab_support_v1 import install_dependencies
    install_dependencies(source,output)
    return source


def run_colab(batch='TEST-A',source_directory=None,source_ref=SOURCE_REF,output_root=OUTPUT_ROOT,
              max_sources=None,max_hours=None,previous_runtime_stopped=False):
    root=Path(output_root);selected=ids(batch)
    def needs(sid):
        path=root/sid/'attempt01';result=json.loads((path/'result.json').read_text()) if (path/'result.json').is_file() else {}
        if path.is_dir() and any(path.iterdir()) and not (path/'checkpoint_control.json').is_file():return False
        return result.get('status') not in ('COMPLETED','STOPPED_FIXED_ATTEMPT')
    pending=[sid for sid in selected if needs(sid)]
    if not pending:
        show_results(batch,root);return dict(status='READ_ONLY',output=str(root),batch=batch)
    import uuid,time
    preparation=WORK/'preparations'/uuid.uuid4().hex
    try:source=prepare_source(source_directory,source_ref,preparation)
    except BaseException as error:
        interrupted=isinstance(error,(KeyboardInterrupt,InterruptedError))
        return dict(status='BATCH_INTERRUPTED' if interrupted else 'PREPARATION_FAILED',interrupted=interrupted,
            error=str(error),output=str(root),source_attempt_consumed=False)
    from runtime.wan.b_paper_cohort_resume_v1 import run_source
    from runtime.wan.paper_checkpoint_v1 import Boundary,BoundaryPause
    from runtime.wan.paper_execution_v1 import check_full_owner
    check_full_owner(previous_runtime_stopped)
    boundary=Boundary(max_sources,None if max_hours is None else max_hours*3600);results=[];status='BATCH_RETURNED'
    try:
        for sid in pending:
            boundary.before()
            result=run_source(sid,output_root=root,seconds=None if boundary.deadline is None else max(.001,boundary.deadline-time.monotonic()),
                previous_runtime_stopped=previous_runtime_stopped)
            results.append(dict(source=sid,result=result))
            if result.get('interrupted'):
                status='BATCH_INTERRUPTED';break
            if result['status']=='PAUSED_BOUNDARY':raise BoundaryPause('saved source boundary')
            boundary.returned()
    except BoundaryPause:status='PAUSED_BOUNDARY'
    show_results(batch,root)
    return dict(status=status,interrupted=status=='BATCH_INTERRUPTED',batch=batch,output=str(root),source_results=results,
        checkpoint='actual stage outputs plus separate bounded infrastructure recovery; original unknown calls retained')
