"""Ordinary movable zip/directory handoffs and isolated user GPU subprocesses."""
from pathlib import Path,PurePosixPath
import importlib.util
import json
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

STEM='b_joint_time_packet_a_physical_v1'


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def saved(output):
    result=dict(status='READ_ONLY_CONSUMED_STAGE',physical_calls_this_invocation=0,saved={},read_errors={})
    for name in ('attempt','state','counts','resources','blind_terminal','result'):
        try:result['saved'][name]=json.loads((Path(output)/(name+'.json')).read_text())
        except (OSError,ValueError) as error:result['read_errors'][name]=str(error)
    return result


def unpack(source,destination,receipt):
    """No identity verification. Reject unsafe archive paths and actual malformed input."""
    source=Path(source).expanduser().resolve()
    if source.is_dir():root=source
    else:
        if not source.is_file():raise FileNotFoundError(source)
        root=Path(destination).resolve();root.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            for entry in archive.infolist():
                name=PurePosixPath(entry.filename.replace('\\','/'))
                mode=entry.external_attr>>16
                if (name.is_absolute() or '..' in name.parts or any(':' in p for p in name.parts)
                        or stat.S_ISLNK(mode) or not (root/str(name)).resolve().is_relative_to(root)):
                    raise ValueError('unsafe handoff archive path: '+entry.filename)
            archive.extractall(root)
    if (root/receipt).is_file():return root
    candidates=[p.parent for p in root.glob('*/'+receipt)]
    if len(candidates)==1:return candidates[0]
    raise ValueError('handoff terminal receipt not found: '+receipt)


def default_gpu_config():
    drive=Path('/content/drive/MyDrive/Video-WM')
    old=drive/'B-Native-Y-Source-Confirmation-V1/fixed-n01-v1-rerun01'
    return dict(model_id='Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
        n01_base=str(old/'time_chain/materialization/A/BASE_preclip.npy'),
        n01_features=str(old/'time_chain/materialization/A/features'),
        n01_old_bank=str(old/'payload/bank'),
        e02_terminal=str(drive/'B-Terminal-Three-Group-V1/fixed-four-new-v1/private/b2-m3-e02/P0_terminal.pt'))


def run_worker(command,root,log,receipt,stage_output,*,process_factory=None):
    """One owned child; explicit TERM/30s/KILL/reap, without stage recovery or retry."""
    log,receipt,stage_output=Path(log),Path(receipt),Path(stage_output)
    receipt.parent.mkdir(parents=True,exist_ok=True);log.parent.mkdir(parents=True,exist_ok=True)
    began=time.monotonic();worker=None
    state=dict(status='STARTING',terminate_sent=False,kill_sent=False,reaped=False,
               terminate_wait_seconds=30,worker_restarted=False)
    def save():
        state.update(wall_seconds=time.monotonic()-began,attempt_present=(stage_output/'attempt.json').is_file(),
                     blind_terminal_present=(stage_output/'blind_terminal.json').is_file(),
                     missing_blind_terminal='retained as missing; no fabricated success or retry')
        temporary=receipt.with_name(receipt.name+'.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(receipt)
    save()
    with log.open('w') as stream:
        try:
            worker=(process_factory or subprocess.Popen)(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
            state.update(status='RUNNING',pid=worker.pid);save()
            code=worker.wait()
            state.update(status='WORKER_EXITED',returncode=code,reaped=True);return code
        except BaseException as error:
            state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'WORKER_WAIT_FAILED',
                         error=type(error).__name__+': '+str(error))
            if worker is not None:
                try:
                    if worker.poll() is None:
                        try:worker.terminate();state['terminate_sent']=True
                        except ProcessLookupError:state['exited_before_terminate']=True
                        save()
                        try:code=worker.wait(timeout=30)
                        except (subprocess.TimeoutExpired,KeyboardInterrupt):
                            worker.kill();state['kill_sent']=True;save();code=worker.wait()
                    else:code=worker.wait()
                    state.update(returncode=code,reaped=True)
                except BaseException as cleanup_error:
                    state['cleanup_error']=type(cleanup_error).__name__+': '+str(cleanup_error)
            raise
        finally:save()


def launch(cpu_handoff,cpu_media,output,source_root,model_id=None):
    output,root=Path(output).expanduser().resolve(),Path(source_root).resolve()
    results={};pending=[]
    for stage,input_path,receipt in (('gpu1',cpu_handoff,'cpu_handoff.json'),('gpu2',cpu_media,'blind_terminal.json')):
        if (output/stage/'attempt.json').exists():results[stage]=saved(output/stage)
        elif not Path(input_path).exists():results[stage]=dict(status='WAITING_INPUT',input=str(input_path),physical_calls_this_invocation=0)
        else:pending.append((stage,input_path,receipt))
    if not pending:return dict(status='STAGED_STATUS',stages=results)
    setup=load_module(root/'scripts'/f'{STEM}_setup.py','joint_a_setup')
    setup.setup_dependencies(root,output/'preparation')
    for stage,input_path,receipt in pending:
        try:
            with tempfile.TemporaryDirectory(prefix='joint-a-handoff-') as temporary:
                folder=unpack(input_path,Path(temporary)/'input',receipt)
                cfg=default_gpu_config()
                cfg['cpu_handoff' if stage=='gpu1' else 'cpu_media']=str(folder)
                if model_id is not None:cfg['model_id']=model_id
                path=Path(temporary)/'config.json';path.write_text(json.dumps(cfg,indent=2))
                # Each stage releases its VAE/process. No GPU waits for local media.
                log=output/'preparation'/(stage+'.console.log')
                code=run_worker([sys.executable,'-B','-m','experiments.wan_state_clock.'+STEM+'_run',stage,
                    '--config',str(path),'--output',str(output/stage)],root,log,
                    output/'preparation'/(stage+'.worker.json'),output/stage)
                if (output/stage/'attempt.json').exists():results[stage]=saved(output/stage)
                else:
                    try:results[stage]=json.loads(log.read_text())
                    except ValueError:results[stage]=dict(status='PREPARATION_FAILED',returncode=code,log=str(log))
        except Exception as error:results[stage]=dict(status='PREPARATION_FAILED',error=type(error).__name__+': '+str(error))
    return dict(status='STAGED_STATUS',stages=results,first_run='GPU-1 only; local CPU media then a later user GPU-2 run')
