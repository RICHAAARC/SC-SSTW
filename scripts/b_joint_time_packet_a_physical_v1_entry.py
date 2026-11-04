"""One Colab Run all: GPU1, CPU media, GPU2, then blind-after-save evaluation."""
from pathlib import Path,PurePosixPath
import importlib.util
import json
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

STEM='b_joint_time_packet_a_physical_v1'
STAGES=('gpu1','cpu_media','gpu2','post')
DEPENDENCIES={'gpu1':(), 'cpu_media':('gpu1',), 'gpu2':('cpu_media',), 'post':('cpu_media','gpu2')}


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def saved(output):
    result=dict(status='READ_ONLY_CONSUMED_STAGE',physical_calls_this_invocation=0,saved={},read_errors={})
    for name in ('attempt','state','counts','resources','blind_terminal','result'):
        try:result['saved'][name]=json.loads((Path(output)/(name+'.json')).read_text())
        except (OSError,ValueError) as error:result['read_errors'][name]=str(error)
    return result


def extract(source,destination,prefixes=None):
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
            selected=[entry for entry in archive.infolist() if prefixes is None or
                      any(entry.filename.replace('\\','/').startswith(p) for p in prefixes)]
            archive.extractall(root,members=selected)
    return root


def unpack(source,destination,receipt):
    root=extract(source,destination)
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
    began=time.monotonic();worker=None;reader=None
    state=dict(status='STARTING',terminate_sent=False,kill_sent=False,reaped=False,
               terminate_wait_seconds=30,worker_restarted=False)
    def save():
        state.update(wall_seconds=time.monotonic()-began,attempt_present=(stage_output/'attempt.json').is_file(),
                     blind_terminal_present=(stage_output/'blind_terminal.json').is_file(),
                     missing_blind_terminal='retained as missing; no fabricated success or retry')
        temporary=receipt.with_name(receipt.name+'.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(receipt)
    save()
    with log.open('w') as stream:
        def relay():
            try:
                for line in worker.stdout:
                    stream.write(line);stream.flush();print(line,end='',flush=True)
            except Exception as error:state['console_error']=type(error).__name__+': '+str(error)
        try:
            worker=(process_factory or subprocess.Popen)(command,cwd=root,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,bufsize=1)
            state.update(status='RUNNING',pid=worker.pid);save()
            print(f'[{stage_output.name}] Worker started: PID={worker.pid}; console log={log}',flush=True)
            if getattr(worker,'stdout',None) is not None:
                reader=threading.Thread(target=relay,daemon=True);reader.start()
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
        finally:
            if reader is not None:
                reader.join(timeout=5);state['console_drained']=not reader.is_alive()
                if not reader.is_alive():worker.stdout.close()
            save()
            print(f'[{stage_output.name}] Worker finished: status={state["status"]}; returncode={state.get("returncode")}; console log={log}',flush=True)


def terminal(folder):
    try:return json.loads((Path(folder)/'blind_terminal.json').read_text()).get('stage_terminal') is True
    except (OSError,ValueError,AttributeError):return False


def package_config(packages,destination,stage):
    """Blind stages extract only received inputs; H/L and post config stay unopened."""
    prefixes=(('media_inputs.json','layer_raw/','time_raw/','e02_old_media/') if stage=='cpu_media'
              else ('post_inputs.json','old_time/'))
    packages=[packages] if isinstance(packages,(str,Path)) else packages
    if len(packages)==1 and Path(packages[0]).is_dir():folder=Path(packages[0]).resolve()
    else:
        folder=Path(destination).resolve()
        for package in packages:
            if Path(package).is_dir():raise ValueError('use one input directory or the input ZIP files')
            extract(package,folder,prefixes)
    name='media_inputs.json' if stage=='cpu_media' else 'post_inputs.json'
    cfg=json.loads((folder/name).read_text())
    def resolve(value):
        if isinstance(value,dict):return {k:resolve(v) for k,v in value.items()}
        path=Path(value)
        if path.is_absolute() or not (folder/path).resolve().is_relative_to(folder):
            raise ValueError('input member must stay inside the ordinary input package')
        return str(folder/path)
    keys=('layer_raw','time_raw','e02_old_media') if stage=='cpu_media' else ('old_time',)
    result={key:resolve(cfg[key]) for key in keys}
    if stage=='post':result['writer_protocol']=cfg['writer_protocol']
    return result


def launch(cpu_handoff,input_packages,output,source_root,model_id=None):
    output,root=Path(output).expanduser().resolve(),Path(source_root).resolve()
    if any(not (output/stage/'attempt.json').exists() for stage in STAGES):
        # Keep the pure display helper available for offline metadata-only reentry.
        (output/'preparation').mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/'scripts'/f'{STEM}_report.py',output/'preparation'/f'{STEM}_report.py')
    results={}
    with tempfile.TemporaryDirectory(prefix='joint-a-inputs-') as temporary:
        scratch=Path(temporary)
        for stage in STAGES:
            folder=output/stage
            if (folder/'attempt.json').exists():
                print(f'[{stage}] Consumed stage: reading saved metadata only.',flush=True)
                results[stage]=saved(folder);continue
            missing=[dep for dep in DEPENDENCIES[stage] if not terminal(output/dep)]
            if missing:
                results[stage]=dict(status='WAITING_INPUT',dependencies=missing,physical_calls_this_invocation=0)
                print(f'[{stage}] Waiting for saved terminal: {missing}; no attempt consumed.',flush=True);continue
            try:
                print(f'[{stage}] Preparing unstarted stage.',flush=True)
                if stage=='gpu1':
                    print(f'[{stage}] Loading CPU handoff: {cpu_handoff}',flush=True)
                    handoff=unpack(cpu_handoff,scratch/'cpu','cpu_handoff.json')
                    cfg=dict(default_gpu_config(),cpu_handoff=str(handoff))
                elif stage=='cpu_media':
                    cfg=dict(package_config(input_packages,scratch/'received',stage),gpu1=str(output/'gpu1'))
                elif stage=='gpu2':cfg=dict(default_gpu_config(),cpu_media=str(output/'cpu_media'))
                else:
                    # Both blind terminals were checked above, before any post config or H/L extraction.
                    cfg=dict(package_config(input_packages,scratch/'post',stage),
                             cpu_media=str(output/'cpu_media'),gpu2=str(output/'gpu2'))
                if model_id is not None:cfg['model_id']=model_id
                setup=load_module(root/'scripts'/f'{STEM}_setup.py','joint_a_setup')
                setup.setup_dependencies(root,output/'preparation',stage=stage)
                path=scratch/(stage+'.json');path.write_text(json.dumps(cfg,indent=2))
                log=output/'preparation'/(stage+'.console.log')
                code=run_worker([sys.executable,'-u','-B','-m','experiments.wan_state_clock.'+STEM+'_run',stage,
                    '--config',str(path),'--output',str(folder)],root,log,
                    output/'preparation'/(stage+'.worker.json'),folder)
                results[stage]=saved(folder) if (folder/'attempt.json').exists() else dict(
                    status='PREPARATION_FAILED',returncode=code,log=str(log),physical_calls_this_invocation=0)
            except Exception as error:
                results[stage]=dict(status='PREPARATION_FAILED',error=type(error).__name__+': '+str(error))
                print(f'[{stage}] {results[stage]["error"]}',flush=True)
    result=dict(status='RUN_ALL_STATUS',stages=results)
    report=load_module(root/'scripts'/f'{STEM}_report.py','joint_a_report')
    result['summary']=report.build(output,results)
    return result
