"""Standalone stdlib bootstrap for a thin user-run Colab notebook."""
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

WORK = Path('/content/video-wm-native-baselines-paper-v3')
OUTPUT = Path('/content/drive/MyDrive/Video-WM_Result/baselines/native-wam-pixelseal-paper-fpr1-v3/attempt01')
SOURCE_REF = 'dev/local-joint-state-payload-v1'


def source_download(repository, ref, destination):
    from urllib.parse import quote
    destination = Path(destination)
    if destination.is_dir():
        return destination
    WORK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=WORK, prefix='source-') as temporary:
        stage = Path(temporary); archive = stage/'source.zip'; unpack = stage/'unpacked'
        urllib.request.urlretrieve(f'https://codeload.github.com/{repository}/zip/refs/heads/{quote(ref, safe="/")}', archive)
        unpack.mkdir()
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                name = member.filename.replace('\\', '/')
                target = (unpack/name).resolve()
                if (name.startswith('/') or any(':' in x for x in name.split('/'))
                        or not target.is_relative_to(unpack.resolve()) or stat.S_ISLNK(member.external_attr >> 16)):
                    raise ValueError('Unsafe source archive member: '+name)
            zipped.extractall(unpack)
        folders = list(unpack.iterdir())
        if len(folders) != 1 or not folders[0].is_dir():
            raise ValueError('Invalid source archive layout')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folders[0]), str(destination))
    return destination


def stop_group(worker):
    """Kill descendants even if the process leader already returned or failed."""
    if worker is None:
        return
    try:
        os.killpg(worker.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(worker.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    worker.wait(timeout=5)


def run_colab(source_directory=None, source_ref=SOURCE_REF):
    WORK.mkdir(parents=True, exist_ok=True); OUTPUT.mkdir(parents=True, exist_ok=True)
    worker = None
    try:
        source = Path(source_directory) if source_directory else source_download('RICHAAARC/SC-SSTW', source_ref, WORK/'project')
        sys.path.insert(0, str(source))
        from runtime.wan.native_baselines_paper_v3 import DEFAULT_CONFIG, load_inputs
        from runtime.wan.native_baseline_storage_v3 import state
        from runtime.wan.native_baseline_storage_v1 import write_json
        config, sources = load_inputs(DEFAULT_CONFIG)
        if (OUTPUT/'sources.json').exists():
            sources = json.loads((OUTPUT/'sources.json').read_text())
        pending = any(state(OUTPUT/'units'/kind/s['id']) == 'NEVER_STARTED'
                      for s in sources for kind in ['prepare', 'wam', 'pixelseal'])
        if not pending and (OUTPUT/'result.json').exists() and (OUTPUT/'evaluation.json').exists():
            display = show_results()
            return dict(status='READ_ONLY_OUTPUT_UNREADABLE' if display['read_errors'] else 'READ_ONLY_COMPLETED',
                        output=str(OUTPUT), display_read_errors=display['read_errors'])
        vendor = WORK/'vendor'
        environment = os.environ.copy(); environment['PYTHONPATH'] = str(source)
        os.environ.update(HF_HOME=str(WORK/'cache/huggingface'), TORCH_HOME=str(WORK/'cache/torch'))
        environment.update(HF_HOME=os.environ['HF_HOME'], TORCH_HOME=os.environ['TORCH_HOME'])
        if pending:
            source_download('facebookresearch/videoseal', 'main', vendor/'videoseal')
            source_download('facebookresearch/watermark-anything', 'main', vendor/'watermark-anything')
            from runtime.wan.native_baseline_temporal_colab_v2 import _install_dependencies
            environment = _install_dependencies(source, vendor)
            weight = vendor/'watermark-anything/checkpoints/wam_mit.pth'
            if not weight.is_file():
                urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth', weight)
        # Statistics dependency is checked by actual import, without version comparison.
        checked = subprocess.run([sys.executable, '-c', 'import numpy, scipy.stats'], capture_output=True)
        if checked.returncode:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'numpy', 'scipy'], check=True)
        args = [sys.executable, '-u', str(source/'experiments/wan_state_clock/native_baselines_paper_fpr1_v3_run.py'),
                '--output', str(OUTPUT), '--vendor', str(vendor)]
        with (OUTPUT/'runner.log').open('a') as log:
            worker = subprocess.Popen(args, cwd=source, env=environment, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            while worker.poll() is None:
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    progress = OUTPUT/'progress.json'
                    print(progress.read_text() if progress.is_file() else 'Worker running; see runner.log', flush=True)
        status = 'PHYSICAL_PLAN_CLOSED' if worker.returncode == 0 else 'INTERRUPTED_RETAINED'
        write_json(OUTPUT/'bootstrap.json', dict(status=status, returncode=worker.returncode, at=time.time()))
    except BaseException as e:
        status = 'PENDING_INPUT' if 'PENDING_INPUT' in str(e) else 'INTERRUPTED_RETAINED'
        value = dict(status=status, error=f'{type(e).__name__}: {e}', at=time.time())
        with (OUTPUT/'bootstrap_failure.json').open('w') as f:
            json.dump(value, f, ensure_ascii=False); f.flush(); os.fsync(f.fileno())
        print(value)
    finally:
        stop_group(worker)
    show_results()
    return dict(status=status, output=str(OUTPUT))


def show_results():
    print('保留结果目录:', OUTPUT)
    read_errors = []
    def read_saved(path):
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError('Saved receipt must be a JSON object')
            return value
        except (OSError, ValueError) as error:
            read_errors.append(dict(file=path.name, error=f'{type(error).__name__}: {error}'))
            print('结果文件读取失败，原文件保留:', path.name, str(error))
            return None
    for name in ['result.json', 'interruption.json', 'bootstrap_failure.json']:
        path = OUTPUT/name
        if path.is_file():
            value = read_saved(path)
            if value is not None:
                print(name, json.dumps({k: v for k, v in value.items() if k != 'physical_calls'}, ensure_ascii=False, indent=2))
    path = OUTPUT/'evaluation.json'
    if path.is_file():
        value = read_saved(path)
        if value is not None:
            try:
                print('CAL状态:', {m: c['status'] for m, c in value['calibration'].items()})
                for r in value['summaries']:
                    if r['channel'] == 'FAMILY_22':
                        print(r['method'], r['arm'], r['verification'],
                              {k: r[k] for k in ('planned', 'positive', 'unknown', 'rate_lower', 'rate_upper', 'conservative_cp95')})
                print('实际TPR/FPR、缺失与区间分别解释；nominal 1%不是已证明总体保证。')
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                read_errors.append(dict(file=path.name, error=f'{type(error).__name__}: {error}'))
                print('结果字段读取失败，原文件保留:', path.name, str(error))
    return dict(read_errors=read_errors, physical_retry=False)
