"""Optional notebook entry: ordinary source/cache, setup, worker and saved display.

This module imports only the standard library. The experiment CLI never imports
it. An existing output always returns through the read-only path before setup.
"""
from pathlib import Path
import importlib
import importlib.util
import json
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import types
import urllib.request
import uuid
import zipfile

ENTRY = 'scripts/b_framewise_qim_v1_entry.py'
CONFIG = 'experiments/wan_state_clock/configs/b_framewise_qim_v1.json'
CACHE_FILES = (ENTRY, 'experiments/wan_state_clock/b_framewise_qim_v1_records.py',
               'experiments/wan_state_clock/b_framewise_qim_v1_report.py')


def existing_output(output):
    output = Path(output)
    return output.exists() and (not output.is_dir() or any(output.iterdir()))


def raw_status(output):
    """Fallback for old outputs or unavailable optional reporting code; never write."""
    print('详细报告代码不可用；仅显示已有原始状态，不下载、不安装、不重跑。')
    for name in ('attempt', 'result'):
        try:
            value = json.loads((Path(output)/(name+'.json')).read_text())
            print(name, value)
        except (OSError, ValueError) as error:
            print(name, '未知：', str(error))


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_report(root, record_module='b_framewise_qim_v1_records', report_module='b_framewise_qim_v1_report'):
    # The ordinary package tree can be either editable source or the small cache.
    root = Path(root)
    for relative in (f'experiments/wan_state_clock/{record_module}.py', f'experiments/wan_state_clock/{report_module}.py'):
        if not (root/relative).is_file():
            raise FileNotFoundError('report code unavailable: '+str(root/relative))
    return load_saved_module(root, report_module)


def load_saved_module(root, module):
    # Isolate ordinary files so an earlier session's sys.modules
    # cannot silently supply a report or records module from another directory.
    name = '_qim_saved_report_'+uuid.uuid4().hex
    package = types.ModuleType(name)
    package.__path__ = [str(Path(root)/'experiments/wan_state_clock')]
    sys.modules[name] = package
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module(name+'.'+module)
    finally:
        sys.dont_write_bytecode = previous
        for key in list(sys.modules):
            if key == name or key.startswith(name+'.'): sys.modules.pop(key, None)


def cache_entry(source_root, cache, files=CACHE_FILES):
    """Copy ordinary readable modules outside OUTPUT; no version/digest admission."""
    for relative in files:
        destination = cache/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (source_root/relative).resolve() != destination.resolve():
            shutil.copyfile(source_root/relative, destination)
    for relative in ('experiments/__init__.py', 'experiments/wan_state_clock/__init__.py'):
        path = cache/relative
        if not path.exists(): path.write_text('')


def prepare_source(source_directory, source_ref, scratch, required_files=None):
    required_files = required_files or (CONFIG, 'experiments/wan_state_clock/b_framewise_qim_v1_run.py',
                                       'runtime/wan/b_framewise_qim_v1_backend.py', 'scripts/b_framewise_qim_v1_setup.py')
    if source_directory is not None:
        root = Path(source_directory).expanduser().resolve()
    else:
        scratch.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix='run-', dir=scratch))
        url = 'https://codeload.github.com/RICHAAARC/SC-SSTW/zip/refs/heads/'+source_ref
        archive = workspace/'source.zip'
        urllib.request.urlretrieve(url, archive)
        destination = (workspace/'source').resolve(); destination.mkdir()
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                name = member.filename.replace('\\', '/')
                target = (destination/name).resolve()
                if (name.startswith('/') or any(':' in part for part in name.split('/'))
                        or not target.is_relative_to(destination)
                        or stat.S_ISLNK(member.external_attr >> 16)):
                    raise ValueError('unsafe source archive path: '+member.filename)
            source.extractall(destination)
        entries = list(destination.glob('*/'+required_files[0]))
        if len(entries) != 1: raise FileNotFoundError('available source lacks the QIM runner: '+url)
        root = entries[0]
        for _ in Path(required_files[0]).parts: root = root.parent
    for relative in required_files:
        if not (root/relative).is_file(): raise FileNotFoundError('usable QIM source missing: '+str(root/relative))
    return root


class Session:
    def __init__(self, source, output, source_ref, source_directory=None,
                 scratch='/content/video-wm-framewise-qim-v1'):
        self.source, self.output = Path(source), Path(output)
        self.source_ref, self.source_directory = source_ref, source_directory
        self.scratch = Path(scratch)
        self.cache = self.output.with_name(self.output.name+'-entry-cache')
        self.logs = self.output.with_name(self.output.name+'-logs')
        self.report = None
        self.preparation_error = None

    def _saved_report(self):
        try: self.report = load_report(self.cache)
        except Exception as error:
            self.preparation_error = f'{type(error).__name__}: {error}'
            print(self.preparation_error)

    def execute(self):
        if existing_output(self.output):
            print('已有输出：只读恢复；跳过下载、setup、模型和科学调用。')
            self._saved_report()
            return
        missing = [str(path) for path in (self.source/'private/b2-m3-e01/P0_terminal.pt',
                                         self.source/'protocol.json') if not path.is_file()]
        if missing:
            self.preparation_error = 'INPUT_MISSING: '+', '.join(missing)
            print(self.preparation_error)
            return
        try:
            root = prepare_source(self.source_directory, self.source_ref, self.scratch)
            print('使用源码：', root)
            try: cache_entry(root, self.cache)
            except OSError as error: print('入口缓存写入失败；以后可只读原始状态：', str(error))
            try: self.report = load_report(root)
            except Exception as error:
                print('可选报告不可用，继续固定运行；结果仍保存：', str(error))
            if existing_output(self.output): return
            self.logs.mkdir(parents=True, exist_ok=True)
            setup = load_file(root/'scripts/b_framewise_qim_v1_setup.py', 'qim_dependency_setup')
            setup.setup_dependencies(root, self.logs)
            if existing_output(self.output): return
        except Exception as error:
            self.preparation_error = f'{type(error).__name__}: {error}'
            print('准备失败，未启动：', self.preparation_error)
            return
        self._worker(root)

    def _command(self, root):
        return [sys.executable, '-u', '-m',
                'experiments.wan_state_clock.b_framewise_qim_v1_run', '--source', str(self.source),
                '--output', str(self.output), '--config', str(root/CONFIG)]

    def _worker(self, root):
        with (self.logs/'runner.log').open('a') as stream:
            worker = subprocess.Popen(self._command(root),
                cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while worker.poll() is None:
                time.sleep(30)
                if self.report is None:
                    print('worker 运行中；详细报告不可用，原始记录目录:', self.output, flush=True)
                    continue
                try:
                    snapshot = self.report.recovery_summary(self.report.recover(self.output))
                    print('记录状态:', snapshot['state_counts'], '求解:', snapshot['solver'].get('status'),
                          '迭代:', snapshot['solver'].get('iterations'), flush=True)
                except Exception as error:
                    print('当前记录读取未决:', str(error), flush=True)
        except KeyboardInterrupt:
            worker.terminate()
            try: worker.wait(timeout=30)
            except subprocess.TimeoutExpired: worker.kill(); worker.wait()
            print('子进程已停止；已有尝试保持消耗，下次 Run all 只读。')
            raise
        print('worker exit:', worker.returncode, '日志:', self.logs/'runner.log')

    def display(self):
        if existing_output(self.output):
            if self.report is None: self._saved_report()
            if self.report is None: raw_status(self.output)
            else:
                try: return self.report.display(self.output)
                except Exception as error:
                    print('详细报告读取未决：', str(error)); raw_status(self.output)
        else: print('没有尝试记录。准备状态:', self.preparation_error or '尚未启动')
        print('输出目录:', self.output)
