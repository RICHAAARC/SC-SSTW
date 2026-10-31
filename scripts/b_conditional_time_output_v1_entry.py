"""Ordinary source/setup/worker entry; consumed Drive output recovers before setup."""
from pathlib import Path
import importlib.util
import json
import sys

STEM = 'b_conditional_time_output_v1'
SOURCE_REF = 'dev/local-joint-state-payload-v1'
CACHE_FILES = ('scripts/'+STEM+'_entry.py', 'scripts/b_framewise_qim_v1_entry.py',
               'experiments/wan_state_clock/'+STEM+'_records.py',
               'experiments/wan_state_clock/b_minimal_l_v1_records.py')


def _load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def saved_status(output):
    saved, errors = {}, {}
    for name in ('attempt', 'state', 'counts', 'resources', 'evaluation', 'result'):
        try:
            value = _load(Path(output)/(name+'.json'))
            if not isinstance(value, dict):
                raise ValueError('saved object required')
            saved[name] = value
        except (OSError, ValueError) as error:
            errors[name] = str(error)
    return dict(status='READ_ONLY_EXISTING_ATTEMPT', physical_calls_this_invocation=0, saved=saved, read_errors=errors)


def _existing(output,recover=saved_status):
    path = Path(output)
    if (path/'attempt.json').exists():
        return recover(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        return dict(status='OUTPUT_CONFLICT', physical_calls_this_invocation=0)
    return None


def _common():
    path = Path(__file__).with_name('b_framewise_qim_v1_entry.py')
    spec = importlib.util.spec_from_file_location('conditional_source_common', path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


class Session:
    stem = STEM
    cache_files = CACHE_FILES
    recover = staticmethod(saved_status)

    def __init__(self, common, config, config_path, output, source_directory, source_ref, scratch):
        self.common, self.config = common, config
        self.config_path, self.output = Path(config_path).resolve(), Path(output).expanduser().resolve()
        self.source_directory, self.source_ref = source_directory, source_ref
        self.scratch = Path(scratch).expanduser().resolve()
        self.cache = self.output.with_name(self.output.name+'-entry-cache')
        self.logs = self.output.with_name(self.output.name+'-logs')
        self.report = None  # Reuse the exercised worker lifecycle; saved JSON is authoritative.

    def _command(self, root):
        return [sys.executable, '-B', '-u', '-m', 'experiments.wan_state_clock.'+self.stem+'_run',
                '--config', str(self.config_path), '--output', str(self.output)]

    def required_files(self):
        return ('experiments/wan_state_clock/'+self.stem+'_run.py',
                'runtime/wan/'+self.stem+'.py','scripts/'+self.stem+'_setup.py')

    def prepare_dependencies(self,setup,root):
        setup.setup_dependencies(root,self.logs,self.config['entry'])

    def execute(self):
        existing = _existing(self.output,self.recover)
        if existing is not None:
            return existing
        try:
            records = self.common.load_saved_module(Path(__file__).resolve().parents[1], self.stem+'_records')
            check = records.preflight(self.config, self.output)
            if check['status'] != 'PREFLIGHT_READY':
                return check
            # Protect scientific inputs before downloads/cache/log writes, not by source identity.
            writable = [self.cache, self.logs]
            if self.source_directory is None:
                writable.append(self.scratch)
            for path in writable:
                protected_roots=check.get('protected_roots',[str(Path(filename).parent) for filename in check['input_paths']])
                for protected in map(Path,protected_roots):
                    if path == protected or path.is_relative_to(protected) or protected.is_relative_to(path):
                        raise ValueError('preparation write directory overlaps read-only input: '+str(path))
                if self.config_path == path or self.config_path.is_relative_to(path):
                    raise ValueError('preparation write directory contains the input configuration')
            root = self.common.prepare_source(self.source_directory, self.source_ref, self.scratch,required_files=self.required_files())
            print('使用源码：', root, flush=True)
            existing = _existing(self.output,self.recover)
            if existing is not None:
                return existing
            try:
                self.common.cache_entry(root, self.cache, self.cache_files)
            except OSError as error:
                print('可选入口缓存未写入：', str(error), flush=True)
            self.logs.mkdir(parents=True, exist_ok=True)
            setup = self.common.load_file(root/'scripts'/(self.stem+'_setup.py'), 'conditional_dependency_setup')
            self.prepare_dependencies(setup,root)
            existing = _existing(self.output,self.recover)
            if existing is not None:
                return existing
        except Exception as error:
            return dict(status='PREPARATION_FAILED', error=f'{type(error).__name__}: {error}',
                        physical_calls_this_invocation=0, physical_attempt_reserved=False)
        # Inherits the already exercised process lifecycle and terminate/wait/kill on interruption.
        try:
            self.common.Session._worker(self, root)
            result = _load(self.output/'result.json')
            if not isinstance(result, dict):
                raise ValueError('runner result object required')
            return result
        except Exception as error:
            return dict(status='RUNNER_RESULT_UNRESOLVED', error=str(error),
                        attempt_consumed=(self.output/'attempt.json').exists(), log=str(self.logs/'runner.log'))


def launch(source_directory=None, config_path=None, output=None, source_ref=SOURCE_REF,
           scratch='/content/video-wm-conditional-time-output-v1',session_type=Session):
    if output:
        existing = _existing(Path(output).expanduser().resolve(),session_type.recover)
        if existing is not None:
            return existing
    if config_path is None:
        return dict(status='UNCONFIGURED', physical_calls_this_invocation=0)
    try:
        config_path = Path(config_path).expanduser().resolve()
        config = _load(config_path)
        if not isinstance(config, dict):
            raise ValueError('configuration object required')
        output = output or config.get('output')
        if not output:
            return dict(status='UNCONFIGURED', physical_calls_this_invocation=0)
        existing = _existing(Path(output).expanduser().resolve(),session_type.recover)
        if existing is not None:
            return existing
        return session_type(_common(), config, config_path, output, source_directory, source_ref, scratch).execute()
    except (OSError, ValueError, ImportError) as error:
        return dict(status='PREPARATION_FAILED', error=f'{type(error).__name__}: {error}',
                    physical_calls_this_invocation=0, physical_attempt_reserved=False)
