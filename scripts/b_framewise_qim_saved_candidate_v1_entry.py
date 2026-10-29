"""Optional thin-notebook helper for the independent saved-candidate attempt."""
from pathlib import Path
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location('qim_common_entry', Path(__file__).with_name('b_framewise_qim_v1_entry.py'))
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
STEM = 'b_framewise_qim_saved_candidate_v1'
CACHE_FILES = ('scripts/'+STEM+'_entry.py', 'scripts/b_framewise_qim_v1_entry.py',
               'experiments/wan_state_clock/b_framewise_qim_v1_records.py',
               'experiments/wan_state_clock/'+STEM+'_records.py',
               'experiments/wan_state_clock/'+STEM+'_report.py')


class Session(common.Session):
    def __init__(self, saved_attempt, truth_source, output, model_id=None,
                 source_directory=None, source_ref='dev/local-joint-state-payload-v1'):
        super().__init__(saved_attempt, output, source_ref, source_directory)
        self.truth_source, self.model_id = Path(truth_source), model_id

    def _saved_report(self):
        try: self.report = common.load_report(self.cache, STEM+'_records', STEM+'_report')
        except Exception as error: print('详细报告代码不可用：', str(error))

    def execute(self):
        saved, output = self.source.resolve(), self.output.resolve()
        protected = (saved, self.truth_source.resolve())
        if any(output == source or output.is_relative_to(source) for source in protected):
            raise ValueError('new output must be outside read-only inputs')
        self.source, self.output = saved, output
        self.cache = output.with_name(output.name+'-entry-cache')
        self.logs = output.with_name(output.name+'-logs')
        if (self.output/'attempt.json').exists():
            self._saved_report(); return
        if common.existing_output(self.output):
            self.preparation_error = 'OUTPUT_CONFLICT: existing content without attempt; no overwrite'
            print(self.preparation_error); return
        try:
            writable = [output, self.cache, self.logs, output.with_name(output.name+'-preflight-v1')]
            writable += [Path(self.source_directory) if self.source_directory is not None else self.scratch]
            for path in writable:
                resolved = path.expanduser().resolve()
                if any(resolved == source or resolved.is_relative_to(source) for source in protected):
                    raise ValueError('write destination must be outside read-only inputs: '+str(resolved))
            root = common.prepare_source(self.source_directory, self.source_ref, self.scratch)
            records = common.load_saved_module(root, STEM+'_records')
            result = records.preflight(self.source, self.output, self.truth_source)
            print('预检:', result['status'], result['errors'], '固定分母:', result['denominators'])
            if result['status'] != 'PREFLIGHT_READY': return
            try: common.cache_entry(root, self.cache, CACHE_FILES)
            except OSError as error: print('可选入口缓存写入未决：', str(error))
            try: self.report = common.load_report(root, STEM+'_records', STEM+'_report')
            except Exception as error: print('可选报告不可用，固定运行仍可继续：', str(error))
            if records.output_state(self.output) != 'NEW': return
            self.logs.mkdir(parents=True, exist_ok=True)
            setup = common.load_file(root/'scripts/b_framewise_qim_v1_setup.py', 'qim_saved_setup')
            setup.setup_dependencies(root, self.logs)
            if records.output_state(self.output) != 'NEW': return
        except Exception as error:
            self.preparation_error = f'{type(error).__name__}: {error}'
            print('准备失败，未启动：', self.preparation_error); return
        self._worker(root)

    def _command(self, root):
        command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.'+STEM+'_run',
                   '--saved-attempt', str(self.source), '--truth-source', str(self.truth_source), '--output', str(self.output)]
        if self.model_id is not None: command += ['--model-id', str(self.model_id)]
        return command
