"""Optional notebook helper. CPU-only setup; consumed output uses stdlib only."""
from pathlib import Path
import importlib.util
import subprocess
import sys
import shutil

_spec=importlib.util.spec_from_file_location('qim_common_entry',Path(__file__).with_name('b_framewise_qim_v1_entry.py'))
common=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(common)
STEM='b_framewise_qim_time_recovery_v1'
CACHE_FILES=('scripts/'+STEM+'_entry.py','scripts/b_framewise_qim_v1_entry.py',
             'experiments/wan_state_clock/'+STEM+'_records.py','experiments/wan_state_clock/'+STEM+'_report.py')
REQUIRED=('experiments/wan_state_clock/'+STEM+'_run.py','experiments/wan_state_clock/'+STEM+'_experiment.py',
          'experiments/wan_state_clock/'+STEM+'_records.py','experiments/wan_state_clock/'+STEM+'_evaluation.py',
          'runtime/wan/'+STEM+'.py','main/tube_state/'+STEM+'.py')


def setup_dependencies(logs):
    missing=[name for name in ('numpy','psutil') if importlib.util.find_spec(name) is None]
    with (logs/'setup.log').open('a') as stream:
        if missing: subprocess.run([sys.executable,'-m','pip','install',*missing],stdout=stream,stderr=subprocess.STDOUT,check=True)
        if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
            subprocess.run(['apt-get','update','-qq'],stdout=stream,stderr=subprocess.STDOUT,check=True)
            subprocess.run(['apt-get','install','-y','ffmpeg'],stdout=stream,stderr=subprocess.STDOUT,check=True)


class Session(common.Session):
    def __init__(self,qim_media,base_media,output,source_directory=None,source_ref='dev/local-joint-state-payload-v1'):
        super().__init__(qim_media,output,source_ref,source_directory,scratch='/content/video-wm-time-recovery-v1')
        self.base=Path(base_media)

    def _saved_report(self):
        try: self.report=common.load_report(self.cache,STEM+'_records',STEM+'_report')
        except Exception as error: print('详细报告不可用:',str(error))

    def execute(self):
        self.output=self.output.expanduser().resolve()
        self.cache=self.output.with_name(self.output.name+'-entry-cache');self.logs=self.output.with_name(self.output.name+'-logs')
        if (self.output/'attempt.json').exists(): self._saved_report();return
        if common.existing_output(self.output): self.preparation_error='OUTPUT_CONFLICT';return
        try:
            # Before any source/cache/setup write, protect both original attempt directories.
            protected=[self.source.expanduser().resolve().parent,self.base.expanduser().resolve().parent]
            writes=[self.output,self.cache,self.logs,self.output.with_name(self.output.name+'-preflight-v1'),
                    Path(self.source_directory) if self.source_directory is not None else self.scratch]
            for path in writes:
                p=path.expanduser().resolve()
                if any(p==s or p.is_relative_to(s) or s.is_relative_to(p) for s in protected): raise ValueError('write path overlaps input attempt')
            root=common.prepare_source(self.source_directory,self.source_ref,self.scratch,REQUIRED)
            records=common.load_saved_module(root,STEM+'_records')
            preflight=records.preflight(self.source,self.base,self.output)
            print('预检:',preflight)
            if preflight['status']!='PREFLIGHT_READY': return
            try: common.cache_entry(root,self.cache,CACHE_FILES)
            except OSError as error: print('可选缓存未写入:',str(error))
            try: self.report=common.load_report(root,STEM+'_records',STEM+'_report')
            except Exception as error: print('可选报告不可用:',str(error))
            if records.output_state(self.output)!='NEW': return
            self.logs.mkdir(parents=True,exist_ok=True);setup_dependencies(self.logs)
            if records.output_state(self.output)!='NEW': return
        except Exception as error: self.preparation_error=f'{type(error).__name__}: {error}';print(self.preparation_error);return
        self._worker(root)

    def _command(self,root):
        return [sys.executable,'-u','-m','experiments.wan_state_clock.'+STEM+'_run',
                '--qim-media',str(self.source),'--base-media',str(self.base),'--output',str(self.output)]
